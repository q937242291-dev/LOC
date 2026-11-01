"""Paired complete-attempt counts and task-cluster uncertainty contracts."""
import math
from fractions import Fraction
from itertools import product
import unittest
from unittest.mock import Mock, patch

from locus.analysis import (compare_operating_points, highest_baseline_comparison,
                            paired_comparison, repeat_statistics, resolution_interval,
                            task_cluster_bootstrap)


def attempt(task, repeat, system, resolved, *, benchmark="test", model="m"):
    return dict(instance_id=task, repeat=repeat, system=system, resolved=resolved,
                benchmark=benchmark, model=model)


def paired_rows(outcomes, repeats=range(5)):
    return [attempt(task, repeat, system, values[index])
            for task, values in outcomes.items() for repeat in repeats
            for index, system in enumerate(("left", "right"))]


def compare(rows, **kwargs):
    return paired_comparison(rows, "left", "right", samples=40, seed=7, **kwargs)[0]


def exact_quantile(values, probability):
    """Independent rational interpolation over a finite resampling population."""
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = position.numerator // position.denominator
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def enumerated_random(tasks, instances):
    """Give each bootstrap invocation every ordered task draw exactly once."""
    def factory(seed):
        rng = Mock()
        rng.choices.side_effect = iter(product(tasks, repeat=len(tasks)))
        instances.append(rng)
        return rng
    return factory


class ExactResamplingTests(unittest.TestCase):
    def test_all_27_task_resamples_match_independent_paired_arithmetic(self):
        tasks = ("a", "b", "c")
        left = ((1, 1, 1, 0, 0), (0, 1, 0, 1, 0), (0, 0, 0, 0, 1))
        right = ((0, 0, 1, 0, 0), (1, 0, 0, 0, 0), (1, 1, 1, 0, 0))
        rows = [attempt(task, repeat, system, bool(outcomes[index][repeat]))
                for index, task in enumerate(tasks) for repeat in range(5)
                for system, outcomes in (("left", left), ("right", right))]
        # Each integer indexes an entire five-repeat vector in both methods.
        reference = []
        reference_runs = [[] for _ in range(5)]
        for draw in product(range(3), repeat=3):
            reference.append(Fraction(100, 15) *
                sum(sum(left[index]) - sum(right[index]) for index in draw))
            for repeat in range(5):
                reference_runs[repeat].append(Fraction(100, 3) *
                    sum(left[index][repeat] - right[index][repeat] for index in draw))
        instances = []
        with patch("locus.analysis.random.Random", side_effect=enumerated_random(tasks, instances)):
            result = paired_comparison(rows, "left", "right", samples=27, confidence=.5)[0]
        self.assertEqual(result["left"]["successes"], 6)
        self.assertEqual(result["right"]["successes"], 5)
        self.assertAlmostEqual(result["resolution_delta_pp"], 100 / 15)
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].choices.call_count, 27)
        intervals = [result["delta_interval"]] + [
            result["by_repeat"][str(repeat)]["delta_interval"] for repeat in range(5)]
        for interval, values in zip(intervals, [reference] + reference_runs):
            self.assertEqual(interval["valid_samples"], 27)
            self.assertAlmostEqual(interval["lower"], float(exact_quantile(values, Fraction(1, 4))))
            self.assertAlmostEqual(interval["upper"], float(exact_quantile(values, Fraction(3, 4))))

    def test_all_four_resamples_reselect_highest_baseline(self):
        outcomes = {"LOCUS": ((1, 1, 1, 0, 0), (1, 1, 1, 1, 0)),
                    "A": ((1, 1, 1, 1, 1), (0, 0, 0, 0, 0)),
                    "B": ((0, 0, 0, 0, 0), (1, 1, 1, 1, 1)),
                    "C": ((1, 1, 0, 0, 0), (1, 1, 0, 0, 0))}
        rows = [dict(attempt(task, repeat, system, bool(vectors[index][repeat])),
                     ledger={"cost": 1 if system == "A" else 2})
                for system, vectors in outcomes.items() for index, task in enumerate(("a", "b"))
                for repeat in range(5)]
        instances = []
        with patch("locus.analysis.random.Random", side_effect=enumerated_random(("a", "b"), instances)):
            result = highest_baseline_comparison(rows, samples=4, confidence=.5)[0]
        # aa, ab, ba, bb have deltas -40, 20, 20, -20 respectively.
        self.assertEqual(result["highest_resolution_baseline"], "A")
        self.assertEqual(result["resolution_delta_pp"], 20)
        self.assertEqual(result["delta_interval"]["lower"], -25)
        self.assertEqual(result["delta_interval"]["upper"], 20)
        self.assertEqual([rng.choices.call_count for rng in instances], [4, 4])
        expected = ((37.5, 62.5), (37.5, 62.5), (37.5, 62.5), (-25, 25), (-62.5, -37.5))
        for repeat, (low, high) in enumerate(expected):
            interval = result["by_repeat"][str(repeat)]["delta_interval"]
            self.assertEqual((interval["lower"], interval["upper"]), (low, high))
            self.assertTrue(interval["conditional_on_comparator"])

    def test_same_task_label_in_different_benchmarks_is_two_clusters(self):
        rows = [dict(instance_id="same", benchmark="one", value=0),
                dict(instance_id="same", benchmark="two", value=1)]
        instances = []
        keys = (("one", "same"), ("two", "same"))
        with patch("locus.analysis.random.Random", side_effect=enumerated_random(keys, instances)):
            interval = task_cluster_bootstrap(rows, lambda sample: sum(r["value"] for r in sample) / len(sample),
                                              samples=4, confidence=.5)
        self.assertEqual(interval["task_clusters"], 2)
        self.assertEqual(interval["estimate"], .5)
        self.assertEqual((interval["lower"], interval["upper"]), (.375, .625))

    def test_resolution_interval_rejects_duplicates_and_mixed_methods(self):
        row = attempt("a", 0, "LOCUS", True)
        for records in ([row, dict(row)], [row, dict(row, system="locus")]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                resolution_interval(records, samples=4)
        for options in (dict(samples=True), dict(samples=1.5), dict(confidence=True), dict(seed=False)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                task_cluster_bootstrap([row], lambda _: 1., **options)
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                task_cluster_bootstrap([row], lambda _: value, samples=4)


class OperatingPointTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [dict(benchmark="b", model="m", system="LOCUS", resolution_pct=60,
                     normalized_cost_mean=2, tokens_mean=100),
                dict(benchmark="b", model="m", system="high", resolution_pct=70,
                     normalized_cost_mean=None, tokens_mean=200),
                dict(benchmark="b", model="m", system="priced", resolution_pct=40,
                     normalized_cost_mean=5, tokens_mean=300)]

    def test_cheapest_requires_every_candidate_cost(self):
        rows = self.rows()
        comparison = compare_operating_points(rows)[0]
        self.assertEqual(comparison["highest_resolution_baseline"], "high")
        self.assertEqual(comparison["resolution_delta_pp"], -10)
        self.assertFalse(comparison["baseline_costs_complete"])
        self.assertIsNone(comparison["lowest_cost_baseline"])
        self.assertIsNone(comparison["cost_saved_vs_cheapest_pct"])
        rows[1]["normalized_cost_mean"] = 1
        comparison = compare_operating_points(rows)[0]
        self.assertTrue(comparison["baseline_costs_complete"])
        self.assertEqual(comparison["lowest_cost_baseline"], "high")
        self.assertEqual(comparison["cost_saved_vs_cheapest_pct"], -100)

    def test_summary_group_duplicates_are_not_treated_as_baselines(self):
        rows = self.rows()
        for row in rows:
            with self.subTest(system=row["system"]), self.assertRaisesRegex(ValueError, "Duplicate summary"):
                compare_operating_points(rows + [dict(row)])

    def test_system_case_matches_the_exact_paired_comparison_contract(self):
        rows = self.rows()
        rows[0]["system"] = "locus"
        self.assertEqual(compare_operating_points(rows), [])
        comparison = compare_operating_points(rows, locus_name="locus")[0]
        self.assertEqual(comparison["resolution_delta_pp"], -10)
        self.assertIsNone(compare_operating_points(rows, locus_name="locus",
                                                  quality_reference="HIGH")[0]["quality_floor_met"])

    def test_invalid_summary_numbers_do_not_select_a_comparator(self):
        for field, value in (("resolution_pct", float("nan")), ("resolution_pct", True),
                             ("resolution_pct", 101), ("normalized_cost_mean", -1),
                             ("normalized_cost_mean", float("inf"))):
            rows = self.rows()
            rows[1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                compare_operating_points(rows)


class PairedAnalysisTests(unittest.TestCase):
    def test_exact_per_repeat_success_and_four_cell_counts(self):
        rows = paired_rows({"a": (True, True), "b": (True, False),
                            "c": (False, True), "d": (False, False)})
        result = compare(rows, expected_repeats=range(5))
        self.assertTrue(result["complete"])
        self.assertTrue(result["five_repeat_complete"])
        self.assertEqual(result["left"]["successes"], 10)
        self.assertEqual(result["left"]["attempts"], 20)
        self.assertEqual(result["right"]["successes"], 10)
        self.assertEqual(result["resolution_delta_pp"], 0)
        for cell in ("both", "left_only", "right_only", "neither"):
            self.assertEqual(result[cell], 5)
            self.assertEqual(result["by_repeat"]["0"][cell], 1)
        self.assertEqual(result["by_repeat"]["0"]["left"]["attempts"], 4)
        self.assertEqual(result["by_repeat"]["0"]["left"]["successes"], 2)
        self.assertEqual(result["by_repeat"]["0"]["delta_interval"]["valid_samples"], 40)

    def test_delta_direction_is_left_minus_explicit_named_right(self):
        rows = paired_rows({"a": (True, False), "b": (True, True)})
        result = compare(rows)
        self.assertEqual(result["resolution_delta_pp"], 50)
        self.assertEqual(result["comparator_system"], "right")
        reverse = paired_comparison(rows, "right", "left", samples=40, seed=7)[0]
        self.assertEqual(reverse["resolution_delta_pp"], -50)
        self.assertEqual(reverse["delta_interval"]["lower"], -result["delta_interval"]["upper"])

    def test_false_grades_are_known_zero_successes(self):
        result = compare(paired_rows({"a": (False, False)}))
        self.assertTrue(result["complete"])
        self.assertEqual(result["left"]["successes"], 0)
        self.assertEqual(result["left"]["resolution_pct"], 0)
        self.assertEqual(result["neither"], 5)
        self.assertEqual(result["delta_interval"]["lower"], 0)
        self.assertEqual(result["delta_interval"]["upper"], 0)

    def test_ungraded_pair_is_not_counted_as_a_failure_or_dropped(self):
        rows = paired_rows({"a": (True, False), "b": (False, True)})
        rows[0]["resolved"] = None
        result = compare(rows)
        self.assertFalse(result["complete"])
        self.assertEqual(result["expected_pairs"], 10)
        self.assertEqual(result["graded_pairs"], 9)
        self.assertIsNone(result["left"]["successes"])
        self.assertEqual(result["left"]["known_successes"], 4)
        self.assertEqual(result["left"]["ungraded_attempts"], 1)
        self.assertIsNone(result["resolution_delta_pp"])
        self.assertIsNone(result["left_only"])
        self.assertEqual(result["delta_interval"]["valid_samples"], 0)
        self.assertIsNone(result["delta_interval"]["lower"])
        self.assertTrue(result["by_repeat"]["1"]["complete"])
        self.assertEqual(result["by_repeat"]["1"]["delta_interval"]["valid_samples"], 40)
        self.assertIsNone(result["by_repeat"]["0"]["delta_interval"]["lower"])

    def test_missing_partner_has_separate_missing_counts(self):
        rows = paired_rows({"a": (True, False), "b": (False, True)})
        rows = [row for row in rows if not (row["instance_id"] == "b" and
                row["repeat"] == 0 and row["system"] == "right")]
        result = compare(rows)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_right"], 1)
        self.assertEqual(result["unpaired_left"], 1)
        self.assertEqual(result["matched_attempts"], 9)
        self.assertEqual(result["left"]["attempts"], 10)
        self.assertEqual(result["right"]["attempts"], 9)
        self.assertIsNone(result["resolution_delta_pp"])
        self.assertIsNone(result["by_repeat"]["0"]["left_only"])

    def test_missing_pair_on_both_methods_does_not_look_complete(self):
        rows = paired_rows({"a": (True, False), "b": (False, True)})
        rows = [row for row in rows if not (row["instance_id"] == "b" and row["repeat"] == 0)]
        result = compare(rows)
        self.assertEqual(result["unpaired_left"], 0)
        self.assertEqual(result["unpaired_right"], 0)
        self.assertEqual(result["missing_left"], 1)
        self.assertEqual(result["missing_right"], 1)
        self.assertFalse(result["balanced_roster"])
        self.assertIsNone(result["delta_interval"]["estimate"])

    def test_explicit_roster_detects_entirely_absent_repeat_and_task(self):
        rows = paired_rows({"a": (True, False)}, repeats=range(4))
        result = compare(rows, expected_repeats=range(5), expected_tasks=("a", "b"))
        self.assertEqual(result["expected_pairs"], 10)
        self.assertEqual(result["missing_left"], 6)
        self.assertEqual(result["by_repeat"]["4"]["left"]["attempts"], 0)
        self.assertFalse(result["five_repeat_complete"])
        self.assertIsNone(result["delta_interval"]["upper"])

    def test_absent_named_method_is_reported_and_others_are_not_substituted(self):
        rows = [attempt("a", 0, "left", True), attempt("a", 0, "unrelated", False)]
        result = compare(rows)
        self.assertEqual(result["right"]["attempts"], 0)
        self.assertEqual(result["missing_right"], 1)
        self.assertFalse(result["complete"])

    def test_benchmark_model_groups_do_not_cross_pair_same_task_ids(self):
        rows = []
        for benchmark in ("one", "two"):
            for model in ("a", "b"):
                rows += [attempt("same", 0, "left", True, benchmark=benchmark, model=model),
                         attempt("same", 0, "right", False, benchmark=benchmark, model=model)]
        results = paired_comparison(rows, "left", "right", samples=5)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result["resolution_delta_pp"] == 100 for result in results))
        self.assertTrue(all(result["expected_pairs"] == 1 for result in results))

    def test_bootstrap_draws_whole_task_clusters_and_both_methods(self):
        rows = paired_rows({"a": (True, False), "b": (False, True)})
        with patch("locus.analysis.random.Random") as random_type:
            random_type.return_value.choices.side_effect = lambda population, k: ["a"] * k
            result = compare(rows)
        self.assertEqual(result["resolution_delta_pp"], 0)
        self.assertEqual(result["delta_interval"]["lower"], 100)
        self.assertEqual(result["delta_interval"]["upper"], 100)
        self.assertEqual(result["delta_interval"]["task_clusters"], 2)
        self.assertEqual(result["by_repeat"]["0"]["delta_interval"]["lower"], 100)
        self.assertEqual(random_type.return_value.choices.call_count, 40)
        for call in random_type.return_value.choices.call_args_list:
            self.assertEqual(call.args[0], ["a", "b"])
            self.assertEqual(call.kwargs["k"], 2)

    def test_seeded_interval_is_reproducible_and_does_not_split_repeats(self):
        rows = paired_rows({"a": (True, False), "b": (False, True)})
        one, two = compare(rows), compare(reversed(rows))
        self.assertEqual(one, two)
        self.assertEqual(one["delta_interval"]["lower"], -100)
        self.assertEqual(one["delta_interval"]["upper"], 100)

    def test_pooled_and_per_repeat_intervals_share_task_multiplicities(self):
        rows = []
        for task in ("a", "b"):
            for repeat in (0, 1):
                left_success = (task == "a") == (repeat == 0)
                rows += [attempt(task, repeat, "left", left_success),
                         attempt(task, repeat, "right", not left_success)]
        with patch("locus.analysis.random.Random") as random_type:
            random_type.return_value.choices.side_effect = lambda population, k: ["a"] * k
            result = compare(rows)
        self.assertEqual(result["delta_interval"]["lower"], 0)
        self.assertEqual(result["by_repeat"]["0"]["delta_interval"]["lower"], 100)
        self.assertEqual(result["by_repeat"]["1"]["delta_interval"]["lower"], -100)

    def test_duplicate_attempts_raise_even_if_rows_are_identical(self):
        rows = paired_rows({"a": (True, False)})
        with self.assertRaisesRegex(ValueError, "Duplicate attempt"):
            compare(rows + [dict(rows[0])])

    def test_repeat_keys_and_grades_are_strict_and_boolean_aware(self):
        for bad_repeat in (True, False, "0", 0.0, -1, None):
            with self.subTest(repeat=bad_repeat):
                rows = paired_rows({"a": (True, False)}, repeats=(0,))
                rows[0]["repeat"] = bad_repeat
                with self.assertRaises(ValueError):
                    compare(rows)
        for bad_grade in (0, 1, 0.0, "false", float("nan")):
            with self.subTest(grade=bad_grade):
                rows = paired_rows({"a": (True, False)}, repeats=(0,))
                rows[0]["resolved"] = bad_grade
                with self.assertRaises(ValueError):
                    compare(rows)

    def test_invalid_rosters_and_bootstrap_settings_raise(self):
        rows = paired_rows({"a": (True, False)}, repeats=(0,))
        for options in (dict(expected_repeats=(True,)), dict(expected_repeats=(0, 0)),
                        dict(expected_repeats=(1,)), dict(expected_tasks=("b",)),
                        dict(expected_tasks=("a", "a")), dict(samples=0), dict(samples=True),
                        dict(samples=1.5), dict(confidence=True), dict(confidence=0),
                        dict(confidence=1), dict(confidence=float("nan")), dict(seed=True)):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    paired_comparison(rows, "left", "right", **options)
        with self.assertRaises(ValueError):
            paired_comparison(rows, "left", "left")

    def test_empty_input_has_no_inferred_comparison_group(self):
        self.assertEqual(paired_comparison([], "left", "right"), [])


class RepeatStatisticsTests(unittest.TestCase):
    def test_asymmetric_five_repeat_variance_has_four_degrees_of_freedom(self):
        successes = (1, 2, 2, 4, 5)
        rows = [attempt(str(task), repeat, "left", task < successes[repeat])
                for task in range(5) for repeat in range(5)]
        result = repeat_statistics(reversed(rows), expected_repeats=range(5))
        # Rates 20,40,40,80,100 have mean 56 and squared deviations totaling 4320.
        self.assertEqual(result["mean"], 56)
        self.assertEqual(result["sample_variance"], 1080)
        self.assertEqual(result["sample_std"], math.sqrt(1080))
        self.assertEqual([result["counts_by_repeat"][str(run)]["successes"] for run in range(5)],
                         list(successes))

    def test_five_repeat_sample_variance_and_exact_success_counts(self):
        rows = [attempt(task, repeat, "left", index < repeat)
                for repeat in range(5) for index, task in enumerate(("a", "b", "c", "d"))]
        result = repeat_statistics(rows, expected_repeats=range(5))
        self.assertEqual(result["mean"], 50)
        self.assertEqual(result["sample_variance"], 1562.5)
        self.assertEqual(result["sample_std"], math.sqrt(1562.5))
        self.assertTrue(result["balanced_roster"])
        self.assertTrue(result["five_repeat_complete"])
        for repeat in range(5):
            counts = result["counts_by_repeat"][str(repeat)]
            self.assertEqual(counts["successes"], repeat)
            self.assertEqual(counts["attempts"], 4)

    def test_zero_variance_is_known_but_one_repeat_variance_is_undefined(self):
        rows = [attempt("a", repeat, "left", False) for repeat in range(5)]
        result = repeat_statistics(rows)
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["sample_variance"], 0)
        self.assertEqual(result["sample_std"], 0)
        self.assertIsNone(repeat_statistics(rows[:1])["sample_variance"])

    def test_missing_attempts_or_grades_do_not_shrink_variance_denominator(self):
        rows = [attempt(task, repeat, "left", True) for task in ("a", "b") for repeat in range(5)]
        result = repeat_statistics(rows[:-1], expected_repeats=range(5))
        self.assertFalse(result["balanced_roster"])
        self.assertEqual(result["missing_attempts"], 1)
        self.assertIsNone(result["by_repeat"]["4"])
        self.assertIsNone(result["mean"])
        rows[-1]["resolved"] = None
        result = repeat_statistics(rows)
        self.assertEqual(result["counts_by_repeat"]["4"]["ungraded_attempts"], 1)
        self.assertIsNone(result["counts_by_repeat"]["4"]["successes"])
        self.assertIsNone(result["sample_variance"])

    def test_expected_five_repeat_roster_reports_missing_whole_repeat(self):
        rows = [attempt("a", repeat, "left", True) for repeat in range(4)]
        result = repeat_statistics(rows, expected_repeats=range(5))
        self.assertEqual(result["repeat_count"], 4)
        self.assertEqual(result["expected_repeat_count"], 5)
        self.assertEqual(result["counts_by_repeat"]["4"]["attempts"], 0)
        self.assertIsNone(result["sample_variance"])

    def test_unknown_grades_do_not_block_a_known_resource_metric(self):
        rows = [dict(attempt("a", repeat, "left", None), ledger={"cost": 0}) for repeat in range(5)]
        result = repeat_statistics(rows, "normalized_cost_mean")
        self.assertFalse(result["grading_complete"])
        self.assertTrue(result["metric_complete"])
        self.assertEqual(result["sample_variance"], 0)

    def test_duplicates_invalid_keys_grades_and_mixed_groups_raise(self):
        row = attempt("a", 0, "left", True)
        for rows in ([row, dict(row)], [dict(row, repeat=True)], [dict(row, repeat="0")],
                     [dict(row, resolved=1)], [row, dict(row, system="right")]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    repeat_statistics(rows)

    def test_boolean_metric_is_not_averaged_as_a_numeric_measure(self):
        rows = [attempt("a", repeat, "left", True) for repeat in range(5)]
        result = repeat_statistics(rows, "usage_complete")
        self.assertEqual(result["known_repeat_count"], 0)
        self.assertIsNone(result["mean"])

    def test_empty_input_keeps_all_estimates_unknown(self):
        result = repeat_statistics([])
        self.assertEqual(result["repeat_count"], 0)
        self.assertFalse(result["complete"])
        self.assertFalse(result["balanced_roster"])
        self.assertIsNone(result["sample_variance"])
        self.assertIsNone(result["sample_std"])
        self.assertIsNone(result["mean"])


class HighestBaselineTests(unittest.TestCase):
    @staticmethod
    def rows(*, costs=True):
        rows = []
        for task in ("a", "b"):
            for repeat in range(5):
                rows += [attempt(task, repeat, "LOCUS", True),
                         attempt(task, repeat, "base-a", task == "a"),
                         attempt(task, repeat, "base-b", task == "b")]
        if costs:
            for row in rows:
                row["ledger"] = {"cost": 1 if row["system"] == "base-a" else 2}
        return rows

    def test_bootstrap_reselects_baseline_with_shared_task_weights(self):
        with patch("locus.analysis.random.Random") as random_type:
            random_type.return_value.choices.side_effect = lambda population, k: ["b"] * k
            result = highest_baseline_comparison(self.rows(), samples=40, seed=7)[0]
        self.assertTrue(result["complete"])
        self.assertEqual(result["highest_resolution_baseline"], "base-a")
        self.assertEqual(result["highest_resolution_candidates"], ["base-a", "base-b"])
        self.assertEqual(result["resolution_delta_pp"], 50)
        self.assertTrue(result["delta_interval"]["comparator_reselected"])
        self.assertEqual(result["delta_interval"]["lower"], 0)
        self.assertEqual(result["delta_interval"]["upper"], 0)
        run = result["by_repeat"]["0"]
        self.assertEqual(run["right"]["successes"], 1)
        self.assertEqual(run["right"]["attempts"], 2)
        self.assertEqual(run["resolution_delta_pp"], 50)
        self.assertTrue(run["delta_interval"]["conditional_on_comparator"])
        self.assertEqual(run["delta_interval"]["comparator_system"], "base-a")
        self.assertEqual(run["delta_interval"]["lower"], 100)

    def test_selection_aware_interval_is_not_fixed_baseline_interval(self):
        result = highest_baseline_comparison(self.rows(), samples=100, seed=7)[0]
        fixed = paired_comparison(self.rows(), "LOCUS", "base-a", samples=100, seed=7)[0]
        self.assertEqual(result["delta_interval"]["lower"], 0)
        self.assertEqual(result["delta_interval"]["upper"], 50)
        self.assertEqual(fixed["delta_interval"]["upper"], 100)

    def test_unique_quality_winner_is_kept_when_its_cost_is_unknown(self):
        rows = self.rows(costs=False)
        for row in rows:
            if row["system"] == "base-a":
                row["resolved"] = True
            if row["system"] == "base-b":
                row["ledger"] = {"cost": 0}
        result = highest_baseline_comparison(rows, samples=5)[0]
        self.assertEqual(result["highest_resolution_baseline"], "base-a")
        self.assertEqual(result["resolution_delta_pp"], 0)
        self.assertEqual(result["by_repeat"]["0"]["right"]["successes"], 2)

    def test_unknown_tied_cost_does_not_arbitrarily_select_a_named_comparator(self):
        result = highest_baseline_comparison(self.rows(costs=False), samples=40)[0]
        self.assertTrue(result["complete"])
        self.assertFalse(result["comparator_identified"])
        self.assertIsNone(result["highest_resolution_baseline"])
        self.assertEqual(result["resolution_delta_pp"], 50)
        self.assertEqual(result["delta_interval"]["valid_samples"], 40)
        self.assertEqual(result["by_repeat"], {})
        self.assertEqual(result["counts_by_repeat"]["0"]["base-a"]["successes"], 1)

    def test_ungraded_or_missing_baseline_is_not_removed_from_candidate_set(self):
        rows = self.rows()
        rows[-1]["resolved"] = None
        result = highest_baseline_comparison(rows, samples=5)[0]
        self.assertFalse(result["complete"])
        self.assertIn("base-b", result["baseline_candidates"])
        self.assertIsNone(result["resolution_delta_pp"])
        self.assertEqual(result["delta_interval"]["valid_samples"], 0)
        self.assertIsNone(result["method_counts"]["base-b"]["successes"])
        missing = highest_baseline_comparison(rows[:-1], samples=5)[0]
        self.assertFalse(missing["complete"])
        self.assertEqual(missing["expected_pairs"], 10)

    def test_missing_locus_no_baseline_and_empty_are_explicit(self):
        result = highest_baseline_comparison([attempt("a", 0, "baseline", False)], samples=5)[0]
        self.assertFalse(result["complete"])
        self.assertEqual(result["left"]["attempts"], 0)
        result = highest_baseline_comparison([attempt("a", 0, "LOCUS", False)], samples=5)[0]
        self.assertEqual(result["baseline_candidates"], [])
        self.assertIsNone(result["resolution_delta_pp"])
        self.assertEqual(highest_baseline_comparison([]), [])


if __name__ == "__main__":
    unittest.main()
