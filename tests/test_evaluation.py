import unittest
from fractions import Fraction

from locus.analysis import (agreement_correlation, allocation_summary, compare_operating_points,
                            make_agreement_cohorts, pearson, repeat_statistics,
                            resolution_interval, task_cluster_bootstrap)
from locus.baselines import import_predictions
from locus.evaluation import EXPLORATION_FIELDS, aggregate, evidence_metrics, normalized_cost, summarize


def attempt(identifier, resolved, cost):
    return {"instance_id": identifier, "repeat": 0, "benchmark": "test", "model": "m", "system": "LOCUS",
            "resolved": resolved, "ledger": {"cost": cost, "uncached_input": 10, "cached_input": 90,
            "output": 10, "worker_hours": 1, "elapsed_seconds": 100, "calls": 3,
            "llm_seconds": 70, "tool_seconds": 10}}


class EvaluationTests(unittest.TestCase):
    def test_imported_baseline_without_telemetry_aggregates(self):
        records = import_predictions(
            [{"instance_id": "a", "model_patch": ""}],
            [{"instance_id": "a", "benchmark": "lite"},
             {"instance_id": "b", "benchmark": "lite"}],
            system="Agentless", model="model", repeat=0,
        )
        result = aggregate(records)[0]
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["ungraded_attempts"], 2)
        self.assertEqual(result["usage_incomplete_attempts"], 2)
        self.assertIsNone(result["normalized_cost_mean"])
        self.assertIsNone(result["tokens_mean"])
        self.assertIsNone(result["time_minutes_mean"])
        self.assertIsNone(result["known_resources_total"]["tokens"])
        self.assertIsNone(result["known_normalized_cost_total"])
        self.assertIsNone(result["file_at_1_pct"])
        self.assertIsNone(result["pooled_recall_pct"])
        records[0]["resolved"], records[1]["resolved"] = True, False
        graded = aggregate(records)[0]
        self.assertEqual(graded["resolution_pct"], 50)
        self.assertIsNone(graded["cost_per_success"])

    def test_explicit_none_optional_telemetry_collections(self):
        row = attempt("a", True, .2)
        row["ledger"].update(totals=None, steps=None, requests=None)
        row["evidence_metrics"] = None
        result = summarize([row])
        self.assertEqual(result["resolution_pct"], 100)
        self.assertEqual(result["normalized_cost_mean"], .2)
        self.assertIsNone(result["file_at_1_pct"])
        self.assertIsNone(result["request_cache_hit_rate_pct"])

    def test_failures_remain_in_success_cost(self):
        result = summarize([attempt("a", True, .2), attempt("b", False, .8)])
        self.assertEqual(result["resolution_pct"], 50)
        self.assertEqual(result["cost_per_success"], 1)
        self.assertEqual(result["normalized_cost_mean"], .5)

    def test_tokens_per_success_includes_failed_complete_task_tokens(self):
        one, two = attempt("a", True, .2), attempt("b", False, .8)
        one["ledger"].update(uncached_input=100_000, cached_input=800_000, output=100_000)
        two["ledger"].update(uncached_input=1_000_000, cached_input=7_000_000, output=1_000_000)
        result = summarize([one, two])
        self.assertEqual(result["tokens_mean"], 5_000_000)
        self.assertEqual(result["tokens_per_success"], 10_000_000)
        self.assertEqual(result["tokens_million_per_success"], 10)

    def test_tokens_per_success_pools_charges_and_successes_across_repeats(self):
        rows = [attempt("a", True, .2), attempt("a", False, .4), attempt("b", True, .6)]
        rows[1]["repeat"] = 1
        for index, row in enumerate(rows, 1):
            row["ledger"].update(uncached_input=index * 1_000_000, cached_input=0, output=0)
        result = aggregate(rows)[0]
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["successes"], 2)
        self.assertEqual(result["tokens_per_success"], 3_000_000)
        self.assertEqual(result["tokens_million_per_success"], 3)

    def test_tokens_per_success_is_null_with_zero_successes_or_pending_grades(self):
        for rows in ([], [attempt("a", False, .2)],
                     [attempt("a", True, .2), attempt("b", None, .8)]):
            with self.subTest(rows=rows):
                result = summarize(rows)
                self.assertIsNone(result["tokens_per_success"])
                self.assertIsNone(result["tokens_million_per_success"])

    def test_tokens_per_success_requires_complete_token_accounting(self):
        one, two = attempt("a", True, .2), attempt("b", False, .8)
        two["ledger"].update(usage_complete=False, tokens=0, known_uncached_input=1,
                             known_cached_input=2, known_output=3)
        result = summarize([one, two])
        self.assertIsNone(result["tokens_per_success"])
        self.assertIsNone(result["tokens_million_per_success"])
        two["ledger"] = {"usage_complete": True}
        self.assertIsNone(summarize([one, two])["tokens_per_success"])

    def test_known_total_tokens_need_no_cost_or_component_estimate(self):
        row = attempt("a", True, None)
        row["ledger"] = {"usage_complete": True, "tokens": 2_000_000}
        result = summarize([row])
        self.assertIsNone(result["cost_per_success"])
        self.assertEqual(result["tokens_per_success"], 2_000_000)
        self.assertEqual(result["tokens_million_per_success"], 2)
        row["ledger"]["tokens"] = 0
        self.assertEqual(summarize([row])["tokens_per_success"], 0)

    def test_unknown_grade_is_not_failure_or_success(self):
        result = summarize([attempt("a", True, .2), attempt("b", None, .8)])
        self.assertIsNone(result["resolution_pct"])
        self.assertIsNone(result["cost_per_success"])
        self.assertEqual(result["graded_resolution_pct"], 100)
        self.assertEqual(result["ungraded_attempts"], 1)

    def test_unknown_annotation_metric_does_not_shrink_denominator(self):
        rows = [attempt("a", True, .2), attempt("b", False, .8)]
        rows[0]["evidence_metrics"] = {"file_at_1": True}
        result = summarize(rows)
        self.assertIsNone(result["file_at_1_pct"])
        self.assertEqual(result["metric_attempts"]["file_at_1"], 1)

    def test_action_timing_schema(self):
        row = attempt("a", True, .2)
        row["ledger"]["calls"] = 2
        row["ledger"]["steps"] = [{"kind": "model", "seconds": 12}, {"kind": "tool", "seconds": 2}]
        result = summarize([row])
        self.assertEqual(result["step_wall_seconds_mean"], 7)
        self.assertEqual(result["step_llm_seconds_mean"], 6)
        self.assertEqual(result["step_tool_seconds_mean"], 1)
        self.assertEqual(result["step_residual_seconds_mean"], 0)

    def test_zero_success_cost_is_undefined(self):
        result = summarize([attempt("a", False, .2)])
        self.assertIsNone(result["cost_per_success"])
        self.assertEqual(result["resolution_pct"], 0)

    def test_incomplete_usage_overrides_default_zero_counters(self):
        row = attempt("a", True, 0)
        row["ledger"].update(usage_complete=False, uncached_input=0,
            cached_input=0, output=0, tokens=0, known_uncached_input=30,
            known_cached_input=40, known_output=50, known_cost=.123)
        result = summarize([row])
        self.assertIsNone(result["normalized_cost_total"])
        self.assertIsNone(result["normalized_cost_mean"])
        self.assertIsNone(result["cost_per_success"])
        self.assertIsNone(result["tokens_mean"])
        self.assertIsNone(result["token_shares_pct"]["output"])
        self.assertFalse(result["usage_complete"])
        self.assertEqual(result["usage_incomplete_attempts"], 1)
        self.assertEqual(result["resolution_pct"], 100)
        self.assertEqual(result["elapsed_seconds_mean"], 100)
        self.assertEqual(result["known_resources_total"]["tokens"], 120)
        self.assertEqual(result["known_normalized_cost_total"], .123)

    def test_incomplete_usage_makes_pooled_totals_unknown(self):
        one, two = attempt("a", True, .2), attempt("b", False, None)
        two["ledger"].update(usage_complete=False, known_uncached_input=4,
            known_cached_input=6, known_output=3, known_cost=.01)
        result = summarize([one, two])
        self.assertIsNone(result["resources_total"]["tokens"])
        self.assertIsNone(result["normalized_cost_total"])
        self.assertIsNone(result["cost_per_success"])
        self.assertAlmostEqual(result["known_normalized_cost_total"], .21)
        self.assertEqual(result["known_resources_total"]["tokens"], 123)

    def test_unknown_model_step_marks_usage_incomplete(self):
        row = attempt("a", False, 0)
        row["ledger"]["steps"] = [{"kind": "model", "seconds": 3,
                                     "usage_known": False}]
        result = summarize([row])
        self.assertFalse(result["usage_complete"])
        self.assertIsNone(result["normalized_cost_total"])

    def test_ungraded_resolution_does_not_get_partial_bootstrap_interval(self):
        result = resolution_interval([attempt("a", True, 1), attempt("b", None, 2)], samples=20)
        self.assertIsNone(result["estimate"])
        self.assertIsNone(result["lower"])
        self.assertIsNone(result["upper"])

    def test_annotation_metric_denominators(self):
        metric = evidence_metrics(["a.py:1:1", "b.py:2:2", "x.py:1:1"],
            ["a.py:1:1", "b.py:2:2", "c.py:1:1", "d.py:1:1"], ["a.py:1:1"],
            ["a.py"], 2000, 200, False)
        self.assertEqual(metric["recall"], .5)
        self.assertAlmostEqual(metric["precision"], 2 / 3)
        self.assertEqual(metric["duplication"], .1)
        self.assertEqual(metric["yield"], .5)
        self.assertTrue(metric["edit_coverage"])
        self.assertFalse(metric["joint"])
        self.assertTrue(metric["file_at_1"])

    def test_empty_and_missing_annotations(self):
        metric = evidence_metrics([], [], [])
        self.assertIsNone(metric["recall"])
        self.assertEqual(metric["precision"], 0)
        self.assertIsNone(metric["joint"])
        self.assertIsNone(evidence_metrics([], None, None)["precision"])

    def test_cached_token_share_has_input_denominator(self):
        result = summarize([attempt("a", True, 1)])
        self.assertEqual(result["token_shares_pct"]["cached_share_of_input"], 90)
        self.assertAlmostEqual(result["token_shares_pct"]["cached_input"], 100 * 90 / 110)
        self.assertIsNone(result["request_cache_hit_rate_pct"])

    def test_tokens_use_disjoint_component_sum(self):
        row = attempt("a", True, .2)
        row["ledger"]["tokens"] = 0
        self.assertEqual(summarize([row])["tokens_mean"], 110)

    def test_partial_request_cache_events_do_not_shrink_denominator(self):
        row = attempt("a", True, .2)
        row["ledger"].update(requests=[{"cache_hit": True}, {"cache_hit": None}],
            calls=2,
            steps=[{"kind": "model", "seconds": 1}, {"kind": "model", "seconds": 2}])
        self.assertIsNone(summarize([row])["request_cache_hit_rate_pct"])
        row["ledger"]["requests"][1]["cache_hit"] = False
        self.assertEqual(summarize([row])["request_cache_hit_rate_pct"], 50)

    def test_unknown_step_time_is_not_dropped_from_step_denominator(self):
        row = attempt("a", True, .2)
        row["ledger"]["calls"] = 2
        row["ledger"]["steps"] = [{"kind": "model", "seconds": 10}, {"kind": "tool", "seconds": None}]
        result = summarize([row])
        self.assertEqual(result["step_count"], 2)
        self.assertIsNone(result["step_wall_seconds_mean"])
        self.assertEqual(result["step_llm_seconds_mean"], 5)
        self.assertIsNone(result["step_tool_seconds_mean"])

    def test_repeat_mean_does_not_drop_ungraded_repeat(self):
        one, two = attempt("a", True, .2), attempt("a", None, .3)
        two["repeat"] = 1
        result = repeat_statistics([one, two])
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["sample_std"])
        self.assertEqual(result["repeat_count"], 2)
        self.assertEqual(result["known_repeat_count"], 1)

    def test_quality_comparator_keeps_highest_resolution_without_cost(self):
        rows = [dict(benchmark="b", model="m", system="LOCUS", resolution_pct=60,
                     normalized_cost_mean=.2, tokens_mean=100),
                dict(benchmark="b", model="m", system="High", resolution_pct=61,
                     normalized_cost_mean=None, tokens_mean=None),
                dict(benchmark="b", model="m", system="Low", resolution_pct=40,
                     normalized_cost_mean=.5, tokens_mean=200)]
        result = compare_operating_points(rows)[0]
        self.assertEqual(result["highest_resolution_baseline"], "High")
        self.assertEqual(result["resolution_delta_pp"], -1)
        self.assertIsNone(result["cost_saved_pct"])
        self.assertIsNone(result["lowest_cost_baseline"])
        self.assertIsNone(result["cost_saved_vs_cheapest_pct"])
        self.assertEqual(result["quality_delta_pp"], -1)
        self.assertTrue(result["quality_floor_met"])
        rows[1]["normalized_cost_mean"] = .3
        priced = compare_operating_points(rows)[0]
        self.assertEqual(priced["lowest_cost_baseline"], "High")
        self.assertAlmostEqual(priced["cost_saved_vs_cheapest_pct"], 100 / 3)
        rows[0]["resolution_pct"] = 59.9
        self.assertFalse(compare_operating_points(rows)[0]["quality_floor_met"])

    def test_quality_floor_uses_explicit_fixed_reference(self):
        rows = [dict(benchmark="b", model="m", system="LOCUS", resolution_pct=60),
                dict(benchmark="b", model="m", system="Broad-context", resolution_pct=50),
                dict(benchmark="b", model="m", system="High", resolution_pct=80)]
        result = compare_operating_points(rows, quality_reference="Broad-context")[0]
        self.assertEqual(result["resolution_delta_pp"], -20)
        self.assertEqual(result["quality_delta_pp"], 10)
        self.assertTrue(result["quality_floor_met"])
        rows[0]["resolution_pct"] = None
        self.assertIsNone(compare_operating_points(rows)[0]["quality_floor_met"])

    def test_unknown_baseline_grade_keeps_quality_floor_unknown(self):
        rows = [dict(benchmark="b", model="m", system="LOCUS", resolution_pct=60),
                dict(benchmark="b", model="m", system="Ungraded", resolution_pct=None),
                dict(benchmark="b", model="m", system="Graded", resolution_pct=50)]
        result = compare_operating_points(rows)[0]
        self.assertIsNone(result["resolution_delta_pp"])
        self.assertIsNone(result["quality_floor_met"])

    def test_allocation_tie_counts_require_valid_labels_and_unique_tasks(self):
        rows = [dict(instance_id="a", budget_class="H", agreement=.5,
                     votes={"one": "L", "two": "M", "bad": "X"})]
        result = allocation_summary(rows)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["promoted_high"], 0)
        with self.assertRaises(ValueError):
            allocation_summary(rows * 5)

    def test_standard_price_schedule(self):
        self.assertAlmostEqual(normalized_cost(1_000_000, 1_000_000, 1_000_000, 1), 7.41)

    def test_groups_pool_repeats(self):
        rows = [attempt("a", True, 1), attempt("a", False, 2)]
        rows[1]["repeat"] = 1
        self.assertEqual(aggregate(rows)[0]["attempts"], 2)
        self.assertEqual(aggregate(rows)[0]["resolution_pct"], 50)

    def test_bootstrap_resamples_tasks_with_all_repeats(self):
        rows = [{"instance_id": key, "repeat": repeat} for key in ["a", "b", "c"] for repeat in range(5)]
        def statistic(sample):
            counts = {}
            for row in sample:
                counts.setdefault(row["instance_id"], []).append(row["repeat"])
            for values in counts.values():
                self.assertEqual(len(values) % 5, 0)
                for repeat in range(5):
                    self.assertEqual(values.count(repeat), len(values) // 5)
            return float(len(sample))
        result = task_cluster_bootstrap(rows, statistic, samples=30, seed=7)
        self.assertEqual(result["lower"], 15)
        self.assertEqual(result["task_clusters"], 3)

    def test_cohorts_use_shared_task_cluster_weights(self):
        tasks = [{"instance_id": f"{cls}{i}", "budget_class": cls,
                  "agreement": i / 12, "resolution": i / 12}
                 for cls in ["L", "M", "H"] for i in range(12)]
        cohorts = make_agreement_cohorts(tasks, seed=3)
        self.assertEqual(len(cohorts), 45)
        self.assertTrue(all(len(set(c)) == 30 for c in cohorts))
        result = agreement_correlation(tasks, cohorts, samples=25, seed=5)
        self.assertAlmostEqual(result["pearson_r"], 1)
        self.assertAlmostEqual(result["lower"], 1)
        self.assertAlmostEqual(result["upper"], 1)
        self.assertIsNone(pearson([1, 1], [1, 2]))


class NumericalAggregationTests(unittest.TestCase):
    def test_five_repeats_pool_integer_successes_and_all_failure_charges(self):
        outcomes = {"a": (True, True, False, True, False),
                    "b": (False, True, False, False, False)}
        rows = []
        for identity, grades in outcomes.items():
            for repeat, grade in enumerate(grades):
                charge, tokens = (1, 10) if identity == "a" else (4, 40)
                row = attempt(identity, grade, charge)
                row["repeat"] = repeat
                row["ledger"].update(uncached_input=tokens, cached_input=0, output=0)
                rows.append(row)
        result = aggregate(rows)[0]
        # Ten attempts, four successes, 25 USD and 250 tokens.
        self.assertEqual((result["attempts"], result["successes"]), (10, 4))
        self.assertEqual(result["resolution_pct"], 40)
        self.assertEqual(result["normalized_cost_mean"], 2.5)
        self.assertEqual(result["cost_per_success"], 6.25)
        self.assertEqual(result["tokens_mean"], 25)
        self.assertEqual(result["tokens_per_success"], 62.5)

    def test_exact_source_units_deduplicate_before_coverage_and_file_rates(self):
        first = evidence_metrics(
            ["a.py:1:1", "a.py:1:1", {"path": "a.py", "start": 1, "end": 1},
             ("b.py", 2, 2), "extra.py:1:1"],
            ["a.py:1:1", "b.py:2:2", "support.py:1:1", "support.py:1:1"],
            ["a.py:1:1", "b.py:2:2", "a.py:1:1"],
            ["extra.py", "a.py"], 1000, 100, False,
        )
        second = evidence_metrics(
            ["a.py:1:1", "b.py:2:2", "support.py:1:1"],
            ["a.py:1:1", "b.py:2:2", "support.py:1:1"],
            ["a.py:1:1", "b.py:2:2"], ["b.py"], 1000, 0, True,
        )
        self.assertEqual((first["selected"], first["required"], first["hits"]), (3, 3, 2))
        self.assertEqual((first["support_hits"], second["support_hits"]), (0, 1))
        rows = [attempt("a", True, 1), attempt("b", False, 2)]
        rows[0]["evidence_metrics"], rows[1]["evidence_metrics"] = first, second
        result = summarize(rows)
        self.assertEqual(result["file_at_1_pct"], 50)
        self.assertEqual(result["edit_pct"], 100)
        self.assertEqual(result["joint_pct"], 50)
        self.assertEqual(result["recovery_pct"], 50)
        self.assertAlmostEqual(result["recall_pct"], float(Fraction(250, 3)))
        self.assertEqual(result["yield_mean"], .5)

    def test_macro_and_pooled_ratios_keep_their_own_denominators(self):
        rows = [attempt("a", True, 1), attempt("b", False, 2)]
        counts = ((0, 1, 0, 0, 100, 0), (2, 4, 1, 1, 300, 60))
        for row, (selected, required, hits, support_hits, admitted, repeated) in zip(rows, counts):
            row["evidence_metrics"] = dict(
                selected=selected, required=required, hits=hits, support_hits=support_hits,
                admitted_tokens=admitted, repeated_tokens=repeated,
                recall=float(Fraction(hits, required)),
                precision=float(Fraction(hits, max(1, selected))),
                duplication=float(Fraction(repeated, admitted)),
                **{"yield": float(Fraction(1000 * support_hits, admitted))},
            )
        result = summarize(rows)
        self.assertEqual(result["recall_pct"], 12.5)
        self.assertEqual(result["precision_pct"], 25)
        self.assertEqual(result["pooled_recall_pct"], 20)
        self.assertEqual(result["precision_denominator_total"], 3)
        self.assertAlmostEqual(result["pooled_precision_pct"], float(Fraction(100, 3)))
        self.assertEqual(result["duplication_mean_pct"], 10)
        self.assertEqual(result["pooled_duplication_pct"], 15)
        self.assertAlmostEqual(result["yield_mean"], float(Fraction(5, 3)))
        self.assertEqual(result["pooled_yield"], 2.5)

    def test_complete_derived_cost_is_also_a_known_cost(self):
        row = attempt("a", True, None)
        row["ledger"].update(uncached_input=1_000_000, cached_input=2_000_000,
                             output=1_000_000, worker_hours=2)
        result = summarize([row])
        # 1.25 + 0.20 + 6.00 + 0.12 = 7.57 USD.
        self.assertAlmostEqual(result["normalized_cost_total"], 7.57)
        self.assertAlmostEqual(result["known_normalized_cost_total"], 7.57)
        self.assertEqual(result["known_resources_total"]["tokens"], 4_000_000)

    def test_total_only_tokens_remain_known_and_partial_usage_stays_separate(self):
        one, two = attempt("a", True, None), attempt("b", False, None)
        one["ledger"] = dict(usage_complete=True, tokens=200, cost=.5)
        two["ledger"] = dict(usage_complete=False, tokens=900, cost=10,
                             known_uncached_input=10, known_cached_input=20,
                             known_output=30, known_cost=.2)
        complete = summarize([one])
        self.assertEqual(complete["known_resources_total"]["tokens"], 200)
        result = summarize([one, two])
        self.assertIsNone(result["resources_total"]["tokens"])
        self.assertIsNone(result["tokens_per_success"])
        self.assertIsNone(result["cost_per_success"])
        self.assertEqual(result["known_resources_total"]["tokens"], 260)
        self.assertAlmostEqual(result["known_normalized_cost_total"], .7)

    def test_invalid_evidence_value_does_not_reduce_metric_denominator(self):
        for invalid in (float("nan"), float("inf"), -1, "pending", 2):
            with self.subTest(invalid=invalid):
                rows = [attempt("a", True, 1), attempt("b", False, 2)]
                rows[0]["evidence_metrics"] = dict(recall=.5, file_at_1=True)
                rows[1]["evidence_metrics"] = dict(recall=invalid, file_at_1=invalid)
                result = summarize(rows)
                self.assertIsNone(result["recall_pct"])
                self.assertIsNone(result["file_at_1_pct"])
                self.assertEqual(result["metric_attempts"]["recall"], 1)

    def test_partial_steps_do_not_become_complete_timing_or_cache_events(self):
        cases = (
            dict(calls=2),
            dict(calls=1, steps_complete=False),
        )
        for controls in cases:
            with self.subTest(controls=controls):
                row = attempt("a", True, .2)
                row["ledger"].update(controls)
                row["ledger"].update(
                    steps=[dict(kind="model", phase="setup", seconds=10)],
                    requests=[dict(cache_hit=True)],
                )
                result = summarize([row])
                self.assertEqual(result["known_step_count"], 1)
                self.assertIsNone(result["step_count"])
                self.assertIsNone(result["request_cache_hit_rate_pct"])
                self.assertIsNone(result["exploration_calls_mean"])
                for field in ("wall_seconds", "llm_seconds", "tool_seconds"):
                    self.assertIsNone(result["step_" + field + "_mean"])
                self.assertEqual(result["tokens_mean"], 110)

    def test_explicitly_partial_request_list_does_not_become_a_hit_rate(self):
        row = attempt("a", True, .2)
        row["ledger"].update(requests=[dict(cache_hit=True)], requests_complete=False)
        self.assertIsNone(summarize([row])["request_cache_hit_rate_pct"])

    def test_cache_hit_rate_uses_requests_rather_than_an_average_of_attempt_rates(self):
        one, two = attempt("a", True, .2), attempt("b", False, .8)
        one["ledger"].update(calls=1, requests=[dict(cache_hit=True)],
                             steps=[dict(kind="model", seconds=1)])
        two["ledger"].update(calls=3, requests=[dict(cache_hit=value) for value in (False, False, True)],
                             steps=[dict(kind="model", seconds=1) for _ in range(3)])
        self.assertEqual(summarize([one, two])["request_cache_hit_rate_pct"], 50)

    def test_nonfinite_step_duration_is_not_dropped(self):
        row = attempt("a", True, .2)
        row["ledger"].update(calls=2, steps=[
            dict(kind="model", phase="setup", seconds=10),
            dict(kind="tool", phase="acquisition", seconds=float("nan")),
        ])
        result = summarize([row])
        self.assertEqual(result["step_count"], 2)
        self.assertIsNone(result["step_wall_seconds_mean"])
        self.assertEqual(result["step_llm_seconds_mean"], 5)
        self.assertIsNone(result["step_tool_seconds_mean"])
        self.assertEqual(result["exploration_llm_seconds_mean"], 10)
        self.assertIsNone(result["exploration_tool_seconds_mean"])

    def test_invalid_resource_totals_cannot_be_negative_or_nonfinite(self):
        for invalid in (-1, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                row = attempt("a", True, invalid)
                row["ledger"] = dict(usage_complete=True, tokens=invalid, cost=invalid)
                result = summarize([row])
                self.assertIsNone(result["tokens_per_success"])
                self.assertIsNone(result["cost_per_success"])

    def test_evidence_instrumentation_requires_finite_nonnegative_counts(self):
        for invalid in (-1, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    evidence_metrics([], [], [], admitted_tokens=invalid)
                with self.assertRaises(ValueError):
                    evidence_metrics([], [], [], admitted_tokens=100, repeated_tokens=invalid)
        with self.assertRaises(ValueError):
            evidence_metrics([], [], [], recovery=1)


class ExplorationEvaluationTests(unittest.TestCase):
    def test_setup_and_acquisition_are_separate_from_complete_task(self):
        row = attempt("a", False, .2)
        row["ledger"].update(calls=6, llm_seconds=107, tool_seconds=210, steps=[
            dict(kind="model", phase="setup", seconds=2),
            dict(kind="tool", phase="setup", seconds=3),
            dict(kind="model", phase="acquisition", seconds=5, success=False),
            dict(kind="tool", phase="acquisition", seconds=7),
            dict(kind="model", phase="repair", seconds=100),
            dict(kind="tool", phase="repair", seconds=200),
        ])
        result = summarize([row])
        self.assertEqual(result["calls_mean"], 6)
        self.assertEqual(result["llm_seconds_mean"], 107)
        self.assertEqual(result["tool_seconds_mean"], 210)
        for field, value in zip(EXPLORATION_FIELDS, (4, 2, 2, 7, 10)):
            self.assertEqual(result["resources_total"][field], value)
            self.assertEqual(result[field + "_mean"], value)

    def test_explicit_exploration_totals_pool_over_all_attempts(self):
        one, two = attempt("a", True, .2), attempt("b", False, .8)
        for row, values in ((one, (4, 1, 3, 2, 5)), (two, (8, 3, 5, 6, 7))):
            row["ledger"].update(zip(EXPLORATION_FIELDS, values))
        result = aggregate([one, two])[0]
        for field, total in zip(EXPLORATION_FIELDS, (12, 4, 8, 8, 12)):
            self.assertEqual(result["resources_total"][field], total)
            self.assertEqual(result[field + "_mean"], total / 2)
        self.assertEqual(result["cost_per_success"], 1)

    def test_phase_is_not_inferred_from_role_or_full_task_totals(self):
        row = attempt("a", True, .2)
        row["ledger"].update(calls=1, steps=[
            dict(kind="model", role="acquisition", seconds=5),
        ])
        result = summarize([row])
        self.assertEqual(result["calls_mean"], 1)
        for field in EXPLORATION_FIELDS:
            self.assertIsNone(result[field + "_mean"])

    def test_partial_phase_or_step_list_leaves_exploration_unknown(self):
        valid = dict(kind="model", phase="setup", seconds=5)
        cases = (
            dict(calls=2, steps=[valid]),
            dict(calls=1, steps=[valid], steps_complete=False),
            dict(calls=2, steps=[valid, dict(kind="tool", seconds=3)]),
            dict(calls=1, steps=[dict(kind="model", phase="unknown", seconds=3)]),
        )
        for telemetry in cases:
            with self.subTest(telemetry=telemetry):
                row = attempt("a", True, .2)
                row["ledger"].update(telemetry)
                result = summarize([row])
                for field in EXPLORATION_FIELDS:
                    self.assertIsNone(result[field + "_mean"])

    def test_unknown_token_usage_keeps_known_failed_call_timing(self):
        row = attempt("a", False, .2)
        row["ledger"].update(calls=2, usage_complete=False, steps=[
            dict(kind="model", phase="acquisition", seconds=4, success=False, usage_known=False),
            dict(kind="tool", phase="repair", seconds=10),
        ])
        result = summarize([row])
        self.assertIsNone(result["tokens_mean"])
        self.assertIsNone(result["normalized_cost_mean"])
        self.assertEqual(result["exploration_calls_mean"], 1)
        self.assertEqual(result["exploration_model_calls_mean"], 1)
        self.assertEqual(result["exploration_llm_seconds_mean"], 4)
        self.assertEqual(result["exploration_tool_seconds_mean"], 0)

    def test_missing_exploration_timing_preserves_call_counts_and_other_component(self):
        row = attempt("a", True, .2)
        row["ledger"].update(calls=3, steps=[
            dict(kind="model", phase="setup", seconds=None),
            dict(kind="tool", phase="acquisition", seconds=3),
            dict(kind="model", phase="repair", seconds=None),
        ])
        result = summarize([row])
        self.assertEqual(result["exploration_calls_mean"], 2)
        self.assertEqual(result["exploration_tool_calls_mean"], 1)
        self.assertEqual(result["exploration_tool_seconds_mean"], 3)
        self.assertIsNone(result["exploration_llm_seconds_mean"])

    def test_explicit_null_and_missing_attempt_metrics_do_not_shrink_denominators(self):
        one, two = attempt("a", True, .2), attempt("b", False, .8)
        one["ledger"].update(zip(EXPLORATION_FIELDS, (1, 1, 0, 4, 0)))
        result = summarize([one, two])
        for field in EXPLORATION_FIELDS:
            self.assertIsNone(result["resources_total"][field])
            self.assertIsNone(result[field + "_mean"])
        one["ledger"].update(calls=1, steps=[dict(kind="model", phase="setup", seconds=4)],
                             exploration_calls=None)
        result = summarize([one])
        self.assertIsNone(result["exploration_calls_mean"])
        self.assertEqual(result["exploration_llm_seconds_mean"], 4)

    def test_empty_complete_action_list_has_zero_exploration(self):
        row = attempt("a", True, .2)
        row["ledger"].update(calls=0, steps=[])
        result = summarize([row])
        for field in EXPLORATION_FIELDS:
            self.assertEqual(result[field + "_mean"], 0)
        row["ledger"]["steps"] = None
        for field in EXPLORATION_FIELDS:
            self.assertIsNone(summarize([row])[field + "_mean"])

    def test_boolean_or_negative_flat_telemetry_is_unknown(self):
        row = attempt("a", True, .2)
        row["ledger"].update(exploration_calls=True, exploration_model_calls=-1,
                             exploration_tool_calls=0, exploration_llm_seconds=float("nan"),
                             exploration_tool_seconds=-1)
        result = summarize([row])
        self.assertEqual(result["exploration_tool_calls_mean"], 0)
        for field in ("exploration_calls", "exploration_model_calls",
                      "exploration_llm_seconds", "exploration_tool_seconds"):
            self.assertIsNone(result[field + "_mean"])


if __name__ == "__main__":
    unittest.main()
