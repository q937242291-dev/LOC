import unittest

from locus.interventions import build_interventions, frozen_edit_intervention, observation_units


def source(identifier, tokens, units, text=None):
    return {"id": identifier, "tokens": tokens, "units": units, "text": text or identifier}


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.frontier = [{"path": "edit.py", "start": 1, "end": 10}]
        self.selected = [source("edit", 2000, ["edit.py:1:10"]),
                         source("support", 6000, ["support.py:1:10"]),
                         source("duplicate", 5000, ["support.py:1:10"])]
        self.pool = self.selected + [source("other", 1000, ["other.py:1:10"])]

    def test_all_conditions_fix_frontier_and_disable_recovery(self):
        result = build_interventions(self.frontier, self.pool, self.selected,
            mandatory_ids=["edit"], reference_support_units=["support.py:1:10"], seed=8)
        self.assertEqual(len(result), 5)
        for packet in result.values():
            self.assertEqual(packet["frontier"], self.frontier)
            self.assertFalse(packet["recovery_allowed"])
            self.assertLessEqual(packet["tokens"], packet["token_cap"])
            self.assertIn("edit", [x["id"] for x in packet["observations"]])
        self.assertEqual(result["required-evidence-removed"]["tokens"], 2000)

    def test_duplicate_removal_preserves_unit_coverage(self):
        result = frozen_edit_intervention("duplicates-removed", self.frontier, self.pool,
            self.selected, mandatory_ids=["edit"])
        before = set().union(*(observation_units(r) for r in self.selected))
        after = set().union(*(observation_units(r) for r in result["observations"]))
        self.assertEqual(before, after)
        self.assertLess(len(result["observations"]), len(self.selected))

    def test_required_removal_needs_external_support(self):
        with self.assertRaises(ValueError):
            frozen_edit_intervention("required-evidence-removed", self.frontier, self.pool, self.selected,
                                     mandatory_ids=["edit"])

    def test_impossible_redundancy_removal_does_not_drop_required_source(self):
        rows = [source("edit", 1000, ["edit.py:1:10"]), source("unique", 12000, ["b.py:1:10"])]
        with self.assertRaises(ValueError):
            frozen_edit_intervention("duplicates-removed", self.frontier, rows, rows, mandatory_ids=["edit"])

    def test_frontier_support_is_retained_without_explicit_mandatory_ids(self):
        pool = [source("noise", 31000, ["other.py:1:10"]), *self.selected]
        result = frozen_edit_intervention("broad", self.frontier, pool, self.selected)
        self.assertIn("edit", [r["id"] for r in result["observations"]])

    def test_selected_must_support_every_frozen_edit(self):
        frontier = [*self.frontier, {"path": "missing.py", "start": 1, "end": 10}]
        with self.assertRaises(ValueError):
            frozen_edit_intervention("selected", frontier, self.pool, self.selected)

    def test_symptom_role_cannot_supply_mandatory_edit_source(self):
        rows = [source("edit", 100, ["edit.py:1:10"])]
        rows[0]["roles"] = ["symptom"]
        with self.assertRaises(ValueError):
            frozen_edit_intervention("selected", self.frontier, rows, rows)

    def test_frontier_support_can_span_multiple_records(self):
        rows = [source("one", 100, ["edit.py:1:4"]), source("two", 100, ["edit.py:5:10"])]
        result = frozen_edit_intervention("same-size-random", self.frontier, rows, rows)
        self.assertEqual(len(result["observations"]), 2)

    def test_redundant_mandatory_fragment_does_not_bind_random_selection(self):
        fragment = source("fragment", 100, ["edit.py:1:4"])
        complete = source("complete", 200, ["edit.py:1:10"])
        alternative = source("alternative", 100, ["other.py:1:4"])
        pool = [fragment, complete, alternative]
        selected = [fragment, complete]
        options = set()
        for seed in range(8):
            packet = frozen_edit_intervention("same-size-random", self.frontier, pool, selected, seed=seed)
            options.add(tuple(row["id"] for row in packet["observations"]))
        self.assertTrue(any("fragment" not in option for option in options))
        self.assertTrue(all("complete" in option for option in options))

    def test_random_seed_is_repeatable(self):
        a = frozen_edit_intervention("same-size-random", self.frontier, self.pool, self.selected, seed=7)
        b = frozen_edit_intervention("same-size-random", self.frontier, self.pool, self.selected, seed=7)
        self.assertEqual(a, b)

    def test_random_matches_actual_selected_tokens_not_just_nominal_cap(self):
        edit = source("edit", 1000, ["edit.py:1:10"])
        selected_support = source("selected", 1500, ["a.py:1:10"])
        alternative = source("alternative", 1500, ["b.py:1:10"])
        oversized = source("oversized", 12000, ["c.py:1:10"])
        pool = [edit, selected_support, alternative, oversized]
        selected = [edit, selected_support]
        choices = set()
        for seed in range(8):
            packet = frozen_edit_intervention("same-size-random", self.frontier, pool, selected, seed=seed)
            self.assertEqual(packet["tokens"], 2500)
            self.assertEqual(packet["selection_budget"], 2500)
            self.assertEqual(packet["token_cap"], 16000)
            self.assertTrue(all(row in pool for row in packet["observations"]))
            choices.add(tuple(row["id"] for row in packet["observations"]))
        self.assertGreater(len(choices), 1)

    def test_fact_redundancy_removal_preserves_fixed_incidence(self):
        edit = source("edit", 2000, ["edit.py:1:10"])
        one = source("one", 7000, ["first.py:1:10"])
        two = source("two", 7000, ["second.py:1:10"])
        rows = [edit, one, two]
        requirements = [{"id": "covered", "source_ids": ["one", "two"]},
                        {"id": "uncovered", "source_ids": ["absent"]}]
        packet = frozen_edit_intervention("duplicates-removed", self.frontier, rows, rows,
                                         requirements=requirements)
        retained = {row["id"] for row in packet["observations"]}
        self.assertEqual(packet["tokens"], 9000)
        self.assertIn("edit", retained)
        self.assertTrue(retained.intersection({"one", "two"}))
        self.assertNotIn("absent", retained)

    def test_fact_preservation_does_not_expand_twelve_k_cap(self):
        edit = source("edit", 1000, ["edit.py:1:10"])
        support = source("support", 12000, ["support.py:1:10"])
        rows = [edit, support]
        with self.assertRaisesRegex(ValueError, "12k"):
            frozen_edit_intervention("duplicates-removed", self.frontier, rows, rows,
                                    requirements=[{"id": "fact", "source_ids": ["support"]}])

    def test_required_removal_does_not_remove_all_unrelated_reference_support(self):
        rows = [source("edit", 2000, ["edit.py:1:10"]),
                source("a", 3000, ["a.py:1:10"]),
                source("b", 3000, ["b.py:1:10"]),
                source("noise", 5000, ["noise.py:1:10"])]
        packet = frozen_edit_intervention("required-evidence-removed", self.frontier, rows, rows,
            reference_support_units=["a.py:1:10", "b.py:1:10"])
        retained = {row["id"] for row in packet["observations"]}
        self.assertNotIn("a", retained)
        self.assertIn("b", retained)
        self.assertIn("noise", retained)

    def test_required_removal_never_drops_unrelated_source_to_fit_cap(self):
        rows = [source("edit", 1000, ["edit.py:1:10"]),
                source("support", 1000, ["support.py:1:10"]),
                source("unrelated", 12000, ["noise.py:1:10"])]
        with self.assertRaisesRegex(ValueError, "12k"):
            frozen_edit_intervention("required-evidence-removed", self.frontier, rows, rows,
                reference_support_units=["support.py:1:10"])


if __name__ == "__main__":
    unittest.main()
