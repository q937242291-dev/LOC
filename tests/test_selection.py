"""Contrasts fixed-rank source records with coverage and exact deduplication."""
from dataclasses import replace
import unittest

from locus.packet import materialize
from locus.selection import select_packet
from locus.types import Candidate, Grounding, Location, Observation, Requirement, supported


def source(text, path, *, start=1, end=2, roles=("support",), tokens=10):
    item = Observation.create(text, Location("repo", "base", path, start, end), len, frozenset(roles))
    return replace(item, tokens=tokens)


def grounding(extra=(), requirements=()):
    edit = source("edit source", "edit.py", roles=("editable",))
    frontier = (Candidate("edit", edit.locations[0], "edit", "repair", 1),)
    pool = {s.id: s for s in (edit, *extra)}
    return Grounding(frontier, (), tuple(requirements), pool), edit


class SelectionTests(unittest.TestCase):
    def test_coverage_and_edit_only_delegate_to_algorithm_four(self):
        context = source("support", "support.py")
        req = Requirement("support", "dependent behavior", "behavior", frozenset({context.id}))
        ground, _ = grounding([context], [req])
        for policy in ("coverage", "edit_only"):
            self.assertEqual(select_packet(ground, 20, policy), materialize(ground, 20, policy=policy))

    def test_static_uses_fixed_order_instead_of_new_requirement_gain(self):
        irrelevant = source("unrelated", "first.py")
        support = source("necessary", "second.py")
        req = Requirement("required", "behavior", "behavior", frozenset({support.id}))
        ground, edit = grounding([support, irrelevant], [req])
        static, missing = select_packet(ground, 20, "static_topk", [irrelevant, edit, support])
        coverage, covered = select_packet(ground, 20, "coverage")
        self.assertEqual([s.id for s in static.sources], [irrelevant.id, edit.id])
        self.assertEqual(missing, (req,))
        self.assertEqual([s.id for s in coverage.sources], [edit.id, support.id])
        self.assertEqual(covered, ())

    def test_static_retains_equal_text_at_distinct_coordinates(self):
        one = source("identical helper body", "first.py")
        two = source("identical helper body", "second.py")
        merged = replace(one.merge(two, len), tokens=15)
        ground, edit = grounding([merged])
        packet, _ = select_packet(ground, 30, "static_topk", [edit, one, two])
        self.assertEqual(len(packet.sources), 3)
        self.assertEqual(packet.sources[1].id, packet.sources[2].id)
        self.assertNotEqual(packet.sources[1].version, packet.sources[2].version)
        self.assertEqual(packet.sources[1].locations, one.locations)
        self.assertEqual(packet.sources[2].locations, two.locations)

    def test_exact_dedup_merges_hash_roles_locations_and_recounts(self):
        one = source("same", "first.py", roles=("support",))
        two = source("same", "second.py", roles=("symptom",))
        canonical = one.merge(two, lambda text: 17)
        ground, edit = grounding([canonical])
        packet, _ = select_packet(ground, 100, "exact_dedup", [one, two], count=lambda text: 17)
        self.assertEqual(len(packet.sources), 1)
        helper = packet.sources[0]
        self.assertEqual(helper.id, one.id)
        self.assertEqual(set(helper.locations), set(one.locations + two.locations))
        self.assertEqual(helper.roles, frozenset({"support", "symptom"}))
        self.assertEqual(helper.tokens, 17)
        self.assertEqual(packet.tokens, 17)

    def test_static_raw_roles_follow_current_pool(self):
        raw = source("same", "support.py", roles=("editable",))
        current = replace(raw, roles=frozenset({"support"}), tokens=19)
        ground, _ = grounding([current])
        packet, _ = select_packet(ground, 100, "static_topk", [raw], count=lambda text: 19)
        self.assertEqual(packet.sources[-1].roles, frozenset({"support"}))
        self.assertEqual(packet.sources[-1].locations, raw.locations)
        self.assertEqual(packet.sources[-1].tokens, 19)

    def test_static_does_not_apply_exact_dedup_to_ranked_records(self):
        ground, edit = grounding()
        packet, _ = select_packet(ground, 100, "static_topk", [edit, edit])
        self.assertEqual(packet.sources, (edit, edit))

    def test_static_does_not_insert_merged_edit_record_ahead_of_raw_rank(self):
        first = source("edit source", "edit.py", roles=("editable",))
        second = source("edit source", "other.py", roles=("support",))
        merged = first.merge(second, lambda text: 20)
        ground, _ = grounding([merged])
        packet, _ = select_packet(ground, 100, "static_topk", [first, second], count=lambda text: 10)
        self.assertEqual(packet.sources[0].locations, first.locations)
        self.assertEqual(packet.sources[1].locations, second.locations)
        self.assertEqual(len(packet.sources), 2)

    def test_overweight_ranked_source_returns_no_source_and_all_unmet(self):
        helper = source("helper", "helper.py")
        req = Requirement("r", "fact", "behavior", frozenset({helper.id}))
        ground, _ = grounding([helper], [req])
        packet, missing = select_packet(ground, 9, "static_topk", [helper])
        self.assertEqual(packet.sources, ())
        self.assertEqual(missing, (req,))

    def test_overweight_records_are_not_truncated(self):
        large = source("large source", "large.py", tokens=100)
        small = source("small source", "small.py", tokens=3)
        ground, edit = grounding([large, small])
        packet, _ = select_packet(ground, 13, "static_topk", [large, small])
        self.assertEqual(packet.sources, (small,))
        self.assertEqual(packet.tokens, 3)
        self.assertFalse(supported(ground.frontier, packet.sources))

    def test_pool_order_is_fallback_and_broad_uses_supplied_cap(self):
        one, two = source("first", "first.py"), source("second", "second.py")
        ground, edit = grounding([two, one])
        packet, _ = select_packet(ground, 20, "broad")
        self.assertEqual(packet.sources, (edit, two))

    def test_exact_merge_can_reuse_existing_validated_count(self):
        one, two = source("same", "first.py"), source("same", "second.py")
        canonical = one.merge(two, lambda text: 13)
        ground, _ = grounding([canonical])
        packet, _ = select_packet(ground, 13, "exact_dedup", [one, two])
        self.assertEqual(packet.sources[-1].tokens, 13)
        self.assertEqual(packet.sources[-1].locations, canonical.locations)

    def test_unvalidated_raw_coordinates_are_excluded(self):
        raw = source("helper", "unvalidated.py")
        canonical = source("helper", "validated.py")
        ground, edit = grounding([canonical])
        packet, _ = select_packet(ground, 100, "static_topk", [raw])
        self.assertEqual(packet.sources, ())

    def test_coordinate_free_raw_source_is_excluded(self):
        canonical = source("helper", "helper.py")
        ground, edit = grounding([canonical])
        packet, _ = select_packet(ground, 100, "static_topk", [replace(canonical, locations=())])
        self.assertEqual(packet.sources, ())

    def test_fixed_rank_baselines_do_not_force_edit_support(self):
        support = source("support", "support.py")
        ground, edit = grounding([support])
        for policy in ("static_topk", "exact_dedup", "broad"):
            packet, _ = select_packet(ground, 10, policy, [support, edit])
            self.assertEqual(packet.sources, (support,))
            self.assertFalse(supported(ground.frontier, packet.sources))

    def test_roles_requiring_new_count_are_not_estimated(self):
        one, two = source("same", "first.py"), source("same", "second.py")
        canonical = replace(one.merge(two, lambda text: 15), roles=frozenset({"symptom"}))
        ground, _ = grounding([canonical])
        with self.assertRaisesRegex(ValueError, "count=tokenizer"):
            select_packet(ground, 100, "static_topk", [one, two])


if __name__ == "__main__":
    unittest.main()
