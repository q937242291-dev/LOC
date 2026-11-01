"""New counterexamples for fixed-map coverage, visibility and task boundaries."""
from dataclasses import replace
import unittest

from locus.config import Profile, Settings
from locus.gate import evidence_gate, visible_packet
from locus.ledger import BudgetExhausted, Ledger
from locus.packet import materialize
from locus.types import Alternative, Candidate, Grounding, Location, Observation, Packet, Requirement


def observation(text, path, *, editable=False):
    return Observation.create(text, Location("repository", "base", path, 1, 1), len,
                              frozenset({"editable" if editable else "support"}))


def candidate(source):
    return Candidate("edit", source.locations[0], "f", "repair", 1)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class PacketContracts(unittest.TestCase):
    def test_positive_small_weight_fact_survives_float_sum_rounding(self):
        edit = observation("implementation", "edit.py", editable=True)
        proof = observation("dependent behavior", "helper.py")
        requirements = (
            Requirement("large", "edit", "edit", frozenset({edit.id}), weight=1e16),
            Requirement("small", "preserve behavior", "behavior", frozenset({proof.id}), weight=1),
        )
        ground = Grounding((candidate(edit),), (), requirements, {s.id: s for s in (edit, proof)})
        packet, missing = materialize(ground, edit.tokens + proof.tokens)
        self.assertIn(proof.id, packet.ids)
        self.assertEqual(missing, ())

    def test_hash_merge_counts_union_metadata_at_exact_cap(self):
        edit = observation("same text", "edit.py", editable=True)
        alias = observation("same text", "caller.py")
        merged = edit.merge(alias, len)
        requirement = Requirement("edit", "implementation", "edit", frozenset({edit.id}))
        ground = Grounding((candidate(edit),), (), (requirement,), {"first": edit, "second": alias})
        packet, missing = materialize(ground, merged.tokens, count=len)
        self.assertEqual(len(packet.sources), 1)
        self.assertEqual(set(packet.sources[0].locations), set(merged.locations))
        self.assertEqual(packet.tokens, merged.tokens)
        self.assertEqual(missing, ())
        too_small, unmet = materialize(ground, merged.tokens - 1, count=len)
        self.assertEqual(too_small.sources, ())
        self.assertEqual(unmet, (requirement,))

    def test_existing_full_canonical_record_supplies_exact_merge_count(self):
        edit = observation("same text", "edit.py", editable=True)
        alias = observation("same text", "caller.py")
        canonical = edit.merge(alias, len)
        ground = Grounding((candidate(edit),), (), (), {"part": edit, "full": canonical})
        packet, _ = materialize(ground, canonical.tokens)
        self.assertEqual(packet.sources, (canonical,))

    def test_unmerged_metadata_requires_actual_counter(self):
        edit = observation("same text", "edit.py", editable=True)
        alias = observation("same text", "caller.py")
        ground = Grounding((candidate(edit),), (), (), {"one": edit, "two": alias})
        with self.assertRaisesRegex(ValueError, "count"):
            materialize(ground, 10000)

    def test_invalid_low_level_token_counts_do_not_enter_packet_budget(self):
        edit = observation("implementation", "edit.py", editable=True)
        for invalid in (0, -1, True, 1.5, float("nan")):
            with self.subTest(tokens=invalid):
                dirty = replace(edit, tokens=invalid)
                ground = Grounding((candidate(edit),), (), (), {dirty.id: dirty})
                with self.assertRaises(ValueError):
                    materialize(ground, 10000)

    def test_counter_remeasures_low_level_record_before_admission(self):
        edit = observation("implementation", "edit.py", editable=True)
        underestimated = replace(edit, tokens=1)
        required = Requirement("edit", "implementation", "edit", frozenset({edit.id}))
        ground = Grounding((candidate(edit),), (), (required,), {edit.id: underestimated})
        packet, unmet = materialize(ground, 1, count=len)
        self.assertEqual(packet.sources, ())
        self.assertEqual(unmet, (required,))

    def test_payload_hash_is_validated_before_selection(self):
        edit = observation("implementation", "edit.py", editable=True)
        dirty = replace(edit, text="different implementation")
        ground = Grounding((candidate(edit),), (), (), {dirty.id: dirty})
        with self.assertRaisesRegex(ValueError, "hash"):
            materialize(ground, 10000)

    def test_fixed_map_never_infers_incidence_from_matching_text(self):
        edit = observation("implementation", "edit.py", editable=True)
        unrelated = observation("preserve caller contract", "helper.py")
        unknown = Requirement("missing", unrelated.text, "behavior", frozenset())
        ground = Grounding((candidate(edit),), (), (unknown,), {s.id: s for s in (edit, unrelated)})
        _, unmet = materialize(ground, 10000)
        self.assertEqual(unmet, (unknown,))
        self.assertEqual(ground.requirements[0].source_ids, frozenset())


class GateContracts(unittest.TestCase):
    def test_intended_change_requires_a_new_full_stability_window(self):
        edit = observation("implementation", "edit.py", editable=True)
        current = candidate(edit)
        previous = replace(current, change="different intended behavior")
        packet = Packet((current,), (edit,))
        history = [(previous.rank_key,), (current.rank_key,)]
        self.assertFalse(evidence_gate(packet, (), (), history, 2).rank_stable)
        history.append((current.rank_key,))
        self.assertTrue(evidence_gate(packet, (), (), history, 2).rank_stable)

    def test_rank_signatures_accept_json_lists_and_ignore_score_changes(self):
        edit = observation("implementation", "edit.py", editable=True)
        current = candidate(edit)
        previous = replace(current, score=0.3, change="  repair  ")
        packet = Packet((current,), (edit,))
        history = [[list(previous.rank_key)], [list(current.rank_key)]]
        self.assertTrue(evidence_gate(packet, (), (), history, 2).rank_stable)
        self.assertFalse(evidence_gate(packet, (), (), history[-1:], 2).rank_stable)

    def test_retained_alternative_loses_declared_check_when_proof_is_evicted(self):
        edit = observation("implementation", "edit.py", editable=True)
        proof = observation("why competing edit remains", "proof.py")
        edit_candidate = candidate(edit)
        alternative = Alternative("rival", edit_candidate.id, "retained", frozenset({proof.id}))
        packet = Packet((edit_candidate,), (edit, proof))
        history = [(edit_candidate.id,)] * 2
        class Recipient:
            def has(self, source):
                return source.version == edit.version
        delivered = visible_packet(packet, Recipient())
        decision = evidence_gate(delivered, (), (alternative,), history, 2)
        self.assertTrue(decision.edit_support)
        self.assertFalse(decision.alternatives_checked)
        self.assertFalse(decision.passed)
        self.assertTrue(evidence_gate(packet, (), (alternative,), history, 2).passed)


class LedgerContracts(unittest.TestCase):
    def test_eq8_reserve_does_not_shift_with_time_between_clock_reads(self):
        class TickingClock(Clock):
            def __call__(self):
                value = self.now
                self.now += 1
                return value
        clock = TickingClock()
        ledger = Ledger(Settings(deadline_seconds=100, reserve_seconds=10), clock)
        ledger.restrict(Profile(64, 1))
        self.assertEqual(ledger.acq_time_end, 90)

    def test_summary_uses_one_elapsed_snapshot_for_worker_cost_and_wall_time(self):
        class TickingClock(Clock):
            def __call__(self):
                value = self.now
                self.now += 1
                return value
        ledger = Ledger(Settings(), TickingClock())
        summary = ledger.summary()
        self.assertAlmostEqual(summary["worker_hours"] * 3600, summary["elapsed_seconds"])

    def test_submission_timestamp_is_stable_across_repeated_finish(self):
        clock = Clock()
        ledger = Ledger(Settings(), clock)
        clock.now = 3
        ledger.finish()
        submitted = ledger.summary()
        clock.now = 100
        ledger.finish()
        self.assertEqual(ledger.elapsed, 3)
        self.assertEqual(ledger.summary()["worker_hours"], submitted["worker_hours"])
        self.assertEqual(ledger.summary()["cost"], submitted["cost"])

    def test_submitted_task_cannot_consume_calls_with_frozen_elapsed_time(self):
        clock = Clock()
        ledger = Ledger(Settings(), clock)
        ledger.finish()
        with self.assertRaises(BudgetExhausted):
            with ledger.action("tool", "after-submission"):
                clock.now += 5
        self.assertEqual(ledger.summary()["calls"], 0)


if __name__ == "__main__":
    unittest.main()
