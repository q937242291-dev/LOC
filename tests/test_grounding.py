"""Re-grounding updates inferred facts without retaining obsolete gate blockers."""
from dataclasses import replace
import math
import unittest

from locus.config import Settings
from locus.gate import evidence_gate
from locus.grounding import GROUND_SYSTEM, add_reads, ground, parse_grounding
from locus.ledger import Ledger
from locus.models import ModelError, Recipient
from locus.packet import materialize
from locus.repository import RepositoryIndex
from locus.types import Candidate, Grounding, Location, Observation, Requirement, compact, uncovered


REPO, COMMIT = "example/repository", "base"


def source(text, path="module.py", start=1):
    return Observation.create(text, Location(REPO, COMMIT, path, start,
                              start + len(text.splitlines()) - 1), len)


def answer(items):
    return dict(roles=[], requirements=[], alternatives=[], candidates=[
        dict(id=str(n), path=s.locations[0].path, start=s.locations[0].start,
             end=s.locations[0].end, symbol="repair", change="correct behavior",
             signals=dict(q=1, b=1, r=0, c=1, h=0),
             signal_sources=dict(q=[s.id], b=[s.id], r=[], c=[s.id], h=[]))
        for n, s in enumerate(items)])


class GroundingFixture:
    def setUp(self):
        self.settings = Settings()
        self.edit = source("def repair():\n    return 1")
        self.proof = source("contract = 'required'", "contract.toml")
        self.index = RepositoryIndex({s.locations[0].path:s.text
                                      for s in (self.edit, self.proof)}, self.settings)

    def parse(self, response, previous=None):
        pool = previous.pool if previous else {s.id:s for s in (self.edit, self.proof)}
        return parse_grounding(response, pool, REPO, COMMIT, self.index,
                               self.settings, len, previous=previous)


class GroundingTransitionsTests(GroundingFixture, unittest.TestCase):
    def test_source_rejection_replaces_old_unchecked_requirement(self):
        initial = answer((self.edit,))
        initial["alternatives"] = [dict(id="rival", candidate="other", status="unchecked")]
        first = self.parse(initial)
        self.assertIn("alternative:rival", {r.id for r in first.requirements})
        updated = answer((self.edit,))
        updated["alternatives"] = [dict(id="rival", candidate="other", status="rejected",
                                        source_ids=[self.proof.id], explanation="contract rules it out")]
        current = self.parse(updated, first)
        self.assertNotIn("alternative:rival", {r.id for r in current.requirements})
        packet, missing = materialize(current, 10000)
        self.assertFalse(missing)
        self.assertIn(self.proof.id, packet.ids)
        history = [tuple(c.id for c in current.frontier)] * 2
        self.assertTrue(evidence_gate(packet, current.requirements,
                                     current.alternatives, history, 2).passed)

    def test_retained_competing_edit_replaces_old_unchecked_requirement(self):
        initial = answer((self.edit, self.proof))
        initial["alternatives"] = [dict(id="rival", candidate="1", status="unchecked")]
        first = self.parse(initial)
        updated = answer((self.edit, self.proof))
        updated["alternatives"] = [dict(id="rival", candidate="1", status="retained")]
        current = self.parse(updated, first)
        packet, missing = materialize(current, 10000)
        self.assertFalse(missing)
        self.assertNotIn("alternative:rival", {r.id for r in current.requirements})
        history = [tuple(c.id for c in current.frontier)] * 2
        self.assertTrue(evidence_gate(packet, current.requirements,
                                     current.alternatives, history, 2).passed)

    def test_displaced_competing_edit_needs_a_new_source_check(self):
        initial = answer((self.edit, self.proof))
        initial["alternatives"] = [dict(id="rival", candidate="1", status="retained")]
        first = self.parse(initial)
        current = self.parse(answer((self.edit,)), first)
        self.assertEqual(current.alternatives[0].status, "unchecked")
        self.assertIn("alternative:rival", {r.id for r in uncovered(
            current.pool.values(), current.requirements)})

    def test_new_rejection_drops_obsolete_proof_requirements(self):
        initial = answer((self.edit,))
        initial["alternatives"] = [dict(id="rival", candidate="other", status="rejected",
                                        source_ids=[self.proof.id])]
        first = self.parse(initial)
        updated = answer((self.edit,))
        updated["alternatives"] = [dict(id="rival", candidate="other", status="rejected",
                                        source_ids=[self.edit.id])]
        current = self.parse(updated, first)
        self.assertNotIn("alternative:rival:"+self.proof.id,
                         {r.id for r in current.requirements})

    def test_nested_protocol_errors_become_model_errors(self):
        for response in (dict(candidates=None), dict(roles=None),
                         dict(candidates=[None]), dict(alternatives=[None])):
            with self.subTest(response=response), self.assertRaises(ModelError):
                self.parse(response)

    def test_read_covered_by_adjacent_valid_intervals_is_not_repeated(self):
        items = (source("first\nsecond"), source("third\nfourth", start=3))
        index = RepositoryIndex({"module.py":"first\nsecond\nthird\nfourth"}, self.settings)

        class Workspace:
            repo, commit = REPO, COMMIT
            def read(self, *args):
                raise AssertionError("A fully seen interval must not be reread")

        ledger = Ledger(self.settings)
        pool, added = add_reads([dict(path="module.py", start=2, end=4)],
                               {s.id:s for s in items}, Workspace(), index, ledger,
                               self.settings, len)
        self.assertEqual(len(pool), 2)
        self.assertFalse(added)
        self.assertEqual(len(ledger.steps), 0)


class EditableConfigurationTests(unittest.TestCase):
    def test_source_validation_requires_coordinates_and_valid_roles(self):
        item = source("implementation")
        index = RepositoryIndex({"module.py":item.text}, Settings())
        for invalid in (replace(item, locations=()), replace(item, roles=frozenset()),
                        replace(item, roles=frozenset({"unknown"}))):
            self.assertFalse(index.validate(invalid, REPO, COMMIT))

    def test_build_and_module_configuration_remain_available_to_grounding(self):
        texts = {"go.mod":"module example.org/project\ngo 1.23",
                 "Makefile":"repair:\n\tpython -m unittest",
                 "src/worker.mts":"export function repair() { return 1; }",
                 "app.conf":"repair.enabled = true"}
        settings = Settings()
        index = RepositoryIndex(texts, settings)
        self.assertEqual(set(index.texts), set(texts))
        items = tuple(source(text, path) for path, text in texts.items())
        state = parse_grounding(answer(items), {s.id:s for s in items}, REPO, COMMIT,
                                index, settings, len)
        packet, missing = materialize(state, 10000)
        self.assertEqual(len(packet.frontier), len(texts))
        self.assertFalse(missing)


class MandatoryReservationTests(unittest.TestCase):
    def test_overlapping_fragment_does_not_make_full_source_exceed_cap(self):
        complete = source("first\nsecond\nthird\nfourth")
        fragment = source("first")
        complete = complete.with_roles({"editable"}, len)
        fragment = fragment.with_roles({"editable"}, len)
        self.assertLess(fragment.tokens, complete.tokens)
        candidate = Candidate("edit", complete.locations[0], "repair", "correct behavior", 1)
        required = Requirement("edit", "editable implementation", "edit", frozenset({complete.id}))
        state = Grounding((candidate,), (), (required,),
                          {s.id:s for s in (fragment, complete)})
        packet, missing = materialize(state, complete.tokens)
        self.assertEqual(packet.ids, frozenset({complete.id}))
        self.assertFalse(missing)


class GroundingProtocolTests(GroundingFixture, unittest.TestCase):
    def test_grounding_cannot_cite_shared_source_absent_from_its_request(self):
        for part in ("signals", "requirements", "alternatives", "roles"):
            response = answer((self.edit,))
            if part == "signals":
                response["candidates"][0]["signal_sources"]["q"] = [self.proof.id]
            elif part == "requirements":
                response[part] = [dict(id="contract", candidate="0", description="caller contract",
                                       source_ids=[self.proof.id])]
            elif part == "alternatives":
                response[part] = [dict(id="rival", candidate="other", status="rejected",
                                       source_ids=[self.proof.id])]
            else:
                response[part] = [dict(source_id=self.proof.id, roles=["support"])]
            with self.subTest(part=part), self.assertRaises(ModelError):
                parse_grounding(response, {s.id:s for s in (self.edit, self.proof)},
                                REPO, COMMIT, self.index, self.settings, len,
                                visible_ids={self.edit.id})

    def test_candidate_cannot_use_shared_coordinates_absent_from_its_request(self):
        with self.assertRaises(ModelError):
            parse_grounding(answer((self.proof,)), {s.id:s for s in (self.edit, self.proof)},
                            REPO, COMMIT, self.index, self.settings, len,
                            visible_ids={self.edit.id})

    def test_same_location_competing_changes_have_distinct_candidate_identities(self):
        response = answer((self.edit,))
        rival = {**response["candidates"][0], "id":"rival", "change":"preserve the old behavior"}
        response["candidates"].append(rival)
        current = self.parse(response)
        self.assertEqual(len(current.frontier), 2)
        self.assertEqual(len({c.id for c in current.frontier}), 2)
        self.assertEqual(len({c.location for c in current.frontier}), 1)

    def test_reordered_same_candidates_keep_unresolved_behavioral_fact(self):
        response = answer((self.edit, self.proof))
        response["requirements"] = [dict(id="caller", candidate="0", description="caller contract",
                                         source_ids=[])]
        first = self.parse(response)
        later = answer((self.edit, self.proof))
        later["candidates"][1]["signals"]["q"] = .1
        current = self.parse(later, first)
        self.assertNotEqual(tuple(c.id for c in first.frontier), tuple(c.id for c in current.frontier))
        self.assertTrue(any(r.id.endswith(":caller") for r in current.requirements))

    def test_retained_alternative_proof_is_materialized_for_visibility_gate(self):
        response = answer((self.edit,))
        response["alternatives"] = [dict(id="rival", candidate="0", status="retained",
                                          source_ids=[self.proof.id])]
        current = self.parse(response)
        packet, missing = materialize(current, 10000)
        self.assertIn(self.proof.id, packet.ids)
        self.assertFalse(missing)

    def test_fractional_or_boolean_source_coordinates_are_rejected(self):
        for value in (True, 1.5, "1"):
            response = answer((self.edit,))
            response["candidates"][0]["start"] = value
            with self.subTest(value=value), self.assertRaises(ModelError):
                self.parse(response)

    def test_non_numeric_signals_are_rejected_without_coercion(self):
        for value in (True, "0.5", None):
            response = answer((self.edit,))
            response["candidates"][0]["signals"]["q"] = value
            with self.subTest(value=value), self.assertRaises(ModelError):
                self.parse(response)

    def test_invalid_serialized_token_counts_cannot_bypass_packet_cap(self):
        for value in (0, -1, True, .5, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Observation.create(self.edit.text, self.edit.locations[0], lambda _:value)
        with self.assertRaises(ValueError):
            self.edit.with_roles({"editable"}, lambda _:0)

    def test_nonfinite_limits_cannot_disable_the_resource_contract(self):
        for settings in (Settings(deadline_seconds=math.nan), Settings(reserve_seconds=math.inf),
                         Settings(total_calls=2.5), Settings(frontier_size=True)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                settings.validate()


class GroundingAcquisitionTests(unittest.TestCase):
    class Model:
        def __init__(self, settings, ledger, response):
            self.recipient = Recipient(GROUND_SYSTEM, len, settings.request_input_tokens)
            self.ledger, self.response = ledger, response

        def session(self, role, system):
            return self.recipient

        def ask(self, role, system, prompt, sources=()):
            if not self.recipient.deliver(sources, extra=prompt):
                raise ModelError("Source exceeds context")
            self.recipient.request_messages(prompt, frozenset(s.id for s in sources))
            with self.ledger.action("model", role):
                self.recipient.complete(prompt, compact(self.response))
                return self.response

    class Workspace:
        repo, commit = REPO, COMMIT

        def __init__(self, texts, ledger):
            self.texts, self.ledger, self.read_paths = texts, ledger, []

        def read(self, path, start, end):
            with self.ledger.action("tool", "source-read"):
                self.read_paths.append(path)
                return ("\n".join(self.texts[path].splitlines()[start-1:end]),
                        Location(REPO, COMMIT, path, start, end))

    def test_unchecked_alternative_relations_are_acquired_within_depth(self):
        main = source("def main():\n    return 1", "main.py")
        rival = source("def rival():\n    return discriminate()", "rival.py")
        texts = {"main.py":main.text, "rival.py":rival.text,
                 "proof.py":"def discriminate():\n    return False"}
        settings = Settings(frontier_size=1)
        ledger = Ledger(settings)
        response = answer((main, rival))
        response["alternatives"] = [dict(id="rival", candidate="1", status="unchecked")]
        workspace = self.Workspace(texts, ledger)
        state = ground("repair main", {s.id:s for s in (main, rival)}, workspace,
                       RepositoryIndex(texts, settings), self.Model(settings, ledger, response),
                       ledger, settings, len, depth=1)
        self.assertEqual(len(state.frontier), 1)
        self.assertEqual(state.frontier[0].location.path, "main.py")
        self.assertIn("proof.py", workspace.read_paths)

    def test_request_context_bound_is_enforced_in_actual_grounding_loop(self):
        main = source("def main():\n    return 1", "main.py")
        hidden = source("x" * 200000, "hidden.py")
        settings = Settings()
        ledger = Ledger(settings)
        texts = {"main.py":main.text, "hidden.py":hidden.text}
        response = answer((main,))
        response["candidates"][0]["signal_sources"]["q"] = [hidden.id]
        workspace = self.Workspace(texts, ledger)
        with self.assertRaises(ModelError):
            ground("repair main", {s.id:s for s in (main, hidden)}, workspace,
                   RepositoryIndex(texts, settings), self.Model(settings, ledger, response),
                   ledger, settings, len, depth=0)


if __name__ == "__main__":
    unittest.main()
