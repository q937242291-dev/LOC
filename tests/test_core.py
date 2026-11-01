"""Executable contracts for the LOCUS controller's deterministic components."""
from __future__ import annotations

from dataclasses import replace
import io
import json
import math
import unittest
from unittest.mock import patch

from locus.config import DEBATE_ROLES, PROFILES, Profile, Settings
from locus.gate import evidence_gate, visible_packet
from locus.ledger import BudgetExhausted, Ledger
from locus.models import ModelError, Recipient, usage_counts
from locus.packet import coverage, materialize
from locus.types import (
    Alternative, Candidate, Grounding, Location, Observation, Packet,
    Requirement, covers_location, supported, uncovered,
)
from locus.voting import debate_vote, plurality


def source(text, *, start=1, end=1, path="module.py", roles=("support",), tokens=None):
    observation = Observation.create(
        text, Location("example/repository", "base", path, start, end),
        len, frozenset(roles),
    )
    return replace(observation, tokens=tokens) if tokens is not None else observation


def candidate(*, start=1, end=1, path="module.py", name="edit"):
    return Candidate(
        name, Location("example/repository", "base", path, start, end),
        "function", "repair behavior", 1.0,
    )


def requirement(name, *sources, weight=1.0):
    return Requirement(name, name, "behavior", frozenset(x.id for x in sources), weight=weight)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class LedgerTests(unittest.TestCase):
    def test_eq8_subtracts_debate_and_structure_and_protects_reserves(self):
        clock = Clock()
        settings = Settings(total_calls=20, deadline_seconds=100, reserve_calls=4, reserve_seconds=15)
        ledger = Ledger(settings, clock)
        for _ in range(5):
            with ledger.action("tool", "structure"):
                clock.advance(2)
        ledger.restrict(Profile(18, 3))
        self.assertEqual(ledger.allowance(), (11, 75))
        for _ in range(11):
            with ledger.action("tool", "acquisition"):
                pass
        with self.assertRaises(BudgetExhausted):
            with ledger.action("tool", "acquisition"):
                pass
        ledger.repair_phase()
        self.assertEqual(ledger.allowance(), (4, 90))

    def test_profile_is_additional_acquisition_allowance(self):
        ledger = Ledger(Settings(total_calls=250, reserve_calls=40), Clock())
        for _ in range(16):
            with ledger.action("model", "debate"):
                pass
        ledger.restrict(PROFILES["L"])
        self.assertEqual(ledger.allowance()[0], 64)
        self.assertEqual(ledger.acq_call_end, 80)

    def test_exploration_includes_setup_and_failed_acquisition_before_repair(self):
        clock = Clock()
        ledger = Ledger(Settings(), clock)
        with ledger.action("tool", "structure"):
            clock.advance(2)
        with ledger.action("model", "debate"):
            clock.advance(3)
        ledger.restrict(PROFILES["L"])
        with self.assertRaises(ModelError):
            with ledger.action("model", "grounding"):
                clock.advance(5)
                raise ModelError("invalid response")
        with ledger.action("tool", "read"):
            clock.advance(7)
        ledger.repair_phase()
        with ledger.action("model", "patch"):
            clock.advance(11)
        with ledger.action("tool", "recovery"):
            clock.advance(13)
        ledger.finish()
        summary = ledger.summary()
        self.assertEqual(summary["calls"], 6)
        self.assertEqual(summary["llm_seconds"], 19)
        self.assertEqual(summary["exploration_calls"], 4)
        self.assertEqual(summary["exploration_model_calls"], 2)
        self.assertEqual(summary["exploration_tool_calls"], 2)
        self.assertEqual(summary["exploration_llm_seconds"], 8)
        self.assertEqual(summary["exploration_tool_seconds"], 9)
        self.assertEqual([step["phase"] for step in summary["steps"]],
                         ["setup", "setup", "acquisition", "acquisition", "repair", "repair"])
        self.assertFalse(summary["steps"][2]["success"])

    def test_negative_allowance_is_clamped(self):
        clock = Clock()
        settings = Settings(total_calls=3, deadline_seconds=10, reserve_calls=5, reserve_seconds=20)
        ledger = Ledger(settings, clock)
        ledger.restrict(PROFILES["H"])
        self.assertEqual(ledger.allowance(), (0, 0))
        with self.assertRaises(BudgetExhausted):
            ledger.check()

    def test_failed_request_keeps_billed_tokens_and_consumed_call(self):
        clock = Clock()
        ledger = Ledger(Settings(total_calls=2), clock)
        with self.assertRaises(ModelError):
            with ledger.action("model", "grounding") as step:
                step.uncached_input = 100
                step.cached_input = 800
                step.output = 50
                clock.advance(3)
                raise ModelError("response invalid")
        summary = ledger.summary()
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["tokens"], 950)
        self.assertFalse(summary["steps"][0]["success"])
        self.assertEqual(summary["llm_seconds"], 3)
        expected = 1e-6*(100*1.25 + 800*.10 + 50*6) + .06*3/3600
        self.assertAlmostEqual(summary["cost"], expected)

    def test_worker_hours_include_overlapping_workers_separately(self):
        clock = Clock()
        ledger = Ledger(Settings(), clock)
        clock.advance(3600)
        ledger.additional_worker_hours = 2
        ledger.finish()
        clock.advance(100)
        self.assertEqual(ledger.summary()["elapsed_seconds"], 3600)
        self.assertEqual(ledger.summary()["worker_hours"], 3)

    def test_usage_categories_are_disjoint(self):
        self.assertEqual(
            usage_counts({"prompt_tokens":1000, "prompt_tokens_details":{"cached_tokens":900},
                          "completion_tokens":50}, "openai"), (100,900,50))
        self.assertEqual(
            usage_counts({"prompt_tokens":1000, "prompt_cache_hit_tokens":900,
                          "prompt_cache_miss_tokens":100, "completion_tokens":50}, "deepseek"),
            (100,900,50))
        self.assertEqual(
            usage_counts({"input_tokens":100, "cache_creation_input_tokens":200,
                          "cache_read_input_tokens":900, "output_tokens":50}, "anthropic"),
            (300,900,50))

    def test_inconsistent_provider_input_totals_are_rejected(self):
        with self.assertRaises(ModelError):
            usage_counts({"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":11}},"openai")
        with self.assertRaises(ModelError):
            usage_counts({"prompt_tokens":100,"prompt_cache_hit_tokens":80,
                          "prompt_cache_miss_tokens":30},"deepseek")

    def test_missing_or_noninteger_provider_counts_are_not_inferred_as_zero(self):
        invalid = [
            {},dict(prompt_tokens=10),dict(completion_tokens=2),
            dict(prompt_tokens=True,completion_tokens=2),
            dict(prompt_tokens=10.5,completion_tokens=2),
            dict(prompt_tokens=10,completion_tokens=-1),
        ]
        for usage in invalid:
            with self.subTest(usage=usage):
                with self.assertRaises(ModelError):
                    usage_counts(usage,"openai")
        with self.assertRaises(ModelError):
            usage_counts(dict(input_tokens=10),"anthropic")
        self.assertEqual(usage_counts(dict(prompt_tokens=0,completion_tokens=0),"openai"),(0,0,0))

    def test_reserve_is_enforced_on_the_consuming_action_not_just_the_vote_loop(self):
        clock = Clock()
        ledger = Ledger(Settings(total_calls=3,reserve_calls=2,deadline_seconds=20,
                                 reserve_seconds=5),clock)
        with ledger.action("model","debate:role",reserve=True):
            clock.advance(1)
        with self.assertRaises(BudgetExhausted):
            with ledger.action("model","debate:role",reserve=True):
                pass
        self.assertEqual(ledger.summary()["calls"],1)
        ledger.repair_phase()
        self.assertEqual(ledger.allowance(),(2,19))

    def test_missing_usage_is_unknown_instead_of_zero_cost(self):
        ledger = Ledger(Settings(),Clock())
        with self.assertRaises(ModelError):
            with ledger.action("model","patch") as step:
                step.usage_known = False
                raise ModelError("connection interrupted")
        summary = ledger.summary()
        self.assertFalse(summary["usage_complete"])
        self.assertIsNone(summary["tokens"])
        self.assertIsNone(summary["cost"])
        self.assertEqual(summary["calls"],1)

    def test_post_call_time_limit_rejects_late_result_but_keeps_its_charges(self):
        clock = Clock()
        ledger = Ledger(Settings(total_calls=10,deadline_seconds=10,
                                 reserve_calls=2,reserve_seconds=2),clock)
        ledger.restrict(PROFILES["L"])
        with self.assertRaises(BudgetExhausted):
            with ledger.action("model","grounding") as step:
                step.uncached_input,step.cached_input,step.output = 100,200,5
                clock.advance(9)
        summary = ledger.summary()
        self.assertEqual(summary["calls"],1)
        self.assertEqual(summary["tokens"],305)
        self.assertEqual(summary["llm_seconds"],9)
        self.assertEqual(ledger.allowance()[1],0)
        ledger.repair_phase()
        self.assertEqual(ledger.allowance(),(9,1))

    def test_last_allowed_call_and_exact_deadline_completion_are_accepted(self):
        clock = Clock()
        ledger = Ledger(Settings(total_calls=1,deadline_seconds=1,
                                 reserve_calls=0,reserve_seconds=0),clock)
        with ledger.action("tool","submission"):
            clock.advance(1)
        self.assertEqual(ledger.summary()["calls"],1)
        self.assertTrue(ledger.summary()["steps"][0]["success"])
        self.assertEqual(ledger.allowance(),(0,0))
        with self.assertRaises(BudgetExhausted):
            ledger.check()


class VotingTests(unittest.TestCase):
    def test_debate_roles_receive_distinct_operational_responsibilities(self):
        from locus.voting import ROLE_RESPONSIBILITIES
        settings = Settings(debate_rounds=1)
        systems = {}

        class Model:
            def ask(self,role,system,prompt):
                systems[role] = system
                return dict(label="L",argument="one local implementation")

        debate_vote("issue",{},Model(),Ledger(settings,Clock()),settings)
        self.assertEqual(len(systems),len(DEBATE_ROLES))
        for role in DEBATE_ROLES:
            self.assertIn(ROLE_RESPONSIBILITIES[role],systems["debate:"+role])
        self.assertEqual(len(set(systems.values())),len(DEBATE_ROLES))

    def test_plurality_promotes_highest_tied_label(self):
        self.assertEqual(plurality({"a":"L","b":"L","c":"M","d":"M","e":"H"}), ("M",.4))
        self.assertEqual(plurality({"a":"L","b":"H"}), ("H",.5))

    def test_no_valid_votes_selects_high_with_zero_agreement(self):
        self.assertEqual(plurality({}), ("H",0))
        self.assertEqual(plurality({"a":"invalid"}), ("H",0))

    def test_latest_valid_vote_persists_after_invalid_later_response(self):
        settings = Settings(debate_rounds=3)
        ledger = Ledger(settings, Clock())
        per_round = (
            ["L","L","M","H","H"],
            ["M","M","bad","H","H"],
            ["bad","bad","bad","H","H"],
        )

        class FixtureModel:
            def __init__(self):
                self.calls = []

            def ask(self, role, system, prompt):
                import json
                self.calls.append(json.loads(prompt))
                index = len(self.calls)-1
                return {"label": per_round[index//5][index%5], "argument":str(index)}

        model = FixtureModel()
        label, agreement, latest = debate_vote("issue", {}, model, ledger, settings)
        self.assertEqual(label, "M")
        self.assertEqual(agreement, .6)
        self.assertEqual(len(latest), 5)
        for item in model.calls[:5]:
            self.assertEqual(item["previous_arguments"], {})
        preceding = model.calls[5]["previous_arguments"]
        self.assertEqual(len(preceding), 5)
        self.assertTrue(all(x["previous_arguments"] == preceding for x in model.calls[5:10]))
        self.assertEqual(latest["debate:" + DEBATE_ROLES[0]] if "debate:" + DEBATE_ROLES[0] in latest
                         else latest[DEBATE_ROLES[0]], "M")


class SourceAndPacketTests(unittest.TestCase):
    def test_hash_merge_unions_roles_and_coordinates_and_recounts_tokens(self):
        a = source("same content", path="a.py", roles=("symptom",))
        b = source("same content", path="b.py", roles=("editable","support"))
        merged = a.merge(b, len)
        self.assertEqual(merged.id, a.id)
        self.assertEqual(len(merged.locations), 2)
        self.assertEqual(merged.roles, frozenset(("editable","support","symptom")))
        self.assertEqual(merged.tokens, len(merged.serialize()))
        self.assertNotEqual(a.version, merged.version)

    def test_edit_support_requires_correct_commit_path_role_and_full_interval(self):
        c = candidate(start=1,end=4)
        a = source("a", start=1,end=2, roles=("editable",))
        b = source("b", start=3,end=4, roles=("editable",))
        self.assertTrue(supported((c,), (a,b)))
        self.assertFalse(supported((c,), (a,)))
        self.assertFalse(supported((c,), (replace(a,roles=frozenset(("symptom",))),b)))
        self.assertFalse(supported((replace(c,location=replace(c.location,commit="other")),),(a,b)))
        self.assertFalse(supported((), (a,b)))

    def test_coverage_counts_each_requirement_once(self):
        a, b = source("a"), source("b")
        requirements = (requirement("shared",a,b,weight=2), requirement("specific",b,weight=3))
        self.assertEqual(coverage((a,),requirements), 2)
        self.assertEqual(coverage((a,b),requirements),5)
        self.assertEqual(coverage((b,),requirements),5)
        self.assertEqual(coverage((a,a),requirements),2)

    def test_mandatory_support_over_cap_returns_empty_payload_and_all_unmet(self):
        c = candidate()
        mandatory = source("implementation",roles=("editable",),tokens=6)
        r = requirement("edit",mandatory)
        g = Grounding((c,),(),(r,),{mandatory.id:mandatory})
        packet, missing = materialize(g,5)
        self.assertEqual(packet.sources,())
        self.assertEqual(missing,(r,))

    def test_greedy_selects_positive_missing_support_per_token(self):
        edit = source("implementation",roles=("editable",),tokens=3)
        cheap = source("cheap",path="support.py",tokens=2)
        large = source("large",path="large.py",tokens=8)
        redundant = source("redundant",path="duplicate.py",tokens=1)
        requirements = (
            requirement("edit",edit), requirement("behavior",cheap,large),
        )
        g = Grounding((candidate(),),(),requirements,
                      {s.id:s for s in (edit,large,redundant,cheap)})
        packet, missing = materialize(g,6)
        self.assertEqual(packet.ids, frozenset((edit.id,cheap.id)))
        self.assertEqual(packet.tokens,5)
        self.assertEqual(missing,())

    def test_pruning_preserves_coverage_and_edit_support(self):
        edit = source("implementation",roles=("editable",),tokens=2)
        a = source("a",path="a.py",tokens=2)
        b = source("b",path="b.py",tokens=3)
        # a wins the first marginal gain; b then makes a redundant.
        requirements = (requirement("edit",edit), requirement("r1",a,b),
                        requirement("r2",b))
        g = Grounding((candidate(),),(),requirements,{s.id:s for s in (edit,a,b)})
        packet, missing = materialize(g,7)
        self.assertTrue(supported(packet.frontier,packet.sources))
        self.assertEqual(coverage(packet.sources,requirements),3)
        self.assertFalse(missing)
        self.assertIn(edit.id,packet.ids)
        self.assertNotIn(a.id,packet.ids)

    def test_partial_packet_reports_exact_unmet_set(self):
        edit = source("implementation",roles=("editable",),tokens=2)
        support = source("support",path="support.py",tokens=10)
        required = requirement("behavior",support)
        g = Grounding((candidate(),),(),(requirement("edit",edit),required),
                      {s.id:s for s in (edit,support)})
        packet, missing = materialize(g,5)
        self.assertEqual(packet.ids,frozenset((edit.id,)))
        self.assertEqual(missing,(required,))

class GroundingTests(unittest.TestCase):
    @staticmethod
    def answer(sources,*,end=1):
        ids = [s.id for s in sources]
        return dict(
            roles=[],
            candidates=[dict(id="c",path="module.py",start=1,end=end,symbol="f",
                             change="repair behavior",signals=dict(q=1,b=1,r=0,c=1,h=0),
                             signal_sources={k:ids if k in {"q","b","c"} else []
                                             for k in ("q","b","r","c","h")})],
            requirements=[],alternatives=[])

    def test_grounding_score_uses_paper_weights_and_finite_range(self):
        from locus.grounding import score_signals
        settings = Settings()
        self.assertAlmostEqual(score_signals(dict(q=1,b=1,r=1,c=1,h=0),settings),1)
        self.assertAlmostEqual(score_signals(dict(q=0,b=0,r=0,c=0,h=1),settings),-.1)
        self.assertAlmostEqual(score_signals(dict(q=.5,b=.2,r=.4,c=.8,h=.3),settings),
                               .78*.5 + .22*(.5*.2+.25*.4+.25*.8)-.1*.3)
        with self.assertRaises(ModelError):
            score_signals(dict(q=math.nan,b=0,r=0,c=0,h=0),settings)

    def test_fragmented_edit_source_is_typed_and_mandatory(self):
        from locus.grounding import parse_grounding
        from locus.repository import RepositoryIndex
        settings = Settings()
        a = source("first\nsecond",start=1,end=2)
        b = source("third\nfourth",start=3,end=4)
        index = RepositoryIndex({"module.py":"first\nsecond\nthird\nfourth"},settings)
        g = parse_grounding(self.answer((a,b),end=4),{s.id:s for s in (a,b)},
                            "example/repository","base",index,settings,len)
        packet,missing = materialize(g,10000)
        self.assertTrue(supported(packet.frontier,packet.sources))
        self.assertFalse(missing)
        self.assertEqual(packet.ids,frozenset((a.id,b.id)))

    def test_normalized_read_interval_does_not_reread_eof(self):
        from locus.grounding import add_reads
        from locus.repository import RepositoryIndex
        settings = Settings()
        ledger = Ledger(settings,Clock())
        existing = source("first\nsecond",start=1,end=2)
        workspace = FixtureWorkspace(ledger,{"module.py":existing.text})
        index = RepositoryIndex(workspace.texts,settings)
        pool,added = add_reads([dict(path="module.py",start=1,end=100)],
                              {existing.id:existing},workspace,index,ledger,settings,len)
        self.assertEqual(added,[])
        self.assertEqual(workspace.read_paths,[])
        self.assertEqual(pool,{existing.id:existing})

    def test_same_frontier_keeps_omitted_support_and_rejected_alternative(self):
        from locus.grounding import parse_grounding
        from locus.repository import RepositoryIndex
        settings = Settings()
        implementation = source("implementation")
        index = RepositoryIndex({"module.py":implementation.text},settings)
        answer = self.answer((implementation,))
        answer["requirements"] = [dict(id="contract",candidate="c",kind="dependency",
                                        description="preserve caller contract",source_ids=[],targets=[])]
        answer["alternatives"] = [dict(id="other-explanation",candidate="other",status="rejected",
                                       source_ids=[implementation.id],explanation="source rejects rival")]
        initial = parse_grounding(answer,{implementation.id:implementation},
                                  "example/repository","base",index,settings,len)
        later = parse_grounding(self.answer((implementation,)),initial.pool,
                                "example/repository","base",index,settings,len,previous=initial)
        self.assertEqual({r.id for r in initial.requirements},{r.id for r in later.requirements})
        self.assertEqual(later.alternatives,initial.alternatives)
        self.assertTrue(any(r.id.endswith(":contract") for r in uncovered(later.pool.values(),
                                                                         later.requirements)))


class GateAndVisibilityTests(unittest.TestCase):
    def test_gate_needs_all_four_conditions_and_full_rank_window(self):
        c = candidate()
        edit = source("implementation",roles=("editable",))
        packet = Packet((c,),(edit,))
        requirements = (requirement("edit",edit),)
        self.assertFalse(evidence_gate(packet,requirements,(),[(c.id,)],2).passed)
        result = evidence_gate(packet,requirements,(),[(c.id,),(c.id,)],2)
        self.assertTrue(result.passed)
        self.assertFalse(evidence_gate(packet,requirements,(),[("old",),(c.id,)],2).passed)
        unknown = requirement("missing")
        self.assertFalse(evidence_gate(packet,requirements+(unknown,),(),[(c.id,)]*2,2).passed)
        symptom = replace(edit,roles=frozenset(("symptom",)))
        self.assertFalse(evidence_gate(Packet((c,),(symptom,)),requirements,(),[(c.id,)]*2,2).passed)

    def test_rejected_alternative_requires_its_sources_visible_in_packet(self):
        c = candidate()
        edit = source("implementation",roles=("editable",))
        proof = source("counterevidence",path="other.py")
        alt = Alternative("hypothesis","other","rejected",frozenset((proof.id,)))
        history = [(c.id,)]*2
        requirements = (requirement("edit",edit),)
        without = Packet((c,),(edit,))
        self.assertFalse(evidence_gate(without,requirements,(alt,),history,2).passed)
        with_proof = Packet((c,),(edit,proof))
        self.assertTrue(evidence_gate(with_proof,requirements,(alt,),history,2).passed)

    def test_retained_alternative_must_be_a_competing_edit(self):
        c = candidate()
        edit = source("implementation",roles=("editable",))
        packet = Packet((c,),(edit,))
        history = [(c.id,)]*2
        invalid = Alternative("a","outside","retained",frozenset())
        valid = replace(invalid,candidate_id=c.id)
        self.assertFalse(evidence_gate(packet,(),(invalid,),history,2).passed)
        self.assertTrue(evidence_gate(packet,(),(valid,),history,2).passed)

    def test_shared_pool_is_not_recipient_visibility(self):
        c = candidate()
        edit = source("implementation",roles=("editable",))
        required = requirement("edit",edit)
        recipient = Recipient("system",len,5000)
        packet = Packet((c,),(edit,))
        filtered = visible_packet(packet,recipient)
        self.assertFalse(evidence_gate(filtered,(required,),(),[(c.id,)]*2,2).passed)
        self.assertTrue(recipient.deliver((edit,)))
        filtered = visible_packet(packet,recipient)
        self.assertTrue(evidence_gate(filtered,(required,),(),[(c.id,)]*2,2).passed)

    def test_evict_and_retransmit_updates_visibility_and_repeated_tokens(self):
        edit = source("implementation",roles=("editable",))
        recipient = Recipient("system",len,5000)
        self.assertTrue(recipient.deliver((edit,)))
        self.assertTrue(recipient.has(edit))
        first_charge = recipient.transmitted_tokens
        self.assertTrue(recipient.deliver((edit,)))
        self.assertEqual(recipient.transmitted_tokens,first_charge)
        self.assertEqual(recipient.repeated_tokens,0)
        recipient.evict(0)
        self.assertFalse(recipient.has(edit))
        self.assertTrue(recipient.deliver((edit,)))
        self.assertEqual(recipient.transmitted_tokens,2*edit.tokens)
        self.assertEqual(recipient.repeated_tokens,edit.tokens)

    def test_failed_admission_is_atomic_and_keeps_last_admitted_state(self):
        edit = source("implementation",roles=("editable",))
        huge = source("x"*4000,path="huge.py")
        recipient = Recipient("system",len,1000)
        self.assertTrue(recipient.deliver((edit,)))
        before = (list(recipient.messages),dict(recipient.visible),
                  recipient.transmitted_tokens,recipient.repeated_tokens,
                  set(recipient.ever_delivered))
        self.assertFalse(recipient.deliver((huge,)))
        after = (recipient.messages,recipient.visible,recipient.transmitted_tokens,
                 recipient.repeated_tokens,recipient.ever_delivered)
        self.assertEqual(before,after)
        self.assertTrue(recipient.has(edit))
        self.assertFalse(recipient.has(huge))

    def test_metadata_changes_require_updated_delivery(self):
        edit = source("implementation",roles=("support",))
        typed = edit.with_roles(("editable",),len)
        recipient = Recipient("system",len,5000)
        self.assertTrue(recipient.deliver((edit,)))
        self.assertFalse(recipient.has(typed))
        self.assertTrue(recipient.deliver((typed,)))
        self.assertTrue(recipient.has(typed))
        self.assertEqual(sum(m.source_id == edit.id for m in recipient.messages),1)

    def test_preserved_static_context_cannot_evict_source_to_fit_larger_prompt(self):
        item = source("x"*100)
        recipient = Recipient("system",len,1000)
        self.assertTrue(recipient.deliver((item,)))
        recipient.preserve_empty_sources = True
        with self.assertRaises(ModelError):
            recipient.deliver((),extra="p"*550)
        self.assertTrue(recipient.has(item))


class FixtureWorkspace:
    """An in-memory repository worker; every public action consumes a tool call."""
    def __init__(self, ledger, texts):
        self.ledger, self.texts = ledger, texts
        self.repo, self.commit = "example/repository", "base"
        self.patch = ""
        self.read_paths = []

    def snapshot(self):
        with self.ledger.action("tool","structure"):
            return dict(self.texts)

    def read(self,path,start,end):
        with self.ledger.action("tool","source-read"):
            self.read_paths.append(path)
            lines = self.texts[path].splitlines()
            end = min(end,len(lines))
            return "\n".join(lines[start-1:end]),Location(self.repo,self.commit,path,start,end)

    def apply_patch(self,value):
        with self.ledger.action("tool","patch"):
            self.patch = value

    def run(self,command,timeout=120):
        with self.ledger.action("tool","validation"):
            return dict(exit_code=0,output="local tests passed")

    def diff(self):
        with self.ledger.action("tool","submission"):
            return self.patch

    def __enter__(self):
        return self

    def __exit__(self,*args):
        return None


class FixtureModel:
    def __init__(self,settings,ledger,*,actions=None,analysis_targets=(),planned_targets=()):
        self.settings,self.ledger = settings,ledger
        self.sessions = {}
        self.actions = iter(actions or [dict(action="finish")])
        self.patch_visibility = []
        self.roles_seen = []
        self.prompts = []
        self.patch_source_counts = []
        self.patch_source_records = []
        self.analysis_targets,self.planned_targets = analysis_targets,planned_targets

    def session(self,role,system):
        if role not in self.sessions:
            self.sessions[role] = Recipient(
                system,len,self.settings.request_input_tokens,self.settings.memory)
        return self.sessions[role]

    def ask(self,role,system,prompt,sources=()):
        self.roles_seen.append(role)
        self.prompts.append((role,json.loads(prompt)))
        recipient = self.session(role,system)
        if not recipient.deliver(sources,extra=prompt):
            raise ModelError("fixture context capacity exceeded")
        if role == "patch":
            requested_ids = set(json.loads(prompt).get("source_ids",[]))
            present_ids = {m.source_id for m in recipient.messages if m.source_id}
            self.patch_visibility.append(bool(requested_ids) and requested_ids <= present_ids)
            self.patch_source_counts.append({
                sid:sum(m.source_id == sid for m in recipient.messages)
                for sid in present_ids
            })
            self.patch_source_records.append(tuple(sources))
        with self.ledger.action("model",role) as step:
            step.uncached_input = recipient.input_size(prompt)
            step.output = 20
            if role.startswith("debate:"):
                answer = dict(label="L",argument="direct local implementation")
            elif role == "grounding":
                matching = [s for s in sources
                            if any(l.path == "module.py" for l in s.locations)]
                implementation = max(matching,key=lambda s:len(s.text))
                location = next(l for l in implementation.locations if l.path == "module.py")
                answer = dict(
                    roles=[dict(source_id=implementation.id,roles=["editable"])],
                    candidates=[dict(id="local",path=location.path,start=location.start,
                                     end=location.end,symbol="normalize",
                                     change="repair normalization",
                                     signals=dict(q=1,b=1,r=0,c=1,h=0),
                                     signal_sources={k:([implementation.id] if k in {"q","b","c"} else [])
                                                     for k in ("q","b","r","c","h")})],
                    requirements=[],alternatives=[])
            elif role == "trajectory":
                answer = dict(analysis="current evidence reviewed",unresolved=[],
                              suggested_targets=list(self.analysis_targets))
            elif role == "planner":
                answer = dict(targets=list(self.planned_targets))
            elif role == "patch":
                answer = next(self.actions)
            else:
                raise AssertionError(role)
            recipient.complete(prompt,json.dumps(answer))
            return answer


class ControllerTests(unittest.TestCase):
    def test_malformed_action_is_charged_and_does_not_reach_worker(self):
        from locus.controller import Agent
        from locus.repository import RepositoryIndex
        invalid_actions = [
            dict(action="apply_patch",patch=None),dict(action="apply_patch",patch={"text":"diff"}),
            dict(action="test",command={"command":"python -m unittest"}),
            dict(action="test",command="python -m unittest",timeout=1.5),
            dict(action="test",command="python -m unittest",timeout=True),
            dict(action="read",path="module.py",start=1.5,end=2),
            dict(action="read",path="module.py",start=True,end=2),
            dict(action="read",path="module.py",start=1,end="2"),
            dict(action="read",path=None,start=1,end=2),
            dict(action="finish",patch="unexecuted"),dict(action=["finish"]),["finish"],
        ]
        for invalid in invalid_actions:
            with self.subTest(action=invalid):
                settings = Settings(total_calls=20)
                ledger = Ledger(settings,Clock())
                implementation = source("implementation",roles=("editable",))
                workspace = FixtureWorkspace(ledger,{"module.py":implementation.text})
                model = FixtureModel(settings,ledger,actions=[invalid,dict(action="finish")])
                patch_text,recovery,tests,stop = Agent(settings,model,ledger,workspace,len)._repair(
                    dict(problem_statement="Repair module.py"),Packet((candidate(),),(implementation,)),
                    dict(passed=False),(),RepositoryIndex(workspace.texts,settings))
                self.assertEqual((patch_text,tests,stop),("",[],"submitted"))
                self.assertEqual(workspace.read_paths,[])
                self.assertEqual(ledger.summary()["calls"],3)
                self.assertFalse(ledger.steps[0].success)
                self.assertEqual(ledger.steps[0].output,20)
                self.assertIn("error",model.prompts[-1][1]["feedback"])
                self.assertEqual(recovery,isinstance(invalid,dict) and invalid.get("action") == "read")

    def test_patch_action_preserves_calls_for_validation_and_submission(self):
        from locus.controller import Agent
        from locus.repository import RepositoryIndex
        for calls in (4,5):
            with self.subTest(calls=calls):
                settings = Settings(total_calls=calls)
                ledger = Ledger(settings,Clock())
                implementation = source("implementation",roles=("editable",))
                workspace = FixtureWorkspace(ledger,{"module.py":implementation.text})
                model = FixtureModel(settings,ledger,actions=[
                    dict(action="apply_patch",patch="current patch"),
                    dict(action="test",command="python -m unittest"),dict(action="finish"),
                ])
                patch_text,_,tests,stop = Agent(settings,model,ledger,workspace,len)._repair(
                    dict(problem_statement="Repair module.py"),Packet((candidate(),),(implementation,)),
                    dict(passed=False),(),RepositoryIndex(workspace.texts,settings))
                self.assertEqual(patch_text,"current patch" if calls == 5 else "")
                self.assertEqual(tests,[dict(exit_code=0)] if calls == 5 else [])
                self.assertLessEqual(ledger.summary()["calls"],calls)
                self.assertNotEqual(stop,"submitted")

    def test_budget_exhaustion_exports_unvalidated_patch_with_explicit_status(self):
        from locus.controller import Agent
        from locus.repository import RepositoryIndex
        settings = Settings(total_calls=6)
        ledger = Ledger(settings,Clock())
        implementation = source("implementation",roles=("editable",))
        workspace = FixtureWorkspace(ledger,{"module.py":implementation.text})
        model = FixtureModel(settings,ledger,actions=[
            dict(action="apply_patch",patch="unfinished patch"),
            dict(action="finish"),dict(action="finish"),
        ])
        patch_text,_,tests,stop = Agent(settings,model,ledger,workspace,len)._repair(
            dict(problem_statement="Repair module.py"),Packet((candidate(),),(implementation,)),
            dict(passed=False),(),RepositoryIndex(workspace.texts,settings))
        self.assertEqual((patch_text,tests,stop),("unfinished patch",[],"validation_incomplete"))
        self.assertLessEqual(ledger.summary()["calls"],settings.total_calls)

    def test_a_changed_patch_requires_a_new_local_validation_before_finish(self):
        from locus.controller import Agent
        from locus.repository import RepositoryIndex
        settings = Settings(total_calls=50)
        ledger = Ledger(settings,Clock())
        implementation = source("implementation",roles=("editable",))
        workspace = FixtureWorkspace(ledger,{"module.py":implementation.text})
        model = FixtureModel(settings,ledger,actions=[
            dict(action="apply_patch",patch="first patch"),dict(action="finish"),
            dict(action="test",command="first validation"),
            dict(action="apply_patch",patch="second patch"),dict(action="finish"),
            dict(action="test",command="second validation"),dict(action="finish"),
        ])
        agent = Agent(settings,model,ledger,workspace,len)
        patch_text,_,tests,stop = agent._repair(
            dict(problem_statement="Repair module.py"),Packet((candidate(),),(implementation,)),
            dict(passed=False),(),RepositoryIndex(workspace.texts,settings))
        self.assertEqual(patch_text,"second patch")
        self.assertEqual(stop,"submitted")
        self.assertEqual(tests,[dict(exit_code=0),dict(exit_code=0)])
        feedback_errors = [p["feedback"].get("error","") for role,p in model.prompts if role == "patch"]
        self.assertEqual(sum("local validation" in message for message in feedback_errors),2)

    def test_recovery_of_equal_payload_merges_edit_coordinates_and_roles(self):
        from locus.controller import Agent
        from locus.repository import RepositoryIndex
        settings = Settings(total_calls=30)
        ledger = Ledger(settings,Clock())
        implementation = source("shared implementation",roles=("editable",))
        workspace = FixtureWorkspace(ledger,{"module.py":implementation.text,"other.py":implementation.text})
        model = FixtureModel(settings,ledger,actions=[
            dict(action="read",path="other.py",start=1,end=1),
            dict(action="finish"),
        ])
        agent = Agent(settings,model,ledger,workspace,len)
        agent._repair(dict(problem_statement="Repair module.py"),Packet((candidate(),),(implementation,)),
                      dict(passed=False),(),RepositoryIndex(workspace.texts,settings))
        self.assertTrue(all(counts[implementation.id] == 1 for counts in model.patch_source_counts))
        delivered = model.patch_source_records[-1]
        self.assertEqual(len(delivered),1)
        self.assertEqual({l.path for l in delivered[0].locations},{"module.py","other.py"})
        self.assertIn("editable",delivered[0].roles)

    def test_static_recovery_adds_source_without_collapsing_original_raw_records(self):
        from locus.controller import Agent,PATCH_SYSTEM
        from locus.repository import RepositoryIndex
        for memory in (True,False):
            with self.subTest(memory=memory):
                settings = Settings(total_calls=30,selection="static_topk",memory=memory)
                ledger = Ledger(settings,Clock())
                implementation = source("implementation",roles=("editable",))
                a,b = source("shared contract",path="a.py"),source("shared contract",path="b.py")
                workspace = FixtureWorkspace(ledger,{"module.py":implementation.text,
                                                   "a.py":a.text,"b.py":b.text,"c.py":a.text})
                packet = Packet((candidate(),),(implementation,a,b))
                model = FixtureModel(settings,ledger,actions=[
                    dict(action="read",path="c.py",start=1,end=1),
                    dict(action="finish"),
                ])
                recipient = model.session("patch",PATCH_SYSTEM)
                self.assertTrue(recipient.deliver(packet.sources,deduplicate=False))
                Agent(settings,model,ledger,workspace,len)._repair(
                    dict(problem_statement="Repair module.py"),packet,dict(passed=False),(),
                    RepositoryIndex(workspace.texts,settings))
                self.assertEqual([counts[a.id] for counts in model.patch_source_counts],[2,3])
                self.assertTrue(recipient.has(a))
                self.assertTrue(recipient.has(b))

    def test_in_memory_end_to_end_repairs_tests_and_submits(self):
        from locus.controller import Agent
        patch_text = (
            "diff --git a/module.py b/module.py\n"
            "--- a/module.py\n+++ b/module.py\n"
            "@@ -1,2 +1,2 @@\n def normalize(value):\n"
            "-    return value.strip()\n+    return value.strip() if value else value\n"
        )
        for memory in (True,False):
            with self.subTest(memory=memory):
                settings = Settings(total_calls=100,reserve_calls=20,reserve_seconds=1,
                                    debate_rounds=1,initial_files=1,frontier_size=1,memory=memory)
                ledger = Ledger(settings,Clock())
                workspace = FixtureWorkspace(ledger,{"module.py":"def normalize(value):\n    return value.strip()\n"})
                model = FixtureModel(settings,ledger,actions=[
                    dict(action="apply_patch",patch=patch_text),
                    dict(action="test",command="python -m unittest",timeout=10),
                    dict(action="finish"),
                ])
                result = Agent(settings,model,ledger,workspace,len).run(
                    dict(instance_id="unit",problem_statement="Repair normalize in module.py"))
                self.assertEqual(result["model_patch"],patch_text)
                self.assertEqual(result["repair_stop"],"submitted")
                self.assertEqual(result["local_tests"],[dict(exit_code=0)])
                self.assertEqual(result["file_predictions"],["module.py"])
                self.assertIsNone(result["resolved"])
                self.assertEqual(model.patch_visibility,[True,True,True])
                self.assertGreater(result["ledger"]["tokens"],0)
                self.assertEqual(result["ledger"]["calls"],len(ledger.steps))

    def _snapshot_inputs(self):
        settings = Settings(total_calls=100,reserve_calls=20,reserve_seconds=1,
                            debate_rounds=1,initial_files=1,frontier_size=1,
                            packet_tokens=2800,request_input_tokens=3200)
        ledger = Ledger(settings,Clock())
        old = source("old implementation",roles=("editable",))
        large = source("x"*2200,path="extra.py",roles=("editable",))
        workspace = FixtureWorkspace(ledger,{"module.py":old.text,"extra.py":large.text})
        old_candidate = candidate()
        new_candidate = candidate(path="extra.py",name="new-edit")
        old_state = Grounding(
            (old_candidate,),(),(requirement("old-required",old),),{old.id:old})
        new_state = Grounding(
            (new_candidate,),(),(requirement("new-required"),),
            {old.id:old,large.id:large})
        model = FixtureModel(settings,ledger)
        from locus.controller import Agent
        agent = Agent(settings,model,ledger,workspace,len)
        return agent,workspace,old_state,new_state

    def test_failed_later_admission_hands_off_last_successful_snapshot(self):
        agent,workspace,old_state,new_state = self._snapshot_inputs()
        with patch("locus.controller.ground",side_effect=[old_state,new_state]):
            with patch.object(agent,"_targets",return_value=[("extra.py",1,1)]):
                result = agent.run(dict(instance_id="unit",problem_statement="Repair module.py"))
        self.assertEqual(result["acquisition_stop"],"admission_failed")
        self.assertEqual(result["file_predictions"],["module.py"])
        self.assertEqual(result["unmet_requirements"],[])
        exported = agent.packet_for_export
        self.assertEqual([r["id"] for r in exported["requirements"]],["old-required"])
        self.assertEqual(exported["frontier"][0]["id"],"edit")
        self.assertIn("extra.py",workspace.read_paths)
        self.assertEqual(agent.model.patch_visibility,[True])

    def test_failed_regrounding_preserves_delivered_requirement_map(self):
        agent,workspace,old_state,new_state = self._snapshot_inputs()
        with patch("locus.controller.ground",
                   side_effect=[old_state,ModelError("new grounding failed")]):
            with patch.object(agent,"_targets",return_value=[("extra.py",1,1)]):
                result = agent.run(dict(instance_id="unit",problem_statement="Repair module.py"))
        self.assertEqual(result["acquisition_stop"],"acquisition_failed")
        self.assertEqual(result["file_predictions"],["module.py"])
        self.assertEqual(result["unmet_requirements"],[])
        self.assertEqual([r["id"] for r in agent.packet_for_export["requirements"]],["old-required"])
        self.assertFalse(result["gate"]["passed"])
        self.assertIn("extra.py",workspace.read_paths)

    def test_same_coordinates_with_new_behavior_do_not_pass_rank_stability(self):
        agent,workspace,old_state,new_state = self._snapshot_inputs()
        changed = replace(old_state.frontier[0],change="preserve whitespace instead of stripping it")
        new_state = replace(new_state,frontier=(changed,),requirements=old_state.requirements)
        with patch("locus.controller.ground",side_effect=[old_state,new_state]):
            with patch.object(agent,"_targets",return_value=[("extra.py",1,1)]):
                result = agent.run(dict(instance_id="unit",problem_statement="Repair module.py"))
        self.assertFalse(result["gate"]["rank_stable"])
        self.assertEqual(result["acquisition_stop"],"no_new_source")
        self.assertEqual(result["file_predictions"],["module.py"])


class AcquisitionTargetTests(unittest.TestCase):
    def make_case(self,*,analysis=(),planned=()):
        from locus.controller import Agent
        from locus.repository import RepositoryIndex
        settings = Settings(total_calls=100)
        ledger = Ledger(settings,Clock())
        text = "def main():\n    return helper()"
        implementation = source(text,start=1,end=2,roles=("editable",))
        texts = {"module.py":text,"helper.py":"def helper():\n    return 1",
                 "extra.py":"def extra():\n    return 2"}
        workspace = FixtureWorkspace(ledger,texts)
        model = FixtureModel(settings,ledger,analysis_targets=analysis,planned_targets=planned)
        state = Grounding((candidate(end=2),),(),(),{implementation.id:implementation})
        packet = Packet(state.frontier,(implementation,))
        return Agent(settings,model,ledger,workspace,len),state,packet,RepositoryIndex(texts,settings)

    def test_unmet_read_targets_precede_other_planned_targets(self):
        extra = dict(path="extra.py",start=1,end=2)
        agent,state,packet,index = self.make_case(planned=[extra])
        missing = (Requirement("helper","helper behavior","dependency",frozenset(),
                               (("helper.py",1,2),)),)
        targets = agent._targets(state,missing,packet,[("edit",)],index,"Repair module.py")
        self.assertEqual([t["path"] for t in targets],["helper.py","extra.py"])

    def test_exhausted_planner_targets_use_pending_trajectory_suggestion(self):
        seen = dict(path="module.py",start=1,end=2)
        suggested = dict(path="extra.py",start=1,end=2)
        agent,state,packet,index = self.make_case(analysis=[suggested],planned=[seen])
        missing = (Requirement("fact","unresolved fact","dependency",frozenset(),
                               (("module.py",1,2),)),)
        targets = agent._targets(state,missing,packet,[("edit",)],index,"Repair module.py")
        self.assertEqual([t["path"] for t in targets],["extra.py"])

    def test_seen_target_lists_do_not_block_relation_expansion(self):
        seen = dict(path="module.py",start=1,end=2)
        agent,state,packet,index = self.make_case(analysis=[seen],planned=[seen])
        missing = (Requirement("fact","unresolved fact","dependency",frozenset(),
                               (("module.py",1,2),)),)
        targets = agent._targets(state,missing,packet,[("edit",)],index,"Repair module.py")
        self.assertEqual({t["path"] for t in targets},{"helper.py"})
        self.assertNotIn("extra.py",{t["path"] for t in targets})

    def test_target_exhaustion_uses_union_of_source_intervals(self):
        agent,state,packet,index = self.make_case()
        first = source("def main():",start=1,end=1)
        second = source("    return helper()",start=2,end=2)
        fragmented = replace(state,pool={s.id:s for s in (first,second)})
        targets = agent._pending_targets(
            [dict(path="module.py",start=1,end=100),dict(path="missing.py",start=1,end=1),
             dict(path="module.py",start=True,end=2),{"path":"module.py"}],
            fragmented,index)
        self.assertEqual(targets,[])

    def test_unchecked_alternative_retains_discriminating_targets_and_coordinates(self):
        agent,state,packet,index = self.make_case()
        location = Location(agent.workspace.repo,agent.workspace.commit,"extra.py",1,2)
        alternative = Alternative("other-explanation","other-edit","unchecked",frozenset(),
                                  "Distinguish helper behavior",location,(("helper.py",1,2),))
        state = replace(state,alternatives=(alternative,))
        targets = agent._targets(state,(),packet,[("edit",)],index,"Repair module.py")
        self.assertEqual(targets,[dict(path="helper.py",start=1,end=2)])
        for _,prompt in agent.model.prompts:
            self.assertEqual(prompt["alternatives"],json.loads(json.dumps([alternative.as_dict()])))

    def test_exhausted_targets_expand_unchecked_alternative_relations_before_frontier(self):
        agent,state,packet,index = self.make_case()
        location = Location(agent.workspace.repo,agent.workspace.commit,"extra.py",1,2)
        state = replace(state,alternatives=(
            Alternative("other-explanation","other-edit","unchecked",frozenset(),
                        "Need its caller",location),))
        with patch.object(index,"relations",side_effect=lambda entry:
                          [dict(path="helper.py",start=1,end=2)] if entry.path == "extra.py" else []):
            targets = agent._targets(state,(),packet,[("edit",)],index,"Repair module.py")
        self.assertEqual(targets,[dict(path="helper.py",start=1,end=2)])


class FrozenAndOuterRunTests(unittest.TestCase):
    def test_frozen_attempt_resets_omitted_source_and_prior_conversation_facts(self):
        from locus.controller import Agent,PATCH_SYSTEM
        settings = Settings(total_calls=20)
        ledger = Ledger(settings,Clock())
        implementation = source("implementation",roles=("editable",))
        omitted = source("omitted dependent behavior",path="support.py")
        workspace = FixtureWorkspace(ledger,{"module.py":implementation.text,"support.py":omitted.text})
        model = FixtureModel(settings,ledger)
        recipient = model.session("patch",PATCH_SYSTEM)
        self.assertTrue(recipient.deliver((omitted,)))
        recipient.complete("previous issue","the missing fact must not carry over")
        condition = dict(token_cap=16000,frontier=[candidate().as_dict()],
                         observations=[implementation.as_dict()],
                         removed_ids=[omitted.id],
                         reference_support_units=["the missing fact must not enter prompts"])
        result = Agent(settings,model,ledger,workspace,len).run_frozen(
            dict(instance_id="unit",problem_statement="Repair module.py"),condition)
        self.assertFalse(recipient.has(omitted))
        self.assertNotIn(omitted.id,recipient.ever_delivered)
        self.assertEqual(result["evidence_metrics"]["admitted_tokens"],implementation.tokens)
        self.assertFalse(any("missing fact" in message.content for message in recipient.messages))
        for _,prompt in model.prompts:
            self.assertEqual(prompt["unmet"],[])
            self.assertNotIn("missing fact",json.dumps(prompt))
            self.assertNotIn(omitted.id,json.dumps(prompt))

    def test_frozen_contract_rejects_coerced_coordinates_values_and_wrong_hash(self):
        from copy import deepcopy
        from locus.controller import Agent
        implementation = source("implementation",roles=("editable",))
        original = dict(token_cap=16000,frontier=[candidate().as_dict()],
                        observations=[implementation.as_dict()])
        variants = []
        for value in (True,16000.5,"16000"):
            variants.append(dict(original,token_cap=value))
        for mutate in (
            lambda c:c["observations"][0].update(id="wrong source hash"),
            lambda c:c["observations"][0].update(text=None),
            lambda c:c["observations"][0].update(roles="editable"),
            lambda c:c["observations"][0]["locations"][0].update(start=1.5),
            lambda c:c["frontier"][0].update(change=None),
            lambda c:c["frontier"][0].update(id=None),
            lambda c:c["frontier"][0].update(score=float("nan")),
            lambda c:c["frontier"][0]["location"].update(start=True),
        ):
            variant = deepcopy(original)
            mutate(variant)
            variants.append(variant)
        for variant in variants:
            with self.subTest(condition=variant):
                settings = Settings(total_calls=20)
                ledger = Ledger(settings,Clock())
                workspace = FixtureWorkspace(ledger,{"module.py":implementation.text})
                model = FixtureModel(settings,ledger)
                with self.assertRaises(ModelError):
                    Agent(settings,model,ledger,workspace,len).run_frozen(
                        dict(instance_id="unit",problem_statement="Repair module.py"),variant)
                self.assertEqual(model.roles_seen,[])
                self.assertIn(ledger.summary()["calls"],(0,2))

    def test_frozen_frontier_repairs_without_relocalization_or_recovery_reads(self):
        from locus.controller import Agent
        settings = Settings(total_calls=40,reserve_calls=8,reserve_seconds=1)
        ledger = Ledger(settings,Clock())
        implementation = source("implementation",roles=("editable",))
        c = candidate()
        workspace = FixtureWorkspace(ledger,{"module.py":implementation.text})
        model = FixtureModel(settings,ledger,actions=[
            dict(action="read",path="unavailable.py",start=1,end=1),
            dict(action="apply_patch",patch="frozen patch"),
            dict(action="test",command="python -m unittest",timeout=10),
            dict(action="finish"),
        ])
        condition = dict(
            condition="selected16k",token_cap=16000,
            frontier=[c.as_dict()],
            observations=[dict(**implementation.as_dict(),tokens=implementation.tokens)],
            reference_support_units=["external annotation must stay outside agent inputs"],
        )
        agent = Agent(settings,model,ledger,workspace,len)
        result = agent.run_frozen(
            dict(instance_id="unit",problem_statement="Repair module.py"),condition)
        self.assertEqual(result["model_patch"],"frozen patch")
        self.assertEqual(result["file_predictions"],["module.py"])
        self.assertEqual(result["local_tests"],[dict(exit_code=0)])
        self.assertEqual(result["repair_stop"],"submitted")
        self.assertEqual(workspace.read_paths,[])
        self.assertTrue(all(role == "patch" for role in model.roles_seen))
        for _,prompt in model.prompts:
            self.assertEqual(prompt["frontier"],[c.as_dict()])
            self.assertFalse(prompt["recovery_allowed"])
            self.assertNotIn("external annotation",json.dumps(prompt))
        self.assertFalse(agent.settings.recovery)
        self.assertIsNone(result["resolved"])

    def test_frozen_source_must_match_current_base_coordinates(self):
        from locus.controller import Agent
        settings = Settings()
        ledger = Ledger(settings,Clock())
        workspace = FixtureWorkspace(ledger,{"module.py":"current base"})
        wrong = source("different source",roles=("editable",))
        condition = dict(condition="selected",token_cap=16000,
                         frontier=[candidate().as_dict()],
                         observations=[wrong.as_dict()])
        model = FixtureModel(settings,ledger)
        with self.assertRaises(ModelError):
            Agent(settings,model,ledger,workspace,len).run_frozen(
                dict(instance_id="unit",problem_statement="Repair module.py"),condition)
        self.assertEqual(model.roles_seen,[])
        self.assertEqual(ledger.summary()["calls"],2)

    def test_outer_worker_failure_preserves_consumption_and_unknown_grade(self):
        from locus.config import ModelSettings
        from locus.controller import run_agent
        from locus.environment import WorkerError
        clock = Clock()
        settings = Settings(seed=17)
        ledgers = []

        def ledger_factory(actual_settings):
            result = Ledger(actual_settings,clock)
            ledgers.append(result)
            return result

        def failing_workspace(image,commit,ledger,**kwargs):
            with ledger.action("tool","worker-setup"):
                clock.advance(3)
                raise WorkerError("worker startup failed")

        def model_factory(model_settings,actual_settings,ledger,count):
            return FixtureModel(actual_settings,ledger)

        task = dict(instance_id="unit",repeat=2,repo="example/repository",
                    base_commit="base",docker_image="worker:unit",
                    problem_statement="Repair module.py")
        with patch("locus.controller.Ledger",side_effect=ledger_factory):
            result = run_agent(task,settings,ModelSettings(model="unit-model"),
                               workspace_factory=failing_workspace,
                               model_factory=model_factory,count=len)
        self.assertEqual(result["status"],"execution_failed")
        self.assertEqual(result["repeat"],2)
        self.assertEqual(result["model"],"unit-model")
        self.assertEqual(result["settings"]["seed"],17)
        self.assertIsNone(result["resolved"])
        self.assertEqual(result["ledger"]["calls"],1)
        self.assertEqual(result["ledger"]["elapsed_seconds"],3)
        self.assertAlmostEqual(result["ledger"]["cost"],.06*3/3600)
        self.assertFalse(result["ledger"]["steps"][0]["success"])

    def test_model_timeout_is_bounded_by_current_phase_time_allowance(self):
        from locus.config import ModelSettings
        from locus.models import HTTPModel
        settings = Settings(deadline_seconds=20,reserve_seconds=5)
        clock = Clock()
        ledger = Ledger(settings,clock)
        ledger.restrict(PROFILES["L"])
        clock.advance(4)
        payload = dict(usage=dict(prompt_tokens=10,completion_tokens=2),
                       choices=[dict(message=dict(content='{"action":"finish"}'),finish_reason="stop")])
        response = io.StringIO(json.dumps(payload))
        model = HTTPModel(ModelSettings(base_url="http://127.0.0.1/v1",timeout_seconds=180),
                          settings,ledger,len)
        with patch("locus.models.urllib.request.urlopen",return_value=response) as transport:
            result = model.ask("patch","JSON only","Return JSON")
        self.assertEqual(result,dict(action="finish"))
        self.assertLessEqual(transport.call_args.kwargs["timeout"],11)
        self.assertEqual(ledger.summary()["tokens"],12)

    def test_debate_http_timeout_preserves_repair_time_reserve(self):
        from locus.config import ModelSettings
        from locus.models import HTTPModel
        settings = Settings(deadline_seconds=20,reserve_seconds=5)
        clock = Clock()
        clock.advance(4)
        ledger = Ledger(settings,clock)
        clock.advance(4)
        payload = dict(usage=dict(prompt_tokens=10,completion_tokens=2),
                       choices=[dict(message=dict(content='{"label":"L"}'),finish_reason="stop")])
        model = HTTPModel(ModelSettings(base_url="http://127.0.0.1/v1"),settings,ledger,len)
        with patch("locus.models.urllib.request.urlopen",return_value=io.StringIO(json.dumps(payload))) as call:
            result = model.ask("debate:Implementation Analyst","JSON only","Return JSON")
        self.assertEqual(result,dict(label="L"))
        self.assertLessEqual(call.call_args.kwargs["timeout"],11)

    def test_extra_body_cannot_override_sampling_but_allows_reasoning_control(self):
        from locus.config import ModelSettings
        from locus.models import HTTPModel
        settings = Settings(temperature=.7,top_p=1.0)
        for field,value in (("temperature",0.0),("top_p",.5)):
            with self.subTest(field=field):
                ledger = Ledger(settings,Clock())
                model = HTTPModel(ModelSettings(base_url="http://127.0.0.1/v1",
                                                extra_body={field:value}),settings,ledger,len)
                with patch("locus.models.urllib.request.urlopen") as transport:
                    with self.assertRaises(ValueError):
                        model.ask("patch","JSON only","Return JSON")
                transport.assert_not_called()
                self.assertEqual(ledger.summary()["calls"],0)
        ledger = Ledger(settings,Clock())
        model = HTTPModel(ModelSettings(base_url="http://127.0.0.1/v1",
                                        extra_body={"reasoning_effort":"none"}),settings,ledger,len)
        payload = dict(usage=dict(prompt_tokens=10,completion_tokens=2),
                       choices=[dict(message=dict(content='{"action":"finish"}'))])
        with patch("locus.models.urllib.request.urlopen",return_value=io.StringIO(json.dumps(payload))) as transport:
            model.ask("patch","JSON only","Return JSON")
        body = json.loads(transport.call_args.args[0].data)
        self.assertEqual(body["temperature"],settings.temperature)
        self.assertEqual(body["top_p"],settings.top_p)
        self.assertEqual(body["reasoning_effort"],"none")

    def test_provider_protocol_errors_preserve_known_or_unknown_usage(self):
        from locus.config import ModelSettings
        from locus.models import HTTPModel
        payloads = [
            ("not JSON",False),
            ("[]",False),
            (json.dumps(dict(usage={},choices=[])),False),
            (json.dumps(dict(usage=dict(prompt_tokens=10,completion_tokens=2),
                             choices=[{}])),True),
            (json.dumps(dict(usage=dict(prompt_tokens=10,completion_tokens=2),
                             choices=dict(message="invalid"))),True),
        ]
        for raw,known in payloads:
            with self.subTest(raw=raw):
                settings = Settings()
                ledger = Ledger(settings,Clock())
                model = HTTPModel(ModelSettings(base_url="http://127.0.0.1/v1"),settings,ledger,len)
                with patch("locus.models.urllib.request.urlopen",return_value=io.StringIO(raw)):
                    with self.assertRaises(ModelError):
                        model.ask("patch","JSON only","Return JSON")
                summary = ledger.summary()
                self.assertEqual(summary["calls"],1)
                self.assertEqual(summary["usage_complete"],known)
                self.assertEqual(summary["tokens"],12 if known else None)

    def test_reported_request_cap_violation_is_charged_and_rejected(self):
        from locus.config import ModelSettings
        from locus.models import HTTPModel
        settings = Settings(packet_tokens=500,request_input_tokens=1000,output_tokens=10)
        for usage in (dict(prompt_tokens=1001,completion_tokens=1),
                      dict(prompt_tokens=10,completion_tokens=11)):
            with self.subTest(usage=usage):
                ledger = Ledger(settings,Clock())
                model = HTTPModel(ModelSettings(base_url="http://127.0.0.1/v1"),settings,ledger,len)
                body = dict(usage=usage,choices=[dict(message=dict(content='{"action":"finish"}'))])
                with patch("locus.models.urllib.request.urlopen",return_value=io.StringIO(json.dumps(body))):
                    with self.assertRaises(ModelError):
                        model.ask("patch","JSON only","Return JSON")
                self.assertTrue(ledger.summary()["usage_complete"])
                self.assertEqual(ledger.summary()["tokens"],sum(usage.values()))

    def test_worker_command_timeout_uses_remaining_task_seconds(self):
        import subprocess
        from locus.environment import DockerWorkspace
        settings = Settings(deadline_seconds=10)
        clock = Clock()
        ledger = Ledger(settings,clock)
        clock.advance(3)
        worker = object.__new__(DockerWorkspace)
        worker.ledger,worker.docker = ledger,"docker"
        worker.container,worker.workdir = "unit-worker","/workspace"
        completed = subprocess.CompletedProcess([],0,b"tests passed")
        with patch("locus.environment.subprocess.run",return_value=completed) as execute:
            result = worker.run("python -m unittest",timeout=120)
        self.assertEqual(result["exit_code"],0)
        self.assertLessEqual(execute.call_args.kwargs["timeout"],7)
        self.assertEqual(ledger.summary()["calls"],1)

    def test_static_and_broad_repair_keep_raw_duplicate_context_with_and_without_memory(self):
        from locus.controller import Agent,PATCH_SYSTEM
        from locus.repository import RepositoryIndex
        for policy in ("static_topk","broad"):
            for memory in (True,False):
                with self.subTest(policy=policy,memory=memory):
                    settings = Settings(total_calls=40,selection=policy,memory=memory)
                    ledger = Ledger(settings,Clock())
                    implementation = source("implementation",roles=("editable",))
                    a = source("dependent contract",path="a.py")
                    b = source("dependent contract",path="b.py")
                    workspace = FixtureWorkspace(ledger,{"module.py":implementation.text,
                                                       "a.py":a.text,"b.py":b.text})
                    model = FixtureModel(settings,ledger,actions=[
                        dict(action="test",command="python -m unittest",timeout=10),
                        dict(action="finish"),
                    ])
                    packet = Packet((candidate(),),(implementation,a,b))
                    recipient = model.session("patch",PATCH_SYSTEM)
                    self.assertTrue(recipient.deliver(packet.sources,deduplicate=False))
                    agent = Agent(settings,model,ledger,workspace,len)
                    index = RepositoryIndex(workspace.texts,settings)
                    _,_,tests,stop = agent._repair(
                        dict(problem_statement="Repair module.py"),packet,
                        dict(passed=False),(),index)
                    self.assertEqual(stop,"submitted")
                    self.assertEqual(tests,[dict(exit_code=0)])
                    self.assertEqual(model.patch_visibility,[True,True])
                    self.assertTrue(all(counts[a.id] == 2 for counts in model.patch_source_counts))
                    self.assertTrue(all(counts[implementation.id] == 1
                                        for counts in model.patch_source_counts))


if __name__ == "__main__":
    unittest.main()
