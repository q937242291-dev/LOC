"""Offline pairing controls; fixtures are constructed in memory."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import unittest

from locus.benchmarks import merge_grades, repeat_plan
from locus.config import ModelSettings, Settings
from locus.run_design import (attempt_design, attempt_seed, audit_run_design,
                              build_run_manifest, validate_manifest)


def task(identity="example-1", benchmark="lite"):
    return dict(instance_id=identity, benchmark=benchmark, repo="example/repository",
                base_commit="base-commit", problem_statement="Repair the example behavior",
                docker_image="public/example:fixture", dataset_revision="release-commit")


class RepeatSeedTests(unittest.TestCase):
    def test_seeds_do_not_depend_on_order_subset_system_or_number_of_repeats(self):
        tasks = [task("a"),task("b"),task("a","pro-v2")]
        full = repeat_plan(tasks,3,17)
        expected = {(r["benchmark"],r["instance_id"],r["repeat"]):r["seed"] for r in full}
        for rows in (tasks[::-1],tasks[1:],[dict(t,system="ablation") for t in tasks]):
            for attempt in repeat_plan(rows,2,17):
                key = attempt["benchmark"],attempt["instance_id"],attempt["repeat"]
                self.assertEqual(attempt["seed"],expected[key])
                self.assertGreaterEqual(attempt["seed"],0)
                self.assertLess(attempt["seed"],2**31)
        self.assertNotEqual(attempt_seed(17,"lite","a",0),attempt_seed(18,"lite","a",0))

    def test_unique_identity_includes_benchmark_and_requires_true_integer_controls(self):
        self.assertEqual(len(repeat_plan([task("a"),task("a","pro-v2")],1)),2)
        with self.assertRaises(ValueError):
            repeat_plan([task(),task()],1)
        for value in (True,False,1.5,"2",None):
            with self.subTest(repeats=value),self.assertRaises(ValueError):
                repeat_plan([task()],value)
            with self.subTest(seed=value),self.assertRaises(ValueError):
                repeat_plan([task()],1,value)
        for value in (True,1.5,"0",-1):
            with self.subTest(repeat=value),self.assertRaises(ValueError):
                attempt_seed(0,"lite","a",value)
        for row in (dict(instance_id=None),dict(instance_id=""),dict(instance_id="a",benchmark=False)):
            with self.subTest(row=row),self.assertRaises(ValueError):
                repeat_plan([row],1)


class GradeScopeTests(unittest.TestCase):
    def test_grades_cannot_be_shared_across_methods_models_or_releases(self):
        original = dict(instance_id="a",repeat=0,benchmark="lite",model="fixture",system="LOCUS")
        for field,value in (("system","ablation"),("model","other-model"),("benchmark","pro-v2")):
            with self.subTest(field=field),self.assertRaises(ValueError):
                merge_grades([original,dict(original,**{field:value})],
                             [dict(instance_id="a",repeat=0,resolved=True)])
            with self.subTest(qualified_grade=field),self.assertRaises(ValueError):
                merge_grades([original],[dict(instance_id="a",repeat=0,resolved=True,**{field:value})])

    def test_grading_repeats_are_true_integers_and_duplicate_attempts_are_rejected(self):
        original = dict(instance_id="a",repeat=0)
        for repeat in (True,False,0.5,"0",-1):
            with self.subTest(record=repeat),self.assertRaises(ValueError):
                merge_grades([dict(original,repeat=repeat)],[])
            with self.subTest(grade=repeat),self.assertRaises(ValueError):
                merge_grades([original],[dict(instance_id="a",repeat=repeat,resolved=False)])
            with self.subTest(default_repeat=repeat),self.assertRaises(ValueError):
                merge_grades([original],[],repeat=repeat)
        with self.assertRaises(ValueError):
            merge_grades([original,dict(original)],[])
        with self.assertRaises(ValueError):
            merge_grades([original],[dict(instance_id="a",resolved=True),dict(instance_id="a",resolved=True)])

    def test_single_method_grade_join_preserves_unknown_and_explicit_failures(self):
        records = [dict(instance_id="a",repeat=1),dict(instance_id="b",repeat=1),
                   dict(instance_id="c",repeat=1)]
        rows = merge_grades(records,[dict(instance_id="a",resolved=False),
                                     dict(instance_id="b",resolved=None,error="incomplete")],repeat=1)
        self.assertIs(rows[0]["resolved"],False)
        self.assertIsNone(rows[1]["resolved"])
        self.assertEqual(rows[1]["grading_error"],"incomplete")
        self.assertNotIn("resolved",rows[2])
        self.assertNotIn("resolved",records[0])


class RunDesignTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [task("a"),task("b","pro-v2")]
        self.settings = Settings(seed=17)
        self.model = ModelSettings(model="fixture-model",base_url="http://127.0.0.1/v1")

    def manifest(self,settings=None,**kwargs):
        options = dict(repeats=2,batch_id="paired-batch",reference_settings=self.settings,
                       deadline_by_benchmark={"lite":7200,"pro-v2":3000},code_version="fixture-v1")
        options.update(kwargs)
        return build_run_manifest(self.tasks,settings or self.settings,self.model,**options)

    def rows(self,settings=None,**kwargs):
        actual = settings or self.settings
        manifest = self.manifest(actual,**kwargs)
        rows = []
        for planned in repeat_plan(self.tasks,2,17):
            deadline = manifest["outer_budget"]["deadline_by_benchmark"][planned["benchmark"]]
            runtime = replace(actual,seed=planned["seed"],deadline_seconds=deadline)
            rows.append(dict(instance_id=planned["instance_id"],benchmark=planned["benchmark"],
                             repeat=planned["repeat"],seed=planned["seed"],model=self.model.model,
                             settings=asdict(runtime),resolved=None,
                             run_design=attempt_design(manifest,planned,runtime)))
        return manifest,rows

    def test_design_id_excludes_factors_and_order_but_batch_and_condition_remain_explicit(self):
        full = self.manifest()
        ablation = self.manifest(replace(self.settings,packet_tokens=4000,gate_enabled=False,allocation="L"))
        reversed_tasks = build_run_manifest(self.tasks[::-1],self.settings,self.model,repeats=2,
            batch_id="another-batch",reference_settings=self.settings,
            deadline_by_benchmark={"lite":7200.0,"pro-v2":3000.0},code_version="fixture-v1")
        self.assertEqual(full["design_id"],ablation["design_id"])
        self.assertEqual(full["design_id"],reversed_tasks["design_id"])
        self.assertEqual(full["attempts"],ablation["attempts"])
        self.assertNotEqual(full["condition_id"],ablation["condition_id"])
        self.assertNotEqual(full["batch_id"],reversed_tasks["batch_id"])
        self.assertEqual(ablation["reference"],dict(packet_tokens=16000,frontier_size=5))
        validate_manifest(json.loads(json.dumps(full)))

    def test_roster_issue_revision_budget_sampling_model_and_code_change_design(self):
        original = self.manifest()["design_id"]
        for field,value in (("base_commit","other-commit"),("dataset_revision","other-release"),
                            ("problem_statement","Different task instructions")):
            tasks = deepcopy(self.tasks)
            tasks[0][field] = value
            changed = build_run_manifest(tasks,self.settings,self.model,repeats=2,
                batch_id="paired-batch",reference_settings=self.settings,
                deadline_by_benchmark={"lite":7200,"pro-v2":3000},code_version="fixture-v1")
            self.assertNotEqual(original,changed["design_id"])
        for settings in (replace(self.settings,total_calls=249),replace(self.settings,temperature=.8),
                         replace(self.settings,reserve_calls=39),replace(self.settings,seed=18)):
            self.assertNotEqual(original,self.manifest(settings)["design_id"])
        self.assertNotEqual(original,self.manifest(code_version="fixture-v2")["design_id"])
        self.assertNotEqual(original,self.manifest(deadline_by_benchmark={"lite":7200,"pro-v2":3100})["design_id"])
        changed = build_run_manifest(self.tasks,self.settings,replace(self.model,model="other-model"),
                                    repeats=2,code_version="fixture-v1")
        self.assertNotEqual(original,changed["design_id"])

    def test_manifest_serializes_only_hashes_of_issue_and_generation_options(self):
        model = replace(self.model,extra_body={"secret_option":"never-copy-this-value"})
        manifest = build_run_manifest(self.tasks,self.settings,model,repeats=1)
        serialized = json.dumps(manifest)
        self.assertNotIn("never-copy-this-value",serialized)
        self.assertNotIn(self.tasks[0]["problem_statement"],serialized)
        self.assertTrue(manifest["code_version"].startswith("sha256:"))
        for endpoint in ("https://user:private-value@localhost/v1",
                         "https://example.org/v1?key=private-value"):
            with self.assertRaises(ValueError) as failure:
                build_run_manifest(self.tasks,self.settings,replace(self.model,base_url=endpoint))
            self.assertNotIn("private-value",str(failure.exception))

    def test_resume_validation_detects_roster_seed_controls_and_condition_changes(self):
        original = self.manifest()
        mutations = (
            lambda m:m["roster"].reverse(),
            lambda m:m["roster"][0].update(base_commit="changed"),
            lambda m:m["attempts"][0].update(seed=1),
            lambda m:m["attempts"][0].update(repeat=False),
            lambda m:m["outer_budget"].update(total_calls=1),
            lambda m:m["factors"].update(packet_tokens=1000),
            lambda m:m.update(repeats=True),
        )
        for mutate in mutations:
            changed = deepcopy(original)
            mutate(changed)
            with self.subTest(manifest=changed),self.assertRaises(ValueError):
                validate_manifest(changed)

    def test_attempt_metadata_is_compact_and_checks_actual_task_seed_and_settings(self):
        manifest,rows = self.rows()
        metadata = rows[0]["run_design"]
        self.assertNotIn("roster",metadata)
        self.assertNotIn("attempts",metadata)
        self.assertEqual(metadata["logical_seed"],rows[0]["seed"])
        self.assertIsNone(metadata["sampling_requested"]["provider_seed_root"])
        self.assertEqual(metadata["sampling_requested"]["provider_seed_mode"],"off")
        planned = repeat_plan(self.tasks,2,17)[0]
        for changed in (dict(planned,seed=1),dict(planned,base_commit="other"),
                        dict(planned,problem_statement="changed"),dict(planned,repeat=True)):
            with self.subTest(task=changed),self.assertRaises(ValueError):
                attempt_design(manifest,changed)
        runtime = replace(self.settings,seed=planned["seed"],total_calls=249)
        with self.assertRaises(ValueError):
            attempt_design(manifest,planned,runtime)
        model = {**asdict(self.model),"seed_mode":"request"}
        seeded = build_run_manifest(self.tasks,self.settings,model,repeats=2)
        request = attempt_design(seeded,planned)["sampling_requested"]
        self.assertEqual(request["provider_seed_root"],planned["seed"])
        self.assertEqual(request["provider_request_seed_policy"],"locus-request-seed-v1")
        self.assertEqual(request["provider_seed_mode"],"request")

    def test_audit_separates_controls_factors_and_resource_outcomes(self):
        _,full = self.rows()
        _,ablated = self.rows(replace(self.settings,packet_tokens=4000,gate_enabled=False))
        full[0]["ledger"] = dict(cost=100,calls=250)
        ablated[0]["ledger"] = dict(cost=1,calls=20)
        full[0]["resolved"] = True
        audit = audit_run_design(full,ablated)
        self.assertEqual(audit["status"],"matched")
        self.assertEqual(audit["metadata_matched_pairs"],4)
        self.assertEqual(len(audit["factor_differences"]),4)
        self.assertFalse(audit["checks_outcomes"])
        self.assertFalse(audit["compares_actual_spending"])

    def test_audit_rejects_actual_seed_budget_sampling_batch_missing_and_duplicate_pairs(self):
        _,full = self.rows()
        mutations = (
            lambda rows:rows[0].update(seed=0),
            lambda rows:rows[0]["settings"].update(total_calls=249),
            lambda rows:rows[0]["settings"].update(temperature=.8),
            lambda rows:rows[0]["run_design"].update(batch_id="different-batch"),
            lambda rows:rows.pop(),
            lambda rows:rows.append(deepcopy(rows[0])),
        )
        for mutate in mutations:
            other = deepcopy(full)
            mutate(other)
            with self.subTest(rows=other):
                result = audit_run_design(full,other)
                self.assertEqual(result["status"],"mismatch")
                self.assertTrue(any(row["kind"] == "mismatch" for row in result["evidence"]))

    def test_audit_unknown_legacy_and_mixed_batches_cannot_be_certified(self):
        legacy = [dict(instance_id="a",benchmark="lite",repeat=0)]
        self.assertEqual(audit_run_design(legacy,legacy)["status"],"unknown")
        self.assertEqual(audit_run_design([],[])["status"],"unknown")
        _,full = self.rows()
        other = deepcopy(full)
        for rows in (full,other):
            rows[1]["run_design"]["batch_id"] = "second-batch"
        audit = audit_run_design(full,other)
        self.assertEqual(audit["status"],"mismatch")
        self.assertTrue(any(row["field"].endswith("mixed_run_designs") for row in audit["evidence"]))

    def test_request_observations_compare_wire_controls_without_equal_call_count_assumption(self):
        _,full = self.rows()
        other = deepcopy(full)
        for row in (full[0],other[0]):
            metadata = row["run_design"]
            row["model_runtime"] = dict(provider=self.model.provider,configured_model=self.model.model,
                seed_mode="off",local_seed=row["seed"],requests=[dict(
                    role="patch",ordinal=0,requested_model=self.model.model,
                    requested_temperature=.7,requested_top_p=1.0,sent_temperature=.7,sent_top_p=1.0,
                    seed_mode="off",requested_seed=None,
                    served_model="provider-model-revision",system_fingerprint="fixture-fingerprint")])
        other[0]["model_runtime"]["requests"].append(dict(
            other[0]["model_runtime"]["requests"][0],role="trajectory"))
        result = audit_run_design(full[:1],other[:1])
        self.assertEqual(result["status"],"matched")
        self.assertEqual(result["request_observations"][0]["status"],"matched")
        self.assertEqual(result["request_observations"][0]["common_requests"],1)
        other[0]["model_runtime"]["requests"][0]["sent_temperature"] = .8
        mismatch = audit_run_design(full[:1],other[:1])
        self.assertEqual(mismatch["status"],"mismatch")
        self.assertTrue(any("sent_temperature" in row["field"] for row in mismatch["evidence"]))
        other[0]["model_runtime"]["requests"][0]["sent_temperature"] = .7
        other[0]["model_runtime"]["requests"][0]["system_fingerprint"] = None
        unknown = audit_run_design(full[:1],other[:1])
        self.assertEqual(unknown["status"],"matched")
        self.assertEqual(unknown["request_observations"][0]["status"],"unknown")

    def test_provider_request_seed_audit_checks_role_and_ordinal(self):
        self.model = replace(self.model,seed_mode="request")
        _,full = self.rows()
        for row in full[:1]:
            encoded = json.dumps(["locus-request-seed-v1",row["seed"],"patch",0],separators=(",",":"))
            seed = int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:4],"big") & 0x7fffffff
            row["model_runtime"] = dict(provider=self.model.provider,configured_model=self.model.model,
                seed_mode="request",local_seed=row["seed"],requests=[dict(
                    role="patch",ordinal=0,requested_model=self.model.model,
                    requested_temperature=.7,requested_top_p=1.0,sent_temperature=.7,sent_top_p=1.0,
                    seed_mode="request",requested_seed=seed,
                    served_model=None,system_fingerprint=None)])
        other = deepcopy(full[:1])
        self.assertEqual(audit_run_design(full[:1],other)["status"],"matched")
        self.assertNotEqual(seed,full[0]["seed"])
        other[0]["model_runtime"]["requests"][0]["requested_seed"] = full[0]["seed"]
        result = audit_run_design(full[:1],other)
        self.assertEqual(result["status"],"mismatch")
        self.assertTrue(any("requested_seed" in row["field"] for row in result["evidence"]))


if __name__ == "__main__":
    unittest.main()
