"""CLI integration tests for tasks, patches, reports, and model responses."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from locus import cli
from locus.benchmarks import read_jsonl, write_jsonl
from locus.config import load_config
from locus.environment import DockerWorkspace
from locus.ledger import Ledger
from locus.models import HTTPModel


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="locus-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def json(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def jsonl(self, name, rows):
        path = self.root / name
        write_jsonl(path, rows)
        return path

    def invoke(self, *arguments, expected=0):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main([str(argument) for argument in arguments])
        self.assertEqual(status, expected, stderr.getvalue())
        return stdout.getvalue()

    def test_all_help_commands_require_no_runtime_dependencies(self):
        commands = (None, "fetch", "run", "sweep", "repair-frozen", "interventions",
                    "evaluate", "export", "grade-lite", "grade-pro-v1", "grade-pro-v2",
                    "read-grades", "merge-grades", "import-baseline", "matrix")
        for command in commands:
            with self.subTest(command=command):
                output = io.StringIO()
                arguments = ([command] if command else []) + ["--help"]
                with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
                    cli.main(arguments)
                self.assertEqual(raised.exception.code, 0)
                self.assertIn("usage:", output.getvalue())

    def test_matrix_and_all_sweep_plans(self):
        matrix = self.root / "matrix.json"
        self.invoke("matrix", "--output", matrix)
        rows = json.loads(matrix.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 81)
        for study in ("selection", "components", "allocation", "width", "capacity"):
            output = self.root / study
            self.invoke("sweep", "--config", ROOT / "configs/gpt52.toml", "--tasks", "unused.jsonl",
                        "--output", output, "--study", study, "--plan-only")
            cases = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            self.assertTrue(cases)
            self.assertTrue(all("name" in case and "settings" in case for case in cases))

    def test_fetch_dispatches_public_preparation_only_when_requested(self):
        with patch("locus.cli.fetch_public", return_value={"count": 1}) as fetch:
            output = self.invoke("fetch", "--benchmark", "pro-v2", "--output", self.root,
                                 "--revision", "v2.0.0", "--limit", 1)
        self.assertEqual(json.loads(output), {"count": 1})
        fetch.assert_called_once_with("pro-v2", self.root, revision="v2.0.0", limit=1)

    def test_baseline_import_export_official_reports_merge_and_evaluate(self):
        tasks = self.jsonl("tasks.jsonl", [{"instance_id": key, "benchmark": "lite"} for key in ("a", "b")])
        predictions = self.json("predictions.json", {"a": {"model_patch": "diff"}})
        attempts = self.root / "attempts.jsonl"
        self.invoke("import-baseline", "--predictions", predictions, "--tasks", tasks,
                    "--system", "mini-swe-agent", "--model", "gpt-5.2", "--repeat", 2,
                    "--output", attempts)
        imported = read_jsonl(attempts)
        self.assertEqual(len(imported), 2)
        self.assertEqual(imported[1]["status"], "empty_patch")
        self.assertIsNone(imported[0]["ledger"])
        exported = self.root / "official.jsonl"
        self.invoke("export", "--attempts", attempts, "--output", exported, "--repeat", 2)
        self.assertEqual(set(read_jsonl(exported)[0]), {"instance_id", "model_name_or_path", "model_patch"})
        patches = self.root / "official-pro1.json"
        self.invoke("export", "--attempts", attempts, "--output", patches, "--repeat", 2,
                    "--format", "pro-v1")
        self.assertEqual(len(json.loads(patches.read_text(encoding="utf-8"))), 2)
        for key, resolved in (("a", True), ("b", False)):
            self.json(f"reports/{key}/report.json", {key: {
                "patch_is_None": False, "patch_exists": True,
                "patch_successfully_applied": True, "resolved": resolved}})
        grades = self.root / "grades.jsonl"
        self.invoke("read-grades", "--format", "lite", "--input", self.root / "reports",
                    "--repeat", 2, "--output", grades)
        merged = self.root / "merged.jsonl"
        self.invoke("merge-grades", "--attempts", attempts, "--grades", grades, "--output", merged)
        self.assertEqual([row["resolved"] for row in read_jsonl(merged)], [True, False])
        summary = self.root / "summary.json"
        self.invoke("evaluate", "--attempts", attempts, "--grades", grades, "--output", summary)
        group = json.loads(summary.read_text(encoding="utf-8"))["groups"][0]
        self.assertEqual(group["resolution_pct"], 50)
        self.assertEqual(group["attempts"], 2)
        pro1_report = self.json("eval_results.json", {"a": True, "b": False})
        self.invoke("read-grades", "--format", "pro-v1", "--input", pro1_report,
                    "--repeat", 2, "--output", grades)
        self.assertEqual({row["instance_id"]: row["resolved"] for row in read_jsonl(grades)},
                         {"a": True, "b": False})

    def test_evaluate_annotation_signature_and_pooled_evidence(self):
        locations = [{"path": "edit.py", "start": 1, "end": 2},
                     {"path": "support.py", "start": 1, "end": 2}]
        attempts = self.jsonl("attempts.jsonl", [{
            "instance_id": "a", "benchmark": "lite", "system": "LOCUS", "model": "model",
            "repeat": 0, "resolved": False, "file_predictions": ["edit.py"],
            "packet": {"sources": [{"locations": locations}]},
            "evidence_metrics": {"admitted_tokens": 1000, "repeated_tokens": 100, "recovery": True}}])
        annotations = self.jsonl("annotations.jsonl", [{"instance_id": "a",
            "reference_units": locations, "edit_units": locations[:1]}])
        evaluated, summary = self.root / "evaluated.jsonl", self.root / "summary.json"
        self.invoke("evaluate", "--attempts", attempts, "--annotations", annotations,
                    "--output", summary, "--graded-output", evaluated)
        metrics = read_jsonl(evaluated)[0]["evidence_metrics"]
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["yield"], 1)
        self.assertEqual(metrics["duplication"], .1)
        self.assertIs(metrics["joint"], True)
        self.assertIs(metrics["file_at_1"], True)

    def test_evaluate_preserves_imported_evidence_without_packet_or_annotation(self):
        supplied = {"selected": 2, "required": 2, "hits": 2, "support_hits": 1,
                    "recall": 1., "precision": 1., "file_at_1": True, "edit_coverage": True,
                    "joint": True, "admitted_tokens": 1000, "repeated_tokens": 100,
                    "duplication": .1, "yield": 1., "recovery": False}
        attempts = self.jsonl("attempts.jsonl", [{"instance_id": "a", "repeat": 0,
            "benchmark": "lite", "model": "model", "system": "upstream", "resolved": True,
            "evidence_metrics": supplied}])
        evaluated = self.root / "evaluated.jsonl"
        self.invoke("evaluate", "--attempts", attempts, "--output", self.root / "summary.json",
                    "--graded-output", evaluated)
        self.assertEqual(read_jsonl(evaluated)[0]["evidence_metrics"], supplied)

    def test_file_at_one_includes_deleted_reference_files(self):
        attempts = self.jsonl("attempts.jsonl", [{"instance_id": "a", "repeat": 0,
            "benchmark": "lite", "model": "model", "system": "LOCUS", "resolved": True,
            "file_predictions": ["obsolete.py"]}])
        grading = self.jsonl("grading.jsonl", [{"instance_id": "a", "patch":
            "diff --git a/obsolete.py b/obsolete.py\ndeleted file mode 100644\n"
            "--- a/obsolete.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n"}])
        evaluated = self.root / "evaluated.jsonl"
        self.invoke("evaluate", "--attempts", attempts, "--grading-metadata", grading,
                    "--output", self.root / "summary.json", "--graded-output", evaluated)
        self.assertIs(read_jsonl(evaluated)[0]["evidence_metrics"]["file_at_1"], True)

    def test_interventions_cli_emits_five_frozen_conditions(self):
        def source(identifier, path, roles):
            return {"id": identifier, "text": identifier, "tokens": 100, "roles": roles,
                    "locations": [{"repo": "owner/repo", "commit": "base", "path": path,
                                   "start": 1, "end": 2}]}
        selected = [source("edit", "edit.py", ["editable"]),
                    source("support", "support.py", ["support"]),
                    source("duplicate", "support.py", ["support"])]
        frontier = [{"id": "edit", "location": selected[0]["locations"][0], "score": 1}]
        packet = self.json("packet.json", {"frontier": frontier, "sources": selected, "pool": selected})
        support = self.json("support.json", ["support.py:1:2"])
        output = self.root / "conditions"
        self.invoke("interventions", "--packet", packet, "--reference-support", support,
                    "--output", output, "--seed", 7)
        files = list(output.glob("*.json"))
        self.assertEqual(len(files), 5)
        for path in files:
            condition = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(condition["frontier"], frontier)
            self.assertIs(condition["recovery_allowed"], False)
            self.assertLessEqual(condition["tokens"], condition["token_cap"])

    def test_run_five_repeats_and_resume_without_starting_workers(self):
        tasks = self.jsonl("tasks.jsonl", [{"instance_id": "a", "benchmark": "pro-v2"}])
        output = self.root / "attempts.jsonl"
        seen = []

        def run_agent(task, settings, model, **kwargs):
            seen.append((task, settings))
            return {"instance_id": task["instance_id"], "repeat": task["repeat"], "resolved": None}

        arguments = ("run", "--config", ROOT / "configs/gpt52.toml", "--tasks", tasks,
                     "--output", output)
        with patch("locus.controller.run_agent", side_effect=run_agent):
            self.invoke(*arguments)
            self.invoke(*arguments, "--resume")
        self.assertEqual(len(seen), 5)
        self.assertEqual({item[0]["repeat"] for item in seen}, set(range(5)))
        self.assertTrue(all(settings.deadline_seconds == 3000 for _, settings in seen))

    def test_grade_commands_and_pristine_v2_dispatch(self):
        output = json.loads(self.invoke("grade-lite", "--predictions", self.root / "p.jsonl",
                                       "--run-id", "test-r0", "--workers", 2))
        self.assertEqual(output["argv"][1:3], ["-m", "swebench.harness.run_evaluation"])
        output = json.loads(self.invoke("grade-pro-v1", "--harness", self.root / "upstream",
            "--grading-metadata", self.root / "grading.jsonl", "--patches", self.root / "p.json",
            "--output", self.root / "grades"))
        self.assertIn("--use_local_docker", output["argv"])
        attempts = self.jsonl("attempts.jsonl", [{"instance_id": "a", "repeat": r,
            "model_patch": "diff"} for r in range(2)])
        with patch("locus.cli.grade_pro_v2", return_value={"instance_id": "a", "repeat": 1,
                                                         "resolved": True}) as grade:
            self.invoke("grade-pro-v2", "--attempts", attempts, "--tasks-dir", self.root / "tasks",
                        "--repeat", 1, "--output", self.root / "grades.jsonl")
        grade.assert_called_once_with(self.root / "tasks/a", read_jsonl(attempts)[1], network="bridge")

    def test_allocation_sweep_shares_matching_complete_task_records(self):
        tasks = self.jsonl("allocation-tasks.jsonl", [
            dict(instance_id=label.lower(), expected=label, benchmark="pro-v2")
            for label in ("L", "M", "H")])
        output, packets = self.root / "allocation", self.root / "packets"
        seen = []

        def run_agent(task, settings, model, **kwargs):
            seen.append((task["instance_id"], task["repeat"], settings.allocation))
            packet = kwargs["packet_output"]
            packet.parent.mkdir(parents=True, exist_ok=True)
            packet.write_text(task["instance_id"] + str(task["repeat"]))
            return dict(instance_id=task["instance_id"], repeat=task["repeat"],
                        system=task["system"], resolved=False,
                        voted_class=task["expected"],
                        budget_class=task["expected"] if settings.allocation == "adaptive"
                                     else settings.allocation,
                        settings=dict(allocation=settings.allocation),
                        ledger=dict(calls=10, cost=len(seen)))

        with patch("locus.controller.run_agent", side_effect=run_agent):
            self.invoke("sweep", "--study", "allocation", "--config", ROOT / "configs/gpt52.toml",
                        "--tasks", tasks, "--repeats", 2, "--output", output, "--packet-dir", packets)
        self.assertEqual(len(seen), 18)
        adaptive = {(row["instance_id"], row["repeat"]):row
                    for row in read_jsonl(output / "LOCUS/attempts.jsonl")}
        for label in ("L", "M", "H"):
            rows = read_jsonl(output / ("Uniform-"+label) / "attempts.jsonl")
            self.assertEqual(len(rows), 6)
            for row in rows:
                key = row["instance_id"], row["repeat"]
                if row["instance_id"] == label.lower():
                    self.assertEqual(row["shared_attempt"]["system"], "LOCUS")
                    self.assertEqual(row["ledger"], adaptive[key]["ledger"])
                    self.assertEqual(row["settings"]["allocation"], label)
                    filename = f"{row['instance_id']}-r{row['repeat']}.json"
                    self.assertEqual((packets / ("Uniform-"+label) / filename).read_bytes(),
                                     (packets / "LOCUS" / filename).read_bytes())
                else:
                    self.assertNotIn("shared_attempt", row)

    def test_selection_sweep_keeps_upstream_broad_context_separate(self):
        from locus.config import Settings
        from locus.experiments import cases
        self.assertEqual({case.name for case in cases("selection", Settings())},
                         {"EditOnly", "StaticTopK", "ExactDedup", "LOCUS"})

    def main_width_reference(self):
        tasks = self.jsonl("width-tasks.jsonl", [
            dict(instance_id=identity, benchmark="pro-v2", repo="example/repository",
                 base_commit="base-commit", problem_statement="Repair " + identity,
                 docker_image="public/example:fixture") for identity in ("a", "b", "c")])
        output = self.root / "main-attempts.jsonl"

        def measured_attempt(task, settings, model, **kwargs):
            return dict(instance_id=task["instance_id"], repeat=task["repeat"],
                        system=task["system"], resolved=task["repeat"] == 0,
                        model_patch="main-patch-" + task["instance_id"],
                        settings=asdict(settings), frontier_size=settings.frontier_size,
                        ledger=dict(calls=13 + task["repeat"], elapsed_seconds=42.5,
                                    tokens=1200, cost=.012),
                        model_runtime=dict(provider=model.provider, configured_model=model.model,
                                           local_seed=settings.seed, requests=[]))

        with patch("locus.controller.run_agent", side_effect=measured_attempt):
            self.invoke("run", "--config", ROOT / "configs/gpt52.toml", "--tasks", tasks,
                        "--repeats", 2, "--output", output)
        return tasks, output, read_jsonl(output)

    def test_width_reuses_only_main_k5_with_larger_reference_roster_and_resume(self):
        tasks, reference, originals = self.main_width_reference()
        output = self.root / "width"
        seen = []

        def run_agent(task, settings, model, **kwargs):
            seen.append((task["system"], task["instance_id"], task["repeat"]))
            return dict(instance_id=task["instance_id"], repeat=task["repeat"],
                        model_patch="new-width-patch", resolved=False,
                        settings=asdict(settings), frontier_size=settings.frontier_size,
                        ledger=dict(calls=20, cost=.03))

        arguments = ("sweep", "--study", "width", "--config", ROOT / "configs/gpt52.toml",
                     "--tasks", tasks, "--repeats", 2, "--limit", 2,
                     "--reference-attempts", reference, "--output", output)
        with patch("locus.controller.run_agent", side_effect=run_agent):
            self.invoke(*arguments)
            self.invoke(*arguments, "--resume")
        self.assertEqual(len(seen), 20)
        self.assertEqual({system for system, _, _ in seen},
                         {"LOCUS-K1", "LOCUS-K3", "LOCUS-K8", "LOCUS-K12", "LOCUS-K18"})
        by_key = {(row["instance_id"], row["repeat"]): row for row in originals}
        reused = read_jsonl(output / "LOCUS-K5/attempts.jsonl")
        self.assertEqual(len(reused), 4)
        for row in reused:
            original = by_key[row["instance_id"], row["repeat"]]
            self.assertEqual(row["system"], "LOCUS-K5")
            for field in ("model_patch", "resolved", "ledger", "seed", "settings", "model_runtime"):
                self.assertEqual(row[field], original[field])
            self.assertEqual(row["shared_attempt"]["source"], str(reference.resolve()))
            self.assertEqual(row["shared_attempt"]["system"], "LOCUS")
            self.assertEqual(row["shared_attempt"]["run_design"], original["run_design"])
            self.assertEqual(row["run_design"]["code_version"], original["run_design"]["code_version"])
            self.assertNotEqual(row["run_design"]["batch_id"], original["run_design"]["batch_id"])
        self.assertEqual(read_jsonl(reference), originals)
        for width in (1, 3, 8, 12, 18):
            for row in read_jsonl(output / f"LOCUS-K{width}/attempts.jsonl"):
                self.assertNotIn("shared_attempt", row)
                self.assertEqual(row["model_patch"], "new-width-patch")

    def test_width_requires_reference_before_running_any_case(self):
        with patch("locus.controller.run_agent") as run:
            self.invoke("sweep", "--study", "width", "--config", ROOT / "configs/gpt52.toml",
                        "--tasks", "unused.jsonl", "--output", self.root / "missing", expected=2)
        run.assert_not_called()

    def test_width_rejects_reference_control_and_identity_mismatches_before_execution(self):
        tasks, _, originals = self.main_width_reference()
        changes = {
            "benchmark": lambda rows: rows[0].update(benchmark="lite"),
            "model": lambda rows: rows[0].update(model="different-model"),
            "seed": lambda rows: rows[0].update(seed=rows[0]["seed"] + 1),
            "settings": lambda rows: rows[0]["settings"].update(packet_tokens=8000),
            "base_commit": lambda rows: rows[0]["run_design"]["task"].update(base_commit="other"),
            "endpoint": lambda rows: rows[0]["run_design"]["model"].update(endpoint="https://other.test"),
            "code_version": lambda rows: rows[0]["run_design"].update(code_version="sha256:other"),
            "missing_repeat": lambda rows: rows.pop(0),
            "duplicate": lambda rows: rows.append(deepcopy(rows[0])),
        }
        for label, change in changes.items():
            with self.subTest(mismatch=label):
                rows = deepcopy(originals)
                change(rows)
                reference = self.jsonl(f"reference-{label}.jsonl", rows)
                output = self.root / ("width-" + label)
                with patch("locus.controller.run_agent") as run:
                    self.invoke("sweep", "--study", "width", "--config", ROOT / "configs/gpt52.toml",
                                "--tasks", tasks, "--repeats", 2, "--reference-attempts", reference,
                                "--output", output, expected=2)
                run.assert_not_called()
                self.assertFalse(output.exists())

    def test_pro_v1_execute_uses_official_harness_working_directory(self):
        harness = self.root / "official-harness"
        harness.mkdir()
        (harness / "swe_bench_pro_eval.py").write_text("# temporary test entrypoint\n")
        with patch("locus.cli.subprocess.run") as run:
            self.invoke("grade-pro-v1", "--harness", harness, "--grading-metadata",
                        self.root / "grading.jsonl", "--patches", self.root / "patches.json",
                        "--output", self.root / "grades", "--execute")
        self.assertEqual(Path(run.call_args.kwargs.get("cwd", Path.cwd())).resolve(), harness.resolve())

    def test_agent_worker_falls_back_to_git_checkout_and_disables_network(self):
        settings, _ = load_config(ROOT / "configs/gpt52.toml")
        commands = []

        def host(worker, argv, **kwargs):
            commands.append(argv)
            if "for d in /app /testbed /workspace; " in " ".join(argv):
                return b"/testbed"
            if "printf yes" in " ".join(argv):
                return b""
            if "rev-parse" in argv:
                return b"base\n"
            return b""

        with patch.object(DockerWorkspace, "_host", host), \
             patch("locus.environment.subprocess.run"):
            with DockerWorkspace("public/image:demo", "base", Ledger(settings),
                                 repo="owner/repo", workdir="/app") as worker:
                self.assertEqual(worker.workdir, "/testbed")
        launch = commands[0]
        self.assertEqual(launch[launch.index("--network") + 1], "none")
        repository_commands = [argv for argv in commands if "git" in argv]
        self.assertTrue(repository_commands)
        self.assertTrue(all(argv[argv.index("--workdir") + 1] == "/testbed"
                            for argv in repository_commands))

    def test_provider_configs_construct_compatible_requests_offline(self):
        for name in ("gpt52", "sonnet"):
            with self.subTest(config=name):
                settings, model = load_config(ROOT / f"configs/{name}.toml")
                model = replace(model, base_url="http://localhost:1234")
                captured = []
                raw = ({"content": [{"type": "text", "text": '{"ok": true}'}],
                        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
                       if model.provider == "anthropic" else
                       {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

                def response(request, **kwargs):
                    captured.append(json.loads(request.data))
                    return io.BytesIO(json.dumps(raw).encode())

                with patch.dict(os.environ, {model.api_key_env: ""}), \
                     patch("locus.models.urllib.request.urlopen", side_effect=response):
                    result = HTTPModel(model, settings, Ledger(settings), len).ask(
                        "test", "Return a JSON object.", "Respond with ok=true.")
                self.assertTrue(result["ok"])
                body = captured[0]
                if model.provider == "anthropic":
                    self.assertFalse("temperature" in body and "top_p" in body)
                    self.assertEqual(body["temperature"], .7)
                else:
                    self.assertEqual(body["reasoning_effort"], "none")
                    self.assertEqual(body["temperature"], .7)
                    self.assertEqual(body["top_p"], 1.)


if __name__ == "__main__":
    unittest.main()
