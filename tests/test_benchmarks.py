"""Benchmark adapter and grader tests with temporary fixtures."""

import json
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from locus.baselines import import_predictions, load_predictions, render_argv, run_external
from locus.benchmarks import (BENCHMARKS, export_predictions, export_pro_v1_patches,
                              export_harbor_replay, grade_pro_v2, inference_task,
                              merge_grades, read_lite_grades, repeat_plan)


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def row(self):
        return {"instance_id": "instance_owner__project-abc", "repo": "owner/project",
                "base_commit": "base", "problem_statement": "Repair the parser.",
                "requirements": "Accept empty values.", "interface": "parse(text)",
                "patch": "REFERENCE_MUST_NOT_ENTER_AGENT", "test_patch": "TEST_ONLY",
                "fail_to_pass": ["answer"], "before_repo_set_cmd": "GRADER_ONLY"}

    def test_projection_keeps_task_text_and_excludes_answers(self):
        task = inference_task(self.row(), "pro-v2")
        self.assertIn("Accept empty", task["problem_statement"])
        self.assertEqual(set(task), {"instance_id", "repo", "base_commit", "problem_statement",
                                     "benchmark", "docker_image", "workdir"})
        self.assertNotIn("REFERENCE", json.dumps(task))
        self.assertEqual(BENCHMARKS["pro-v1"]["config"], "v1")
        self.assertEqual(BENCHMARKS["pro-v2"]["tasks"], 642)

    def test_five_repeats_are_independent_and_deterministic(self):
        tasks = [inference_task(self.row(), "pro-v2")]
        plan = repeat_plan(tasks, seed=19)
        self.assertEqual(len(plan), 5)
        self.assertEqual(plan, repeat_plan(tasks, seed=19))
        self.assertEqual(len({row["seed"] for row in plan}), 5)

    def test_export_requires_separate_repeats(self):
        rows = [{"instance_id": "x", "repeat": i, "model_patch": "patch", "model": "model"}
                for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            with self.assertRaises(ValueError):
                export_predictions(rows, output)
            self.assertEqual(export_predictions(rows, output, repeat=1), 1)
            self.assertEqual(json.loads(output.read_text())["model_patch"], "patch")
            array = Path(directory) / "patches.json"
            export_pro_v1_patches(rows, array, repeat=0)
            self.assertIsInstance(json.loads(array.read_text()), list)

    def test_missing_grade_stays_unknown(self):
        rows = [{"instance_id": "a", "repeat": 1}, {"instance_id": "b", "repeat": 1}]
        merged = merge_grades(rows, [{"instance_id": "a", "resolved": False}], repeat=1)
        self.assertIs(merged[0]["resolved"], False)
        self.assertNotIn("resolved", merged[1])

    def test_import_preserves_missing_attempt_and_unknown_cost(self):
        tasks = ({"instance_id": identity, "benchmark": "lite"} for identity in ("a", "b"))
        rows = import_predictions([{"instance_id": "a", "model_patch": "diff"}], tasks,
                                  system="upstream", model="model")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["status"], "empty_patch")
        self.assertIsNone(rows[0]["resolved"])
        self.assertIsNone(rows[0]["ledger"])

    def test_upstream_json_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preds.json"
            path.write_text(json.dumps({"a": {"model_patch": "diff"}}))
            self.assertEqual(load_predictions(path)[0]["instance_id"], "a")
        self.assertEqual(render_argv(["tool", "{output}"], {"output": "a b; c"}),
                         ["tool", "a b; c"])

    def test_null_patch_is_kept_as_an_empty_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "predictions.json"
            source.write_text(json.dumps({"a": None}))
            records = import_predictions(load_predictions(source), [{"instance_id": "a"}],
                                         system="upstream", model="model")
            self.assertEqual(records[0]["model_patch"], "")
            self.assertEqual(records[0]["status"], "empty_patch")
            path = Path(directory) / "official.jsonl"
            export_predictions([{"instance_id": "a", "model_patch": None}], path)
            self.assertEqual(json.loads(path.read_text())["model_patch"], "")
            export_harbor_replay([{"instance_id": "a", "model_patch": None}],
                                 Path(directory) / "replay", repeat=0)
            self.assertEqual((Path(directory) / "replay/instance_a/agent/model.patch").read_text(), "")

    def test_pro_v1_export_rejects_multiple_attempts_of_same_task(self):
        rows = [{"instance_id": "a", "repeat": i, "model_patch": "diff"} for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                export_pro_v1_patches(rows, Path(directory) / "patches.json")

    def test_lite_aggregate_includes_empty_patches_and_preserves_unknown_grading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "a/report.json").write_text(json.dumps({"a": {"resolved": True}}))
            summary = root / "results.json"
            summary.write_text(json.dumps({"resolved_ids": ["a"], "unresolved_ids": ["b"],
                "empty_patch_ids": ["c"], "error_ids": ["d"], "incomplete_ids": ["e"]}))
            grades = read_lite_grades(root, repeat=2)
            self.assertEqual(len(grades), 5)
            values = {row["instance_id"]: row["resolved"] for row in grades}
            self.assertEqual(values, {"a": True, "b": False, "c": False, "d": None, "e": None})
            self.assertTrue(all(row["repeat"] == 2 for row in grades))
            self.assertEqual(read_lite_grades(summary, repeat=2), grades)
            summary.write_text(json.dumps({"resolved_ids": [], "unresolved_ids": ["a"]}))
            with self.assertRaises(ValueError):
                read_lite_grades(root)

    def test_all_catalog_templates_render_and_product_entries_use_official_links(self):
        config_path = ROOT / "configs/baselines.toml"
        with config_path.open("rb") as stream:
            systems = tomllib.load(stream)["systems"]
        self.assertEqual(len(systems), 8)
        context = {"model": "model", "workers": 2, "output": "outside/project",
                   "llm_config": "outside/model.json"}
        for name, config in systems.items():
            if "argv" in config:
                self.assertTrue(render_argv(config["argv"], context))
            else:
                with self.subTest(system=name), patch("locus.baselines.subprocess.run") as run:
                    with self.assertRaisesRegex(ValueError, "official entry"):
                        run_external(config_path, name, context)
                    run.assert_not_called()
        self.assertEqual(systems["live-swe-agent"]["runtime"], "mini-swe-agent~=1.0")
        self.assertTrue(systems["codex"]["source"].startswith("https://developers.openai.com/"))
        self.assertTrue(systems["claude-code"]["source"].startswith("https://code.claude.com/"))

    def test_pristine_grader_reads_reward_and_removes_container(self):
        calls = []

        def grading_response(argv, timeout, **kwargs):
            calls.append(argv)
            text = "1\n" if "cat" in argv else ""
            return subprocess.CompletedProcess(argv, 0, text, "")

        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "instance_demo"
            (task / "tests").mkdir(parents=True)
            (task / "task.toml").write_text('[environment]\ndocker_image="public/image:demo"\n'
                                           'cpus=8\nmemory_mb=32768\n[verifier]\ntimeout_sec=3000\n')
            (task / "tests" / "test.sh").write_text("exit 0\n")
            with patch("locus.benchmarks._docker", side_effect=grading_response):
                grade = grade_pro_v2(task, {"instance_id": "instance_demo", "model_patch": ""})
        self.assertTrue(grade["resolved"])
        self.assertEqual(calls[-1][:2], ["rm", "-f"])
        self.assertTrue(any(argv[:1] == ["run"] for argv in calls))
        self.assertTrue(any("bash /tests/test.sh" in argument for argv in calls for argument in argv))

    def test_v2_verifier_gets_its_full_phase_timeout_and_authoritative_reward(self):
        calls = []

        def grading_response(argv, timeout, **kwargs):
            calls.append((argv, timeout))
            applying = any("git apply" in argument for argument in argv)
            return subprocess.CompletedProcess(argv, 1 if applying else 0,
                                               "1\n" if "cat" in argv else "", "")

        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "instance_demo"
            (task / "tests").mkdir(parents=True)
            (task / "task.toml").write_text('[environment]\ndocker_image="public/image:demo"\n'
                'cpus=1\nmemory_mb=4096\n[verifier]\ntimeout_sec=123\n')
            (task / "tests/test.sh").write_text("exit 0\n")
            with patch("locus.benchmarks._docker", side_effect=grading_response):
                grade = grade_pro_v2(task, {"instance_id": task.name, "model_patch": "diff"})
        self.assertIs(grade["resolved"], True)
        verifier = [(argv, timeout) for argv, timeout in calls
                    if any("bash /tests/test.sh" in argument for argument in argv)]
        self.assertEqual(verifier[0][1], 123)
        applying = next(argv for argv, _ in calls if any("git apply" in argument for argument in argv))
        self.assertIn("/app/.git", " ".join(applying))
        self.assertIn("/testbed/.git", " ".join(applying))
        launch = calls[0][0]
        self.assertEqual(launch[launch.index("--network") + 1], "bridge")
        self.assertEqual(launch[launch.index("--cpus") + 1], "1")
        self.assertEqual(launch[launch.index("--memory") + 1], "4096m")


if __name__ == "__main__":
    unittest.main()
