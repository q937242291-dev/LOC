"""Pairing controls and statistics through the public command interface."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from locus import cli
from locus.baselines import import_predictions
from locus.benchmarks import read_jsonl, write_jsonl
from locus.config import Settings
from locus.reporting import analysis_report


ROOT = Path(__file__).resolve().parents[1]


class ReportingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="locus-pairing-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def invoke(self, *args, status=0):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            actual = cli.main([str(a) for a in args])
        self.assertEqual(actual, status, error.getvalue())
        return output.getvalue(), error.getvalue()

    def task_file(self):
        path = self.root / "tasks.jsonl"
        write_jsonl(path, [dict(instance_id="a", benchmark="pro-v2", repo="example/project",
                               base_commit="base", docker_image="public/image:tag",
                               problem_statement="Repair the example function.")])
        return path

    def fixture_agent(self, task, settings, model, **kwargs):
        return dict(instance_id=task["instance_id"], repeat=task["repeat"],
                    settings=asdict(settings), resolved=True)

    def test_component_sweep_has_one_batch_and_shared_task_seeds(self):
        tasks = self.task_file()
        output = self.root / "components"
        with patch("locus.controller.run_agent", side_effect=self.fixture_agent):
            self.invoke("sweep", "--study", "components", "--config", ROOT / "configs/gpt52.toml",
                        "--tasks", tasks, "--output", output, "--seed", 42, "--batch-id", "test-batch")
        files = sorted(output.glob("*/attempts.jsonl"))
        rows = [row for path in files for row in read_jsonl(path)]
        self.assertEqual(len(rows), 35)
        self.assertEqual({r["run_design"]["batch_id"] for r in rows}, {"test-batch"})
        self.assertEqual(len({r["run_design"]["design_id"] for r in rows}), 1)
        for repeat in range(5):
            self.assertEqual(len({r["seed"] for r in rows if r["repeat"] == repeat}), 1)
        report = analysis_report(rows, samples=10)
        self.assertEqual(len(report["paired_comparisons"]), 6)
        for comparison in report["paired_comparisons"]:
            self.assertEqual(comparison["design_audit"]["status"], "matched")
            self.assertTrue(comparison["five_repeat_complete"])
        for summary in report["repeats"]:
            self.assertEqual(summary["sample_variance"], 0.)

    def test_resume_rejects_changed_seed_without_overwriting_files(self):
        tasks, target = self.task_file(), self.root / "attempts.jsonl"
        args = ("run", "--config", ROOT / "configs/gpt52.toml", "--tasks", tasks,
                "--output", target)
        with patch("locus.controller.run_agent", side_effect=self.fixture_agent) as run:
            self.invoke(*args)
            before = target.read_bytes()
            self.invoke(*args, "--resume", "--seed", 99, status=2)
            self.invoke(*args, "--resume")
        self.assertEqual(run.call_count, 5)
        self.assertEqual(target.read_bytes(), before)

    def test_multi_file_evaluate_exposes_per_run_counts_and_variance(self):
        files = []
        for system, outcomes in (("LOCUS", [True, False, True, False, True]),
                                 ("Baseline", [False] * 5)):
            path = self.root / (system + ".jsonl")
            write_jsonl(path, [dict(instance_id="a", benchmark="b", model="m", system=system,
                                   repeat=i, resolved=outcome) for i, outcome in enumerate(outcomes)])
            files.append(path)
        target = self.root / "summary.json"
        self.invoke("evaluate", "--attempts", *files, "--output", target, "--bootstrap-samples", 10)
        report = json.loads(target.read_text())
        locus = next(r for r in report["repeats"] if r["system"] == "LOCUS")
        self.assertEqual(locus["sample_variance"], 3000.)
        self.assertEqual(locus["counts_by_repeat"]["0"]["successes"], 1)
        delta = report["paired_comparisons"][0]
        self.assertEqual(delta["resolution_delta_pp"], 60.)
        self.assertEqual(delta["design_audit"]["status"], "unknown")

    def test_one_methods_grades_cannot_be_applied_to_multiple_systems(self):
        path = self.root / "attempts.jsonl"
        write_jsonl(path, [dict(instance_id="a", repeat=0, benchmark="b", model="m", system=s)
                           for s in ("LOCUS", "Baseline")])
        grade = self.root / "grades.jsonl"
        write_jsonl(grade, [dict(instance_id="a", repeat=0, resolved=True)])
        self.invoke("evaluate", "--attempts", path, "--grades", grade,
                    "--output", self.root / "summary.json", status=2)

    def test_partial_repeats_do_not_report_five_repeat_variance(self):
        rows = [dict(instance_id="a", repeat=0, benchmark="b", model="m", system="LOCUS", resolved=True)]
        result = analysis_report(rows, samples=10)["repeats"][0]
        self.assertFalse(result["five_repeat_complete"])
        self.assertIsNone(result["sample_variance"])
        self.assertIsNone(result["resolution_interval"])

    def test_explicit_roster_detects_task_missing_from_all_five_repeats(self):
        rows = [dict(instance_id="a", repeat=r, benchmark="b", model="m", system="LOCUS", resolved=True)
                for r in range(5)]
        report = analysis_report(rows, samples=10, task_rosters={"b": ["a", "b"]})
        result = report["repeats"][0]
        self.assertEqual(result["missing_attempts"], 5)
        self.assertFalse(result["five_repeat_complete"])
        self.assertIsNone(result["sample_variance"])
        self.assertEqual(report["statistical_protocol"]["roster_source"], "explicit")

    def test_all_methods_share_the_observed_task_roster(self):
        rows = [dict(instance_id=task, repeat=run, benchmark="b", model="m", system=system,
                     resolved=task == "a", ledger={"cost": 1 if system == "LOCUS" else 2,
                                                    "tokens": 10})
                for system, tasks in (("LOCUS", ("a",)), ("Baseline", ("a", "b")))
                for task in tasks for run in range(5)]
        report = analysis_report(rows, samples=10)
        summaries = {summary["system"]: summary for summary in report["repeats"]}
        self.assertEqual(summaries["LOCUS"]["missing_attempts"], 5)
        self.assertFalse(summaries["LOCUS"]["five_repeat_complete"])
        self.assertIsNone(summaries["LOCUS"]["sample_variance"])
        self.assertIsNone(summaries["LOCUS"]["resolution_interval"])
        self.assertTrue(summaries["Baseline"]["five_repeat_complete"])
        comparison = report["comparisons"][0]
        self.assertFalse(comparison["common_roster_complete"])
        for metric in ("resolution_delta_pp", "tokens_saved_pct", "cost_saved_pct",
                       "cost_saved_vs_cheapest_pct", "quality_floor_met"):
            self.assertIsNone(comparison[metric])
        # Observed group totals remain available independently of comparability.
        observed = next(group for group in report["groups"] if group["system"] == "LOCUS")
        self.assertEqual(observed["attempts"], 5)
        self.assertEqual(observed["normalized_cost_total"], 5)
        paired = report["paired_comparisons"][0]
        self.assertFalse(paired["complete"])
        self.assertEqual(paired["missing_left"], 5)
        self.assertFalse(report["highest_baseline_comparisons"][0]["complete"])

    def test_missing_baseline_task_cannot_produce_quality_or_cheapest_claim(self):
        rows = [dict(instance_id=task, repeat=run, benchmark="b", model="m", system=system,
                     resolved=True, ledger={"cost": 1})
                for system, tasks in (("LOCUS", ("a", "b")), ("Baseline", ("a",)))
                for task in tasks for run in range(5)]
        comparison = analysis_report(rows, samples=10)["comparisons"][0]
        self.assertFalse(comparison["common_roster_complete"])
        self.assertIsNone(comparison["resolution_delta_pp"])
        self.assertIsNone(comparison["quality_floor_met"])
        self.assertIsNone(comparison["lowest_cost_baseline"])
        self.assertIsNone(comparison["cost_saved_vs_cheapest_pct"])

    def test_rosters_are_separate_per_benchmark_and_model(self):
        rows = [dict(instance_id=task, repeat=run, benchmark=benchmark, model=model,
                     system=system, resolved=system == "LOCUS")
                for benchmark, model, task in (("b", "m", "a"), ("b", "other", "b"),
                                               ("other", "m", "c"))
                for system in ("LOCUS", "Baseline") for run in range(5)]
        report = analysis_report(rows, samples=10)
        self.assertEqual(len(report["paired_comparisons"]), 3)
        self.assertTrue(all(item["five_repeat_complete"] for item in report["repeats"]))
        self.assertTrue(all(item["common_roster_complete"] for item in report["comparisons"]))

    def test_mixed_method_case_uses_one_explicit_reference_consistently(self):
        rows = [dict(instance_id="a", repeat=run, benchmark="b", model="m", system=system,
                     resolved=system == "locus")
                for system in ("locus", "Baseline") for run in range(5)]
        missing = analysis_report(rows, samples=10)
        self.assertEqual(missing["comparisons"], [])
        self.assertFalse(missing["highest_baseline_comparisons"][0]["complete"])
        explicit = analysis_report(rows, locus_system="locus", samples=10)
        self.assertEqual(explicit["comparisons"][0]["resolution_delta_pp"], 100)
        self.assertTrue(explicit["paired_comparisons"][0]["complete"])

    def test_invalid_controls_and_duplicate_attempts_fail_before_reporting(self):
        row = dict(instance_id="a", repeat=0, benchmark="b", model="m", system="LOCUS", resolved=True)
        with self.assertRaisesRegex(ValueError, "Duplicate attempt"):
            analysis_report([row, dict(row)], samples=10)
        for options in (dict(samples=True), dict(confidence=True), dict(seed=False),
                        dict(locus_system=""), dict(expected_repeats=False)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                analysis_report([], **options)

    def test_reporting_preserves_observed_control_metadata_without_inventing_matches(self):
        rows = [dict(instance_id="a", repeat=run, benchmark="b", model="m", system=system,
                     seed=run + 12, resolved=system == "LOCUS",
                     model_runtime={"seed_mode": "off", "provider": "observed"})
                for system in ("LOCUS", "Baseline") for run in range(5)]
        before = deepcopy(rows)
        report = analysis_report(rows, samples=10)
        self.assertEqual(rows, before)
        self.assertEqual(report["paired_comparisons"][0]["design_audit"]["status"], "unknown")

    def test_baseline_import_preserves_only_supplied_controls(self):
        task = dict(instance_id="a", benchmark="b")
        controls = dict(seed=42, run_design=dict(batch_id="observed-batch"),
                        model_runtime=dict(seed_mode="off"), settings=dict(temperature=.7))
        rows = import_predictions([], [task], system="Baseline", model="m", telemetry={"a": controls})
        self.assertEqual(rows[0]["run_design"], controls["run_design"])
        self.assertEqual(rows[0]["seed"], 42)
        controls["run_design"]["batch_id"] = "changed"
        self.assertEqual(rows[0]["run_design"]["batch_id"], "observed-batch")
        legacy = import_predictions([], [task], system="Baseline", model="m")
        self.assertNotIn("run_design", legacy[0])
        with self.assertRaises(ValueError):
            import_predictions([], [task], system="Baseline", model="m", telemetry={"a": {"repeat": True}})


if __name__ == "__main__":
    unittest.main()
