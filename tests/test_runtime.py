"""Recipient admission and immutable-worker contract tests."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from locus.config import ModelSettings, Settings
from locus.environment import DockerWorkspace, WorkerError
from locus.ledger import Ledger
from locus.models import HTTPModel, ModelError, Recipient, usage_counts
from locus.types import Location, Observation


def source(text="return 1", *, path="module.py"):
    return Observation.create(text, Location("owner/repo", "base", path, 1, 1), len)


class RecipientRuntimeTests(unittest.TestCase):
    def test_ordered_duplicate_eviction_keeps_remaining_acknowledgments(self):
        first = source()
        updated = first.with_roles({"editable"}, len)
        recipient = Recipient("system", len, 10000)
        self.assertTrue(recipient.deliver((first, updated), deduplicate=False))
        self.assertTrue(recipient.has(first))
        self.assertTrue(recipient.has(updated))
        recipient.evict(1)
        self.assertEqual(recipient.visible, {first.id: first.version})
        self.assertTrue(recipient.has(first))
        self.assertFalse(recipient.has(updated))
        recipient.evict(0)
        self.assertFalse(recipient.visible)
        self.assertEqual(recipient.transmitted_tokens, first.tokens + updated.tokens)

    def test_failed_preserved_empty_admission_restores_non_source_history(self):
        item = source()
        recipient = Recipient("system", len, 10000)
        recipient.deliver((item,))
        recipient.complete("old question", "old answer")
        recipient.preserve_empty_sources = True
        recipient.cap = recipient.input_size() + 10
        before = (list(recipient.messages), dict(recipient.visible),
                  recipient.transmitted_tokens, recipient.repeated_tokens,
                  set(recipient.ever_delivered))
        with self.assertRaises(ModelError):
            recipient.deliver((), extra="x" * (recipient.cap * 2))
        self.assertEqual(before, (recipient.messages, recipient.visible,
                                  recipient.transmitted_tokens, recipient.repeated_tokens,
                                  recipient.ever_delivered))

    def test_consumed_version_survives_post_request_eviction(self):
        item = source()
        recipient = Recipient("system", len, 10000)
        other = Recipient("other", len, 10000)
        recipient.deliver((item,))
        self.assertFalse(recipient.consumed(item))
        recipient.request_messages("Use the source")
        recipient.complete("Use the source", "{}")
        recipient.evict(0)
        self.assertFalse(recipient.has(item))
        self.assertTrue(recipient.consumed(item))
        self.assertFalse(recipient.consumed(item.with_roles({"editable"}, len)))
        self.assertFalse(other.has(item))
        self.assertFalse(other.consumed(item))

    def test_failed_request_or_admission_does_not_replace_consumed_snapshot(self):
        first, second = source(), source("return 2", path="other.py")
        recipient = Recipient("system", len, 10000)
        recipient.deliver((first,))
        recipient.request_messages("First request")
        recipient.deliver((second,))
        recipient.cap = recipient.input_size() + 10
        with self.assertRaises(ModelError):
            recipient.request_messages("x" * recipient.cap,
                                       frozenset({first.id, second.id}))
        self.assertTrue(recipient.consumed(first))
        self.assertFalse(recipient.consumed(second))
        self.assertFalse(recipient.deliver((source("x" * 10000),)))
        self.assertTrue(recipient.consumed(first))
        self.assertFalse(recipient.consumed(second))

    def test_reset_starts_an_independent_attempt(self):
        item = source()
        recipient = Recipient("system", len, 10000, memory=False)
        recipient.deliver((item,))
        recipient.deliver((item,), deduplicate=False)
        recipient.request_messages("Use source")
        recipient.complete("Use source", "{}")
        recipient.preserve_empty_sources = True
        recipient.count_wire_bytes = True
        recipient.reset()
        self.assertEqual(recipient.messages, [])
        self.assertEqual(recipient.visible, {})
        self.assertEqual(recipient.last_request_versions, frozenset())
        self.assertEqual(recipient.ever_delivered, set())
        self.assertEqual((recipient.transmitted_tokens, recipient.repeated_tokens), (0, 0))
        self.assertFalse(recipient.preserve_empty_sources)
        self.assertEqual((recipient.system, recipient.count, recipient.cap, recipient.memory),
                         ("system", len, 10000, False))
        self.assertTrue(recipient.count_wire_bytes)

    def test_cached_aliases_must_report_the_same_disjoint_category(self):
        base = dict(prompt_tokens=100, completion_tokens=5, prompt_cache_hit_tokens=80)
        for details in ({"cached_tokens": 79}, {"cached_tokens": True}):
            with self.subTest(details=details), self.assertRaises(ModelError):
                usage_counts(dict(base, prompt_tokens_details=details), "compatible")
        self.assertEqual(usage_counts(dict(base, prompt_tokens_details={"cached_tokens": 80},
                                          input_tokens_details={"cached_tokens": 80}),
                                      "compatible"), (20, 80, 5))
        self.assertEqual(usage_counts(dict(input_tokens=100, output_tokens=5,
                                          input_tokens_details={"cached_tokens": 80}),
                                      "compatible"), (20, 80, 5))
        with self.assertRaises(ModelError):
            usage_counts(dict(base, prompt_tokens_details=False), "compatible")

    def test_direct_model_construction_validates_provider_and_timeout(self):
        settings = Settings()
        for model in (ModelSettings(provider="unsupported"),
                      ModelSettings(timeout_seconds=float("nan")),
                      ModelSettings(timeout_seconds=0)):
            with self.subTest(model=model), self.assertRaises(ValueError):
                HTTPModel(model, settings, Ledger(settings), len)


class WorkerRuntimeTests(unittest.TestCase):
    def worker(self):
        worker = object.__new__(DockerWorkspace)
        worker.ledger = Ledger(Settings(deadline_seconds=10), clock=lambda: 0)
        worker.docker, worker.container = "docker", "fixture-worker"
        worker.repo, worker.commit, worker.workdir = "owner/repo", "base", "/app"
        return worker

    def test_host_timeout_is_bounded_even_when_caller_supplies_larger_limit(self):
        worker = self.worker()
        with patch("locus.environment.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0, b"ok")) as execute:
            self.assertEqual(worker._host(["docker", "info"], timeout=100), b"ok")
        self.assertEqual(execute.call_args.kwargs["timeout"], 10)
        self.assertEqual(worker.ledger.summary()["calls"], 1)

    def test_nonzero_and_container_timeout_results_are_failed_charged_actions(self):
        for returncode in (1, 124, 137):
            worker = self.worker()
            with self.subTest(returncode=returncode), patch(
                    "locus.environment.subprocess.run",
                    return_value=subprocess.CompletedProcess([], returncode, b"failed")) as execute:
                result = worker.run("fixture-command", timeout=20)
            self.assertEqual(result["exit_code"], returncode)
            self.assertFalse(worker.ledger.steps[0].success)
            self.assertEqual(worker.ledger.summary()["calls"], 1)
            self.assertIn("timeout", execute.call_args.args[0])
            self.assertLessEqual(execute.call_args.kwargs["timeout"], 10)

    def test_invalid_read_interval_is_rejected_before_consuming_a_tool_call(self):
        worker = self.worker()
        for start, end in ((True, 1), (1, 1.0), (0, 2), (2, 1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                worker.read("module.py", start, end)
        self.assertEqual(worker.ledger.summary()["calls"], 0)

    def test_submission_uses_immutable_commit_and_disables_diff_drivers(self):
        worker = self.worker()
        with patch.object(worker, "_exec", return_value=b"patch") as execute:
            self.assertEqual(worker.diff(), "patch")
        argv = execute.call_args.args[0]
        self.assertEqual(argv[-1], worker.commit)
        self.assertNotIn("HEAD", argv[2])
        self.assertIn("--no-replace-objects", argv[2])
        self.assertIn("--no-ext-diff", argv[2])
        self.assertIn("--no-textconv", argv[2])


@unittest.skipUnless(shutil.which("git"), "Local Git is required for immutable-source checks")
class GitSourceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="locus-runtime-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.git("init", "-q")
        self.git("config", "user.email", "anonymous@localhost")
        self.git("config", "user.name", "Runtime Fixture")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.autocrlf", "false")
        for path, text in {
            ".gitattributes": "hidden.py export-ignore\nversion.py export-subst\n",
            "hidden.py": "value = 1\n",
            "copy.py": "value = 1\n",
            "version.py": 'version = "$Format:%H$"\n',
            ".env": "FIXTURE_SETTING=example\n",
            "secret.pem": "runtime fixture\n",
        }.items():
            (self.root / path).write_bytes(text.encode())
        (self.root / "binary.bin").write_bytes(b"\x00fixture binary")
        self.git("add", "--all")
        self.git("commit", "-qm", "base fixture")
        self.base = self.git("rev-parse", "HEAD").decode().strip()
        self.worker = object.__new__(DockerWorkspace)
        self.worker.ledger = Ledger(Settings())
        self.worker.commit, self.worker.repo = self.base, "owner/repo"
        self.worker._exec = self.execute

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    def execute(self, argv, *, stdin=None, timeout=None, role="worker"):
        if argv[0] == "sh":
            shell = shutil.which("sh")
            if not shell:
                bundled = Path(shutil.which("git")).parent.parent / "bin" / "sh.exe"
                if bundled.is_file():
                    shell = str(bundled)
            if not shell:
                self.skipTest("A local POSIX shell is required for the submission command")
            argv = [shell] + argv[1:]
        with self.worker.ledger.action("tool", role):
            return subprocess.run(argv, cwd=self.root, input=stdin, check=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    def test_source_snapshot_preserves_export_attributes_and_filters_binary_and_secrets(self):
        snapshot = self.worker.snapshot()
        self.assertEqual(snapshot["hidden.py"], "value = 1\n")
        self.assertEqual(snapshot["copy.py"], snapshot["hidden.py"])
        self.assertEqual(snapshot["version.py"], 'version = "$Format:%H$"\n')
        self.assertNotIn("binary.bin", snapshot)
        self.assertNotIn(".env", snapshot)
        self.assertNotIn("secret.pem", snapshot)
        self.assertEqual(self.worker.ledger.summary()["calls"], 2)

    def test_replace_refs_and_later_commits_cannot_change_base_source(self):
        (self.root / "hidden.py").write_bytes(b"value = 2\n")
        self.git("add", "--all")
        self.git("commit", "-qm", "validation-created commit")
        replacement = self.git("rev-parse", "HEAD").decode().strip()
        self.git("replace", self.base, replacement)
        self.assertEqual(self.worker.snapshot()["hidden.py"], "value = 1\n")
        text, location = self.worker.read("hidden.py", 1, 1)
        self.assertEqual(text, "value = 1")
        self.assertEqual(location.commit, self.base)

    def test_source_limit_is_checked_before_loading_blob_payloads(self):
        with self.assertRaises(WorkerError):
            self.worker.snapshot(max_bytes=1)
        self.assertEqual(self.worker.ledger.summary()["calls"], 1)

    def test_submission_keeps_changes_committed_by_a_validation_command(self):
        (self.root / "hidden.py").write_bytes(b"value = 2\n")
        (self.root / "created.py").write_bytes(b"created = True\n")
        self.git("add", "--all")
        self.git("commit", "-qm", "validation-created commit")
        self.assertEqual(self.git("diff", "HEAD", "--"), b"")
        result = self.worker.diff()
        self.assertIn("diff --git a/hidden.py b/hidden.py", result)
        self.assertIn("-value = 1", result)
        self.assertIn("+value = 2", result)
        self.assertIn("diff --git a/created.py b/created.py", result)


if __name__ == "__main__":
    unittest.main()
