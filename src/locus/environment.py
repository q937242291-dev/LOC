"""Disposable, offline repository workers; credentials stay in the controller."""
from __future__ import annotations

import subprocess
import uuid

from .types import Location, repository_path


class WorkerError(RuntimeError):
    pass


class DockerWorkspace:
    def __init__(self, image: str, commit: str, ledger, *, repo: str,
                 workdir="", docker="docker", cpus=8, memory="32g"):
        self.ledger, self.repo, self.commit = ledger, repo, commit
        self.docker, self.container = docker, "locus-" + uuid.uuid4().hex[:16]
        self.workdir = workdir
        self.started = False
        if image.startswith("-") or not image:
            raise ValueError("Invalid worker image")
        try:
            self._host([docker, "run", "--detach", "--name", self.container,
                        "--network", "none", "--cpus", str(cpus), "--memory", memory,
                        "--pids-limit", "512", "--cap-drop", "ALL",
                        "--security-opt", "no-new-privileges",
                        "--entrypoint", "/bin/sh", image, "-c", "while :; do sleep 3600; done"])
            self.started = True
            if self.workdir:
                valid_dir = self._host(
                    [docker,"exec",self.container,"sh","-c",
                     'if [ -d "$1/.git" ]; then printf yes; fi', "sh", self.workdir]
                ).decode().strip() if self.workdir else ""
            else:
                valid_dir = ""
            if not valid_dir:
                self.workdir = self._host(
                    [docker, "exec", self.container, "sh", "-c",
                     "for d in /app /testbed /workspace; do "
                     "if [ -d \"$d/.git\" ]; then printf '%s' \"$d\"; exit 0; fi; done; exit 1"]
                ).decode().strip()
            head = self._exec(["git", "--no-replace-objects", "rev-parse", "HEAD"]).decode().strip()
            target = self._exec(["git", "--no-replace-objects", "rev-parse", "--verify",
                                 commit + "^{commit}"]).decode().strip()
            if head != target:
                self._exec(["git", "--no-replace-objects", "checkout", "--detach", target])
            self.commit = target
            if self._exec(["git", "--no-replace-objects", "diff", "--no-ext-diff",
                           "--no-textconv", self.commit, "--"]).strip():
                raise WorkerError("Worker image must start with a clean base checkout")
        except Exception:
            self.close()
            raise

    def _host(self, argv, *, stdin=None, timeout=None, charge=True, role="worker"):
        def execute():
            remaining = self.ledger.allowance()[1]
            seconds = remaining if timeout is None else min(float(timeout), remaining)
            if seconds <= 0:
                raise WorkerError("Worker command has no remaining time allowance")
            try:
                result = subprocess.run(argv, input=stdin, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, timeout=seconds, check=False)
            except (OSError, subprocess.TimeoutExpired):
                raise WorkerError("Worker command failed or timed out") from None
            if result.returncode:
                raise WorkerError("Worker command exited unsuccessfully")
            return result.stdout
        if charge:
            with self.ledger.action("tool", role):
                return execute()
        return execute()

    def _exec(self, argv, *, stdin=None, timeout=None, role="worker"):
        cmd = [self.docker, "exec"]
        if stdin is not None:
            cmd.append("-i")
        cmd += ["--workdir", self.workdir, self.container] + list(argv)
        return self._host(cmd, stdin=stdin, timeout=timeout, role=role)

    def snapshot(self, max_bytes=256_000_000) -> dict[str, str]:
        # Read immutable blobs rather than git archive: export-ignore and
        # export-subst attributes must not alter the repository source view.
        listing = self._exec(["git", "--no-replace-objects", "ls-tree", "-rlz",
                              "--full-tree", self.commit], role="structure")
        paths: dict[str, tuple[str, int]] = {}
        sizes: dict[str, int] = {}
        for record in listing.split(b"\x00"):
            if not record:
                continue
            try:
                header, raw_path = record.split(b"\t", 1)
                mode, kind, raw_oid, raw_size = header.split()
                if mode not in {b"100644", b"100755"} or kind != b"blob":
                    continue
                size = int(raw_size)
                if size > 2_000_000:
                    continue
                path = repository_path(raw_path.decode("utf-8"))
                oid = raw_oid.decode("ascii")
            except (ValueError, UnicodeError):
                raise WorkerError("Worker returned an invalid repository tree") from None
            if size < 0 or not oid or any(c not in "0123456789abcdef" for c in oid):
                raise WorkerError("Worker returned an invalid repository blob")
            if path.rsplit("/", 1)[-1].startswith(".env") or path.endswith((".pem", ".key", ".p12")):
                continue
            paths[path] = (oid, size)
            sizes[oid] = size
        if sum(sizes.values()) > max_bytes:
            raise WorkerError("Repository source exceeds indexing memory allowance")
        if not paths:
            return {}
        raw = self._exec(["git", "--no-replace-objects", "cat-file", "--batch"],
                         stdin="".join(oid+"\n" for oid in sizes).encode("ascii"),
                         role="structure")
        blobs = {}
        offset = 0
        for oid, size in sizes.items():
            end = raw.find(b"\n", offset)
            if end < 0 or raw[offset:end] != f"{oid} blob {size}".encode("ascii"):
                raise WorkerError("Worker returned an invalid repository blob")
            offset = end+1
            data = raw[offset:offset+size]
            offset += size
            if len(data) != size or raw[offset:offset+1] != b"\n":
                raise WorkerError("Worker returned an incomplete repository blob")
            offset += 1
            if b"\x00" not in data:
                blobs[oid] = data.decode("utf-8", errors="replace")
        result = {path: blobs[oid] for path, (oid, _) in paths.items() if oid in blobs}
        return result

    def read(self, path: str, start: int, end: int) -> tuple[str, Location]:
        path = repository_path(path)
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise ValueError("Invalid read interval")
        text = self._exec(["git", "--no-replace-objects", "show", self.commit + ":" + path],
                          role="source-read").decode(
            "utf-8", errors="replace")
        lines = text.splitlines()
        if start > len(lines):
            raise WorkerError("Source read begins beyond the file")
        end = min(end, len(lines))
        return "\n".join(lines[start-1:end]), Location(self.repo, self.commit, path, start, end)

    def apply_patch(self, patch: str):
        # --index includes newly created files in the submitted diff.
        self._exec(["git", "apply", "--index", "--whitespace=nowarn", "-"],
                   stdin=patch.encode(), role="patch")

    def run(self, command: str, timeout=120) -> dict:
        with self.ledger.action("tool", "validation") as step:
            remaining = self.ledger.allowance()[1]
            grace = min(5,max(0.001,remaining/2))
            seconds = max(0.001,min(timeout,remaining-grace))
            # The deadline is enforced inside the container, so terminating the
            # Docker client cannot leave a test process running in the worker.
            argv = [self.docker, "exec", "--workdir", self.workdir, self.container,
                    "timeout", "--signal=TERM", "--kill-after="+str(grace)+"s", str(seconds)+"s",
                    "sh", "-lc", command]
            try:
                completed = subprocess.run(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=max(0.001, min(seconds+grace, self.ledger.allowance()[1])), check=False)
                step.success = completed.returncode == 0
                return dict(exit_code=completed.returncode,
                            output=completed.stdout.decode("utf-8", errors="replace")[-48000:])
            except subprocess.TimeoutExpired:
                step.success = False
                return dict(exit_code=124, output="Local validation timed out")
            except OSError:
                raise WorkerError("Worker is unavailable") from None

    def diff(self) -> str:
        return self._exec(
            ["sh", "-c", "git add --intent-to-add --all && "
             "git --no-replace-objects diff --binary --no-ext-diff --no-textconv \"$1\" --",
             "sh", self.commit],
            role="submission").decode("utf-8", errors="replace")

    def close(self):
        # Cleanup is outside submission accounting and never touches host files.
        if self.started or getattr(self, "container", ""):
            try:
                subprocess.run([self.docker, "rm", "--force", self.container],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.started = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

