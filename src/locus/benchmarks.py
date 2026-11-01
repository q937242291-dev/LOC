"""Public benchmark preparation and independent patch grading.

Inference tasks are a projection of the public release. Grading metadata is
written separately and is only consumed after patch submission.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import tomllib
import uuid
from pathlib import Path
from typing import Any, Iterable


BENCHMARKS: dict[str, dict[str, Any]] = {
    "lite": {
        "dataset": "princeton-nlp/SWE-bench_Lite", "config": None,
        "revision": "main", "split": "test", "tasks": 300, "deadline": 7200,
        "url": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite",
    },
    "pro-v1": {
        "dataset": "ScaleAI/SWE-bench_Pro", "config": "v1",
        "revision": "v2.0.0", "split": "test", "tasks": 731, "deadline": 7200,
        "url": "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro",
    },
    "pro-v2": {
        "dataset": "ScaleAI/SWE-bench_Pro", "config": "default",
        "revision": "v2.0.0", "split": "test", "tasks": 642, "deadline": 3000,
        "url": "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro",
    },
}


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {number} must be an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def inference_task(row: dict[str, Any], benchmark: str) -> dict[str, Any]:
    """Project public task text without reference patches, tests or fix commits."""
    if benchmark not in BENCHMARKS:
        raise ValueError("benchmark must be lite, pro-v1 or pro-v2")
    required = ("instance_id", "repo", "base_commit", "problem_statement")
    for key in required:
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"public task lacks {key}")
    statement = row["problem_statement"].strip()
    for field, heading in (("requirements", "Requirements"), ("interface", "Interfaces")):
        value = row.get(field)
        if value and isinstance(value, str):
            statement += f"\n\n## {heading}\n{value.strip()}"
    image = row.get("docker_image")
    if not image:
        if benchmark == "pro-v2":
            image = "ghcr.io/scaleapi/swe-bench_pro-v2:" + row["instance_id"]
        elif benchmark == "pro-v1":
            tag = row.get("dockerhub_tag")
            if not tag:
                raise ValueError("Pro V1 task lacks dockerhub_tag")
            image = "docker.io/jefzda/sweap-images:" + str(tag)
        else:
            # The public SWE-bench image naming convention escapes double '_'.
            image_id = row["instance_id"].lower().replace("__", "_1776_")
            image = f"swebench/sweb.eval.x86_64.{image_id}:latest"
    return {
        "instance_id": row["instance_id"], "repo": row["repo"],
        "base_commit": row["base_commit"], "problem_statement": statement,
        "benchmark": benchmark, "docker_image": image,
        "workdir": row.get("workdir") or ("/testbed" if benchmark == "lite" else "/app"),
    }


def fetch_public(benchmark: str, output_dir: Path, *, revision: str | None = None,
                 limit: int | None = None) -> dict[str, Any]:
    """Download a public release when explicitly called by the user.

    ``tasks.jsonl`` is the only input to the repair agent. ``grading.jsonl``
    retains the original public metadata for the independent grading harness.
    """
    spec = BENCHMARKS[benchmark]
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the datasets extra: pip install -e '.[benchmarks]'") from exc
    kwargs: dict[str, Any] = {"split": spec["split"], "revision": revision or spec["revision"]}
    data = load_dataset(spec["dataset"], spec["config"], **kwargs)
    if limit is None and len(data) != spec["tasks"]:
        raise ValueError(f"{benchmark} release has {len(data)} tasks; expected {spec['tasks']}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    rows = [dict(row) for index, row in enumerate(data) if limit is None or index < limit]
    output_dir = Path(output_dir)
    tasks_path = output_dir / "tasks.jsonl"
    grading_path = output_dir / "grading.jsonl"
    write_jsonl(tasks_path, (inference_task(row, benchmark) for row in rows))
    write_jsonl(grading_path, rows)
    return {"benchmark": benchmark, "tasks": str(tasks_path), "grading": str(grading_path),
            "count": len(rows), "revision": kwargs["revision"], "dataset": spec["dataset"]}


def repeat_plan(tasks: Iterable[dict[str, Any]], repeats: int = 5,
                seed: int = 0) -> list[dict[str, Any]]:
    """Pair repeat seeds by task identity, independent of ordering or system."""
    from .run_design import attempt_seed, task_identity
    if type(repeats) is not int or repeats < 1 or type(seed) is not int:
        raise ValueError("repeats must be a positive integer and seed must be an integer")
    rows = list(tasks)
    ids = [task_identity(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("task list contains duplicate benchmark/instance_id")
    return [{**row, "benchmark": benchmark, "repeat": repeat,
             "seed": attempt_seed(seed,benchmark,identity,repeat)}
            for repeat in range(repeats) for row,(benchmark,identity) in zip(rows,ids)]


def _submitted_patch(row: dict[str, Any]) -> str:
    value = row.get("model_patch", row.get("patch", ""))
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("model_patch must be a string or null")
    return value


def export_predictions(records: Iterable[dict[str, Any]], output: Path, *,
                       repeat: int | None = None) -> int:
    """Export one repeat to the official SWE-bench JSONL prediction schema."""
    seen: set[str] = set()
    exported = []
    for row in records:
        if repeat is not None and row.get("repeat", 0) != repeat:
            continue
        key = row["instance_id"]
        if key in seen:
            raise ValueError("Export each repeat separately; duplicate instance_id")
        seen.add(key)
        exported.append({"instance_id": key,
                         "model_name_or_path": row.get("model", row.get("model_name_or_path", "LOCUS")),
                         "model_patch": _submitted_patch(row)})
    return write_jsonl(output, exported)


def lite_grade_command(predictions: Path, run_id: str, workers: int = 1,
                       python: str = "python") -> list[str]:
    return [python, "-m", "swebench.harness.run_evaluation", "--dataset_name",
            BENCHMARKS["lite"]["dataset"], "--split", "test", "--predictions_path",
            str(Path(predictions).resolve()), "--max_workers", str(workers), "--run_id", run_id]


def export_pro_v1_patches(records: Iterable[dict[str, Any]], output: Path, *,
                          repeat: int | None = None) -> int:
    """The Pro V1 harness reads a JSON array, rather than JSONL."""
    rows, seen = [], set()
    for row in records:
        if repeat is not None and row.get("repeat", 0) != repeat:
            continue
        identity = str(row["instance_id"])
        if identity in seen:
            raise ValueError("Export each repeat separately; duplicate instance_id")
        seen.add(identity)
        rows.append({"instance_id": identity, "model_patch": _submitted_patch(row),
                     "prefix": f"repeat-{row.get('repeat', 0)}"})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return len(rows)


def pro_v1_grade_command(harness: Path, grading: Path, patches: Path, output: Path,
                         workers: int = 1, python: str = "python") -> list[str]:
    harness = Path(harness).resolve()
    return [python, str(harness / "swe_bench_pro_eval.py"),
            "--raw_sample_path", str(Path(grading).resolve()),
            "--patch_path", str(Path(patches).resolve()), "--output_dir", str(Path(output).resolve()),
            "--dockerhub_username", "jefzda", "--scripts_dir", str(harness / "run_scripts"),
            "--use_local_docker", "--num_workers", str(workers)]


def merge_grades(records: Iterable[dict[str, Any]], grades: Iterable[dict[str, Any]], *,
                 repeat: int | None = None) -> list[dict[str, Any]]:
    """Join one method/backbone/release at a time; missing grades remain unknown."""
    rows = list(records)

    def repeat_number(value):
        if type(value) is not int or value < 0:
            raise ValueError("repeat must be a nonnegative integer")
        return value

    def identity(row):
        value = row.get("instance_id")
        if not isinstance(value,str) or not value.strip():
            raise ValueError("Grading requires a nonempty instance_id")
        return value

    if repeat is not None:
        repeat_number(repeat)
    fields = ("benchmark","model","system")
    groups = set()
    seen_attempts = set()
    for row in rows:
        group = tuple(row.get(field) for field in fields)
        if any(value is not None and (not isinstance(value,str) or not value.strip()) for value in group):
            raise ValueError("Invalid benchmark/model/system grading scope")
        groups.add(group)
        key = identity(row),repeat_number(row.get("repeat",0))
        if key in seen_attempts:
            raise ValueError("Duplicate attempt; grade each method and repeat independently")
        seen_attempts.add(key)
    if len(groups) > 1:
        raise ValueError("Grade one benchmark, model and system at a time")
    group = next(iter(groups),None)
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for grade in grades:
        key = (identity(grade),repeat_number(grade.get("repeat",repeat if repeat is not None else 0)))
        if key in by_key:
            raise ValueError("duplicate grading result")
        if grade.get("resolved") is not None and type(grade["resolved"]) is not bool:
            raise ValueError("resolved must be boolean or null")
        if group is not None and any(field in grade and expected is not None and grade[field] != expected
                                     for field,expected in zip(fields,group)):
            raise ValueError("Grade metadata does not match the attempted benchmark/model/system")
        by_key[key] = grade
    result = []
    for record in rows:
        row = dict(record)
        grade = by_key.get((row["instance_id"],row.get("repeat",0)))
        if grade is not None:
            value = grade.get("resolved")
            if value is not None and type(value) is not bool:
                raise ValueError("resolved must be boolean or null")
            row["resolved"] = value
            if grade.get("error"):
                row["grading_error"] = grade["error"]
        result.append(row)
    return result


def read_lite_grades(root: Path, repeat: int = 0) -> list[dict[str, Any]]:
    """Read official per-instance reports or one run's aggregate JSON report.

    Empty submissions are absent from per-instance reports and appear in the
    official aggregate's ``empty_patch_ids``. Missing or errored verification
    remains unknown until a completed report is supplied.
    """
    root = Path(root)
    paths = ([root] if root.is_file() else sorted(
        set(root.rglob("report.json")) | set(root.rglob("results.json")) | set(root.glob("*.json"))))
    by_id: dict[str, dict[str, Any]] = {}

    def admit(identity: str, value: bool | None, error: str | None = None):
        previous = by_id.get(identity)
        if previous is not None and previous["resolved"] is not None:
            if value is not None and previous["resolved"] != value:
                raise ValueError("Conflicting official reports; select one model and run")
            return
        row = {"instance_id": identity, "repeat": repeat, "resolved": value}
        if error:
            row["error"] = error
        by_id[identity] = row

    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(report, dict):
            continue
        for key, value in report.items():
            if isinstance(value, dict) and type(value.get("resolved")) is bool:
                admit(str(key), value["resolved"])
        for field, outcome, error in (
            ("resolved_ids", True, None), ("unresolved_ids", False, None),
            ("empty_patch_ids", False, None), ("error_ids", None, "official_harness_error"),
            ("incomplete_ids", None, "official_harness_incomplete"),
        ):
            values = report.get(field, [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError("Invalid official run-report instance list")
            for identity in values:
                admit(identity, outcome, error)
    return [by_id[key] for key in sorted(by_id)]


def read_pro_v1_grades(path: Path, repeat: int = 0) -> list[dict[str, Any]]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{"instance_id": key, "repeat": repeat, "resolved": value}
            for key, value in report.items() if type(value) is bool]


def _docker(argv: list[str], timeout: float, *, docker: str = "docker",
            input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([docker, *argv], input=input_text, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=max(1, timeout), check=False)


def grade_pro_v2(task_dir: Path, prediction: dict[str, Any], *, timeout: int = 3000,
                 docker: str = "docker", network: str = "bridge") -> dict[str, Any]:
    """Replay one submitted diff in a fresh V2 image and run its official verifier.

    Only ``tests/`` from the official task directory and the submitted patch are
    copied into this new container. The repair container is never reused.
    The verifier phase may access the network as specified by the V2 protocol.
    """
    task_dir = Path(task_dir).resolve()
    identity = str(prediction["instance_id"])
    if task_dir.name != identity or not re.fullmatch(r"[A-Za-z0-9_.-]+", identity):
        raise ValueError("task directory must match instance_id")
    with (task_dir / "task.toml").open("rb") as stream:
        task = tomllib.load(stream)
    env = task.get("environment", {})
    image = env.get("docker_image")
    if not image:
        raise ValueError("official task.toml has no environment.docker_image")
    if not (task_dir / "tests" / "test.sh").is_file():
        raise ValueError("official task tests/test.sh is missing")
    verifier = task.get("verifier", {})
    limit = min(float(timeout), float(verifier.get("timeout_sec", timeout)))
    cpus = str(env.get("cpus", 8))
    memory = str(int(env.get("memory_mb", 32768))) + "m"
    name = "locus-grade-" + uuid.uuid4().hex[:16]
    result: dict[str, Any] = {"instance_id": identity, "repeat": prediction.get("repeat", 0),
                              "resolved": None, "error": None}
    try:
        with tempfile.TemporaryDirectory(prefix="locus-grade-") as temporary:
            patch_path = Path(temporary) / "model.patch"
            patch = _submitted_patch(prediction)
            patch_path.write_text(patch, encoding="utf-8", newline="\n")
            created = _docker(["run", "-d", "--name", name, "--platform", "linux/amd64",
                               "--network", network, "--cpus", cpus, "--memory", memory,
                               "--pids-limit", "2048", "--security-opt", "no-new-privileges",
                               "--entrypoint", "/bin/sh", image, "-c", "while :; do sleep 60; done"],
                              max(300, limit), docker=docker)
            if created.returncode:
                result["error"] = "container_start_failed"
                return result
            for source, target in ((task_dir / "tests", "/tests"), (patch_path, "/tmp/model.patch")):
                copied = _docker(["cp", str(source), f"{name}:{target}"], 300, docker=docker)
                if copied.returncode:
                    result["error"] = "verifier_copy_failed"
                    return result
            if patch.strip():
                # Same three patch-application modes accepted by official patch replay.
                apply = ("if [ -d /app/.git ]; then cd /app; "
                         "elif [ -d /testbed/.git ]; then cd /testbed; else exit 1; fi; ")
                apply += "git apply /tmp/model.patch || git apply --3way /tmp/model.patch || "
                apply += "patch -p1 --fuzz=3 -i /tmp/model.patch"
                applied = _docker(["exec", name, "bash", "-lc", apply],
                                  300, docker=docker)
                if applied.returncode:
                    result["error"] = "patch_application_incomplete"
            _docker(["exec", name, "bash", "-lc", "mkdir -p /logs/verifier; bash /tests/test.sh"],
                    limit, docker=docker)
            reward = _docker(["exec", name, "cat", "/logs/verifier/reward.txt"], 30, docker=docker)
            value = reward.stdout.strip()
            if reward.returncode == 0 and value in {"0", "0.0", "1", "1.0"}:
                result["resolved"] = float(value) == 1.0
            else:
                result["error"] = "verifier_reward_missing"
    except subprocess.TimeoutExpired:
        result["error"] = "grading_timeout"
    except OSError:
        result["error"] = "docker_unavailable"
    finally:
        try:
            _docker(["rm", "-f", name], 30, docker=docker)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return result


def export_harbor_replay(records: Iterable[dict[str, Any]], output: Path, *, repeat: int) -> int:
    """Export patches in the source-job layout used by official PatchReplayAgent."""
    count = 0
    seen: set[str] = set()
    for row in records:
        if row.get("repeat", 0) != repeat:
            continue
        identity = str(row["instance_id"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", identity) or identity in seen:
            raise ValueError("invalid or duplicate instance_id")
        seen.add(identity)
        trial = Path(output) / (identity if identity.startswith("instance_") else "instance_" + identity)
        (trial / "agent").mkdir(parents=True, exist_ok=True)
        (trial / "agent" / "model.patch").write_text(_submitted_patch(row), encoding="utf-8")
        (trial / "result.json").write_text(json.dumps({"task_name": "swebench-pro/" + identity}), encoding="utf-8")
        count += 1
    return count


def pro_v2_replay_command(tasks: Path, source_job: Path, job_name: str,
                          workers: int = 1) -> list[str]:
    """Run from the official harness with PYTHONPATH=v2/tooling."""
    return ["harbor", "run", "-p", str(tasks), "-e", "modal", "-n", str(workers),
            "-a", "patch_replay:PatchReplayAgent", "-m", "replay", "--ak",
            "source_job=" + str(Path(source_job).resolve()), "--job-name", job_name]
