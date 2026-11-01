"""Launch upstream systems and import their patches into one evaluation schema."""

from __future__ import annotations

import json
from copy import deepcopy
import os
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping


UPSTREAM = {
    "swe-agent": "https://github.com/SWE-agent/SWE-agent",
    "agentless": "https://github.com/OpenAutoCoder/Agentless",
    "openhands": "https://github.com/OpenHands/OpenHands",
    "live-swe-agent": "https://github.com/OpenAutoCoder/live-swe-agent",
    "cca": "https://github.com/facebookresearch/cca-swebench",
    "claude-code": "https://code.claude.com/docs/en/overview",
    "codex": "https://developers.openai.com/codex/cli",
    "mini-swe-agent": "https://github.com/SWE-agent/mini-swe-agent",
}


def load_predictions(path: Path | str) -> list[dict[str, Any]]:
    """Accept upstream JSONL, a JSON array, or an instance-keyed JSON object."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        if "instance_id" in data:
            return [data]
        rows = []
        for identity, value in data.items():
            if isinstance(value, dict):
                rows.append({**value, "instance_id": identity})
            elif value is None or isinstance(value, str):
                rows.append({"instance_id": identity, "model_patch": value})
            else:
                raise ValueError("prediction object values must be objects, patch strings or null")
        return rows
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError("predictions must contain JSON objects")
    return data


def import_predictions(predictions: Iterable[dict[str, Any]], tasks: Iterable[dict[str, Any]], *,
                       system: str, model: str, repeat: int = 0,
                       telemetry: Mapping[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Keep every task in the denominator, including missing/empty patches.

    Resource fields come only from supplied complete-task telemetry; absent
    telemetry stays null. Patch success is populated by independent grading.
    """
    if type(repeat) is not int or repeat < 0:
        raise ValueError("repeat must be a nonnegative integer")
    tasks = list(tasks)
    task_ids = [str(task["instance_id"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task list contains duplicate instance_id")
    by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        identity = str(row["instance_id"])
        if identity in by_id:
            raise ValueError("import each repeat separately; duplicate instance_id")
        by_id[identity] = row
    result = []
    for task in tasks:
        identity = str(task["instance_id"])
        prediction = by_id.get(identity, {})
        patch = prediction.get("model_patch", prediction.get("patch", ""))
        if patch is None:
            patch = ""
        if not isinstance(patch, str):
            raise ValueError("model_patch must be a string")
        measured = (telemetry or {}).get(identity, {})
        if not isinstance(measured, Mapping):
            raise ValueError("Attempt telemetry must be an object")
        if "repeat" in measured and type(measured["repeat"]) is not int:
            raise ValueError("Telemetry repeat must be an integer")
        for field, expected in (("repeat", repeat), ("system", system), ("model", model),
                                ("benchmark", task.get("benchmark"))):
            if field in measured and measured[field] != expected:
                raise ValueError("Telemetry identity does not match the imported attempt")
        result.append({
            "instance_id": identity, "benchmark": task.get("benchmark"),
            "repeat": repeat, "system": system, "model": model, "model_patch": patch,
            "resolved": None, "status": "submitted" if patch else "empty_patch",
            "file_predictions": prediction.get("file_predictions"),
            "ledger": measured.get("ledger"), "evidence_metrics": measured.get("evidence_metrics"),
        })
        # Preserve actual upstream controls when supplied; patch files alone do
        # not establish a batch identity or sampling parameters.
        for field in ("seed", "settings", "run_design", "model_runtime"):
            if field in measured:
                value = measured[field]
                if field == "seed" and type(value) is not int:
                    raise ValueError("Telemetry seed must be an integer")
                if field != "seed" and not isinstance(value, Mapping):
                    raise ValueError("Telemetry control metadata must be an object")
                result[-1][field] = deepcopy(value)
    unknown = set(by_id) - {str(task["instance_id"]) for task in tasks}
    if unknown:
        raise ValueError("predictions contain tasks outside the selected benchmark")
    return result


def render_argv(template: Iterable[str], context: Mapping[str, Any]) -> list[str]:
    """Substitute individual argument strings without constructing a shell command."""
    result = []
    for argument in template:
        if not isinstance(argument, str):
            raise ValueError("each command argument must be a string")
        result.append(argument.format_map({key: str(value) for key, value in context.items()}))
    if not result:
        raise ValueError("configure an argv command for the selected upstream system")
    return result


def run_external(config_path: Path, system: str, context: Mapping[str, Any], *,
                 cwd: Path | None = None, timeout: int | None = None,
                 env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run an upstream argv pipeline. Credentials are inherited through the environment.

    Output streams are not copied into the project. The caller supplies an
    upstream output directory through ``context`` and imports its prediction file.
    """
    with Path(config_path).open("rb") as stream:
        config = tomllib.load(stream)["systems"][system]
    supported = config.get("benchmarks")
    if supported and context.get("benchmark", "lite") not in supported:
        raise ValueError("use the selected release's upstream benchmark runner")
    commands = config.get("commands")
    if commands is None:
        if not config.get("argv"):
            raise ValueError(f"{system}: run the official entry at {config.get('usage', UPSTREAM.get(system, ''))}; "
                             "then import the submitted patches")
        commands = [config["argv"]]
    if not isinstance(commands, list) or not commands:
        raise ValueError("commands must contain at least one argv list")
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    started = time.monotonic()
    for command in commands:
        argv = render_argv(command, context)
        try:
            completed = subprocess.run(argv, cwd=cwd, env=child_env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{system}: upstream process exceeded its time limit") from None
        except OSError:
            raise RuntimeError(f"{system}: upstream executable is unavailable") from None
        if completed.returncode:
            raise RuntimeError(f"{system}: upstream process exited with code {completed.returncode}")
    return {"system": system, "launcher_elapsed_seconds": time.monotonic() - started,
            "prediction_file": context.get("predictions"), "ledger": None}
