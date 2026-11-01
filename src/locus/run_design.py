"""Paired run identities and metadata checks, separate from outcome statistics."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
import uuid

from .config import Settings

SCHEMA_VERSION = 1
SEED_POLICY = {"version": "sha256-attempt-v1", "bits": 31,
               "key_fields": ["root_seed", "benchmark", "instance_id", "repeat"]}
FACTOR_FIELDS = ("packet_tokens", "frontier_size", "gate_enabled", "stability_window",
                 "allocation", "selection", "role_grounding", "alternatives", "memory", "recovery")
OUTER_FIELDS = ("total_calls", "request_input_tokens", "output_tokens")
SAMPLING_FIELDS = ("temperature", "top_p")
DESIGN_FIELDS = ("schema_version", "seed_policy", "root_seed", "repeats", "roster",
                 "model", "sampling_requested", "outer_budget", "runtime_controls",
                 "reference", "code_version")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def task_identity(task: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(task, Mapping):
        raise ValueError("Task must be an object")
    values = task.get("benchmark", "custom"), task.get("instance_id")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Task requires nonempty benchmark and instance_id strings")
    return values


def attempt_seed(root_seed: int, benchmark: str, instance_id: str, repeat: int) -> int:
    """Derive a logical seed from the root seed and exact task/repeat identity."""
    _integer(root_seed, "root_seed")
    _integer(repeat, "repeat", minimum=0)
    task_identity(dict(benchmark=benchmark, instance_id=instance_id))
    digest = hashlib.sha256(_json([SEED_POLICY["version"], root_seed, benchmark,
                                  instance_id, repeat]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << SEED_POLICY["bits"]) - 1)


def _settings(value: Any) -> dict[str, Any]:
    raw = asdict(value) if is_dataclass(value) else dict(value)
    normalized = asdict(Settings(**raw).validate())
    for field in fields(Settings):
        if field.type in (float,"float"):
            normalized[field.name] = float(normalized[field.name])
    return normalized


def _public_model(value: Any) -> dict[str, Any]:
    raw = asdict(value) if is_dataclass(value) else dict(value)
    provider, model = raw.get("provider"), raw.get("model")
    if any(not isinstance(x, str) or not x.strip() for x in (provider, model)):
        raise ValueError("Model metadata requires provider and model identifiers")
    parsed = urlsplit(raw.get("base_url", ""))
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Model metadata requires an endpoint without credentials or query parameters")
    timeout = raw.get("timeout_seconds")
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Model metadata requires its finite positive request timeout")
    # Store endpoint metadata and a fingerprint of generation options.
    return dict(provider=provider, model=model,
                endpoint=urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", "")),
                timeout_seconds=float(timeout),
                extra_body_sha256=_digest(raw.get("extra_body", {})))


def _sampling(model: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
    raw = asdict(model) if is_dataclass(model) else dict(model)
    mode = raw.get("seed_mode", "off")
    if mode not in {"off", "request"}:
        raise ValueError("seed_mode must be off or request")
    return {**{key: settings[key] for key in SAMPLING_FIELDS},
            "provider_seed_mode": mode,
            "provider_request_seed_policy": "locus-request-seed-v1" if mode == "request" else None}


def _runtime(settings: Mapping[str, Any]) -> dict[str, Any]:
    excluded = set(FACTOR_FIELDS + OUTER_FIELDS + SAMPLING_FIELDS + ("deadline_seconds", "seed"))
    return {key: value for key, value in settings.items() if key not in excluded}


def _code_version() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return "sha256:" + digest.hexdigest()


def _attempts(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(benchmark=row["benchmark"], instance_id=row["instance_id"], repeat=repeat,
                 seed=attempt_seed(manifest["root_seed"], row["benchmark"], row["instance_id"], repeat))
            for repeat in range(manifest["repeats"]) for row in manifest["roster"]]


def build_run_manifest(tasks: Iterable[Mapping[str, Any]], settings: Any, model: Any, *,
                       repeats: int = 5, root_seed: int | None = None,
                       batch_id: str | None = None, reference_settings: Any = None,
                       deadline_by_benchmark: Mapping[str, float] | None = None,
                       code_version: str | None = None) -> dict[str, Any]:
    """Build requested controls from actual inputs; factors do not change design_id.

    Pass the same batch_id to compared runs. An omitted batch_id identifies a new
    execution batch, while design_id is a deterministic controlled-design hash.
    Logical seeds control local randomness and pair matching.
    """
    actual = _settings(settings)
    baseline = _settings(reference_settings if reference_settings is not None else Settings())
    _integer(repeats, "repeats", minimum=1)
    root_seed = _integer(actual["seed"] if root_seed is None else root_seed, "root_seed")
    if batch_id is not None and (not isinstance(batch_id, str) or not batch_id.strip()):
        raise ValueError("batch_id must be a nonempty string")
    roster, seen = [], set()
    for task in tasks:
        key = task_identity(task)
        if key in seen:
            raise ValueError("Task roster contains duplicate benchmark/instance_id")
        seen.add(key)
        entry = dict(benchmark=key[0], instance_id=key[1])
        for field in ("repo", "base_commit", "dataset", "dataset_revision", "dataset_config",
                      "dataset_split", "docker_image"):
            value = task.get(field, task.get("revision") if field == "dataset_revision" else None)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Task roster {field} must be a nonempty string or null")
            entry[field] = value
        statement = task.get("problem_statement")
        if statement is not None and not isinstance(statement, str):
            raise ValueError("Task problem_statement must be text")
        entry["issue_sha256"] = hashlib.sha256(statement.encode("utf-8")).hexdigest() \
            if statement is not None else None
        roster.append(entry)
    if not roster:
        raise ValueError("Task roster must not be empty")
    roster.sort(key=task_identity)
    deadlines = {}
    for benchmark, _ in sorted(seen):
        value = (deadline_by_benchmark or {}).get(benchmark, actual["deadline_seconds"])
        # Reuse the runtime's finite, positive deadline validation.
        deadlines[benchmark] = _settings({**actual, "deadline_seconds": value})["deadline_seconds"]
    if code_version is not None and (not isinstance(code_version, str) or not code_version.strip()):
        raise ValueError("code_version must be a nonempty string")
    manifest = dict(schema_version=SCHEMA_VERSION, seed_policy=deepcopy(SEED_POLICY), root_seed=root_seed,
                    repeats=repeats, roster=roster, roster_sha256=_digest(roster),
                    model=_public_model(model), sampling_requested=_sampling(model, actual),
                    outer_budget={**{key: actual[key] for key in OUTER_FIELDS},
                                  "deadline_by_benchmark": deadlines},
                    runtime_controls=_runtime(actual),
                    factors={key: actual[key] for key in FACTOR_FIELDS},
                    reference={key: baseline[key] for key in ("packet_tokens", "frontier_size")},
                    code_version=code_version or _code_version(),
                    batch_id=batch_id or "locus-batch-" + uuid.uuid4().hex)
    manifest["design_id"] = "locus-design-" + _digest({key: manifest[key] for key in DESIGN_FIELDS})
    manifest["condition_id"] = "locus-condition-" + _digest(manifest["factors"])
    manifest["attempts"] = _attempts(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject a changed roster, seed plan or controlled-design hash on resume."""
    try:
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION \
                or manifest["seed_policy"] != SEED_POLICY:
            raise ValueError("Unsupported run manifest or seed policy")
        _integer(manifest["root_seed"], "root_seed")
        _integer(manifest["repeats"], "repeats", minimum=1)
        if not isinstance(manifest["batch_id"],str) or not manifest["batch_id"].strip():
            raise ValueError("Run manifest lacks a batch identity")
        roster = manifest["roster"]
        if not isinstance(roster,list) or not roster:
            raise ValueError("Run manifest lacks its task roster")
        keys = [task_identity(row) for row in roster]
        if keys != sorted(set(keys)) or _digest(roster) != manifest["roster_sha256"]:
            raise ValueError("Run manifest roster is changed, duplicated or unordered")
        expected = "locus-design-" + _digest({key: manifest[key] for key in DESIGN_FIELDS})
        if expected != manifest["design_id"]:
            raise ValueError("Run manifest design_id does not match its controls")
        if manifest["condition_id"] != "locus-condition-" + _digest(manifest["factors"]):
            raise ValueError("Run manifest condition_id does not match its factors")
        for attempt in manifest["attempts"]:
            _integer(attempt["repeat"],"repeat",minimum=0)
            _integer(attempt["seed"],"seed",minimum=0)
        if manifest["attempts"] != _attempts(manifest):
            raise ValueError("Run manifest attempt seed plan is changed")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Malformed run manifest") from exc


def attempt_design(manifest: Mapping[str, Any], task: Mapping[str, Any],
                   settings: Any = None) -> dict[str, Any]:
    """Compact metadata from a built/validated manifest; validate once on resume."""
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("seed_policy") != SEED_POLICY:
        raise ValueError("Unsupported run manifest or seed policy")
    benchmark, identity = task_identity(task)
    repeat = _integer(task.get("repeat"), "repeat", minimum=0)
    if repeat >= manifest["repeats"]:
        raise ValueError("Attempt repeat is outside the manifest plan")
    entry = next((row for row in manifest["roster"] if task_identity(row) == (benchmark,identity)), None)
    if entry is None:
        raise ValueError("Attempt task is outside the manifest roster")
    for field in ("repo","base_commit","dataset","dataset_revision","dataset_config",
                  "dataset_split","docker_image"):
        value = task.get(field,task.get("revision") if field == "dataset_revision" else None)
        if (field in task or field == "dataset_revision" and "revision" in task) and value != entry[field]:
            raise ValueError("Actual task metadata differs from the manifest roster")
    if "problem_statement" in task and (not isinstance(task["problem_statement"],str) or
            hashlib.sha256(task["problem_statement"].encode("utf-8")).hexdigest() != entry["issue_sha256"]):
        raise ValueError("Actual issue text differs from the manifest roster")
    seed = attempt_seed(manifest["root_seed"],benchmark,identity,repeat)
    if "seed" in task and (type(task["seed"]) is not int or task["seed"] != seed):
        raise ValueError("Attempt seed does not match its planned identity")
    outer = {key: manifest["outer_budget"][key] for key in OUTER_FIELDS}
    outer["deadline_seconds"] = manifest["outer_budget"]["deadline_by_benchmark"][benchmark]
    factors = dict(manifest["factors"])
    if settings is not None:
        actual = _settings(settings)
        if actual["seed"] != seed or any(actual[key] != value for key,value in outer.items()) \
                or _runtime(actual) != manifest["runtime_controls"] \
                or any(actual[key] != manifest["sampling_requested"][key] for key in SAMPLING_FIELDS):
            raise ValueError("Actual attempt settings differ from the manifest controls")
        factors = {key: actual[key] for key in FACTOR_FIELDS}
    sampling = dict(manifest["sampling_requested"])
    sampling["provider_seed_root"] = seed if sampling["provider_seed_mode"] == "request" else None
    return dict(schema_version=SCHEMA_VERSION, design_id=manifest["design_id"], batch_id=manifest["batch_id"],
                condition_id="locus-condition-" + _digest(factors),
                roster_sha256=manifest["roster_sha256"], code_version=manifest["code_version"],
                seed_policy=deepcopy(manifest["seed_policy"]), root_seed=manifest["root_seed"],
                task=dict(entry), repeat=repeat, logical_seed=seed, model=dict(manifest["model"]),
                sampling_requested=sampling, outer_budget=outer,
                runtime_controls=dict(manifest["runtime_controls"]), reference=dict(manifest["reference"]),
                factors=factors)


def _request_observations(left, right, lhs, rhs):
    """Compare transmitted sampling controls, API seeds and provider identities."""
    evidence, indexed = [], []

    def record(kind, side, field, request=None, full=None, ablation=None):
        evidence.append(dict(kind=kind,side=side,field=field,
                             request=list(request) if request else None,
                             full=full,ablation=ablation))

    for side,row,metadata in (("full",left,lhs),("ablation",right,rhs)):
        runtime, requests_by_key = row.get("model_runtime"), {}
        indexed.append(requests_by_key)
        if not isinstance(runtime,Mapping):
            record("unknown",side,"model_runtime")
            continue
        try:
            sampling = metadata["sampling_requested"]
            expected = dict(provider=metadata["model"]["provider"],
                            configured_model=metadata["model"]["model"],
                            seed_mode=sampling["provider_seed_mode"],local_seed=metadata["logical_seed"])
            for field,value in expected.items():
                if field not in runtime:
                    record("unknown",side,field)
                elif runtime[field] != value:
                    record("mismatch",side,field)
            requests = runtime.get("requests")
            if not isinstance(requests,list):
                record("unknown",side,"requests")
                continue
            for request in requests:
                role,ordinal = request["role"],request["ordinal"]
                if not isinstance(role,str) or type(ordinal) is not int or ordinal < 0:
                    record("mismatch",side,"request_identity")
                    continue
                key = role,ordinal
                if key in requests_by_key:
                    record("mismatch",side,"duplicate_request_identity",key)
                requests_by_key[key] = request
                seed = None
                if sampling["provider_seed_mode"] == "request":
                    encoded = _json(["locus-request-seed-v1",metadata["logical_seed"],role,ordinal])
                    seed = int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:4],"big") & 0x7fffffff
                required = dict(requested_model=metadata["model"]["model"],
                                requested_temperature=sampling["temperature"],
                                requested_top_p=sampling["top_p"],seed_mode=sampling["provider_seed_mode"],
                                requested_seed=seed)
                for field,value in required.items():
                    if field not in request:
                        record("unknown",side,field,key)
                    elif request[field] != value:
                        record("mismatch",side,field,key)
        except (KeyError,TypeError,ValueError):
            record("unknown",side,"incomplete_request_metadata")
    common = set(indexed[0]) & set(indexed[1])
    if not common:
        record("unknown","pair","no_common_model_requests")
    for key in sorted(common):
        a,b = indexed[0][key],indexed[1][key]
        for field in ("sent_temperature","sent_top_p","served_model","system_fingerprint"):
            if field not in a or field not in b or field in {"served_model","system_fingerprint"} \
                    and (a[field] is None or b[field] is None):
                record("unknown","pair",field,key)
            elif a[field] != b[field]:
                record("mismatch","pair",field,key,a[field],b[field])
    status = "mismatch" if any(r["kind"] == "mismatch" for r in evidence) else \
        "unknown" if evidence else "matched"
    return dict(status=status,common_requests=len(common),evidence=evidence)


def audit_run_design(full_rows: Iterable[Mapping[str, Any]],
                     ablation_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare batch identity, task seeds, model configuration and outer budgets.

    Return the matching status, parameter differences and request observations.
    Experimental factors and actual resource consumption are reported separately.
    """
    evidence, factor_differences, groups, observations = [], [], [], []

    def issue(kind, field, key, left=None, right=None):
        row = dict(kind=kind,field=field,pair=list(key) if key else None)
        if left is not None:
            row["full"] = left
        if right is not None:
            row["ablation"] = right
        evidence.append(row)

    for name, rows in (("full",full_rows),("ablation",ablation_rows)):
        group, first = {}, None
        for row in rows:
            try:
                key = (*task_identity(row),_integer(row.get("repeat"),"repeat",minimum=0))
            except (TypeError,ValueError):
                issue("mismatch",name+".invalid_identity",None)
                continue
            if key in group:
                issue("mismatch",name+".duplicate_pair",key)
            else:
                group[key] = row
            metadata = row.get("run_design")
            if isinstance(metadata,Mapping):
                signature = {field:metadata.get(field) for field in
                             ("design_id","batch_id","roster_sha256","code_version","model",
                              "seed_policy","root_seed","runtime_controls","reference")}
                sampling = metadata.get("sampling_requested")
                signature["sampling_requested"] = {k:v for k,v in sampling.items()
                    if k != "provider_seed_root"} if isinstance(sampling,Mapping) else None
                if first is None:
                    first = signature
                elif signature != first:
                    issue("mismatch",name+".mixed_run_designs",key)
        groups.append(group)
    left_group,right_group = groups
    keys = sorted(set(left_group) | set(right_group))
    if not keys:
        issue("unknown","empty_comparison",None)
    common = ("schema_version","design_id","batch_id","roster_sha256","code_version",
              "seed_policy","root_seed","task","repeat","logical_seed","model",
              "sampling_requested","outer_budget","runtime_controls","reference")
    matched = 0
    for key in keys:
        if key not in left_group or key not in right_group:
            issue("mismatch","missing_pair",key,key in left_group,key in right_group)
            continue
        start = len(evidence)
        a,b = left_group[key],right_group[key]
        lhs,rhs = a.get("run_design"),b.get("run_design")
        if not isinstance(lhs,Mapping) or not isinstance(rhs,Mapping):
            issue("unknown","run_design",key)
            continue
        for field in common:
            if field not in lhs or field not in rhs:
                issue("unknown",field,key)
            elif lhs[field] != rhs[field]:
                issue("mismatch",field,key,lhs[field],rhs[field])
        for side, row, metadata in (("full",a,lhs),("ablation",b,rhs)):
            try:
                if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("seed_policy") != SEED_POLICY:
                    issue("unknown",side+".unsupported_schema_or_seed_policy",key)
                    continue
                expected = attempt_seed(metadata["root_seed"],key[0],key[1],key[2])
                if metadata["logical_seed"] != expected or metadata.get("repeat") != key[2] \
                        or task_identity(metadata["task"]) != key[:2]:
                    issue("mismatch",side+".attempt_identity",key)
                if type(metadata["logical_seed"]) is not int or type(metadata["repeat"]) is not int:
                    issue("mismatch",side+".invalid_seed_or_repeat_type",key)
                for field in ("repo","base_commit","issue_sha256"):
                    if not metadata["task"].get(field):
                        issue("unknown",side+".task."+field,key)
                sampling = metadata["sampling_requested"]
                if sampling.get("provider_seed_mode") not in {"off","request"}:
                    issue("unknown",side+".unsupported_provider_seed_mode",key)
                requested_seed = expected if sampling["provider_seed_mode"] == "request" else None
                policy = "locus-request-seed-v1" if sampling["provider_seed_mode"] == "request" else None
                if sampling.get("provider_seed_root") != requested_seed \
                        or sampling.get("provider_request_seed_policy") != policy:
                    issue("mismatch",side+".provider_seed_request",key)
                actual_settings = row.get("settings",{})
                actual_seed = row.get("seed",actual_settings.get("seed",expected))
                if type(actual_seed) is not int or actual_seed != expected:
                    issue("mismatch",side+".actual_seed",key)
                if row.get("model",metadata["model"]["model"]) != metadata["model"]["model"]:
                    issue("mismatch",side+".actual_model",key)
                for field,value in metadata["outer_budget"].items():
                    if field in actual_settings and actual_settings[field] != value:
                        issue("mismatch",side+".actual_outer_budget."+field,key)
                for field in SAMPLING_FIELDS:
                    if field in actual_settings and actual_settings[field] != metadata["sampling_requested"][field]:
                        issue("mismatch",side+".actual_sampling."+field,key)
                if "condition_id" not in metadata:
                    issue("unknown",side+".condition_id",key)
                elif metadata["condition_id"] != "locus-condition-" + _digest(metadata["factors"]):
                    issue("mismatch",side+".condition_id",key)
                for field in FACTOR_FIELDS:
                    if field in actual_settings and actual_settings[field] != metadata["factors"].get(field):
                        issue("mismatch",side+".actual_factor."+field,key)
            except (KeyError,TypeError,ValueError,AttributeError):
                issue("unknown",side+".incomplete_attempt_metadata",key)
        left_factors,right_factors = lhs.get("factors"),rhs.get("factors")
        if isinstance(left_factors,Mapping) and isinstance(right_factors,Mapping):
            changes = {field:dict(full=left_factors.get(field),ablation=right_factors.get(field))
                       for field in sorted(set(left_factors)|set(right_factors))
                       if left_factors.get(field) != right_factors.get(field)}
            if changes:
                factor_differences.append(dict(pair=list(key),changes=changes))
        else:
            issue("unknown","factors",key)
        observed = _request_observations(a,b,lhs,rhs)
        observations.append(dict(pair=list(key),**observed))
        for difference in observed["evidence"]:
            if difference["kind"] == "mismatch":
                issue("mismatch","model_runtime."+difference["side"]+"."+difference["field"],key,
                      difference["full"],difference["ablation"])
        matched += len(evidence) == start
    status = "mismatch" if any(row["kind"] == "mismatch" for row in evidence) else \
        "unknown" if evidence else "matched"
    return dict(status=status,pairs_compared=len(set(left_group)&set(right_group)),
                metadata_matched_pairs=matched,evidence=evidence,factor_differences=factor_differences,
                request_observations=observations,
                checks_outcomes=False,compares_actual_spending=False)
