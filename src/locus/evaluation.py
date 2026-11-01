"""Complete-attempt repair, evidence and resource metrics (paper Eqs. 15–16).

Counts are pooled before ratios are calculated. Grades use true for success,
false for failure and ``None`` for pending evaluation.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from statistics import fmean
import re
from typing import Any


EXPLORATION_FIELDS = (
    "exploration_calls", "exploration_model_calls", "exploration_tool_calls",
    "exploration_llm_seconds", "exploration_tool_seconds",
)


def unit_id(unit: Any) -> str:
    """Canonical exact unit identity; mappings can use id or path/start/end.

    Unit strings use ``path:start:end`` (or an application-defined exact ID).
    Line interval coordinates are inclusive. A caller may choose line-sized,
    symbol-sized or annotated span-sized units, consistently on both sides.
    """
    if isinstance(unit, str):
        return unit.replace("\\", "/")
    if isinstance(unit, Mapping):
        if "id" in unit:
            return str(unit["id"]).replace("\\", "/")
        path = str(unit["path"]).replace("\\", "/")
        start = unit.get("start", unit.get("start_line"))
        end = unit.get("end", unit.get("end_line", start))
        return f"{path}:{start}:{end}"
    if isinstance(unit, (tuple, list)) and len(unit) == 3:
        return f"{unit[0]}:{unit[1]}:{unit[2]}".replace("\\", "/")
    raise TypeError("Source units must be IDs, coordinate mappings or triples")


def _unit_path(unit: Any) -> str:
    if isinstance(unit, Mapping) and "path" in unit:
        return str(unit["path"]).replace("\\", "/")
    return re.sub(r":\d+:\d+$", "", unit_id(unit))


def evidence_metrics(
    selected_units: Iterable[Any], reference_units: Iterable[Any] | None,
    edit_units: Iterable[Any] | None, predicted_files: Sequence[str] = (),
    admitted_tokens: int | None = None, repeated_tokens: int | None = None,
    recovery: bool | None = None,
) -> dict[str, Any]:
    """Compare a delivered packet to external annotation, outside inference.

    ``None`` means annotation or instrumentation was not supplied. Ratios are
    fractions; yield is matched *support* units per 1,000 admitted tokens.
    ``reference_units`` contains edit and support units; ``edit_units`` is its
    edit subset. Duplicate source units count once here; repeated serialized
    tokens are measured separately by duplication.
    """
    selected = {unit_id(x) for x in selected_units}
    reference = None if reference_units is None else {unit_id(x) for x in reference_units}
    edit_rows = None if edit_units is None else list(edit_units)
    edits = None if edit_rows is None else {unit_id(x) for x in edit_rows}
    if admitted_tokens is not None and (_number(admitted_tokens) is None or admitted_tokens < 0):
        raise ValueError("admitted_tokens must be finite and nonnegative")
    if repeated_tokens is not None and (_number(repeated_tokens) is None or repeated_tokens < 0 or (admitted_tokens is not None and repeated_tokens > admitted_tokens)):
        raise ValueError("repeated_tokens must lie between zero and admitted_tokens")
    if recovery is not None and type(recovery) is not bool:
        raise ValueError("recovery must be true, false or null")
    if edits is not None and reference is not None and not edits <= reference:
        raise ValueError("edit_units must be a subset of reference_units")
    hits = None if reference is None else len(selected & reference)
    support = None if reference is None or edits is None else reference - edits
    support_hits = None if support is None else len(selected & support)
    edited_paths = None if edit_rows is None else {_unit_path(x) for x in edit_rows}
    return {
        "selected": len(selected), "required": None if reference is None else len(reference),
        "hits": hits, "support_hits": support_hits,
        "recall": None if not reference else hits / len(reference),
        "precision": None if reference is None else hits / max(1, len(selected)),
        "file_at_1": None if not edited_paths else bool(predicted_files and predicted_files[0].replace("\\", "/") in edited_paths),
        "edit_coverage": None if not edits else edits <= selected,
        "joint": None if not reference or not edits else reference <= selected,
        "admitted_tokens": admitted_tokens, "repeated_tokens": repeated_tokens,
        "duplication": None if admitted_tokens in (None, 0) or repeated_tokens is None else repeated_tokens / admitted_tokens,
        "yield": None if admitted_tokens in (None, 0) or support_hits is None else 1000 * support_hits / admitted_tokens,
        "recovery": recovery,
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if isfinite(value) else None


def _steps_complete(ledger: Mapping[str, Any]) -> bool:
    """Check whether a supplied action list covers the complete attempt."""
    steps = ledger.get("steps")
    if not isinstance(steps, (list, tuple)) or ledger.get("steps_complete", True) is not True:
        return False
    if not all(isinstance(step, Mapping) for step in steps):
        return False
    calls = ledger.get("calls")
    return calls is None or (type(calls) is int and calls == len(steps))


def _exploration_metrics(ledger: Mapping[str, Any]) -> dict[str, int | float | None]:
    """Use explicit totals or complete phase-labelled action records.

    Exploration contains setup and acquisition, including failed actions. A
    supplied steps list represents the complete action list unless marked
    ``steps_complete=False``; a supplied total call count must match its length.
    Phase and action kind are required for derivation. Resource accounting for
    repair and recovery remains in the complete-task totals.
    """
    derived = dict.fromkeys(EXPLORATION_FIELDS)
    steps = ledger.get("steps")
    complete = _steps_complete(ledger)
    complete = complete and all(
        isinstance(step, Mapping) and step.get("phase") in {"setup", "acquisition", "repair"}
        and step.get("kind") in {"model", "tool"} for step in (steps or ()))
    if complete:
        exploration = [step for step in steps if step["phase"] in {"setup", "acquisition"}]
        derived["exploration_calls"] = len(exploration)
        derived["exploration_model_calls"] = sum(step["kind"] == "model" for step in exploration)
        derived["exploration_tool_calls"] = sum(step["kind"] == "tool" for step in exploration)
        for kind, field in (("model", "exploration_llm_seconds"), ("tool", "exploration_tool_seconds")):
            component = "llm_seconds" if kind == "model" else "tool_seconds"
            durations = [_number(step.get(component, step.get("seconds")))
                         for step in exploration if step["kind"] == kind]
            derived[field] = sum(durations) if all(value is not None and value >= 0 for value in durations) else None
    for field in EXPLORATION_FIELDS:
        if field in ledger:
            value = ledger[field]
            valid = (type(value) is int and value >= 0) if field.endswith("calls") else (
                _number(value) is not None and value >= 0)
            derived[field] = value if valid else None
    return derived


def _ledger(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(record.get("ledger") or {})
    result = {**raw, **(raw.get("totals") or {})}
    steps = result.get("steps")
    valid_steps = steps is None or isinstance(steps, (list, tuple))
    complete = result.get("usage_complete", bool(raw)) is True and valid_steps and all(
        isinstance(step, Mapping) and (step.get("kind") != "model" or step.get("usage_known", True) is True)
        for step in (steps or [])
    )
    result["usage_complete"] = bool(complete)
    if not complete:
        # A failed request may have consumed tokens before transport failed.
        # Default counters of zero are not a complete provider-usage report.
        for field in ("uncached_input", "cached_input", "output", "tokens", "cost"):
            result[field] = None
    result.update(_exploration_metrics(result))
    return result


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [float(v) if isinstance(v, bool) else _number(v) for v in values]
    return fmean(numbers) if numbers and all(value is not None for value in numbers) else None


def _complete_sum(values: Iterable[Any]) -> float | None:
    items = list(values)
    numbers = [_number(x) for x in items]
    return sum(numbers) if items and all(x is not None and x >= 0 for x in numbers) else None


def _ratio(numerator: Any, denominator: Any, scale: float = 1.0) -> float | None:
    return None if numerator is None or denominator in (None, 0) else scale * numerator / denominator


def normalized_cost(uncached_input: int, cached_input: int, output: int, worker_hours: float) -> float:
    """Paper Eq. 5: USD/M tokens 1.25/0.10/6.00 and USD/worker-hour 0.06."""
    if any(_number(x) is None or x < 0 for x in (uncached_input, cached_input, output, worker_hours)):
        raise ValueError("Resource usage must be finite and nonnegative")
    return 1e-6 * (1.25 * uncached_input + .10 * cached_input + 6 * output) + .06 * worker_hours


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool attempts, including failures, with explicit metric denominators."""
    rows = list(records)
    n = len(rows)
    ledgers = [_ledger(x) for x in rows]
    grades = [x.get("resolved") for x in rows]
    if any(g is not True and g is not False and g is not None for g in grades):
        raise ValueError("resolved must be true, false or null")
    graded = sum(g is not None for g in grades)
    success = sum(g is True for g in grades)
    resources = {name: _complete_sum(x.get(name) for x in ledgers) for name in (
        "uncached_input", "cached_input", "output", "worker_hours", "elapsed_seconds", "calls", "llm_seconds", "tool_seconds",
        *EXPLORATION_FIELDS)}
    totals = [x.get("tokens") for x in ledgers]
    for i, ledger in enumerate(ledgers):
        components = _complete_sum(ledger.get(k) for k in ("uncached_input", "cached_input", "output"))
        if components is not None:
            totals[i] = components
    resources["tokens"] = _complete_sum(totals)
    costs = []
    for ledger in ledgers:
        cost = ledger.get("cost")
        if cost is None and all(_number(ledger.get(k)) is not None and ledger[k] >= 0
                                for k in ("uncached_input", "cached_input", "output", "worker_hours")):
            cost = normalized_cost(*(ledger[k] for k in ("uncached_input", "cached_input", "output", "worker_hours")))
        costs.append(cost)
    total_cost = _complete_sum(costs)
    evidence = [x.get("evidence_metrics") or {} for x in rows]
    result: dict[str, Any] = {
        "attempts": n, "graded_attempts": graded, "ungraded_attempts": n - graded,
        "successes": success,
        "resolution_pct": _ratio(success, n, 100) if graded == n else None,
        "graded_resolution_pct": _ratio(success, graded, 100),
        "normalized_cost_total": total_cost, "normalized_cost_mean": _ratio(total_cost, n),
        "cost_per_success": _ratio(total_cost, success) if graded == n else None,
        "tokens_per_success": _ratio(resources["tokens"], success) if graded == n else None,
        "tokens_million_per_success": _ratio(resources["tokens"], success * 1e6) if graded == n else None,
        "resources_total": resources, "tokens_mean": _ratio(resources["tokens"], n),
        "tokens_million_mean": _ratio(resources["tokens"], n * 1e6),
        "elapsed_seconds_mean": _ratio(resources["elapsed_seconds"], n),
        "time_minutes_mean": _ratio(resources["elapsed_seconds"], n * 60),
        "calls_mean": _ratio(resources["calls"], n),
        "llm_seconds_mean": _ratio(resources["llm_seconds"], n),
        "tool_seconds_mean": _ratio(resources["tool_seconds"], n), "metric_attempts": {},
        "usage_complete": all(x["usage_complete"] for x in ledgers) if ledgers else None,
        "usage_complete_attempts": sum(x["usage_complete"] for x in ledgers),
        "usage_incomplete_attempts": sum(not x["usage_complete"] for x in ledgers),
    }
    result.update({field + "_mean": _ratio(resources[field], n) for field in EXPLORATION_FIELDS})
    known_resources = {}
    for name in ("uncached_input", "cached_input", "output"):
        known_resources[name] = _complete_sum(
            ledger.get("known_" + name, ledger.get(name)) for ledger in ledgers
        )
    known_tokens = []
    for ledger, total in zip(ledgers, totals):
        components = _complete_sum(ledger.get("known_" + name, ledger.get(name))
                                   for name in ("uncached_input", "cached_input", "output"))
        known_tokens.append(components if components is not None else ledger.get("known_tokens", total))
    known_resources["tokens"] = _complete_sum(known_tokens)
    result["known_resources_total"] = known_resources
    result["known_normalized_cost_total"] = _complete_sum(
        ledger.get("known_cost", cost) for ledger, cost in zip(ledgers, costs)
    )
    for source, destination in (
        ("file_at_1", "file_at_1_pct"), ("recall", "recall_pct"),
        ("precision", "precision_pct"), ("edit_coverage", "edit_pct"),
        ("joint", "joint_pct"), ("duplication", "duplication_mean_pct"),
        ("recovery", "recovery_pct"), ("yield", "yield_mean"),
    ):
        vals = [e.get(source) for e in evidence]
        vals = [value if (isinstance(value, bool) or _number(value) is not None)
                and value >= 0 and (source == "yield" or value <= 1) else None for value in vals]
        result["metric_attempts"][source] = sum(x is not None for x in vals)
        mean = _mean(vals)
        result[destination] = None if mean is None or any(v is None for v in vals) else mean * (1 if source == "yield" else 100)
    for field in ("hits", "required", "selected", "support_hits", "admitted_tokens", "repeated_tokens"):
        result[field + "_total"] = _complete_sum(e.get(field) for e in evidence)
    result["pooled_recall_pct"] = _ratio(result["hits_total"], result["required_total"], 100)
    # Eq. (16) floors each attempt's precision denominator before pooling.
    result["precision_denominator_total"] = _complete_sum(
        max(1, e["selected"]) if _number(e.get("selected")) is not None and e["selected"] >= 0 else None
        for e in evidence)
    result["pooled_precision_pct"] = _ratio(result["hits_total"], result["precision_denominator_total"], 100)
    result["pooled_duplication_pct"] = _ratio(result["repeated_tokens_total"], result["admitted_tokens_total"], 100)
    result["pooled_yield"] = _ratio(result["support_hits_total"], result["admitted_tokens_total"], 1000)
    u, k, o = (resources[name] for name in ("uncached_input", "cached_input", "output"))
    input_tokens = None if u is None or k is None else u + k
    token_sum = None if input_tokens is None or o is None else input_tokens + o
    result["token_shares_pct"] = {
        "uncached_input": _ratio(u, token_sum, 100), "cached_input": _ratio(k, token_sum, 100),
        "output": _ratio(o, token_sum, 100), "cached_share_of_input": _ratio(k, input_tokens, 100),
    }
    events = [e for ledger in ledgers for e in (ledger.get("requests") or [])]
    def cache_events_complete(ledger):
        requests = ledger.get("requests")
        steps = ledger.get("steps")
        if ledger.get("requests_complete", True) is not True:
            return False
        if steps is not None and not _steps_complete(ledger):
            return False
        if requests is None:
            return _steps_complete(ledger) and not any(step.get("kind") == "model" for step in steps)
        if not isinstance(requests, (list, tuple)) or any(
                not isinstance(event, Mapping) or not isinstance(event.get("cache_hit"), bool)
                for event in requests):
            return False
        return steps is None or len(requests) == sum(step.get("kind") == "model" for step in steps)
    known_hits = [e["cache_hit"] for e in events if isinstance(e, Mapping) and isinstance(e.get("cache_hit"), bool)]
    result["request_cache_hit_rate_pct"] = None if not known_hits or not all(cache_events_complete(ledger) for ledger in ledgers) else 100 * fmean(known_hits)
    steps = [s for ledger in ledgers for s in (ledger.get("steps") or [])]
    steps_complete = all(_steps_complete(ledger) for ledger in ledgers)
    result["step_count"] = len(steps) if steps_complete else None
    result["known_step_count"] = len(steps)
    def step_duration(step, field):
        if field in step:
            value = step[field]
            return value if _number(value) is not None and value >= 0 else None
        if field != "wall_seconds":
            kind = {"llm_seconds": "model", "tool_seconds": "tool"}[field]
            if step.get("kind") in {"model", "tool"} and step["kind"] != kind:
                return 0
        seconds = step.get("seconds")
        if _number(seconds) is None or seconds < 0:
            return None
        if field == "wall_seconds":
            return seconds
        return seconds if step.get("kind") == kind else None
    for field in ("wall_seconds", "llm_seconds", "tool_seconds"):
        durations = [step_duration(s, field) for s in steps] if steps_complete else []
        result[f"step_{field}_mean"] = _mean(durations) if steps_complete and all(value is not None for value in durations) else None
    wall, llm, tool = (result[f"step_{field}_mean"] for field in ("wall_seconds", "llm_seconds", "tool_seconds"))
    result["step_residual_seconds_mean"] = None if None in (wall, llm, tool) else wall - llm - tool
    return result


def aggregate(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize benchmark/backbone/system groups, pooling every repeat."""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(str(record.get(k, "")) for k in ("benchmark", "model", "system"))].append(record)
    return [{"benchmark": key[0], "model": key[1], "system": key[2], **summarize(rows)}
            for key, rows in sorted(groups.items())]
