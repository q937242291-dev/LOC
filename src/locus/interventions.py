"""Frozen-edit context interventions, Table 7.

Reference units select the required-evidence-removal condition. Repair prompts
contain the selected source context and fixed edit frontier. Every condition
disables post-handoff recovery.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import random
import re
from typing import Any

from .evaluation import unit_id


def observation_id(row: Mapping[str, Any]) -> str:
    if "id" in row:
        return str(row["id"])
    if "observation_id" in row:
        return str(row["observation_id"])
    raise ValueError("Every observation requires a unique id")


def observation_units(row: Mapping[str, Any]) -> set[str]:
    """Explicit units, or one path/start/end unit for a source span."""
    if "units" in row:
        return {unit_id(x) for x in row["units"]}
    locations = row.get("locations")
    if locations:
        return {unit_id(loc) for loc in locations}
    if "path" in row:
        return {unit_id(row)}
    return set()


def payload_hash(row: Mapping[str, Any]) -> str:
    return str(row.get("payload_hash", row.get("hash", hashlib.sha256(str(row.get("text", "")).encode()).hexdigest())))


def _tokens(row: Mapping[str, Any]) -> int:
    value = row.get("tokens", row.get("token_count"))
    if type(value) is not int or value < 1:
        raise ValueError("Every observation requires a positive serialized token count")
    return value


def _location(value: Mapping[str, Any]) -> dict[str, Any]:
    location = value.get("location", value)
    path = str(location["path"]).replace("\\", "/")
    start = location.get("start", location.get("start_line"))
    end = location.get("end", location.get("end_line", start))
    if type(start) is not int or type(end) is not int or start < 1 or end < start:
        raise ValueError("Frozen frontier requires valid inclusive source coordinates")
    return {"path": path, "start": start, "end": end,
            "repo": location.get("repo"), "commit": location.get("commit")}


def _source_locations(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    if row.get("locations"):
        return [_location(loc) for loc in row["locations"]]
    if "path" in row:
        return [_location(row)]
    result = []
    for unit in row.get("units", ()):
        if isinstance(unit, Mapping) and "path" in unit:
            result.append(_location(unit))
        elif isinstance(unit, (list, tuple)) and len(unit) == 3:
            result.append(_location(dict(path=unit[0], start=unit[1], end=unit[2])))
        elif isinstance(unit, str):
            match = re.fullmatch(r"(.+):(\d+):(\d+)", unit)
            if match:
                result.append(_location(dict(path=match[1], start=int(match[2]), end=int(match[3]))))
    return result


def _covers(rows: Sequence[Mapping[str, Any]], target: Mapping[str, Any]) -> bool:
    intervals = []
    for row in rows:
        if "roles" in row and "editable" not in row["roles"]:
            continue
        for loc in _source_locations(row):
            if loc["path"] != target["path"]:
                continue
            if any(loc.get(key) is not None and target.get(key) is not None
                   and loc[key] != target[key] for key in ("repo", "commit")):
                continue
            intervals.append((loc["start"], loc["end"]))
    cursor = target["start"]
    for low, high in sorted(intervals):
        if high < cursor:
            continue
        if low > cursor:
            return False
        cursor = max(cursor, high + 1)
        if cursor > target["end"]:
            return True
    return False


def _mandatory_source(frontier: Sequence[Any], selected: Sequence[Mapping[str, Any]],
                      explicit: set[str]) -> set[str]:
    """Keep editable source spanning every frozen edit, not just named IDs."""
    by_id = {observation_id(row): row for row in selected}
    if not explicit <= by_id.keys():
        raise ValueError("Mandatory source is absent from the selected packet")
    mandatory = set(explicit)
    for candidate in frontier:
        if not isinstance(candidate, Mapping):
            raise ValueError("Frozen frontier candidates require source coordinates")
        target = _location(candidate)
        if not _covers(selected, target):
            raise ValueError("Selected packet does not support every frozen edit")
        for row in sorted(selected, key=lambda r: (_tokens(r), observation_id(r))):
            if _covers([by_id[key] for key in mandatory], target):
                break
            if "roles" in row and "editable" not in row["roles"]:
                continue
            if any(loc["path"] == target["path"] and loc["start"] <= target["end"]
                   and loc["end"] >= target["start"]
                   and all(loc.get(key) is None or target.get(key) is None or loc[key] == target[key]
                           for key in ("repo", "commit")) for loc in _source_locations(row)):
                mandatory.add(observation_id(row))
    for key in sorted(mandatory - explicit, key=lambda item: (-_tokens(by_id[item]), item)):
        remaining = [by_id[item] for item in mandatory if item != key]
        if all(_covers(remaining, _location(candidate)) for candidate in frontier):
            mandatory.remove(key)
    return mandatory


def _fit(rows: Sequence[Mapping[str, Any]], mandatory: set[str], cap: int) -> list[Mapping[str, Any]]:
    by_id = {observation_id(r): r for r in rows}
    if not mandatory <= by_id.keys():
        raise ValueError("Mandatory edit source is absent from the source pool")
    result = [r for r in rows if observation_id(r) in mandatory]
    used = sum(_tokens(r) for r in result)
    if used > cap:
        raise ValueError("Mandatory editable source exceeds the intervention cap")
    for row in rows:
        if observation_id(row) in mandatory:
            continue
        if used + _tokens(row) <= cap:
            result.append(row)
            used += _tokens(row)
    return result


def _covered_facts(rows: Sequence[Mapping[str, Any]], requirements: Sequence[Mapping[str, Any]]) -> set[str]:
    ids = {observation_id(row) for row in rows}
    return {str(requirement["id"]) for requirement in requirements
            if ids.intersection(map(str, requirement.get("source_ids", ())))}


def _prune_redundant(rows: Sequence[Mapping[str, Any]], mandatory: set[str],
                     requirements: Sequence[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    """Preserve covered inferred facts, or exact source units without a fact map."""
    result = list(rows)
    covered_facts = None if requirements is None else _covered_facts(rows, requirements)
    for row in list(result):
        if observation_id(row) in mandatory:
            continue
        remaining = [r for r in result if r is not row]
        if requirements is not None:
            redundant = _covered_facts(remaining, requirements) == covered_facts
        else:
            units = set().union(*(observation_units(r) for r in remaining)) if remaining else set()
            same_payload = any(payload_hash(r) == payload_hash(row) for r in remaining)
            covered = observation_units(row)
            redundant = (covered and covered <= units) or (not covered and same_payload)
        if redundant:
            result.remove(row)
    return result


def _same_size_random(pool: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]],
                      mandatory: set[str], seed: int) -> list[Mapping[str, Any]]:
    """Randomized exact subset sum over whole serialized source records."""
    # Reuse the selected view for already-selected identities. This preserves
    # its exact roles, coordinates and serialized cost when pool metadata grew.
    selected_by_id = {observation_id(row): row for row in selected}
    candidates = [selected_by_id.get(observation_id(row), row) for row in pool]
    kept = [row for row in candidates if observation_id(row) in mandatory]
    target = sum(_tokens(row) for row in selected) - sum(_tokens(row) for row in kept)
    if target < 0:
        raise ValueError("Mandatory source exceeds the selected packet's actual size")
    optional = [row for row in candidates if observation_id(row) not in mandatory]
    random.Random(seed).shuffle(optional)
    # Bit s denotes an attainable sum s. Each newly attainable sum receives one
    # predecessor, so backtracking selects each observation at most once.
    reachable = 1
    mask = (1 << (target + 1)) - 1
    previous: list[tuple[int, int] | None] = [None] * (target + 1)
    for index, row in enumerate(optional):
        size = _tokens(row)
        if size > target:
            continue
        shifted = (reachable << size) & mask
        fresh = shifted & ~reachable
        while fresh:
            bit = fresh & -fresh
            total = bit.bit_length() - 1
            previous[total] = (total - size, index)
            fresh ^= bit
        reachable |= shifted
        if reachable & (1 << target):
            break
    if not reachable & (1 << target):
        raise ValueError("The acquired source pool cannot match the selected packet's actual token size")
    chosen = []
    cursor = target
    while cursor:
        predecessor = previous[cursor]
        if predecessor is None:
            raise ValueError("Source packet selection did not reach the requested token size")
        cursor, index = predecessor
        chosen.append(index)
    result = kept + [optional[index] for index in sorted(chosen)]
    assert sum(_tokens(row) for row in result) == sum(_tokens(row) for row in selected)
    return result


def frozen_edit_intervention(
    condition: str, frontier: Sequence[Any], pool: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]], *, mandatory_ids: Iterable[str] = (),
    reference_support_units: Iterable[Any] | None = None, seed: int = 0,
    requirements: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one condition from a previously grounded, frozen edit frontier.

    Observation dictionaries contain ``id``, ``tokens``, ``text`` and either
    ``units`` or source coordinates. ``selected`` is the 16k selected packet,
    while ``pool`` is the acquired source in its original order. Tokens count
    the serialized observation. Source spans are not truncated to manufacture
    a desired packet size. The random condition exactly matches the selected
    packet's actual serialized token count, within the same nominal 16k cap.
    ``requirements`` is an optional fixed incidence map for redundancy removal;
    covered and uncovered facts keep their original incidence throughout.

    Conditions: ``broad``, ``selected``, ``same-size-random``,
    ``required-evidence-removed`` and ``duplicates-removed``.
    """
    aliases = {"broad32k": "broad", "selected16k": "selected", "random": "same-size-random",
               "required-removed": "required-evidence-removed", "duplicate-removed": "duplicates-removed"}
    condition = aliases.get(condition, condition)
    caps = {"broad": 32000, "selected": 16000, "same-size-random": 16000,
            "required-evidence-removed": 12000, "duplicates-removed": 12000}
    if condition not in caps:
        raise ValueError(f"Unknown intervention: {condition}")
    if not frontier:
        raise ValueError("Frozen-edit interventions require a nonempty frontier")
    mandatory = set(map(str, mandatory_ids))
    pool_ids = [observation_id(r) for r in pool]
    selected_ids = [observation_id(r) for r in selected]
    if len(pool_ids) != len(set(pool_ids)) or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Observation IDs must be unique")
    if not set(selected_ids) <= set(pool_ids):
        raise ValueError("Selected source must come from the acquired pool")
    if not mandatory <= set(selected_ids):
        raise ValueError("Selected packet must include mandatory editable source")
    for row in (*pool, *selected):
        _tokens(row)
    mandatory = _mandatory_source(frontier, selected, mandatory)
    if sum(_tokens(r) for r in selected) > 16000:
        raise ValueError("Selected packet exceeds 16k tokens")
    cap = caps[condition]
    requirement_rows = None if requirements is None else tuple(requirements)
    if requirement_rows is not None and len({str(row["id"]) for row in requirement_rows}) != len(requirement_rows):
        raise ValueError("Requirement IDs must be unique")
    removed: list[str] = []
    if condition == "broad":
        result = _fit(pool, mandatory, cap)
    elif condition == "selected":
        result = list(selected)
    elif condition == "same-size-random":
        result = _same_size_random(pool, selected, mandatory, seed)
    elif condition == "required-evidence-removed":
        if reference_support_units is None:
            raise ValueError("Required-evidence removal requires external reference support units")
        support = {unit_id(u) for u in reference_support_units}
        removed_set: set[str] = set()
        for unit in sorted(support):
            matching = [row for row in selected if unit in observation_units(row)]
            if not matching or any(observation_id(row) in mandatory for row in matching):
                continue
            removed_set.update(observation_id(row) for row in matching)
            remaining = [row for row in selected if observation_id(row) not in removed_set]
            if sum(_tokens(row) for row in remaining) <= cap:
                break
        if not removed_set:
            raise ValueError("No removable reference support is present in the selected packet")
        result = [row for row in selected if observation_id(row) not in removed_set]
        if sum(_tokens(row) for row in result) > cap:
            raise ValueError("Required-evidence removal alone does not fit the 12k cap")
        removed = [observation_id(row) for row in selected if observation_id(row) in removed_set]
    else:
        result = _prune_redundant(selected, mandatory, requirement_rows)
        if sum(_tokens(r) for r in result) > cap:
            raise ValueError("Coverage-preserving redundancy removal does not fit the 12k cap")
        removed = [observation_id(r) for r in selected if observation_id(r) not in {observation_id(x) for x in result}]
    if not all(_covers(result, _location(candidate)) for candidate in frontier):
        raise ValueError("Intervention does not support every frozen edit")
    return {"condition": condition, "frontier": deepcopy(list(frontier)),
            "observations": deepcopy(list(result)), "token_cap": cap,
            "tokens": sum(_tokens(r) for r in result), "recovery_allowed": False,
            "selection_budget": sum(_tokens(row) for row in selected) if condition == "same-size-random" else cap,
            "seed": seed, "removed_ids": removed}


def build_interventions(
    frontier: Sequence[Any], pool: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]],
    *, mandatory_ids: Iterable[str] = (), reference_support_units: Iterable[Any] | None = None,
    seed: int = 0, requirements: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build all five fixed-edit conditions from the same source acquisition."""
    mandatory = tuple(mandatory_ids)
    support = None if reference_support_units is None else tuple(reference_support_units)
    requirement_rows = None if requirements is None else tuple(requirements)
    return {name: frozen_edit_intervention(name, frontier, pool, selected,
            mandatory_ids=mandatory, reference_support_units=support, seed=seed,
            requirements=requirement_rows)
            for name in ("broad", "selected", "same-size-random", "required-evidence-removed", "duplicates-removed")}
