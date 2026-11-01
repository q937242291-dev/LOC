"""Algorithm 4: mandatory edits, coverage/token and coverage-preserving pruning."""
from __future__ import annotations

from hashlib import sha256

from .types import Packet, ROLES, supported, uncovered


def coverage(sources, requirements) -> float:
    ids = {s.id for s in sources}
    return sum(r.weight for r in requirements if ids.intersection(r.source_ids))


def _validated_pool(pool, count):
    """Merge payload hashes before selection using exact serialized costs."""
    groups = {}
    for source in pool.values():
        if type(source.tokens) is not int or source.tokens < 1:
            raise ValueError("Serialized source token count must be a positive integer")
        if not source.locations or not source.roles or not source.roles <= ROLES:
            raise ValueError("Source records require coordinates and valid roles")
        if source.id != sha256(source.text.encode("utf-8")).hexdigest():
            raise ValueError("Source payload hash does not match its text")
        groups.setdefault(source.id, []).append(source)

    merged = {}
    for sid, records in groups.items():
        if count is not None:
            # Even a single low-level record is measured with the actual counter.
            source = records[0].merge(records[0], count)
            for other in records[1:]:
                source = source.merge(other, count)
        else:
            roles = frozenset().union(*(record.roles for record in records))
            locations = frozenset(loc for record in records for loc in record.locations)
            canonical = [record for record in records
                         if record.roles == roles and frozenset(record.locations) == locations]
            if not canonical:
                raise ValueError("Hash metadata merging requires the serialized token count callback")
            if len({record.tokens for record in canonical}) != 1:
                raise ValueError("Conflicting serialized counts require the token count callback")
            source = canonical[0]
        merged[sid] = source
    return merged


def mandatory_source(frontier, pool):
    """Retain source covering every edit interval; one hash has one payload."""
    def covered_lines(sources):
        total = 0
        for candidate in frontier:
            target = candidate.location
            intervals = [(max(loc.start, target.start), min(loc.end, target.end))
                         for source in sources if "editable" in source.roles
                         for loc in source.locations
                         if (loc.repo, loc.commit, loc.path) ==
                            (target.repo, target.commit, target.path)
                         and loc.start <= target.end and loc.end >= target.start]
            cursor = target.start
            for start, end in sorted(intervals):
                total += max(0, end - max(start, cursor) + 1)
                cursor = max(cursor, end + 1)
        return total

    selected = {}
    while not supported(frontier, selected.values()):
        current = covered_lines(selected.values())
        choices = [(covered_lines((*selected.values(), source)) - current, source)
                   for source in pool.values() if source.id not in selected]
        choices = [(gain, source) for gain, source in choices if gain > 0]
        if not choices:
            return ()
        _, source = min(choices, key=lambda item:(
            -item[0] / max(1, item[1].tokens), item[1].tokens, item[1].id))
        selected[source.id] = source
    # Do not reject a feasible packet because overlapping implementation
    # fragments have remained in the mandatory reservation unnecessarily.
    for sid in list(selected):
        remaining = tuple(source for key, source in selected.items() if key != sid)
        if supported(frontier, remaining):
            del selected[sid]
    return tuple(selected.values())


def materialize(grounding, cap: int, *, policy="coverage", count=None):
    """Return a capped packet and its exact, fixed-map unmet requirement set.

    ``count`` measures complete serialized source records after hash metadata
    merging. Without it, each metadata union must already exist in the pool as
    an exactly counted canonical record.
    """
    if type(cap) is not int or cap < 0:
        raise ValueError("Packet cap must be a nonnegative integer")
    if policy not in {"coverage", "edit_only"}:
        raise ValueError("Algorithm 4 supports coverage or edit_only selection")
    frontier, requirements = grounding.frontier, grounding.requirements
    pool = _validated_pool(grounding.pool, count)
    mandatory = mandatory_source(frontier, pool)
    if not supported(frontier, mandatory) or sum(x.tokens for x in mandatory) > cap:
        return Packet(frontier, ()), tuple(requirements)
    selected = {s.id: s for s in mandatory}
    budget = cap - sum(s.tokens for s in selected.values())
    if policy == "edit_only":
        return Packet(frontier, tuple(selected.values())), uncovered(selected.values(), requirements)
    while True:
        missing = uncovered(selected.values(), requirements)
        feasible = [s for s in pool.values() if s.id not in selected and s.tokens <= budget]
        if not feasible or not missing:
            break
        def key(s):
            gain = sum(r.weight for r in missing if s.id in r.source_ids)
            return (-gain/max(1, s.tokens), s.tokens, s.id)
        candidate = min(feasible, key=key)
        if not any(candidate.id in r.source_ids for r in missing):
            break
        selected[candidate.id] = candidate
        budget -= candidate.tokens
    for sid in list(selected):
        remaining = tuple(s for k,s in selected.items() if k != sid)
        # Compare incidence sets directly: floating weighted sums can hide
        # the loss of a small positive-weight requirement.
        if (uncovered(remaining, requirements) == uncovered(selected.values(), requirements)
                and supported(frontier, remaining)):
            del selected[sid]
    packet = Packet(frontier, tuple(selected.values()))
    assert packet.tokens <= cap
    return packet, uncovered(packet.sources, requirements)

