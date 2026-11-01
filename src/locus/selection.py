"""Source-selection policies with a shared frozen retrieval order.

StaticTopK retains full original records, including equal text at different
coordinates. ExactDedup merges equal payloads before using that same order.
Mandatory edit-source reservation belongs to coverage and EditOnly selection.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable

from .packet import materialize
from .types import Grounding, Observation, Packet, uncovered


def _valid_tokens(source: Observation) -> Observation:
    if type(source.tokens) is not int or source.tokens < 1:
        raise ValueError("Serialized source token counts must be positive integers")
    return source


def _set_current_roles(source: Observation, canonical: Observation,
                       count: Callable[[str], int] | None) -> Observation:
    """Use current roles while retaining the original observation coordinates."""
    if source.text != canonical.text:
        raise ValueError("A payload identity refers to different source text")
    if source.roles == canonical.roles:
        return _valid_tokens(source)
    if count is not None:
        # with_roles unions roles, so first remove previous-round role metadata.
        return _valid_tokens(replace(source, roles=frozenset()).with_roles(canonical.roles, count))
    updated = replace(source, roles=canonical.roles)
    if updated.version == canonical.version:
        return _valid_tokens(replace(updated, tokens=canonical.tokens))
    raise ValueError("Pass count=tokenizer to recount source records after role updates")


def _fixed_sources(grounding: Grounding, raw_sources: Iterable[Observation],
                   count: Callable[[str], int] | None) -> tuple[Observation, ...]:
    raw = tuple(raw_sources)
    if not raw:
        return tuple(_valid_tokens(s) for s in grounding.pool.values())
    result = []
    for source in raw:
        canonical = grounding.pool.get(source.id)
        if canonical is None:
            continue
        # Only coordinates retained by the validated pool are eligible.
        if not source.locations or not set(source.locations) <= set(canonical.locations):
            continue
        result.append(_set_current_roles(source, canonical, count))
    return tuple(result)


def _deduplicate(sources: Iterable[Observation], grounding: Grounding,
                 count: Callable[[str], int] | None) -> tuple[Observation, ...]:
    merged: dict[str, Observation] = {}
    for source in sources:
        previous = merged.get(source.id)
        if previous is None:
            merged[source.id] = source
            continue
        if previous.text != source.text:
            raise ValueError("A payload identity refers to different source text")
        if set(source.locations) <= set(previous.locations) and source.roles <= previous.roles:
            continue
        if count is not None:
            merged[source.id] = _valid_tokens(previous.merge(source, count))
            continue
        updated = replace(previous,
            locations=tuple(sorted(set(previous.locations + source.locations))),
            roles=previous.roles | source.roles)
        canonical = grounding.pool[source.id]
        if updated.version != canonical.version:
            raise ValueError("Pass count=tokenizer to recount merged source coordinates")
        merged[source.id] = _valid_tokens(replace(updated, tokens=canonical.tokens))
    return tuple(merged.values())


def select_packet(
    grounding: Grounding, cap: int, policy: str, raw_sources: Iterable[Observation] = (),
    *, count: Callable[[str], int] | None = None,
) -> tuple[Packet, tuple]:
    """Return the selected packet and the exact inferred unmet requirements.

    ``coverage`` and ``edit_only`` use Algorithm 4. ``static_topk`` and ``broad``
    append fitting full records in the initial ``raw_sources`` order; there is
    no coverage-gain reranking or support pruning. ``exact_dedup`` merges equal
    payload hashes, roles and coordinates in first-occurrence order, then uses
    the same selection procedure. These fixed-rank policies do not insert edit
    source ahead of their initial ranking. The caller supplies the broad cap.

    Empty ``raw_sources`` uses the validated pool's insertion order. Raw source
    coordinates are retained, while their roles follow the current pool. Pass
    the run's actual tokenizer as ``count`` to recount changed role metadata or
    merged coordinates. Packet sources may share an id under static selection.
    """
    aliases = {"LOCUS": "coverage", "EditOnly": "edit_only",
               "StaticTopK": "static_topk", "ExactDedup": "exact_dedup",
               "Broad context": "broad"}
    policy = aliases.get(policy, policy)
    if type(cap) is not int or cap < 0:
        raise ValueError("Packet token cap must be a nonnegative integer")
    if policy in {"coverage", "edit_only"}:
        return materialize(grounding, cap, policy=policy, count=count)
    if policy not in {"static_topk", "exact_dedup", "broad"}:
        raise ValueError(f"Unknown source-selection policy: {policy}")
    ranked = _fixed_sources(grounding, raw_sources, count)
    if policy == "exact_dedup":
        ranked = _deduplicate(ranked, grounding, count)
    selected = []
    budget = cap
    for source in ranked:
        if source.tokens <= budget:
            selected.append(source)
            budget -= source.tokens
    packet = Packet(grounding.frontier, tuple(selected))
    assert packet.tokens <= cap
    return packet, uncovered(packet.sources, grounding.requirements)
