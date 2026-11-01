"""The four conjunctive evidence conditions in Eq. (14)."""
from __future__ import annotations

from dataclasses import dataclass
from .types import supported, uncovered, Packet


@dataclass(frozen=True)
class GateResult:
    edit_support: bool
    coverage: bool
    alternatives_checked: bool
    rank_stable: bool

    @property
    def passed(self) -> bool:
        return all((self.edit_support, self.coverage, self.alternatives_checked, self.rank_stable))

    def as_dict(self) -> dict:
        return dict(edit_support=self.edit_support, coverage=self.coverage,
                    alternatives_checked=self.alternatives_checked, rank_stable=self.rank_stable,
                    passed=self.passed)


def evidence_gate(packet, requirements, alternatives, history, window: int, *, check_alternatives=True):
    if type(window) is not int or window < 1:
        raise ValueError("Rank stability requires a positive completed-round window")
    ids = packet.ids
    frontier_ids = tuple(c.id for c in packet.frontier)
    frontier_key = tuple(c.rank_key for c in packet.frontier)
    alternatives_ok = all(
        (a.status == "retained" and a.candidate_id in frontier_ids and a.source_ids <= ids)
        or (a.status == "rejected" and bool(a.source_ids) and a.source_ids <= ids)
        for a in alternatives
    ) if check_alternatives else True

    def same_frontier(entry):
        if not isinstance(entry, (tuple, list)):
            return False
        if all(isinstance(item, str) for item in entry):
            # Identifier histories compare the ordered candidate identities.
            return tuple(entry) == frontier_ids
        if not all(isinstance(item, (tuple, list)) and len(item) == 2
                   and all(isinstance(value, str) for value in item) for item in entry):
            return False
        return tuple((item[0], " ".join(item[1].split())) for item in entry) == frontier_key

    stable = (len(history) >= window and
              all(same_frontier(entry) for entry in history[-window:]))
    return GateResult(
        edit_support=supported(packet.frontier, packet.sources),
        coverage=not uncovered(packet.sources, requirements),
        alternatives_checked=alternatives_ok,
        rank_stable=stable,
    )


def visible_packet(packet, recipient):
    return Packet(packet.frontier, tuple(s for s in packet.sources if recipient.has(s)))

