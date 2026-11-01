"""Source coordinates, payload identity, edit frontier and support incidence."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
from pathlib import PurePosixPath
from typing import Callable

ROLES = frozenset({"editable", "symptom", "support"})
ReadTarget = tuple[str, int, int]
TokenCount = Callable[[str], int]


def compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def repository_path(path: str) -> str:
    if not isinstance(path, str):
        raise ValueError("Expected a relative repository path")
    normalized = path.replace("\\", "/")
    p = PurePosixPath(normalized)
    if (not normalized or str(p) == "." or p.is_absolute() or ".." in p.parts
            or ":" in normalized or "\x00" in normalized):
        raise ValueError("Expected a relative repository path")
    return str(p)


@dataclass(frozen=True, order=True)
class Location:
    repo: str
    commit: str
    path: str
    start: int
    end: int

    def __post_init__(self):
        object.__setattr__(self, "path", repository_path(self.path))
        if (type(self.start) is not int or type(self.end) is not int
                or self.start < 1 or self.end < self.start):
            raise ValueError("Invalid source interval")
        if not isinstance(self.repo, str) or not self.repo or not isinstance(self.commit, str) or not self.commit:
            raise ValueError("Source coordinates require a repository and base commit")

    def as_dict(self) -> dict:
        return dict(repo=self.repo, commit=self.commit, path=self.path, start=self.start, end=self.end)


@dataclass(frozen=True)
class Observation:
    id: str
    text: str
    locations: tuple[Location, ...]
    roles: frozenset[str]
    tokens: int

    def _counted(self, count: TokenCount) -> "Observation":
        tokens = count(self.serialize())
        if type(tokens) is not int or tokens < 1:
            raise ValueError("Serialized source token count must be a positive integer")
        return replace(self, tokens=tokens)

    @classmethod
    def create(cls, text: str, location: Location, count: TokenCount,
               roles=frozenset({"support"})) -> "Observation":
        if not roles or not set(roles) <= ROLES:
            raise ValueError("Invalid source roles")
        item = cls(sha256(text.encode("utf-8")).hexdigest(), text, (location,), frozenset(roles), 0)
        return item._counted(count)

    def as_dict(self) -> dict:
        return dict(id=self.id, locations=[x.as_dict() for x in self.locations],
                    roles=sorted(self.roles), text=self.text)

    def serialize(self) -> str:
        return compact(self.as_dict())

    @property
    def version(self) -> str:
        return sha256(self.serialize().encode()).hexdigest()

    def merge(self, other: "Observation", count: TokenCount) -> "Observation":
        if self.id != other.id or self.text != other.text:
            raise ValueError("Cannot merge different source payloads")
        item = replace(self, locations=tuple(sorted(set(self.locations + other.locations))),
                       roles=self.roles | other.roles, tokens=0)
        return item._counted(count)

    def with_roles(self, roles, count) -> "Observation":
        if not set(roles) <= ROLES:
            raise ValueError("Unknown evidence role")
        item = replace(self, roles=self.roles | frozenset(roles), tokens=0)
        return item._counted(count)


@dataclass(frozen=True)
class Candidate:
    id: str
    location: Location
    symbol: str
    change: str
    score: float
    signals: dict[str, float] = field(default_factory=dict)
    signal_sources: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def rank_key(self) -> tuple[str, str]:
        """Stability compares the ordered edit and intended behavioral change."""
        return self.id, " ".join(self.change.split())

    def as_dict(self) -> dict:
        return dict(id=self.id, location=self.location.as_dict(), symbol=self.symbol,
                    change=self.change, score=self.score, signals=self.signals,
                    signal_sources=self.signal_sources)


@dataclass(frozen=True)
class Requirement:
    id: str
    description: str
    kind: str
    source_ids: frozenset[str]
    targets: tuple[ReadTarget, ...] = ()
    weight: float = 1.0

    def __post_init__(self):
        if (type(self.weight) not in (int, float) or not math.isfinite(self.weight)
                or self.weight <= 0):
            raise ValueError("Requirement weights must be finite and positive")

    def as_dict(self) -> dict:
        return dict(id=self.id, description=self.description, kind=self.kind,
                    source_ids=sorted(self.source_ids), targets=self.targets, weight=self.weight)


@dataclass(frozen=True)
class Alternative:
    id: str
    candidate_id: str
    status: str
    source_ids: frozenset[str]
    explanation: str = ""
    location: Location | None = None
    targets: tuple[ReadTarget, ...] = ()

    def as_dict(self) -> dict:
        return dict(id=self.id, candidate=self.candidate_id, status=self.status,
                    source_ids=sorted(self.source_ids), explanation=self.explanation,
                    location=self.location.as_dict() if self.location else None,
                    targets=self.targets)


@dataclass(frozen=True)
class Grounding:
    frontier: tuple[Candidate, ...]
    alternatives: tuple[Alternative, ...]
    requirements: tuple[Requirement, ...]
    pool: dict[str, Observation]


@dataclass(frozen=True)
class Packet:
    frontier: tuple[Candidate, ...]
    sources: tuple[Observation, ...]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(s.id for s in self.sources)

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.sources)

    def as_dict(self, *, include_text=True) -> dict:
        return dict(frontier=[c.as_dict() for c in self.frontier],
                    sources=[s.as_dict() if include_text else
                             dict(id=s.id, locations=[l.as_dict() for l in s.locations],
                                  roles=sorted(s.roles), tokens=s.tokens) for s in self.sources],
                    tokens=self.tokens)


def covers_location(sources, location: Location, *, editable=False) -> bool:
    intervals = []
    for s in sources:
        if editable and "editable" not in s.roles:
            continue
        intervals.extend((l.start, l.end) for l in s.locations
                         if (l.repo, l.commit, l.path) ==
                         (location.repo, location.commit, location.path))
    cursor = location.start
    for lo, hi in sorted(intervals):
        if hi < cursor:
            continue
        if lo > cursor:
            return False
        cursor = max(cursor, hi + 1)
        if cursor > location.end:
            return True
    return False


def supported(frontier, sources) -> bool:
    sources = tuple(sources)
    return bool(frontier) and all(covers_location(sources, c.location, editable=True) for c in frontier)


def uncovered(sources, requirements) -> tuple[Requirement, ...]:
    ids = {s.id for s in sources}
    return tuple(r for r in requirements if not ids.intersection(r.source_ids))

