"""Small structural interfaces shared by the controller and its adapters."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from .types import Location, Observation, TokenCount


class RecipientProtocol(Protocol):
    transmitted_tokens: int
    repeated_tokens: int
    preserve_empty_sources: bool

    def has(self, source: Observation) -> bool: ...

    def consumed(self, source: Observation) -> bool: ...

    def deliver(self, observations: Iterable[Observation], *, extra: str = "",
                deduplicate: bool = True,
                protected: frozenset[str] = frozenset()) -> bool: ...

    def reset(self) -> None: ...

    def request_messages(self, prompt: str,
                         protected: frozenset[str] = frozenset()) -> list[dict[str, str]]: ...


class ModelProtocol(Protocol):
    def session(self, role: str, system: str) -> RecipientProtocol: ...

    def ask(self, role: str, system: str, prompt: str,
            sources: Iterable[Observation] = ()) -> dict[str, Any]: ...


class WorkspaceProtocol(Protocol):
    repo: str
    commit: str

    def snapshot(self) -> dict[str, str]: ...

    def read(self, path: str, start: int, end: int) -> tuple[str, Location]: ...

    def apply_patch(self, patch: str) -> None: ...

    def run(self, command: str, timeout: int = 120) -> dict[str, Any]: ...

    def diff(self) -> str: ...
