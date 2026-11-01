"""Runtime settings for Algorithms 1-4 and the experimental protocol."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from pathlib import Path
import re
import tomllib


@dataclass(frozen=True)
class Profile:
    calls: int
    depth: int


PROFILES = {"L": Profile(64, 1), "M": Profile(128, 2), "H": Profile(210, 3)}
DEBATE_ROLES = (
    "Implementation Analyst", "Dependency Analyst", "Behavior Analyst",
    "Alternative-Hypothesis Analyst", "Validation Analyst",
)


@dataclass(frozen=True)
class Settings:
    total_calls: int = 250
    deadline_seconds: float = 7200
    packet_tokens: int = 16000
    request_input_tokens: int = 128000
    output_tokens: int = 32768
    temperature: float = 0.7
    top_p: float = 1.0
    frontier_size: int = 5
    debate_rounds: int = 3
    stability_window: int = 2
    reserve_calls: int = 40
    reserve_seconds: float = 300
    relevance_weight: float = 0.78
    symptom_penalty: float = 0.10
    behavior_weight: float = 0.50
    relation_weight: float = 0.25
    compatibility_weight: float = 0.25
    uncached_price: float = 1.25
    cached_price: float = 0.10
    output_price: float = 6.00
    worker_hour_price: float = 0.06
    initial_files: int = 8
    read_lines: int = 160
    read_batch: int = 6
    max_file_bytes: int = 2_000_000
    recovery: bool = True
    role_grounding: bool = True
    alternatives: bool = True
    memory: bool = True
    gate_enabled: bool = True
    selection: str = "coverage"
    allocation: str = "adaptive"
    encoding: str = "o200k_base"
    tokenizer_model: str = ""
    seed: int = 0

    def validate(self) -> "Settings":
        for key in ("total_calls", "packet_tokens", "request_input_tokens",
                    "output_tokens", "frontier_size", "stability_window",
                    "debate_rounds", "read_lines", "read_batch", "initial_files", "max_file_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.reserve_calls) is not int or self.reserve_calls < 0 or type(self.seed) is not int:
            raise ValueError("Reserve calls and seed must be integers")
        for key in ("deadline_seconds", "reserve_seconds", "temperature", "top_p",
                    "relevance_weight", "symptom_penalty", "behavior_weight", "relation_weight",
                    "compatibility_weight", "uncached_price", "cached_price", "output_price",
                    "worker_hour_price"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        if self.reserve_seconds < 0 or self.deadline_seconds <= 0:
            raise ValueError("Invalid resource allowance")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid sampling parameters")
        if min(self.uncached_price, self.cached_price, self.output_price, self.worker_hour_price) < 0:
            raise ValueError("Resource prices must be nonnegative")
        for key in ("recovery", "role_grounding", "alternatives", "memory", "gate_enabled"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if self.packet_tokens >= self.request_input_tokens:
            raise ValueError("Packet must leave request space for instructions and issue")
        weights = (self.behavior_weight, self.relation_weight, self.compatibility_weight)
        if min(weights) < 0 or abs(sum(weights) - 1.0) > 1e-8:
            raise ValueError("Grounding weights must be nonnegative and sum to one")
        if not 0 <= self.relevance_weight <= 1 or self.symptom_penalty < 0:
            raise ValueError("Invalid ranking weights")
        if self.allocation not in {"adaptive", "L", "M", "H"}:
            raise ValueError("Unknown allocation")
        if self.selection not in {"coverage", "edit_only", "static_topk", "exact_dedup", "broad"}:
            raise ValueError("Unknown source-selection policy")
        return self


@dataclass(frozen=True)
class ModelSettings:
    provider: str = "compatible"
    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "LOCUS_API_KEY"
    timeout_seconds: float = 180
    seed_mode: str = "off"
    extra_body: dict = field(default_factory=dict)

    def validate(self) -> "ModelSettings":
        if self.provider not in {"openai", "compatible", "anthropic"}:
            raise ValueError("Unknown model provider")
        if self.seed_mode not in {"off", "request"}:
            raise ValueError("seed_mode must be off or request")
        if self.provider == "anthropic" and self.seed_mode != "off":
            raise ValueError("The Anthropic Messages adapter does not support a seed request")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("A model identifier is required")
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("Model timeout must be finite and positive")
        if not isinstance(self.api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ValueError("Invalid credential environment variable name")
        if not isinstance(self.extra_body, dict):
            raise ValueError("Model extra_body must be an object")
        return self


def load_config(path: str | Path | None = None) -> tuple[Settings, ModelSettings]:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    for section, cls in (("agent", Settings), ("model", ModelSettings)):
        extra = set(raw.get(section, {})) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"Unknown {section} settings: {sorted(extra)}")
    return Settings(**raw.get("agent", {})).validate(), ModelSettings(**raw.get("model", {})).validate()

