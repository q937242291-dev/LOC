"""Provider transport, source visibility and charged recipient conversations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .config import ModelSettings, Settings
from .ledger import Ledger
from .types import Observation, compact


class ModelError(RuntimeError):
    pass


class TokenCounter:
    def __init__(self, settings: Settings):
        self.tokenizer = None
        if settings.tokenizer_model:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.tokenizer_model, trust_remote_code=False)
        else:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(settings.encoding)

    def __call__(self, text: str) -> int:
        if hasattr(self.tokenizer, "encode"):
            try:
                return len(self.tokenizer.encode(text, disallowed_special=()))
            except TypeError:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
        raise TypeError("Tokenizer does not implement encode")

    def truncate(self, text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if self(text) <= limit:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1)//2
            if self(text[:mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]


@dataclass
class Message:
    role: str
    content: str
    source_id: str | None = None
    source_version: str | None = None


class Recipient:
    """Visibility changes only on completed delivery or ordered eviction."""
    def __init__(self, system: str, count, cap: int, memory=True):
        self.system = system
        self.count = count
        self.cap = cap
        self.memory = memory
        self.messages: list[Message] = []
        self.visible: dict[str, str] = {}
        self.last_request_versions: frozenset[tuple[str, str]] = frozenset()
        self.transmitted_tokens = 0
        self.repeated_tokens = 0
        self.ever_delivered: set[str] = set()
        self.preserve_empty_sources = False
        self.count_wire_bytes = False

    def reset(self) -> None:
        """Begin an independent attempt without retaining prior source context."""
        self.messages.clear()
        self.visible.clear()
        self.last_request_versions = frozenset()
        self.transmitted_tokens = 0
        self.repeated_tokens = 0
        self.ever_delivered.clear()
        self.preserve_empty_sources = False

    def input_size(self, extra: str = "") -> int:
        messages = [{"role": "system", "content": self.system}]
        messages += [dict(role=m.role, content=m.content) for m in self.messages]
        if extra:
            messages.append(dict(role="user", content=extra))
        # JSON framing plus a per-message allowance includes role/protocol overhead.
        serialized = compact(messages)
        payload_size = self.count(serialized)
        if self.count_wire_bytes:
            payload_size = max(payload_size,len(serialized.encode("utf-8")))
        return payload_size + 64*(len(messages)+1)

    def evict(self, index: int):
        old = self.messages.pop(index)
        if old.source_id is not None:
            remaining = next((m for m in reversed(self.messages)
                              if m.source_id == old.source_id), None)
            if remaining is None:
                self.visible.pop(old.source_id, None)
            else:
                self.visible[old.source_id] = remaining.source_version

    def fit(self, extra="", protected=frozenset()):
        while self.input_size(extra) > self.cap:
            index = next((i for i,m in enumerate(self.messages)
                          if m.source_id not in protected), None)
            if index is None:
                raise ModelError("Required context exceeds the request-input cap")
            self.evict(index)

    def deliver(self, observations, *, extra="", deduplicate=True,
                protected=frozenset()) -> bool:
        observations = tuple(observations)
        snapshot = (list(self.messages), dict(self.visible), self.transmitted_tokens,
                    self.repeated_tokens, set(self.ever_delivered))
        try:
            if not observations and self.preserve_empty_sources:
                self.fit(extra, protected | frozenset(
                    m.source_id for m in self.messages if m.source_id))
                return True
            if not self.memory:
                for i in reversed(range(len(self.messages))):
                    if self.messages[i].source_id:
                        self.evict(i)
            for s in observations:
                if deduplicate and self.has(s):
                    continue
                if deduplicate:
                    for i in reversed(range(len(self.messages))):
                        if self.messages[i].source_id == s.id:
                            self.evict(i)
                self.messages.append(Message("user", "SOURCE\n" + s.serialize(), s.id, s.version))
                self.visible[s.id] = s.version
                self.transmitted_tokens += s.tokens
                if s.id in self.ever_delivered:
                    self.repeated_tokens += s.tokens
                self.ever_delivered.add(s.id)
            self.fit(extra, protected | frozenset(s.id for s in observations))
            return True
        except ModelError:
            (self.messages, self.visible, self.transmitted_tokens,
             self.repeated_tokens, self.ever_delivered) = snapshot
            if not observations and self.preserve_empty_sources:
                raise
            return False

    def has(self, source: Observation) -> bool:
        return any(m.source_id == source.id and m.source_version == source.version
                   for m in self.messages)

    def consumed(self, source: Observation) -> bool:
        return (source.id, source.version) in self.last_request_versions

    def request_messages(self, prompt: str, protected=frozenset()) -> list[dict]:
        self.fit(prompt, protected)
        self.last_request_versions = frozenset(
            (m.source_id, m.source_version) for m in self.messages
            if m.source_id is not None and m.source_version is not None)
        return [dict(role="system", content=self.system)] + [
            dict(role=m.role, content=m.content) for m in self.messages
        ] + [dict(role="user", content=prompt)]

    def complete(self, prompt: str, answer: str):
        self.messages.extend([Message("user", prompt), Message("assistant", answer)])


def usage_counts(usage: dict, provider: str) -> tuple[int, int, int]:
    """Normalize provider-reported uncached input, cached input and output counts."""
    if not isinstance(usage, dict):
        raise ModelError("Provider usage must be an object")

    def count(*names, required=False):
        values = [usage[name] for name in names if name in usage]
        if not values:
            if required:
                raise ModelError("Provider omitted required token usage")
            return 0
        if any(type(value) is not int or value < 0 for value in values):
            raise ModelError("Provider token usage must contain nonnegative integers")
        if any(value != values[0] for value in values[1:]):
            raise ModelError("Provider token usage contains inconsistent totals")
        return values[0]

    if provider == "anthropic":
        uncached = count("input_tokens", required=True) + count("cache_creation_input_tokens")
        cached = count("cache_read_input_tokens")
        output = count("output_tokens", required=True)
    else:
        cached_values = []
        for name in ("prompt_tokens_details", "input_tokens_details"):
            details = usage.get(name)
            if details is None:
                continue
            if not isinstance(details, dict):
                raise ModelError("Provider token details must be an object")
            if "cached_tokens" in details:
                cached_values.append(details["cached_tokens"])
        if "prompt_cache_hit_tokens" in usage:
            cached_values.append(count("prompt_cache_hit_tokens"))
        if any(type(value) is not int or value < 0 for value in cached_values):
            raise ModelError("Provider cached usage must be a nonnegative integer")
        if any(value != cached_values[0] for value in cached_values[1:]):
            raise ModelError("Provider cached usage contains inconsistent totals")
        cached = cached_values[0] if cached_values else 0
        prompt = count("prompt_tokens", "input_tokens", required=True)
        if cached > prompt:
            raise ModelError("Cached input exceeds total provider input")
        uncached = count("prompt_cache_miss_tokens") if "prompt_cache_miss_tokens" in usage else prompt-cached
        if uncached + cached != prompt:
            raise ModelError("Provider input categories do not partition total input")
        output = count("completion_tokens", "output_tokens", required=True)
    return uncached, cached, output


class HTTPModel:
    def __init__(self, model: ModelSettings, settings: Settings, ledger: Ledger, count):
        self.model, self.settings, self.ledger, self.count = model.validate(), settings, ledger, count
        self.sessions: dict[str, Recipient] = {}
        self.role_calls: dict[str, int] = {}
        self.request_metadata: list[dict] = []
        u = urlsplit(model.base_url)
        if u.username or u.password or u.query or u.fragment:
            raise ValueError("Model endpoint must not contain credentials or query parameters")
        if u.scheme not in {"https", "http"} or not u.hostname:
            raise ValueError("Invalid model endpoint")
        if u.scheme == "http" and u.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use HTTPS for remote model endpoints")

    def session(self, role: str, system: str) -> Recipient:
        if role not in self.sessions:
            self.sessions[role] = Recipient(system, self.count, self.settings.request_input_tokens,
                                            self.settings.memory)
            self.sessions[role].count_wire_bytes = self.model.provider == "anthropic"
        return self.sessions[role]

    def ask(self, role: str, system: str, prompt: str, sources=()) -> dict:
        session = self.session(role, system)
        if not session.deliver(sources, extra=prompt):
            raise ModelError("Selected source does not fit the recipient request")
        messages = session.request_messages(prompt, frozenset(s.id for s in sources))
        cfg, settings = self.model, self.settings
        key = os.environ.get(cfg.api_key_env, "")
        if not key and urlsplit(cfg.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ModelError(f"Set the {cfg.api_key_env} environment variable")
        headers = {"Content-Type": "application/json"}
        if cfg.provider == "anthropic":
            system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
            sequence = []
            for m in messages:
                if m["role"] == "system":
                    continue
                if sequence and sequence[-1]["role"] == m["role"]:
                    sequence[-1]["content"] += "\n\n" + m["content"]
                else:
                    sequence.append(dict(m))
            # A breakpoint on the final input block enables provider caching.
            if sequence:
                sequence[-1]["content"] = [
                    {"type": "text", "text": sequence[-1]["content"],
                     "cache_control": {"type": "ephemeral"}}]
            body = dict(model=cfg.model, system=system_text, messages=sequence,
                        max_tokens=settings.output_tokens, temperature=settings.temperature)
            # Sonnet 4.5 accepts one sampling control. Prefer the configured
            # temperature when top_p is the paper's reference value.
            if settings.top_p != 1.0:
                body.pop("temperature")
                body["top_p"] = settings.top_p
            headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
            base = cfg.base_url.rstrip("/")
            url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        else:
            body = dict(model=cfg.model, messages=messages, temperature=settings.temperature,
                        top_p=settings.top_p, response_format={"type": "json_object"})
            if cfg.provider == "openai":
                body["max_completion_tokens"] = settings.output_tokens
            else:
                body["max_tokens"] = settings.output_tokens
            headers["Authorization"] = "Bearer " + key
            base = cfg.base_url.rstrip("/")
            url = base + "/chat/completions"
        protected_fields = {"model", "messages", "system", "max_tokens", "max_completion_tokens",
                            "temperature", "top_p", "seed"}
        if protected_fields.intersection(cfg.extra_body):
            raise ValueError("extra_body cannot replace model, context, output limits, or sampling settings")
        body.update(cfg.extra_body)
        request_seed = None
        ordinal = self.role_calls.get(role, 0)
        if cfg.seed_mode == "request":
            # Role-local numbering keeps common requests aligned when another
            # role makes extra calls in a component comparison.
            encoded = compact(["locus-request-seed-v1", settings.seed, role, ordinal])
            request_seed = int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:4], "big") & 0x7fffffff
            body["seed"] = request_seed
        metadata = dict(role=role, ordinal=ordinal, requested_model=cfg.model,
                        requested_temperature=settings.temperature, requested_top_p=settings.top_p,
                        sent_temperature=body.get("temperature"), sent_top_p=body.get("top_p"),
                        seed_mode=cfg.seed_mode, requested_seed=request_seed,
                        served_model=None, system_fingerprint=None, response_received=False)
        raw = None
        reserve = role.startswith("debate:")
        with self.ledger.action("model", role, reserve=reserve) as step:
            self.role_calls[role] = ordinal + 1
            self.request_metadata.append(metadata)
            step.usage_known = False
            try:
                timeout = min(cfg.timeout_seconds, self.ledger.allowance(reserve=reserve)[1])
                request = urllib.request.Request(url, compact(body).encode("utf-8"), headers)
                with urllib.request.urlopen(request, timeout=max(0.001, timeout)) as response:
                    raw = json.load(response)
            except urllib.error.HTTPError as exc:
                # Some providers return billable usage with an error response.
                try:
                    error_payload = json.loads(exc.read())
                    if "usage" in error_payload:
                        step.uncached_input, step.cached_input, step.output = usage_counts(
                            error_payload["usage"], cfg.provider)
                        step.usage_known = True
                except (ValueError, TypeError, ModelError):
                    pass
                raise ModelError(f"Model request failed (HTTP {exc.code})") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                raise ModelError("Model request failed (connection or timeout)") from None
            except (json.JSONDecodeError, UnicodeError, TypeError):
                raise ModelError("Model endpoint returned an invalid JSON response") from None
            if not isinstance(raw, dict):
                raise ModelError("Model endpoint response must be an object")
            metadata["response_received"] = True
            for field in ("model", "system_fingerprint"):
                value = raw.get(field)
                if isinstance(value, str):
                    metadata["served_model" if field == "model" else field] = value
            if "usage" not in raw:
                raise ModelError("Model endpoint must return token usage")
            step.uncached_input, step.cached_input, step.output = usage_counts(raw["usage"], cfg.provider)
            step.usage_known = True
            if step.uncached_input + step.cached_input > settings.request_input_tokens:
                raise ModelError("Provider input exceeded the request-input cap")
            if step.output > settings.output_tokens:
                raise ModelError("Provider output exceeded the output-token ceiling")
            if cfg.provider == "anthropic":
                content = raw.get("content")
                if not isinstance(content,list) or any(not isinstance(c,dict) for c in content):
                    raise ModelError("Model endpoint returned malformed content blocks")
                texts = [c.get("text") for c in content if c.get("type") == "text"]
                if any(not isinstance(text,str) for text in texts):
                    raise ModelError("Model text content must be a string")
                answer = "\n".join(texts)
                truncated = raw.get("stop_reason") == "max_tokens"
            else:
                choices = raw.get("choices", [])
                if not isinstance(choices,list) or not choices or not isinstance(choices[0],dict):
                    raise ModelError("Model returned no completion")
                message = choices[0].get("message")
                if not isinstance(message,dict) or not isinstance(message.get("content"),str):
                    raise ModelError("Model endpoint returned a malformed completion")
                answer = message["content"]
                truncated = choices[0].get("finish_reason") == "length"
            session.complete(prompt, answer)
            if truncated:
                raise ModelError("Model response reached its output ceiling")
            match = re.fullmatch(r"\s*(?:\x60{3}(?:json)?\s*)?([\s\S]*?)(?:\s*\x60{3})?\s*", answer)
            try:
                result = json.loads(match.group(1) if match else answer)
            except (json.JSONDecodeError, TypeError):
                raise ModelError("Model response must be a JSON object") from None
            if not isinstance(result, dict):
                raise ModelError("Model response must be a JSON object")
            return result

