"""Complete-task accounting, repair reserves and Eq. (5)/(8)."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time

from .config import Settings, Profile


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Step:
    kind: str
    role: str
    seconds: float = 0.0
    uncached_input: int = 0
    cached_input: int = 0
    output: int = 0
    success: bool | None = None
    usage_known: bool = True
    phase: str = "setup"


class Ledger:
    def __init__(self, settings: Settings, clock=time.monotonic):
        self.settings = settings
        self.clock = clock
        self.started = clock()
        self.ended: float | None = None
        self.steps: list[Step] = []
        self.phase = "setup"
        self.acq_call_end = settings.total_calls
        self.acq_time_end = settings.deadline_seconds
        self.additional_worker_hours = 0.0

    @property
    def elapsed(self) -> float:
        return (self.ended if self.ended is not None else self.clock()) - self.started

    def restrict(self, profile: Profile):
        """Eq. (8): profile allowance starts after structure and debate charges."""
        s = self.settings
        spent_seconds = self.elapsed
        a = max(0, min(profile.calls, s.total_calls - len(self.steps) - s.reserve_calls))
        seconds = max(0, min(s.deadline_seconds, s.deadline_seconds - spent_seconds - s.reserve_seconds))
        self.acq_call_end = len(self.steps) + a
        self.acq_time_end = spent_seconds + seconds
        self.phase = "acquisition"

    def allowance(self, *, reserve=False) -> tuple[int, float]:
        if self.ended is not None:
            return 0, 0.0
        s = self.settings
        calls, seconds = s.total_calls - len(self.steps), s.deadline_seconds - self.elapsed
        if self.phase == "acquisition":
            calls = min(calls, self.acq_call_end - len(self.steps))
            seconds = min(seconds, self.acq_time_end - self.elapsed)
        elif reserve:
            calls -= s.reserve_calls
            seconds -= s.reserve_seconds
        return max(0, calls), max(0.0, seconds)

    def check(self, *, reserve=False):
        calls, seconds = self.allowance(reserve=reserve)
        if calls < 1 or seconds <= 0:
            raise BudgetExhausted("Resource allowance exhausted")

    @contextmanager
    def action(self, kind: str, role: str, *, reserve=False):
        self.check(reserve=reserve)
        time_limit = self.acq_time_end if self.phase == "acquisition" \
            else self.settings.deadline_seconds - (self.settings.reserve_seconds if reserve else 0)
        step = Step(kind, role, phase=self.phase)
        self.steps.append(step)  # Failed calls consume their allowance too.
        start = self.clock()
        try:
            yield step
            if step.success is None:
                step.success = True
        except BaseException:
            if step.success is None:
                step.success = False
            raise
        finally:
            step.seconds = max(0.0, self.clock() - start)
        if self.elapsed > time_limit:
            raise BudgetExhausted("Action completed beyond its time allowance")

    def repair_phase(self):
        self.phase = "repair"

    def finish(self):
        if self.ended is None:
            self.ended = self.clock()

    def summary(self) -> dict:
        s = self.settings
        elapsed = self.elapsed
        u = sum(x.uncached_input for x in self.steps if x.usage_known)
        k = sum(x.cached_input for x in self.steps if x.usage_known)
        o = sum(x.output for x in self.steps if x.usage_known)
        hours = elapsed / 3600 + self.additional_worker_hours
        complete = all(x.usage_known for x in self.steps)
        known_cost = 1e-6*(s.uncached_price*u + s.cached_price*k + s.output_price*o) \
            + s.worker_hour_price*hours
        exploration = [step for step in self.steps if step.phase in {"setup", "acquisition"}]
        return dict(
            usage_complete=complete,
            uncached_input=u if complete else None, cached_input=k if complete else None,
            output=o if complete else None, tokens=u+k+o if complete else None,
            worker_hours=hours,
            cost=known_cost if complete else None,
            known_uncached_input=u, known_cached_input=k, known_output=o,
            known_cost=known_cost,
            elapsed_seconds=elapsed, calls=len(self.steps),
            llm_seconds=sum(x.seconds for x in self.steps if x.kind == "model"),
            tool_seconds=sum(x.seconds for x in self.steps if x.kind == "tool"),
            exploration_calls=len(exploration),
            exploration_model_calls=sum(x.kind == "model" for x in exploration),
            exploration_tool_calls=sum(x.kind == "tool" for x in exploration),
            exploration_llm_seconds=sum(x.seconds for x in exploration if x.kind == "model"),
            exploration_tool_seconds=sum(x.seconds for x in exploration if x.kind == "tool"),
            cached_input_share=k/(u+k) if u+k and complete else None,
            steps=[dict(kind=x.kind, role=x.role, phase=x.phase, seconds=x.seconds,
                        uncached_input=x.uncached_input, cached_input=x.cached_input,
                        output=x.output, success=x.success,usage_known=x.usage_known)
                   for x in self.steps],
        )

