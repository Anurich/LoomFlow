"""Token / call / cost budgets.

:class:`StandardBudget` enforces hard limits on tokens, cost, and
wall clock; emits a soft warning at a configurable threshold.
:class:`NoBudget` is the always-allow stub used when the user has
opted out of governance entirely.

**Multi-tenant accounting (M9).** ``StandardBudget`` tracks usage
per-``user_id`` so one user can't exhaust another's quota. Pass
``per_user_max_tokens`` / ``per_user_max_cost_usd`` /
``per_user_max_wall_clock`` in the :class:`BudgetConfig` to enforce
per-user caps in addition to (or instead of) the global ones. The
agent loop forwards ``user_id`` from the live :class:`RunContext`
into every ``allows_step`` / ``consume`` call automatically;
direct callers pass it explicitly via the keyword.

**Wall clock.** ``max_wall_clock`` / ``per_user_max_wall_clock`` cap the
time spent *running* — the agent loop wraps each run in
:meth:`StandardBudget.running` — not the time since the budget was
created: an agent that sat idle between requests isn't blocked for it.
Concurrent or nested runs count once (the clock runs while at least one
is in progress). For a cap on a single run, use ``Agent(timeout=)``.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import anyio

from ..core._eviction import BoundedDict
from ..core.types import BudgetStatus

_DEFAULT_MAX_USERS = 100_000
_DEFAULT_USER_TTL_SECONDS = 24 * 3600  # 24h idle


class NoBudget:
    """Never blocks, never warns."""

    async def allows_step(
        self, *, user_id: str | None = None
    ) -> BudgetStatus:
        return BudgetStatus.ok_()

    async def consume(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        user_id: str | None = None,
    ) -> None:
        return None


@dataclass(slots=True)
class BudgetConfig:
    """Global + per-user budget caps.

    Every ``max_*`` field has a global counterpart and a
    ``per_user_*`` counterpart. The global cap applies to the whole
    Agent (all users combined); the per-user cap applies to each
    user_id's bucket independently. A run is blocked when *either*
    its user's cap or the global cap is exceeded — whichever fires
    first.

    Use one or both depending on what you want to enforce:

    * ``max_tokens=200_000`` — Agent-wide total. Caps the whole tenant.
    * ``per_user_max_tokens=10_000`` — Per user. Caps each user.
    * Both — one user can't hog the global, and the global stops
      runaway aggregate usage.

    The warning threshold (``soft_warning_at``) is shared across
    global and per-user caps.
    """

    # Global caps (apply to all users combined).
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost_usd: float | None = None
    max_wall_clock: timedelta | None = None

    # Per-user caps (apply to each user_id's bucket independently).
    per_user_max_tokens: int | None = None
    per_user_max_input_tokens: int | None = None
    per_user_max_output_tokens: int | None = None
    per_user_max_cost_usd: float | None = None
    per_user_max_wall_clock: timedelta | None = None

    soft_warning_at: float = 0.8  # 80% triggers a warning


@dataclass(slots=True)
class _Clock:
    """Time during which at least one run was in progress."""

    total: timedelta = timedelta(0)
    active: int = 0
    since: datetime | None = None

    def start(self, now: datetime) -> None:
        if self.active == 0:
            self.since = now
        self.active += 1

    def stop(self, now: datetime) -> None:
        self.active = max(0, self.active - 1)
        if self.active == 0 and self.since is not None:
            self.total += now - self.since
            self.since = None

    def elapsed(self, now: datetime) -> timedelta:
        return self.total + (now - self.since if self.since is not None else timedelta(0))


@dataclass(slots=True)
class _UserUsage:
    """Per-user-id bucket. Mirrors the global counters."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    clock: _Clock = field(default_factory=_Clock)

    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class StandardBudget:
    """Hard-limited, thread-safe budget tracker with per-user
    accounting.

    Tracks usage globally AND per-user-id; either limit can fire.
    Multi-tenant production agents should pass ``user_id`` to every
    ``allows_step`` / ``consume`` call (the agent loop does this
    automatically from the live :class:`~loomflow.RunContext`).
    Single-tenant code can omit it; the framework treats unspecified
    user_id as the anonymous bucket.
    """

    def __init__(
        self,
        cfg: BudgetConfig | None = None,
        *,
        max_users: int | None = _DEFAULT_MAX_USERS,
        user_idle_ttl_seconds: float | None = _DEFAULT_USER_TTL_SECONDS,
    ) -> None:
        self._cfg = cfg or BudgetConfig()
        # Global counters.
        self._tokens_in = 0
        self._tokens_out = 0
        self._cost = 0.0
        self._clock = _Clock()
        # Per-user counters. Bounded so a runaway tenant or
        # adversarial caller can't grow this dict without limit
        # (process OOM is the only ceiling otherwise). LRU evicts
        # the least-recently-touched user when ``max_users`` is
        # exceeded; idle TTL drops users who haven't consumed in
        # ``user_idle_ttl_seconds`` (default 24h). Evicting a
        # bucket *resets* that user's running totals — appropriate
        # for in-process accounting where the alternative is
        # unbounded growth. Pass ``max_users=None`` /
        # ``user_idle_ttl_seconds=None`` to disable bounding for
        # single-tenant or small fixed-tenant deployments.
        self._by_user: BoundedDict[str | None, _UserUsage] = BoundedDict(
            max_keys=max_users,
            ttl_seconds=user_idle_ttl_seconds,
        )
        self._lock = anyio.Lock()

    async def allows_step(
        self, *, user_id: str | None = None
    ) -> BudgetStatus:
        async with self._lock:
            blocked = self._first_block_reason(user_id)
            if blocked is not None:
                return BudgetStatus.blocked_(blocked)
            warn = self._first_warning_reason(user_id)
            if warn is not None:
                return BudgetStatus.warn_(warn)
            return BudgetStatus.ok_()

    async def consume(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        user_id: str | None = None,
    ) -> None:
        async with self._lock:
            self._tokens_in += tokens_in
            self._tokens_out += tokens_out
            self._cost += cost_usd
            bucket = self._by_user.setdefault(user_id, _UserUsage())
            bucket.tokens_in += tokens_in
            bucket.tokens_out += tokens_out
            bucket.cost += cost_usd

    @contextlib.asynccontextmanager
    async def running(self, *, user_id: str | None = None) -> AsyncIterator[None]:
        """Count the enclosed run's time toward the wall-clock caps.

        The agent loop wraps every run in this; call it yourself when
        driving ``allows_step`` / ``consume`` directly."""
        async with self._lock:
            now = datetime.now(UTC)
            self._clock.start(now)
            self._by_user.setdefault(user_id, _UserUsage()).clock.start(now)
        try:
            yield
        finally:
            with anyio.CancelScope(shield=True):
                async with self._lock:
                    now = datetime.now(UTC)
                    self._clock.stop(now)
                    bucket = self._by_user.get(user_id)
                    if bucket is not None:
                        bucket.clock.stop(now)

    # ---- introspection (test + ops helper) ------------------------------

    def usage_for(self, user_id: str | None) -> dict[str, float]:
        """Snapshot one user's running totals — for telemetry / ops
        dashboards. Returns an empty bucket for a user who hasn't
        consumed anything yet."""
        bucket = self._by_user.get(user_id, _UserUsage())
        return {
            "tokens_in": bucket.tokens_in,
            "tokens_out": bucket.tokens_out,
            "tokens_total": bucket.total_tokens(),
            "cost_usd": bucket.cost,
        }

    # ---- helpers ---------------------------------------------------------

    def _total_tokens(self) -> int:
        return self._tokens_in + self._tokens_out

    def _elapsed(self) -> timedelta:
        return self._clock.elapsed(datetime.now(UTC))

    def _user_elapsed(self, user_id: str | None) -> timedelta | None:
        bucket = self._by_user.get(user_id)
        if bucket is None:
            return None
        return bucket.clock.elapsed(datetime.now(UTC))

    def _first_block_reason(self, user_id: str | None) -> str | None:
        c = self._cfg

        # Global caps first.
        if c.max_tokens is not None and self._total_tokens() >= c.max_tokens:
            return "max_tokens"
        if c.max_input_tokens is not None and self._tokens_in >= c.max_input_tokens:
            return "max_input_tokens"
        if c.max_output_tokens is not None and self._tokens_out >= c.max_output_tokens:
            return "max_output_tokens"
        if c.max_cost_usd is not None and self._cost >= c.max_cost_usd:
            return "max_cost_usd"
        if c.max_wall_clock is not None and self._elapsed() >= c.max_wall_clock:
            return "max_wall_clock"

        # Per-user caps.
        u = self._by_user.get(user_id)
        if u is None:
            return None
        if (
            c.per_user_max_tokens is not None
            and u.total_tokens() >= c.per_user_max_tokens
        ):
            return "per_user_max_tokens"
        if (
            c.per_user_max_input_tokens is not None
            and u.tokens_in >= c.per_user_max_input_tokens
        ):
            return "per_user_max_input_tokens"
        if (
            c.per_user_max_output_tokens is not None
            and u.tokens_out >= c.per_user_max_output_tokens
        ):
            return "per_user_max_output_tokens"
        if (
            c.per_user_max_cost_usd is not None
            and u.cost >= c.per_user_max_cost_usd
        ):
            return "per_user_max_cost_usd"
        if c.per_user_max_wall_clock is not None:
            elapsed = self._user_elapsed(user_id)
            if elapsed is not None and elapsed >= c.per_user_max_wall_clock:
                return "per_user_max_wall_clock"
        return None

    def _first_warning_reason(self, user_id: str | None) -> str | None:
        c = self._cfg
        threshold = c.soft_warning_at

        # Global warnings.
        if c.max_tokens is not None and self._total_tokens() >= c.max_tokens * threshold:
            return f"tokens at {self._total_tokens() / c.max_tokens:.0%}"
        if c.max_cost_usd is not None and self._cost >= c.max_cost_usd * threshold:
            return f"cost at {self._cost / c.max_cost_usd:.0%}"

        # Per-user warnings.
        u = self._by_user.get(user_id)
        if u is None:
            return None
        if (
            c.per_user_max_tokens is not None
            and u.total_tokens() >= c.per_user_max_tokens * threshold
        ):
            return (
                f"per-user tokens at "
                f"{u.total_tokens() / c.per_user_max_tokens:.0%}"
            )
        if (
            c.per_user_max_cost_usd is not None
            and u.cost >= c.per_user_max_cost_usd * threshold
        ):
            return f"per-user cost at {u.cost / c.per_user_max_cost_usd:.0%}"
        return None


def run_clock(budget: Any, *, user_id: str | None) -> contextlib.AbstractAsyncContextManager[Any]:
    """``budget.running(user_id=...)`` when the budget keeps a run
    clock, else a no-op context."""
    running = getattr(budget, "running", None)
    if running is None:
        return contextlib.nullcontext()
    cm: contextlib.AbstractAsyncContextManager[Any] = running(user_id=user_id)
    return cm
