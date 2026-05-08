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
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import anyio

from ..core.types import BudgetStatus


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
class _UserUsage:
    """Per-user-id bucket. Mirrors the global counters."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    started_at: datetime | None = None

    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class StandardBudget:
    """Hard-limited, thread-safe budget tracker with per-user
    accounting.

    Tracks usage globally AND per-user-id; either limit can fire.
    Multi-tenant production agents should pass ``user_id`` to every
    ``allows_step`` / ``consume`` call (the agent loop does this
    automatically from the live :class:`~jeevesagent.RunContext`).
    Single-tenant code can omit it; the framework treats unspecified
    user_id as the anonymous bucket.
    """

    def __init__(self, cfg: BudgetConfig | None = None) -> None:
        self._cfg = cfg or BudgetConfig()
        # Global counters.
        self._tokens_in = 0
        self._tokens_out = 0
        self._cost = 0.0
        self._started_at = datetime.now(UTC)
        # Per-user counters. ``defaultdict`` lazy-initialises a fresh
        # bucket on first reference; ``started_at`` is set on first
        # consume() so a user's wall-clock cap measures from their
        # first activity, not the Agent's birth.
        self._by_user: dict[str | None, _UserUsage] = defaultdict(_UserUsage)
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
            bucket = self._by_user[user_id]
            if bucket.started_at is None:
                bucket.started_at = datetime.now(UTC)
            bucket.tokens_in += tokens_in
            bucket.tokens_out += tokens_out
            bucket.cost += cost_usd

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
        return datetime.now(UTC) - self._started_at

    def _user_elapsed(self, user_id: str | None) -> timedelta | None:
        bucket = self._by_user.get(user_id)
        if bucket is None or bucket.started_at is None:
            return None
        return datetime.now(UTC) - bucket.started_at

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
