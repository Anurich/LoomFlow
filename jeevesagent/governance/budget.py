"""Token / call / cost budgets.

:class:`StandardBudget` enforces hard limits on tokens, cost, and
wall clock; emits a soft warning at a configurable threshold.
:class:`NoBudget` is the always-allow stub used when the user has
opted out of governance entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import anyio

from ..core.types import BudgetStatus


class NoBudget:
    """Never blocks, never warns."""

    async def allows_step(self) -> BudgetStatus:
        return BudgetStatus.ok_()

    async def consume(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        return None


@dataclass(slots=True)
class BudgetConfig:
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost_usd: float | None = None
    max_wall_clock: timedelta | None = None
    soft_warning_at: float = 0.8  # 80% triggers a warning


class StandardBudget:
    """Hard-limited, thread-safe budget tracker."""

    def __init__(self, cfg: BudgetConfig | None = None) -> None:
        self._cfg = cfg or BudgetConfig()
        self._tokens_in = 0
        self._tokens_out = 0
        self._cost = 0.0
        self._started_at = datetime.now(UTC)
        self._lock = anyio.Lock()

    async def allows_step(self) -> BudgetStatus:
        async with self._lock:
            blocked = self._first_block_reason()
            if blocked is not None:
                return BudgetStatus.blocked_(blocked)
            warn = self._first_warning_reason()
            if warn is not None:
                return BudgetStatus.warn_(warn)
            return BudgetStatus.ok_()

    async def consume(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        async with self._lock:
            self._tokens_in += tokens_in
            self._tokens_out += tokens_out
            self._cost += cost_usd

    # ---- helpers ---------------------------------------------------------

    def _total_tokens(self) -> int:
        return self._tokens_in + self._tokens_out

    def _elapsed(self) -> timedelta:
        return datetime.now(UTC) - self._started_at

    def _first_block_reason(self) -> str | None:
        c = self._cfg
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
        return None

    def _first_warning_reason(self) -> str | None:
        c = self._cfg
        threshold = c.soft_warning_at
        if c.max_tokens is not None and self._total_tokens() >= c.max_tokens * threshold:
            return f"tokens at {self._total_tokens() / c.max_tokens:.0%}"
        if c.max_cost_usd is not None and self._cost >= c.max_cost_usd * threshold:
            return f"cost at {self._cost / c.max_cost_usd:.0%}"
        return None
