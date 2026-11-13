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

**Concurrency (reservations).** A cap is a check *before* a model call
and a charge *after* it, so on its own it is check-then-act: fifty
concurrent runs would all pass the check before any of them is
charged. Inside :meth:`StandardBudget.running` (which the agent loop
always enters) ``allows_step`` therefore *reserves* an estimate of the
coming call under the same lock as the check, so concurrent callers
see each other; ``consume`` swaps the reservation for the real usage,
and leaving ``running`` releases it if the call never reported (it
failed or was cancelled). The estimate is the largest recent call;
until one call has completed, the first caller holds the remaining
headroom exclusively. A caller that finds the headroom fully reserved
waits for those calls to settle (up to ``reserve_wait_seconds``) and
is then allowed or blocked on the real numbers — so a cap is exceeded
by at most about one in-flight call, never by one per concurrent run.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import AsyncIterator
from contextvars import ContextVar
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


@dataclass(slots=True)
class _Amounts:
    """One model call's size along each capped dimension.

    As a reservation, ``math.inf`` means "size unknown — hold the
    whole remaining headroom"."""

    tokens_in: float = 0.0
    tokens_out: float = 0.0
    tokens: float = 0.0
    cost: float = 0.0


@dataclass(slots=True, eq=False)
class _RunScope:
    """One ``StandardBudget.running()`` block: the unit that can hold
    a reservation. ``parent`` is the enclosing run (any budget), so a
    sub-agent sharing its parent's budget doesn't wait on the
    reservation its own parent is holding for it."""

    budget: StandardBudget
    parent: _RunScope | None
    user_id: str | None = None
    reserved: _Amounts | None = None


_RUN_SCOPE: ContextVar[_RunScope | None] = ContextVar(
    "loomflow_budget_run_scope", default=None
)

_DEFAULT_RESERVE_WAIT_SECONDS = 60.0
# How fast the per-call estimate forgets an unusually large call.
_ESTIMATE_DECAY = 0.9


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
        reserve_wait_seconds: float = _DEFAULT_RESERVE_WAIT_SECONDS,
    ) -> None:
        self._cfg = cfg or BudgetConfig()
        # In-flight reservations (see the module docstring): the run
        # scopes currently holding one, the recent per-call size used
        # as the estimate (``None`` until a call has completed), and
        # the event waiters park on until a reservation settles.
        self._reserve_wait = reserve_wait_seconds
        self._holders: set[_RunScope] = set()
        self._estimate: _Amounts | None = None
        self._settled: anyio.Event | None = None
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
        scope = self._current_scope()
        deadline: float | None = None
        while True:
            async with self._lock:
                if scope is not None:
                    # One reservation per run: asking again replaces
                    # whatever the previous step left outstanding.
                    self._release_locked(scope)
                blocked = self._first_block_reason(user_id)
                if blocked is not None:
                    return BudgetStatus.blocked_(blocked)
                contended = (
                    self._contended_reason(user_id, scope)
                    if scope is not None
                    else None
                )
                if contended is None:
                    if scope is not None:
                        # Check + reserve under ONE lock hold, so the
                        # next concurrent caller sees this call coming.
                        self._reserve_locked(scope, user_id)
                    warn = self._first_warning_reason(user_id)
                    if warn is not None:
                        return BudgetStatus.warn_(warn)
                    return BudgetStatus.ok_()
                if self._settled is None:
                    self._settled = anyio.Event()
                settled = self._settled
            # The remaining headroom is fully spoken for by calls that
            # are in flight. Wait for one to settle, then decide on
            # the real numbers (bounded, so a stuck holder can't park
            # this caller forever).
            now = anyio.current_time()
            if deadline is None:
                deadline = now + self._reserve_wait
            if now >= deadline:
                return BudgetStatus.blocked_(contended)
            with anyio.move_on_after(deadline - now):
                await settled.wait()

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
            # Reconcile: the real usage is on the books, so this run's
            # reservation (if it holds one) is no longer needed.
            call = _Amounts(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tokens=tokens_in + tokens_out,
                cost=cost_usd,
            )
            prev = self._estimate
            self._estimate = call if prev is None else _Amounts(
                tokens_in=max(call.tokens_in, prev.tokens_in * _ESTIMATE_DECAY),
                tokens_out=max(call.tokens_out, prev.tokens_out * _ESTIMATE_DECAY),
                tokens=max(call.tokens, prev.tokens * _ESTIMATE_DECAY),
                cost=max(call.cost, prev.cost * _ESTIMATE_DECAY),
            )
            scope = self._current_scope()
            if scope is not None:
                self._release_locked(scope)

    @contextlib.asynccontextmanager
    async def running(self, *, user_id: str | None = None) -> AsyncIterator[None]:
        """Count the enclosed run's time toward the wall-clock caps,
        and make its ``allows_step`` checks atomic.

        Inside this block ``allows_step`` reserves the coming call's
        estimated size so concurrent runs can't all pass the same
        check (see the module docstring); the reservation is released
        by ``consume`` or, if the call never reports, on exit.

        The agent loop wraps every run in this; call it yourself when
        driving ``allows_step`` / ``consume`` directly."""
        scope = _RunScope(budget=self, parent=_RUN_SCOPE.get(), user_id=user_id)
        token = _RUN_SCOPE.set(scope)
        async with self._lock:
            now = datetime.now(UTC)
            self._clock.start(now)
            self._by_user.setdefault(user_id, _UserUsage()).clock.start(now)
        try:
            yield
        finally:
            # Exit may run in another context (a stream finalised
            # elsewhere); the scope is released explicitly below, so a
            # token that can't be reset is harmless.
            with contextlib.suppress(ValueError):
                _RUN_SCOPE.reset(token)
            with anyio.CancelScope(shield=True):
                async with self._lock:
                    # Release on failure / cancellation: a call that
                    # never reached ``consume`` must not keep its
                    # reservation.
                    self._release_locked(scope)
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

    # ---- reservations ----------------------------------------------------

    def _current_scope(self) -> _RunScope | None:
        """The innermost enclosing ``running()`` block of THIS budget."""
        scope = _RUN_SCOPE.get()
        while scope is not None and scope.budget is not self:
            scope = scope.parent
        return scope

    def _reserve_locked(self, scope: _RunScope, user_id: str | None) -> None:
        estimate = self._estimate
        scope.user_id = user_id
        scope.reserved = (
            _Amounts(math.inf, math.inf, math.inf, math.inf)
            if estimate is None
            else _Amounts(
                estimate.tokens_in, estimate.tokens_out, estimate.tokens, estimate.cost
            )
        )
        self._holders.add(scope)

    def _release_locked(self, scope: _RunScope) -> None:
        if scope.reserved is None:
            return
        scope.reserved = None
        self._holders.discard(scope)
        # Wake everyone waiting on the headroom this call was holding.
        if self._settled is not None:
            self._settled.set()
            self._settled = None

    def _reserved_by_others(
        self, scope: _RunScope
    ) -> tuple[_Amounts, dict[str | None, _Amounts]]:
        """Reservations held by other runs: the total, and per user.

        Ancestors of ``scope`` are skipped — a parent run that is
        waiting on this (sub-agent) run holds its reservation FOR it.
        """
        ancestors: set[int] = set()
        node = scope.parent
        while node is not None:
            ancestors.add(id(node))
            node = node.parent
        total = _Amounts()
        by_user: dict[str | None, _Amounts] = {}
        for holder in self._holders:
            held = holder.reserved
            if held is None or holder is scope or id(holder) in ancestors:
                continue
            mine = by_user.setdefault(holder.user_id, _Amounts())
            for target in (total, mine):
                target.tokens_in += held.tokens_in
                target.tokens_out += held.tokens_out
                target.tokens += held.tokens
                target.cost += held.cost
        return total, by_user

    def _contended_reason(
        self, user_id: str | None, scope: _RunScope
    ) -> str | None:
        """Name of the first cap whose remaining headroom is entirely
        covered by OTHER runs' in-flight reservations, else ``None``.

        Called only after ``_first_block_reason`` passed, so real
        usage alone is under every cap."""
        if not self._holders:
            return None
        c = self._cfg
        total, by_user = self._reserved_by_others(scope)
        u = self._by_user.get(user_id) or _UserUsage()
        mine = by_user.get(user_id) or _Amounts()
        checks: tuple[tuple[str, float | None, float], ...] = (
            ("max_tokens", c.max_tokens, self._total_tokens() + total.tokens),
            ("max_input_tokens", c.max_input_tokens, self._tokens_in + total.tokens_in),
            ("max_output_tokens", c.max_output_tokens, self._tokens_out + total.tokens_out),
            ("max_cost_usd", c.max_cost_usd, self._cost + total.cost),
            ("per_user_max_tokens", c.per_user_max_tokens, u.total_tokens() + mine.tokens),
            (
                "per_user_max_input_tokens",
                c.per_user_max_input_tokens,
                u.tokens_in + mine.tokens_in,
            ),
            (
                "per_user_max_output_tokens",
                c.per_user_max_output_tokens,
                u.tokens_out + mine.tokens_out,
            ),
            ("per_user_max_cost_usd", c.per_user_max_cost_usd, u.cost + mine.cost),
        )
        for name, cap, committed in checks:
            if cap is not None and committed >= cap:
                return name
        return None

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
