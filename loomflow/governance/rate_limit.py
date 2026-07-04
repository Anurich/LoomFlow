"""Per-tenant QPS rate limiting (G5).

Cumulative budgets (:class:`~loomflow.governance.budget.StandardBudget`)
cap *total* spend but do nothing about burst rate — a tenant can fire
an arbitrary number of steps per second until the cumulative cap
trips. :class:`TokenBucketRateLimiter` closes that gap: a classic
token bucket, checked once per model step from the shared
:func:`~loomflow.architecture.helpers.budget_gate` choke point, so
every architecture is paced through ONE seam.

Two modes:

* ``"throttle"`` (default) — ``acquire`` waits (``anyio.sleep``)
  until a token is available, smoothing bursts into the configured
  rate instead of failing them.
* ``"raise"`` — ``acquire`` raises
  :class:`~loomflow.core.errors.RateLimitExceeded` immediately when
  the bucket is empty, for callers who'd rather surface "slow down"
  (e.g. propagate a 429 to their own clients) than queue work.

Per-user buckets live in a :class:`~loomflow.core._eviction
.BoundedDict` with the same LRU + idle-TTL bounds ``StandardBudget``
uses for its per-user accounting, so an adversarial stream of fresh
``user_id`` values can't grow the map without limit. Evicting a
bucket resets it to a full burst — appropriate for in-process state
where the alternative is unbounded growth.

Wiring: ``Agent(rate_limiter=TokenBucketRateLimiter(rps=5, burst=10))``.
Default is ``None`` — no limiter, zero overhead (the
``fast_rate_limit`` flag on :class:`~loomflow.architecture.base
.Dependencies` short-circuits the call site entirely).

Time source is ``anyio.current_time()`` throughout, so tests on a
mock clock and production on the monotonic clock both behave.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import anyio

from ..core._eviction import BoundedDict
from ..core.errors import RateLimitExceeded

__all__ = ["RateLimiter", "TokenBucketRateLimiter"]

# Same per-user map bounds StandardBudget uses (budget.py): LRU cap
# of 100k tenants, 24h idle TTL. Kept in lock-step deliberately —
# the rate-limit bucket map and the budget usage map face the same
# multi-tenant growth problem.
_DEFAULT_MAX_USERS = 100_000
_DEFAULT_USER_TTL_SECONDS = 24 * 3600  # 24h idle


@runtime_checkable
class RateLimiter(Protocol):
    """Structural type for per-step rate limiters.

    The agent loop calls :meth:`acquire` once before every model
    step (from ``budget_gate``), passing the run's ``user_id`` so
    per-tenant implementations can bucket independently. Implement
    this to plug in a distributed limiter (Redis, API gateway) in
    place of the in-process :class:`TokenBucketRateLimiter`.
    """

    async def acquire(self, *, user_id: str | None) -> None:
        """Take one permit; wait or raise when none are available."""
        ...


@dataclass(slots=True)
class _Bucket:
    """One tenant's token bucket. ``updated`` is anyio clock time."""

    tokens: float
    updated: float


class TokenBucketRateLimiter:
    """In-process token-bucket limiter with per-user buckets.

    ``rps`` is the sustained refill rate (tokens per second);
    ``burst`` is the bucket capacity — how many steps a quiet tenant
    may fire back-to-back before pacing kicks in. Each
    :meth:`acquire` consumes one token.

    ``per_user=True`` (default) keeps an independent bucket per
    ``user_id`` (the anonymous ``None`` id gets its own bucket);
    ``per_user=False`` shares a single global bucket across all
    callers.

    ``mode="throttle"`` (default) waits via ``anyio.sleep`` until a
    token accrues; ``mode="raise"`` raises
    :class:`~loomflow.core.errors.RateLimitExceeded` instead. Note
    the deliberate distinction from
    :class:`~loomflow.core.errors.RateLimitError` — that one is the
    *provider's* 429 surfacing through a model adapter; this one is
    the framework's own admission gate firing before any provider
    call is made.
    """

    def __init__(
        self,
        rps: float,
        burst: int,
        *,
        per_user: bool = True,
        mode: Literal["throttle", "raise"] = "throttle",
        max_users: int | None = _DEFAULT_MAX_USERS,
        user_idle_ttl_seconds: float | None = _DEFAULT_USER_TTL_SECONDS,
    ) -> None:
        if rps <= 0:
            raise ValueError("rps must be > 0")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        if mode not in ("throttle", "raise"):
            raise ValueError(
                f'mode must be "throttle" or "raise", got {mode!r}'
            )
        self._rps = float(rps)
        self._burst = float(burst)
        self._per_user = per_user
        self._mode: Literal["throttle", "raise"] = mode
        # Per-user buckets — bounded exactly like StandardBudget's
        # per-user usage map so a runaway tenant stream can't OOM
        # the process. BoundedDict holds no lock of its own; all
        # access goes through ``self._lock`` below.
        self._buckets: BoundedDict[str | None, _Bucket] = BoundedDict(
            max_keys=max_users,
            ttl_seconds=user_idle_ttl_seconds,
        )
        self._lock = anyio.Lock()

    @property
    def rps(self) -> float:
        return self._rps

    @property
    def burst(self) -> int:
        return int(self._burst)

    @property
    def mode(self) -> Literal["throttle", "raise"]:
        return self._mode

    async def acquire(self, *, user_id: str | None) -> None:
        """Consume one token from ``user_id``'s bucket.

        Returns immediately while tokens remain (up to ``burst``
        back-to-back). On an empty bucket: ``throttle`` mode sleeps
        until the refill rate frees a token (retrying under the
        lock, so concurrent waiters can't oversubscribe), ``raise``
        mode raises :class:`RateLimitExceeded` carrying the
        ``user_id`` and a retry-in hint.
        """
        key = user_id if self._per_user else None
        while True:
            async with self._lock:
                now = anyio.current_time()
                bucket = self._buckets.get(key)
                if bucket is None:
                    # Fresh (or evicted-and-reset) tenant: full burst.
                    bucket = _Bucket(tokens=self._burst, updated=now)
                    self._buckets[key] = bucket
                else:
                    # Lazy refill — accrue rps * elapsed, capped at
                    # burst.
                    bucket.tokens = min(
                        self._burst,
                        bucket.tokens + (now - bucket.updated) * self._rps,
                    )
                    bucket.updated = now
                if bucket.tokens >= 1.0:
                    bucket.tokens -= 1.0
                    return
                # Empty. Compute the wait for ONE token to accrue.
                wait = (1.0 - bucket.tokens) / self._rps
            if self._mode == "raise":
                raise RateLimitExceeded(
                    f"rate limit exceeded ({self._rps:g} rps, "
                    f"burst {int(self._burst)}); retry in {wait:.3f}s",
                    user_id=user_id,
                    retry_after=wait,
                )
            # Throttle: sleep outside the lock so other tenants'
            # buckets stay reachable, then re-contend. Multiple
            # waiters may wake together; the loop re-checks under
            # the lock so only as many proceed as tokens accrued.
            await anyio.sleep(wait)
