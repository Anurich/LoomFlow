"""29_resilience_governance.py — model fallback chains + per-tenant rate limiting.

Two production-resilience primitives that shipped in v0.11:

**FallbackModel** — an ordered chain of models. When the primary fails
with a *fallback-worthy* error (rate limit, overload, transient network
— never auth errors or content-filter refusals, which rerouting would
mask or bypass), the chain advances to the next model. Streaming fails
over only *before the first chunk*: once tokens have reached the
caller, a mid-stream error propagates rather than silently switching
voices mid-answer. Compose with retries so each model exhausts its own
retry budget first::

    FallbackModel([RetryingModel(primary), RetryingModel(backup)])

Pair it with per-request wall clocks — every adapter now accepts
``request_timeout_s`` so a hung SSE stream is killed instead of
blocking forever::

    AnthropicModel("claude-...", request_timeout_s=60)

**TokenBucketRateLimiter** — per-``user_id`` QPS pacing. Budgets cap
*cumulative* spend; the rate limiter caps *burst rate*. Each user gets
an independent token bucket (LRU+TTL bounded, same discipline as
``StandardBudget``), so one tenant hammering the loop cannot starve
another. ``mode="throttle"`` waits; ``mode="raise"`` raises
``RateLimitExceeded`` (distinct from the provider-429
``RateLimitError``)::

    Agent(..., rate_limiter=TokenBucketRateLimiter(rps=5, burst=10))

This example runs OFFLINE (no API key): the "primary" is a tiny fake
model that always raises ``RateLimitError``, the "backup" answers, and
the limiter demo paces direct ``acquire()`` calls with a stopwatch.

Run with::

    python examples/29_resilience_governance.py
"""

from __future__ import annotations

import time
from typing import Any

import anyio

from loomflow import Agent, TokenBucketRateLimiter, Usage
from loomflow.core import RateLimitError
from loomflow.model import FallbackModel


class _RateLimitedModel:
    """Fake primary: every call fails with a provider 429."""

    name = "primary-under-pressure"

    def __init__(self) -> None:
        self.attempts = 0

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.attempts += 1
        raise RateLimitError("429: slow down")


class _BackupModel:
    """Fake secondary: always answers."""

    name = "steady-backup"

    def __init__(self) -> None:
        self.attempts = 0

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.attempts += 1
        return (
            "Handled by the backup model — primary was rate limited.",
            [],
            Usage(input_tokens=10, output_tokens=12),
            "stop",
        )


async def fallback_demo() -> None:
    print("=" * 60)
    print("1) FallbackModel — rate-limited primary fails over")
    print("=" * 60)

    primary = _RateLimitedModel()
    backup = _BackupModel()
    agent = Agent(
        "You are a resilient assistant.",
        model=FallbackModel([primary, backup]),
    )

    result = await agent.run("Say hello.")
    print(f"  output          : {result.output}")
    print(f"  primary attempts: {primary.attempts} (raised 429 each time)")
    print(f"  backup attempts : {backup.attempts} (served the run)")


async def rate_limit_demo() -> None:
    print()
    print("=" * 60)
    print("2) TokenBucketRateLimiter — per-user pacing")
    print("=" * 60)

    # 5 requests/second, burst of 2: the first two acquires per user
    # are instant, then ~0.2 s of pacing per additional acquire.
    limiter = TokenBucketRateLimiter(rps=5, burst=2, mode="throttle")

    start = time.monotonic()
    for i in range(6):
        await limiter.acquire(user_id="alice")
        print(f"  alice acquire #{i + 1} at t={time.monotonic() - start:.2f}s")

    # Bob has his OWN bucket — alice's burst never slows him down.
    t_bob = time.monotonic()
    await limiter.acquire(user_id="bob")
    print(f"  bob   acquire #1 at +{time.monotonic() - t_bob:.3f}s (instant)")

    print()
    print("  Wire the same limiter into an agent and every model step")
    print("  is paced per tenant:")
    print("      Agent(..., rate_limiter=TokenBucketRateLimiter(rps=5, burst=10))")


async def main() -> None:
    await fallback_demo()
    await rate_limit_demo()


if __name__ == "__main__":
    anyio.run(main)
