"""Per-tenant QPS rate limiting (G5).

Covers :class:`loomflow.governance.rate_limit.TokenBucketRateLimiter`
directly (burst allowance, pacing under concurrency, per-user bucket
isolation, throttle-vs-raise modes, constructor validation) plus the
wiring seams: the ``budget_gate`` choke point in
``architecture/helpers.py`` (acquire called once per step, skipped
entirely on the ``fast_rate_limit`` path) and the end-to-end
``Agent(rate_limiter=...)`` kwarg with :class:`EchoModel`.

Timing tests use duration bounds around ``anyio.current_time()``
rather than a mock clock (the suite runs on the asyncio backend,
which has no autojump clock). Rates are scaled high (e.g. rps=100)
so the pacing window is tens of milliseconds — long enough to assert
a lower bound, short enough to keep the suite fast.
"""

from typing import Any

import anyio
import pytest

from loomflow import Agent
from loomflow.architecture.base import AgentSession, Dependencies
from loomflow.architecture.helpers import budget_gate
from loomflow.core.context import RunContext
from loomflow.core.errors import RateLimitError, RateLimitExceeded
from loomflow.governance.budget import NoBudget
from loomflow.governance.rate_limit import RateLimiter, TokenBucketRateLimiter
from loomflow.memory.inmemory import InMemoryMemory
from loomflow.model.echo import EchoModel
from loomflow.observability.tracing import NoTelemetry
from loomflow.runtime.inproc import InProcRuntime
from loomflow.security.hooks import HookRegistry
from loomflow.security.permissions import AllowAll
from loomflow.tools.registry import InProcessToolHost

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingLimiter:
    """RateLimiter protocol fake that records every acquire."""

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def acquire(self, *, user_id: str | None) -> None:
        self.calls.append(user_id)


class _ExplodingLimiter:
    """Fails the test if the fast path ever reaches ``acquire``."""

    async def acquire(self, *, user_id: str | None) -> None:
        raise AssertionError(
            "rate_limiter.acquire must not be called on the fast path"
        )


def _make_deps(**overrides: Any) -> Dependencies:
    kwargs: dict[str, Any] = dict(
        model=EchoModel(),
        memory=InMemoryMemory(),
        runtime=InProcRuntime(),
        tools=InProcessToolHost(),
        budget=NoBudget(),
        permissions=AllowAll(),
        hooks=HookRegistry(),
        telemetry=NoTelemetry(),
        audit_log=None,
        max_turns=10,
    )
    kwargs.update(overrides)
    return Dependencies(**kwargs)


def _session() -> AgentSession:
    return AgentSession(id="sess_test", instructions="test")


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter — constructor validation
# ---------------------------------------------------------------------------


def test_rps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="rps must be > 0"):
        TokenBucketRateLimiter(0, 10)
    with pytest.raises(ValueError, match="rps must be > 0"):
        TokenBucketRateLimiter(-1.0, 10)


def test_burst_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="burst must be >= 1"):
        TokenBucketRateLimiter(5, 0)


def test_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        TokenBucketRateLimiter(5, 10, mode="explode")  # type: ignore[arg-type]


def test_satisfies_rate_limiter_protocol() -> None:
    limiter = TokenBucketRateLimiter(5, 10)
    assert isinstance(limiter, RateLimiter)
    assert limiter.rps == 5.0
    assert limiter.burst == 10
    assert limiter.mode == "throttle"


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter — burst allowance
# ---------------------------------------------------------------------------


async def test_burst_acquires_are_immediate() -> None:
    """A quiet tenant may fire ``burst`` steps back-to-back with no
    waiting — only the (burst+1)th acquire has to pace."""
    limiter = TokenBucketRateLimiter(rps=1, burst=10)
    start = anyio.current_time()
    for _ in range(10):
        await limiter.acquire(user_id="alice")
    elapsed = anyio.current_time() - start
    assert elapsed < 0.2  # no sleeps happened


async def test_acquire_beyond_burst_waits_for_refill() -> None:
    """Draining the bucket forces the next acquire to wait ~1/rps."""
    limiter = TokenBucketRateLimiter(rps=50, burst=1)
    await limiter.acquire(user_id="alice")  # drains the single token
    start = anyio.current_time()
    await limiter.acquire(user_id="alice")  # must wait ~20ms
    elapsed = anyio.current_time() - start
    assert elapsed >= 0.015


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter — pacing under concurrency
# ---------------------------------------------------------------------------


async def test_twenty_concurrent_acquires_are_paced() -> None:
    """The G5 acceptance shape (20 acquires, burst 10) at a scaled-up
    rate: 10 pass on the initial burst, the other 10 accrue at
    ``rps`` — total duration ≈ 10/rps, never "everything at once"."""
    rps, burst, n = 100.0, 10, 20
    limiter = TokenBucketRateLimiter(rps=rps, burst=burst)
    done_at: list[float] = []

    async def one() -> None:
        await limiter.acquire(user_id="alice")
        done_at.append(anyio.current_time())

    start = anyio.current_time()
    async with anyio.create_task_group() as tg:
        for _ in range(n):
            tg.start_soon(one)
    elapsed = anyio.current_time() - start

    # 10 excess acquires at 100 rps ≈ 0.1s. Assert the pacing floor
    # (with slack for scheduler jitter) and a sane ceiling.
    assert elapsed >= 0.08, f"20 acquires finished too fast: {elapsed:.3f}s"
    assert elapsed < 2.0, f"pacing overshot: {elapsed:.3f}s"
    # The initial burst really was immediate: at least ``burst``
    # acquires completed well before the pacing window ended.
    immediate = [t for t in done_at if t - start < 0.05]
    assert len(immediate) >= burst


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter — per-user isolation
# ---------------------------------------------------------------------------


async def test_second_user_is_unaffected_by_first_users_exhaustion() -> None:
    limiter = TokenBucketRateLimiter(rps=1, burst=2)
    await limiter.acquire(user_id="alice")
    await limiter.acquire(user_id="alice")  # alice's bucket now empty
    start = anyio.current_time()
    await limiter.acquire(user_id="bob")  # independent bucket
    elapsed = anyio.current_time() - start
    assert elapsed < 0.1


async def test_raise_mode_per_user_isolation() -> None:
    limiter = TokenBucketRateLimiter(rps=0.001, burst=1, mode="raise")
    await limiter.acquire(user_id="alice")
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire(user_id="alice")
    # Bob's bucket is untouched.
    await limiter.acquire(user_id="bob")


async def test_anonymous_user_gets_its_own_bucket() -> None:
    limiter = TokenBucketRateLimiter(rps=0.001, burst=1, mode="raise")
    await limiter.acquire(user_id=None)
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire(user_id=None)
    await limiter.acquire(user_id="alice")  # named user unaffected


async def test_per_user_false_shares_one_global_bucket() -> None:
    limiter = TokenBucketRateLimiter(
        rps=0.001, burst=2, per_user=False, mode="raise"
    )
    await limiter.acquire(user_id="alice")
    await limiter.acquire(user_id="bob")
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire(user_id="carol")  # shared bucket drained


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter — raise mode
# ---------------------------------------------------------------------------


async def test_raise_mode_raises_with_metadata() -> None:
    limiter = TokenBucketRateLimiter(rps=2, burst=1, mode="raise")
    await limiter.acquire(user_id="alice")
    with pytest.raises(RateLimitExceeded) as excinfo:
        await limiter.acquire(user_id="alice")
    err = excinfo.value
    assert err.user_id == "alice"
    assert err.retry_after is not None and err.retry_after > 0
    assert "rate limit exceeded" in str(err)


async def test_raise_mode_recovers_after_refill() -> None:
    limiter = TokenBucketRateLimiter(rps=100, burst=1, mode="raise")
    await limiter.acquire(user_id="alice")
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire(user_id="alice")
    await anyio.sleep(0.05)  # 5 tokens' worth at 100 rps, capped at 1
    await limiter.acquire(user_id="alice")  # no raise


def test_rate_limit_exceeded_is_distinct_from_provider_429() -> None:
    """Framework admission gate vs provider 429 stay separate types —
    catching one must not swallow the other."""
    assert not issubclass(RateLimitExceeded, RateLimitError)
    assert not issubclass(RateLimitError, RateLimitExceeded)


# ---------------------------------------------------------------------------
# budget_gate choke point
# ---------------------------------------------------------------------------


async def test_budget_gate_calls_acquire_with_run_user_id() -> None:
    recorder = _RecordingLimiter()
    deps = _make_deps(
        rate_limiter=recorder,
        fast_rate_limit=False,
        context=RunContext(user_id="alice"),
    )
    blocked, events = await budget_gate(deps, _session())
    assert not blocked
    assert events == []
    assert recorder.calls == ["alice"]


async def test_budget_gate_rate_limits_even_with_no_budget() -> None:
    """The QPS gate must fire before the ``fast_budget`` early
    return — a NoBudget agent is still paced."""
    limiter = TokenBucketRateLimiter(rps=0.001, burst=1, mode="raise")
    deps = _make_deps(
        rate_limiter=limiter,
        fast_rate_limit=False,
        fast_budget=True,
        context=RunContext(user_id="alice"),
    )
    session = _session()
    await budget_gate(deps, session)  # consumes the single token
    with pytest.raises(RateLimitExceeded):
        await budget_gate(deps, session)


async def test_budget_gate_fast_path_never_touches_limiter() -> None:
    """Default deps (``fast_rate_limit=True``) skip the acquire call
    site entirely — disabled means zero overhead AND zero calls."""
    deps = _make_deps(rate_limiter=_ExplodingLimiter())  # flag stays True
    blocked, events = await budget_gate(deps, _session())
    assert not blocked
    assert events == []


async def test_budget_gate_no_limiter_is_noop() -> None:
    deps = _make_deps()  # rate_limiter=None, fast_rate_limit=True
    assert deps.fast_rate_limit
    assert deps.rate_limiter is None
    blocked, events = await budget_gate(deps, _session())
    assert not blocked
    assert events == []


# ---------------------------------------------------------------------------
# End-to-end through Agent + EchoModel
# ---------------------------------------------------------------------------


async def test_agent_run_acquires_once_per_step() -> None:
    """EchoModel answers in one turn → exactly one acquire, carrying
    the run's user_id."""
    recorder = _RecordingLimiter()
    agent = Agent("you help", model="echo", rate_limiter=recorder)
    result = await agent.run("hello", user_id="alice")
    assert result.output.startswith("Echo: ")
    assert recorder.calls == ["alice"]

    # A second run adds exactly one more acquire (anonymous bucket).
    await agent.run("again")
    assert recorder.calls == ["alice", None]


async def test_agent_raise_mode_propagates_rate_limit_exceeded() -> None:
    limiter = TokenBucketRateLimiter(rps=0.001, burst=1, mode="raise")
    agent = Agent("you help", model="echo", rate_limiter=limiter)
    await agent.run("first", user_id="alice")  # consumes the token
    with pytest.raises(RateLimitExceeded):
        await agent.run("second", user_id="alice")


async def test_agent_throttle_mode_paces_consecutive_runs() -> None:
    limiter = TokenBucketRateLimiter(rps=50, burst=1)
    agent = Agent("you help", model="echo", rate_limiter=limiter)
    await agent.run("first", user_id="alice")
    start = anyio.current_time()
    await agent.run("second", user_id="alice")  # waits ~20ms for refill
    elapsed = anyio.current_time() - start
    assert elapsed >= 0.015


async def test_agent_default_has_no_limiter() -> None:
    agent = Agent("you help", model="echo")
    assert agent._rate_limiter is None
    result = await agent.run("hello")  # runs fine, zero rate-limit calls
    assert result.output.startswith("Echo: ")
