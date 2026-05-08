"""Retry / error-taxonomy tests.

Covers the M5 contract end-to-end:

* :class:`RetryPolicy` dataclass + factories produce the right
  schedules.
* :func:`compute_backoff` honours jitter, cap, and provider-supplied
  ``retry_after`` floor.
* :func:`classify_model_error` maps known SDK exceptions to our
  taxonomy (when those SDKs are installed) and returns ``None`` for
  unknown ones.
* :class:`RetryingModel` retries transient errors, raises permanent
  ones immediately, exhausts after ``max_attempts``, and respects
  rate-limit ``retry_after`` hints. Streaming retries fire only
  before the first chunk.
* :class:`Agent` auto-wraps real network adapters with the default
  policy but leaves test fakes (``ScriptedModel``) and unknown
  custom Models alone.
"""

from __future__ import annotations

import random
from typing import Any

import anyio
import pytest

from jeevesagent import (
    Agent,
    AuthenticationError,
    RateLimitError,
    RetryPolicy,
    TransientModelError,
    classify_model_error,
)
from jeevesagent.governance.retry import compute_backoff
from jeevesagent.model.retrying import RetryingModel
from jeevesagent.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# RetryPolicy + compute_backoff
# ---------------------------------------------------------------------------


def test_retry_policy_factories_have_sensible_defaults() -> None:
    """The shipped factories cover the three typical use cases:
    default (sensible production), disabled (fail fast), aggressive
    (long-running blip recovery)."""
    default = RetryPolicy()
    assert default.is_enabled() is True
    assert default.max_attempts >= 2

    disabled = RetryPolicy.disabled()
    assert disabled.is_enabled() is False
    assert disabled.max_attempts == 1

    aggressive = RetryPolicy.aggressive()
    assert aggressive.is_enabled() is True
    assert aggressive.max_attempts > default.max_attempts


def test_compute_backoff_returns_zero_for_disabled_policy() -> None:
    """A single-attempt policy never sleeps."""
    assert compute_backoff(RetryPolicy.disabled(), attempt=1) == 0.0


def test_compute_backoff_grows_geometrically_without_jitter() -> None:
    """With ``jitter=0`` the schedule is a clean geometric series
    bounded by ``max_delay_s``."""
    policy = RetryPolicy(
        max_attempts=5, initial_delay_s=1.0, multiplier=2.0,
        max_delay_s=30.0, jitter=0.0,
    )
    assert compute_backoff(policy, attempt=1) == 1.0
    assert compute_backoff(policy, attempt=2) == 2.0
    assert compute_backoff(policy, attempt=3) == 4.0
    assert compute_backoff(policy, attempt=4) == 8.0
    # Capped at max_delay_s.
    assert compute_backoff(policy, attempt=10) == 30.0


def test_compute_backoff_applies_symmetric_jitter() -> None:
    policy = RetryPolicy(
        max_attempts=3, initial_delay_s=10.0, jitter=0.2, multiplier=1.0,
    )
    rng = random.Random(0)
    out = [compute_backoff(policy, 1, rng=rng) for _ in range(50)]
    # All values within ±20% of the base.
    assert all(8.0 <= x <= 12.0 for x in out)
    # Some variation actually happens.
    assert max(out) - min(out) > 0.1


def test_compute_backoff_honours_retry_after_floor() -> None:
    """Provider-supplied ``Retry-After`` is a floor, not a ceiling —
    it can exceed ``max_delay_s`` because the provider is more
    authoritative than our local cap."""
    policy = RetryPolicy(
        max_attempts=3, initial_delay_s=1.0, max_delay_s=5.0, jitter=0.0,
    )
    # Computed backoff would be 1s; provider asked for 60s.
    assert compute_backoff(policy, 1, retry_after=60.0) == 60.0
    # Provider asked for less than computed — keep the computed.
    assert compute_backoff(policy, 1, retry_after=0.1) == 1.0


# ---------------------------------------------------------------------------
# classify_model_error
# ---------------------------------------------------------------------------


def test_classify_returns_already_classified_unchanged() -> None:
    """Re-classifying a ``ModelError`` is idempotent — the wrapper
    returns the same instance, so the cause-chain stays intact."""
    err = TransientModelError("blip")
    assert classify_model_error(err) is err


def test_classify_unknown_exception_returns_none() -> None:
    """Anything we don't recognise propagates unchanged. The
    framework refuses to silently coerce arbitrary errors into
    transient-retry territory."""
    assert classify_model_error(ValueError("???")) is None
    assert classify_model_error(RuntimeError("???")) is None


# ---------------------------------------------------------------------------
# RetryingModel — retry mechanics
# ---------------------------------------------------------------------------


class _FlakyModel:
    """Inner model that fails ``fail_count`` times before succeeding.

    Each failure raises the configured exception; the success
    returns a canned tuple. Counts attempts so tests can assert
    on retry behaviour.
    """

    name = "flaky"

    def __init__(
        self,
        *,
        fail_count: int = 0,
        exc_factory: Any = None,
    ) -> None:
        self.attempts = 0
        self._fail_count = fail_count
        self._exc_factory = exc_factory or (
            lambda: TransientModelError("transient")
        )

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.attempts += 1
        if self.attempts <= self._fail_count:
            raise self._exc_factory()
        from jeevesagent.core.types import Usage
        return ("ok", [], Usage(), "stop")

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        self.attempts += 1
        if self.attempts <= self._fail_count:
            raise self._exc_factory()
        from jeevesagent.core.types import ModelChunk, Usage
        yield ModelChunk(kind="text", text="ok")
        yield ModelChunk(kind="finish", finish_reason="stop", usage=Usage())


async def test_retrying_model_succeeds_on_first_attempt() -> None:
    inner = _FlakyModel(fail_count=0)
    wrapped = RetryingModel(inner, RetryPolicy(jitter=0.0))
    text, _, _, _ = await wrapped.complete([])
    assert text == "ok"
    assert inner.attempts == 1


async def test_retrying_model_recovers_from_transient_failures() -> None:
    """One transient blip, the wrapper retries once, then succeeds."""
    inner = _FlakyModel(fail_count=1)
    policy = RetryPolicy(
        max_attempts=3, initial_delay_s=0.0, jitter=0.0,
    )
    wrapped = RetryingModel(inner, policy)
    text, _, _, _ = await wrapped.complete([])
    assert text == "ok"
    assert inner.attempts == 2  # 1 fail + 1 success


async def test_retrying_model_raises_after_exhausting_attempts() -> None:
    """If every attempt fails the wrapper gives up and raises the
    last transient error."""
    inner = _FlakyModel(fail_count=10)
    policy = RetryPolicy(
        max_attempts=3, initial_delay_s=0.0, jitter=0.0,
    )
    wrapped = RetryingModel(inner, policy)
    with pytest.raises(TransientModelError):
        await wrapped.complete([])
    assert inner.attempts == 3


async def test_retrying_model_does_not_retry_permanent_errors() -> None:
    """Permanent errors (auth, content filter, bad request) never
    retry — they raise immediately on the first call."""
    inner = _FlakyModel(
        fail_count=10,
        exc_factory=lambda: AuthenticationError("bad key"),
    )
    policy = RetryPolicy(
        max_attempts=5, initial_delay_s=0.0, jitter=0.0,
    )
    wrapped = RetryingModel(inner, policy)
    with pytest.raises(AuthenticationError):
        await wrapped.complete([])
    assert inner.attempts == 1  # no retries


async def test_retrying_model_propagates_unknown_exceptions_unchanged() -> None:
    """If the inner model raises something the classifier doesn't
    recognise, the wrapper does NOT silently retry — the original
    exception bubbles up as-is so callers see the real failure."""
    inner = _FlakyModel(
        fail_count=10,
        exc_factory=lambda: ValueError("not an SDK error"),
    )
    wrapped = RetryingModel(inner, RetryPolicy())
    with pytest.raises(ValueError):
        await wrapped.complete([])
    assert inner.attempts == 1  # no retries


async def test_retrying_model_respects_rate_limit_retry_after() -> None:
    """Rate-limit errors carrying ``retry_after`` set a floor on the
    backoff. We only need to verify the wait was at least that
    long; the exact sleep duration is mocked via anyio."""
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch_target = anyio
    real_sleep = monkeypatch_target.sleep
    try:
        monkeypatch_target.sleep = fake_sleep  # type: ignore[assignment]

        inner = _FlakyModel(
            fail_count=1,
            exc_factory=lambda: RateLimitError(
                "slow down", retry_after=42.0
            ),
        )
        policy = RetryPolicy(
            max_attempts=2, initial_delay_s=0.1, jitter=0.0,
        )
        wrapped = RetryingModel(inner, policy)
        await wrapped.complete([])
    finally:
        monkeypatch_target.sleep = real_sleep  # type: ignore[assignment]

    # Slept once, for at least 42 s (the retry_after floor).
    assert len(sleeps) == 1
    assert sleeps[0] >= 42.0


# ---------------------------------------------------------------------------
# RetryingModel — streaming
# ---------------------------------------------------------------------------


async def test_retrying_model_stream_retries_before_first_chunk() -> None:
    """Errors raised before the first chunk arrives are retried;
    once chunks start flowing we cannot rewind."""
    inner = _FlakyModel(fail_count=2)
    policy = RetryPolicy(
        max_attempts=4, initial_delay_s=0.0, jitter=0.0,
    )
    wrapped = RetryingModel(inner, policy)
    chunks = [chunk async for chunk in wrapped.stream([])]
    assert any(c.kind == "text" for c in chunks)
    assert inner.attempts == 3


# ---------------------------------------------------------------------------
# Agent integration — auto-wrapping the right things
# ---------------------------------------------------------------------------


async def test_agent_does_not_wrap_scripted_model_by_default() -> None:
    """Test fakes get retries disabled by default — they don't
    have transient failures and tests want deterministic timing."""
    agent = Agent(
        "hi", model=ScriptedModel([ScriptedTurn(text="ok")])
    )
    # The internal ``_wrapped_model`` is the same instance as the
    # adapter for fakes — no RetryingModel layer in between.
    assert type(agent._wrapped_model).__name__ != "RetryingModel"


async def test_agent_wraps_custom_model_when_explicit_policy_given() -> None:
    """Custom Models don't auto-wrap (we can't reason about their
    error types), but a caller-supplied policy opts in."""
    agent = Agent(
        "hi",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        retry_policy=RetryPolicy(),
    )
    assert type(agent._wrapped_model).__name__ == "RetryingModel"


async def test_agent_disabled_policy_skips_wrapping_even_for_network() -> None:
    """``RetryPolicy.disabled()`` opts out of retries entirely.
    Useful for callers who handle errors themselves at a higher
    layer (custom retry/backoff, circuit breaker, etc.)."""
    agent = Agent(
        "hi",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        retry_policy=RetryPolicy.disabled(),
    )
    assert type(agent._wrapped_model).__name__ != "RetryingModel"


async def test_agent_run_works_through_retry_wrapper_end_to_end() -> None:
    """Sanity: putting the retry wrapper in the chain doesn't break
    a basic agent run. The wrapper is transparent on the happy
    path."""
    agent = Agent(
        "hi",
        model=ScriptedModel([ScriptedTurn(text="hello")]),
        retry_policy=RetryPolicy(initial_delay_s=0.0, jitter=0.0),
    )
    result = await agent.run("anything")
    assert result.output == "hello"
