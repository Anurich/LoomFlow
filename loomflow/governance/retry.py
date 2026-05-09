"""Resilience for model calls.

Two pieces:

* :class:`RetryPolicy` — a small dataclass describing the backoff
  schedule (max attempts, initial delay, multiplier, max cap, jitter).
* :func:`classify_model_error` — inspects a raw exception from any
  model SDK and maps it to our taxonomy
  (:class:`~loomflow.TransientModelError` /
  :class:`~loomflow.RateLimitError` / :class:`~loomflow.AuthenticationError`
  / etc.). Lazy imports so we never require an SDK that isn't
  installed.

The actual retry loop lives in
:class:`~loomflow.model.retrying.RetryingModel`, which wraps any
:class:`~loomflow.Model` and runs every call through this
policy + classifier pair. Splitting policy/classification from the
retry mechanics keeps each piece testable in isolation and lets
callers reuse the classifier for non-Agent code (e.g. cron jobs
that hit the same SDK).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..core.errors import (
    AuthenticationError,
    ContentFilterError,
    InvalidRequestError,
    ModelError,
    PermanentModelError,
    RateLimitError,
    TransientModelError,
)

__all__ = [
    "RetryPolicy",
    "classify_model_error",
    "compute_backoff",
]


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential-backoff-with-jitter retry schedule.

    The default is sensible for production: up to **3 attempts**
    (one initial + two retries), starting at 1 s, doubling each
    attempt, capped at 30 s, with ±10% jitter so synchronised
    clients don't reform a thundering herd.

    Examples::

        # default — sensible for most apps
        RetryPolicy()

        # disable retries (fail fast)
        RetryPolicy.disabled()

        # aggressive — survives long provider blips
        RetryPolicy.aggressive()

        # tuned to a specific SLO
        RetryPolicy(max_attempts=4, initial_delay_s=0.5, max_delay_s=15)

    The schedule applies *between* attempts: the first call has no
    delay, the second is delayed by ``initial_delay_s`` (± jitter),
    the third by ``initial_delay_s * multiplier`` (± jitter), etc.,
    each capped at ``max_delay_s``. Provider-supplied
    ``Retry-After`` hints (carried on
    :class:`~loomflow.RateLimitError.retry_after`) override the
    computed delay when they ask for *more* time — we never sleep
    less than the provider asked for.
    """

    max_attempts: int = 3
    """Maximum total attempts including the first call. ``1`` means
    no retries; the call either succeeds or raises immediately. The
    minimum-meaningful retry policy is therefore ``max_attempts=2``."""

    initial_delay_s: float = 1.0
    """Backoff before the FIRST retry (i.e. between attempts 1 and 2).
    Subsequent retries use ``initial_delay_s * multiplier**n``."""

    max_delay_s: float = 30.0
    """Cap on any single backoff. Prevents runaway sleeps when
    ``multiplier`` is large or ``max_attempts`` is high."""

    multiplier: float = 2.0
    """Geometric growth between successive retries. ``2.0`` doubles
    each time; ``1.0`` makes the policy linear (fixed-interval)."""

    jitter: float = 0.1
    """Fractional ±jitter applied to each computed delay. ``0.1`` =
    ±10%. Set to ``0`` for deterministic backoff (useful in tests)."""

    @classmethod
    def disabled(cls) -> RetryPolicy:
        """Single attempt, no retries — fail fast on any error."""
        return cls(max_attempts=1, initial_delay_s=0.0, jitter=0.0)

    @classmethod
    def aggressive(cls) -> RetryPolicy:
        """Up to 6 attempts, faster initial backoff, longer cap.
        Use when the underlying provider is known-flaky and the
        caller prefers slow success over fast failure."""
        return cls(
            max_attempts=6,
            initial_delay_s=0.5,
            max_delay_s=60.0,
            multiplier=2.0,
        )

    def is_enabled(self) -> bool:
        """``True`` when the policy permits at least one retry."""
        return self.max_attempts >= 2


def compute_backoff(
    policy: RetryPolicy,
    attempt: int,
    *,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Backoff (seconds) before retry number ``attempt`` (1-indexed).

    ``attempt=1`` is the delay before the first *retry* (i.e. between
    attempts 1 and 2 of ``max_attempts``). Returns ``0`` when
    ``policy`` is disabled.

    ``retry_after`` (provider hint, e.g. from a 429 ``Retry-After``
    header) acts as a *floor*: we never wait less than the provider
    asked for, but we still cap at ``policy.max_delay_s``. This
    means a provider-supplied 60-second hint paired with a 30-second
    cap is honoured at 60 seconds (exceeding the cap on purpose —
    the provider is more authoritative than our heuristic).
    """
    if policy.max_attempts <= 1:
        return 0.0

    base = policy.initial_delay_s * (policy.multiplier ** max(0, attempt - 1))
    base = min(base, policy.max_delay_s)
    if policy.jitter > 0:
        r = rng or random
        # Symmetric ±jitter around ``base``.
        base = base * (1.0 + r.uniform(-policy.jitter, policy.jitter))
    base = max(0.0, base)

    if retry_after is not None and retry_after > base:
        # Provider asked for more time — honour it (still bounded
        # below by 0; can exceed our cap when the provider says so).
        return retry_after
    return base


# ---------------------------------------------------------------------------
# Classification — SDK exception → our taxonomy
# ---------------------------------------------------------------------------


def classify_model_error(exc: BaseException) -> ModelError | None:
    """Map an exception from any model SDK to the framework's taxonomy.

    Returns ``None`` when the exception is not recognised as a
    model-call failure — let callers decide whether to wrap it in
    something else or propagate. Returns an instance of one of
    :class:`~loomflow.TransientModelError` /
    :class:`~loomflow.RateLimitError` /
    :class:`~loomflow.AuthenticationError` /
    :class:`~loomflow.InvalidRequestError` /
    :class:`~loomflow.ContentFilterError` /
    :class:`~loomflow.PermanentModelError` otherwise.

    SDK imports are lazy — having e.g. the ``anthropic`` package
    installed is not required for OpenAI classification to work,
    and vice versa.
    """
    # Already classified — pass through unchanged so wrapping
    # double-classification doesn't lose context.
    if isinstance(exc, ModelError):
        return exc

    classified = _classify_openai(exc)
    if classified is not None:
        return classified

    classified = _classify_anthropic(exc)
    if classified is not None:
        return classified

    classified = _classify_httpx(exc)
    if classified is not None:
        return classified

    return None


def _classify_openai(exc: BaseException) -> ModelError | None:
    try:
        import openai
    except ImportError:
        return None

    # Rate limit is a subclass of APIStatusError; check it first.
    if isinstance(exc, getattr(openai, "RateLimitError", ())):
        return RateLimitError(
            f"OpenAI rate limit: {exc}",
            retry_after=_extract_retry_after(exc),
            cause=exc,
        )
    if isinstance(exc, getattr(openai, "AuthenticationError", ())):
        return AuthenticationError(f"OpenAI auth failed: {exc}", cause=exc)
    if isinstance(exc, getattr(openai, "PermissionDeniedError", ())):
        return AuthenticationError(
            f"OpenAI permission denied: {exc}", cause=exc
        )
    if isinstance(exc, getattr(openai, "BadRequestError", ())):
        # OpenAI's BadRequestError covers content-filter rejections
        # too; the body's ``code`` field disambiguates.
        if _is_content_filter(exc):
            return ContentFilterError(
                f"OpenAI content filter: {exc}", cause=exc
            )
        return InvalidRequestError(
            f"OpenAI bad request: {exc}", cause=exc
        )
    if isinstance(
        exc,
        (
            getattr(openai, "APITimeoutError", ()),
            getattr(openai, "APIConnectionError", ()),
            getattr(openai, "InternalServerError", ()),
        ),
    ):
        return TransientModelError(
            f"OpenAI transient: {exc}",
            retry_after=_extract_retry_after(exc),
            cause=exc,
        )
    if isinstance(exc, getattr(openai, "APIStatusError", ())):
        # Anything else with an HTTP status from the OpenAI SDK.
        status = getattr(exc, "status_code", None)
        if status is not None and 500 <= int(status) < 600:
            return TransientModelError(
                f"OpenAI {status}: {exc}",
                retry_after=_extract_retry_after(exc),
                cause=exc,
            )
        return PermanentModelError(f"OpenAI error: {exc}", cause=exc)
    return None


def _classify_anthropic(exc: BaseException) -> ModelError | None:
    try:
        import anthropic
    except ImportError:
        return None

    if isinstance(exc, getattr(anthropic, "RateLimitError", ())):
        return RateLimitError(
            f"Anthropic rate limit: {exc}",
            retry_after=_extract_retry_after(exc),
            cause=exc,
        )
    if isinstance(exc, getattr(anthropic, "AuthenticationError", ())):
        return AuthenticationError(
            f"Anthropic auth failed: {exc}", cause=exc
        )
    if isinstance(exc, getattr(anthropic, "PermissionDeniedError", ())):
        return AuthenticationError(
            f"Anthropic permission denied: {exc}", cause=exc
        )
    if isinstance(exc, getattr(anthropic, "BadRequestError", ())):
        if _is_content_filter(exc):
            return ContentFilterError(
                f"Anthropic content filter: {exc}", cause=exc
            )
        return InvalidRequestError(
            f"Anthropic bad request: {exc}", cause=exc
        )
    if isinstance(
        exc,
        (
            getattr(anthropic, "APITimeoutError", ()),
            getattr(anthropic, "APIConnectionError", ()),
            getattr(anthropic, "InternalServerError", ()),
        ),
    ):
        return TransientModelError(
            f"Anthropic transient: {exc}",
            retry_after=_extract_retry_after(exc),
            cause=exc,
        )
    if isinstance(exc, getattr(anthropic, "APIStatusError", ())):
        status = getattr(exc, "status_code", None)
        if status is not None and 500 <= int(status) < 600:
            return TransientModelError(
                f"Anthropic {status}: {exc}",
                retry_after=_extract_retry_after(exc),
                cause=exc,
            )
        return PermanentModelError(
            f"Anthropic error: {exc}", cause=exc
        )
    return None


def _classify_httpx(exc: BaseException) -> ModelError | None:
    """Bare ``httpx`` exceptions surface from custom transports
    (non-OpenAI/Anthropic providers via LiteLLM, custom HTTP-based
    Models). Only network-layer failures are classified — HTTP
    status responses never raise from ``httpx`` directly."""
    try:
        import httpx
    except ImportError:
        return None

    if isinstance(
        exc,
        (
            getattr(httpx, "TimeoutException", ()),
            getattr(httpx, "ConnectError", ()),
            getattr(httpx, "ReadError", ()),
            getattr(httpx, "WriteError", ()),
            getattr(httpx, "RemoteProtocolError", ()),
        ),
    ):
        return TransientModelError(
            f"network: {exc}", cause=exc
        )
    return None


# ---------------------------------------------------------------------------
# Helpers — best-effort metadata extraction from heterogeneous SDK errors
# ---------------------------------------------------------------------------


def _extract_retry_after(exc: BaseException) -> float | None:
    """Pull a ``Retry-After`` value off a SDK exception when one is
    available. SDKs vary: some carry the response on ``.response``,
    some on ``.body``, some not at all.

    Returns the value in seconds, or ``None`` when nothing is set.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


_CONTENT_FILTER_HINTS = (
    "content_filter",
    "content_policy",
    "safety",
)


def _is_content_filter(exc: BaseException) -> bool:
    """Heuristic check — does this 400 look like a content-filter
    rejection? Both OpenAI and Anthropic surface these as 400s
    with a marker code in the body."""
    msg = str(exc).lower()
    if any(hint in msg for hint in _CONTENT_FILTER_HINTS):
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        code = body.get("code") or body.get("type")
        if isinstance(code, str) and any(
            hint in code.lower() for hint in _CONTENT_FILTER_HINTS
        ):
            return True
    return False
