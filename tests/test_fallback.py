"""FallbackModel + per-request timeout tests (G6 contract).

* :class:`FallbackModel` advances to the next model on fallback-worthy
  errors (rate limit / transient / plain permanent), never on success,
  never on auth / unknown errors, and only BEFORE the first streamed
  chunk — mid-stream errors propagate with no silent switch and no
  duplicated chunks.
* ``fall_on`` customises the trigger predicate.
* ``request_timeout_s`` on the adapters kills a hung stream via the
  anyio wall clock and surfaces as a classified
  :class:`TransientModelError`; the SDK-native ``timeout=`` option is
  threaded into the client call.
* Composition: ``FallbackModel([RetryingModel(A), RetryingModel(B)])``
  exhausts retries on A before falling to B.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from loomflow.core import (
    AuthenticationError,
    ConfigError,
    PermanentModelError,
    RateLimitError,
    TransientModelError,
)
from loomflow.core.types import Message, ModelChunk, Role, Usage
from loomflow.governance import RetryPolicy
from loomflow.model.anthropic import AnthropicModel
from loomflow.model.fallback import FallbackModel, default_fall_on
from loomflow.model.openai import OpenAIModel
from loomflow.model.retrying import RetryingModel

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fakes (pattern mirrors tests/test_retry.py::_FlakyModel)
# ---------------------------------------------------------------------------


class _FakeModel:
    """Inner model that fails ``fail_count`` calls before succeeding.

    ``complete`` and ``stream`` count attempts independently so tests
    can assert exactly how far a chain / retry wrapper walked.
    """

    def __init__(
        self,
        name: str,
        *,
        fail_count: int = 0,
        exc_factory: Any = None,
        text: str = "ok",
    ) -> None:
        self.name = name
        self.attempts = 0
        self.stream_attempts = 0
        self._fail_count = fail_count
        self._exc_factory = exc_factory or (
            lambda: TransientModelError("transient")
        )
        self._text = text

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.attempts += 1
        if self.attempts <= self._fail_count:
            raise self._exc_factory()
        return (self._text, [], Usage(), "stop")

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        self.stream_attempts += 1
        if self.stream_attempts <= self._fail_count:
            raise self._exc_factory()
        yield ModelChunk(kind="text", text=self._text)
        yield ModelChunk(kind="finish", finish_reason="stop", usage=Usage())


class _MidStreamFailModel:
    """Yields ONE chunk, then raises — models a mid-stream drop."""

    name = "midstream"

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        yield ModelChunk(kind="text", text="partial")
        raise TransientModelError("mid-stream blip")


# ---------------------------------------------------------------------------
# default_fall_on — trigger classification
# ---------------------------------------------------------------------------


def test_default_fall_on_triggers() -> None:
    """Transient family (incl. rate limit) and PLAIN permanent errors
    fall over; auth and unclassified exceptions never do."""
    assert default_fall_on(RateLimitError("429")) is True
    assert default_fall_on(TransientModelError("529 overloaded")) is True
    assert default_fall_on(PermanentModelError("odd provider status")) is True
    assert default_fall_on(AuthenticationError("bad key")) is False
    assert default_fall_on(ValueError("not a model error")) is False


def test_empty_chain_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        FallbackModel([])


# ---------------------------------------------------------------------------
# complete() — failover mechanics
# ---------------------------------------------------------------------------


async def test_complete_fails_over_on_rate_limit() -> None:
    primary = _FakeModel(
        "primary",
        fail_count=10,
        exc_factory=lambda: RateLimitError("slow down"),
    )
    secondary = _FakeModel("secondary", text="from-secondary")
    fb = FallbackModel([primary, secondary])

    text, _, _, _ = await fb.complete([])

    assert text == "from-secondary"
    assert primary.attempts == 1
    assert secondary.attempts == 1
    assert fb.last_served == "secondary"


async def test_complete_no_failover_on_success() -> None:
    primary = _FakeModel("primary", text="from-primary")
    secondary = _FakeModel("secondary")
    fb = FallbackModel([primary, secondary])

    text, _, _, _ = await fb.complete([])

    assert text == "from-primary"
    assert secondary.attempts == 0
    assert fb.last_served == "primary"
    # Wrapper keeps a stable name (the primary's) for telemetry.
    assert fb.name == "primary"


async def test_complete_auth_error_does_not_fail_over() -> None:
    """Permanent subclass errors (auth / bad request / content
    filter) raise immediately — a different model won't fix them."""
    primary = _FakeModel(
        "primary",
        fail_count=10,
        exc_factory=lambda: AuthenticationError("bad key"),
    )
    secondary = _FakeModel("secondary")
    fb = FallbackModel([primary, secondary])

    with pytest.raises(AuthenticationError):
        await fb.complete([])
    assert secondary.attempts == 0


async def test_complete_last_model_error_propagates() -> None:
    """When the whole chain is down the LAST model's error surfaces."""
    a = _FakeModel("a", fail_count=10)
    b = _FakeModel("b", fail_count=10)
    fb = FallbackModel([a, b])

    with pytest.raises(TransientModelError):
        await fb.complete([])
    assert a.attempts == 1
    assert b.attempts == 1


async def test_complete_unknown_exception_propagates_unchanged() -> None:
    primary = _FakeModel(
        "primary",
        fail_count=10,
        exc_factory=lambda: ValueError("programming error"),
    )
    secondary = _FakeModel("secondary")
    fb = FallbackModel([primary, secondary])

    with pytest.raises(ValueError):
        await fb.complete([])
    assert secondary.attempts == 0


async def test_custom_fall_on_predicate() -> None:
    """``fall_on`` fully replaces the default trigger set."""
    primary = _FakeModel(
        "primary",
        fail_count=10,
        exc_factory=lambda: ValueError("routable after all"),
    )
    secondary = _FakeModel("secondary", text="rescued")
    fb = FallbackModel(
        [primary, secondary],
        fall_on=lambda exc: isinstance(exc, ValueError),
    )
    text, _, _, _ = await fb.complete([])
    assert text == "rescued"

    # And the inverse: a predicate that refuses rate limits.
    rl_primary = _FakeModel(
        "primary",
        fail_count=10,
        exc_factory=lambda: RateLimitError("429"),
    )
    strict = FallbackModel(
        [rl_primary, _FakeModel("secondary")],
        fall_on=lambda exc: False,
    )
    with pytest.raises(RateLimitError):
        await strict.complete([])


# ---------------------------------------------------------------------------
# stream() — failover only before the first chunk
# ---------------------------------------------------------------------------


async def test_stream_fails_over_before_first_chunk() -> None:
    primary = _FakeModel("primary", fail_count=10)
    secondary = _FakeModel("secondary", text="from-secondary")
    fb = FallbackModel([primary, secondary])

    chunks = [c async for c in fb.stream([])]

    texts = [c.text for c in chunks if c.kind == "text"]
    assert texts == ["from-secondary"]
    assert chunks[-1].kind == "finish"
    assert primary.stream_attempts == 1
    assert fb.last_served == "secondary"


async def test_stream_mid_stream_error_propagates_without_switch() -> None:
    """After the first chunk we are committed: the error surfaces,
    the secondary is never touched, and nothing is double-yielded."""
    secondary = _FakeModel("secondary")
    fb = FallbackModel([_MidStreamFailModel(), secondary])

    seen: list[ModelChunk] = []
    with pytest.raises(TransientModelError):
        async for chunk in fb.stream([]):
            seen.append(chunk)

    assert [c.text for c in seen if c.kind == "text"] == ["partial"]
    assert secondary.stream_attempts == 0
    # Committed to the model that yielded, even though it then died.
    assert fb.last_served == "midstream"


# ---------------------------------------------------------------------------
# Capability duck-typing — delegated to the primary
# ---------------------------------------------------------------------------


async def test_capabilities_delegate_to_primary() -> None:
    class _Capable(_FakeModel):
        supports_native_structured_output = True

        async def count_tokens(self, messages: Any, **kwargs: Any) -> int:
            return 42

    fb = FallbackModel([_Capable("primary"), _FakeModel("secondary")])
    assert fb.supports_native_structured_output is True
    assert await fb.count_tokens([]) == 42

    plain = FallbackModel([_FakeModel("primary")])
    assert plain.supports_native_structured_output is False
    # hasattr-based capability discovery sees the primary's surface.
    assert not hasattr(plain, "count_tokens")


# ---------------------------------------------------------------------------
# Composition with RetryingModel — retries exhaust before failover
# ---------------------------------------------------------------------------


async def test_retries_exhaust_on_primary_before_falling_over() -> None:
    a = _FakeModel("a", fail_count=10)
    b = _FakeModel("b", text="from-b")
    policy = RetryPolicy(max_attempts=3, initial_delay_s=0.0, jitter=0.0)
    fb = FallbackModel([RetryingModel(a, policy), RetryingModel(b, policy)])

    text, _, _, _ = await fb.complete([])

    assert text == "from-b"
    assert a.attempts == 3  # full retry budget spent on the primary
    assert b.attempts == 1
    assert fb.last_served == "b"


async def test_stream_composition_with_retrying_model() -> None:
    a = _FakeModel("a", fail_count=10)
    b = _FakeModel("b", text="from-b")
    policy = RetryPolicy(max_attempts=2, initial_delay_s=0.0, jitter=0.0)
    fb = FallbackModel([RetryingModel(a, policy), RetryingModel(b, policy)])

    chunks = [c async for c in fb.stream([])]

    assert [c.text for c in chunks if c.kind == "text"] == ["from-b"]
    assert a.stream_attempts == 2
    assert b.stream_attempts == 1


# ---------------------------------------------------------------------------
# Per-request timeouts — hung streams die, classified transient
# ---------------------------------------------------------------------------


class _HungAsyncIterator:
    """A stream whose next-chunk await never resolves (hung SSE)."""

    def __aiter__(self) -> _HungAsyncIterator:
        return self

    async def __anext__(self) -> Any:
        await anyio.sleep(3600)
        raise AssertionError("unreachable")


class _HungOAICompletions:
    def __init__(self) -> None:
        self.captured_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _HungAsyncIterator:
        self.captured_kwargs = kwargs
        return _HungAsyncIterator()


class _HungOAIClient:
    def __init__(self) -> None:
        self.chat = type(
            "_Chat", (), {"completions": _HungOAICompletions()}
        )()


async def test_openai_stream_timeout_fires_and_is_transient() -> None:
    client = _HungOAIClient()
    model = OpenAIModel("gpt-4o", client=client, request_timeout_s=0.05)

    with pytest.raises(TransientModelError):
        async for _ in model.stream([Message(role=Role.USER, content="hi")]):
            raise AssertionError("no chunk should ever arrive")

    # SDK-native per-request timeout was threaded into the call too.
    captured = client.chat.completions.captured_kwargs
    assert captured is not None
    assert captured["timeout"] == 0.05


async def test_openai_no_timeout_kwarg_when_unset() -> None:
    class _EmptyStream:
        def __aiter__(self) -> _EmptyStream:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    class _Completions:
        captured_kwargs: dict[str, Any] | None = None

        async def create(self, **kwargs: Any) -> _EmptyStream:
            _Completions.captured_kwargs = kwargs
            return _EmptyStream()

    client = type(
        "_C", (), {"chat": type("_Ch", (), {"completions": _Completions()})()}
    )()
    model = OpenAIModel("gpt-4o", client=client)
    [c async for c in model.stream([Message(role=Role.USER, content="hi")])]
    assert _Completions.captured_kwargs is not None
    assert "timeout" not in _Completions.captured_kwargs


class _HungAnthropicStream:
    async def __aenter__(self) -> _HungAnthropicStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> _HungAnthropicStream:
        return self

    async def __anext__(self) -> Any:
        await anyio.sleep(3600)
        raise AssertionError("unreachable")


class _HungAnthropicMessages:
    def __init__(self) -> None:
        self.captured_kwargs: dict[str, Any] | None = None

    def stream(self, **kwargs: Any) -> _HungAnthropicStream:
        self.captured_kwargs = kwargs
        return _HungAnthropicStream()


class _HungAnthropicClient:
    def __init__(self) -> None:
        self.messages = _HungAnthropicMessages()


async def test_anthropic_stream_timeout_fires_and_is_transient() -> None:
    client = _HungAnthropicClient()
    model = AnthropicModel(
        "claude-opus-4-7", client=client, request_timeout_s=0.05
    )

    with pytest.raises(TransientModelError):
        async for _ in model.stream([Message(role=Role.USER, content="hi")]):
            raise AssertionError("no chunk should ever arrive")

    captured = client.messages.captured_kwargs
    assert captured is not None
    assert captured["timeout"] == 0.05


async def test_openai_complete_timeout_fires_and_is_transient() -> None:
    class _HungCompletions:
        async def create(self, **kwargs: Any) -> Any:
            await anyio.sleep(3600)
            raise AssertionError("unreachable")

    client = type(
        "_C",
        (),
        {"chat": type("_Ch", (), {"completions": _HungCompletions()})()},
    )()
    model = OpenAIModel("gpt-4o", client=client, request_timeout_s=0.05)

    with pytest.raises(TransientModelError):
        await model.complete([Message(role=Role.USER, content="hi")])


async def test_timeout_composes_with_fallback_chain() -> None:
    """A hung primary times out (transient) → the chain falls over."""
    hung = OpenAIModel(
        "gpt-hung", client=_HungOAIClient(), request_timeout_s=0.05
    )
    secondary = _FakeModel("secondary", text="rescued")
    fb = FallbackModel([hung, secondary])

    chunks = [c async for c in fb.stream([Message(role=Role.USER, content="hi")])]

    assert [c.text for c in chunks if c.kind == "text"] == ["rescued"]
    assert fb.last_served == "secondary"
