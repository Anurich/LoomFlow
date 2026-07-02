"""Regression tests for the model-adapter review fixes (wsd batch).

Covers:

1. Anthropic extended thinking: thinking / redacted_thinking blocks
   are captured (complete + stream), emitted as ``kind="thinking"``
   chunks while streaming, and replayed as leading blocks on the
   assistant turn of the NEXT request (required by the API when
   thinking is enabled and the turn contains tool_use).
2. ``budget_tokens`` never exceeds ``max_tokens``; temperature is
   dropped when thinking is on.
3. Prompt caching places a breakpoint on the conversation tail and
   stays within the 4-marker API budget.
4. ``complete()`` only falls back to streaming on duck-typing
   failures; real SDK errors propagate to the retry layer.
5. ``RetryingModel.stream`` never re-runs a stream after chunks
   were yielded.
6. OpenAI reasoning models get ``max_completion_tokens``.
7. ``cost_per_mtoken`` override threads through to cost math, and
   unknown-model pricing warns that budgets aren't enforced.
8. ``tool_result`` blocks carry ``is_error`` for failed tools.
9. LiteLLM never claims / emits native structured output.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any

import pytest

from loomflow.core.types import (
    Message,
    ModelChunk,
    PromptCacheConfig,
    Role,
    ToolCall,
)
from loomflow.model.anthropic import AnthropicModel, _to_anthropic_messages
from loomflow.model.openai import OpenAIModel

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fakes — non-streaming Anthropic client (messages.create)
# ---------------------------------------------------------------------------


class _FakeCreateMessages:
    """Fake ``client.messages`` that answers ``create()`` from a
    scripted list of responses and records every request's kwargs."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeCreateClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = _FakeCreateMessages(responses)


def _response(blocks: list[Any], stop_reason: str = "end_turn") -> Any:
    return NS(
        content=blocks,
        usage=NS(input_tokens=10, output_tokens=5),
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# Fakes — streaming Anthropic client (messages.stream)
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> _FakeStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration as e:
            raise StopAsyncIteration from e


class _FakeStreamMessages:
    def __init__(self, event_batches: list[list[Any]]) -> None:
        self._batches = list(event_batches)
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.calls.append(kwargs)
        return _FakeStream(self._batches.pop(0))


class _FakeStreamClient:
    def __init__(self, event_batches: list[list[Any]]) -> None:
        self.messages = _FakeStreamMessages(event_batches)


async def _collect(stream: AsyncIterator[ModelChunk]) -> list[ModelChunk]:
    return [c async for c in stream]


# ---------------------------------------------------------------------------
# 1. Extended thinking — capture + replay (complete path)
# ---------------------------------------------------------------------------


async def test_complete_thinking_blocks_replayed_on_next_turn() -> None:
    """Two-turn tool-use round trip: the thinking block from turn 1
    must LEAD the rebuilt assistant turn in turn 2's request."""
    turn1 = _response(
        [
            NS(type="thinking", thinking="let me check", signature="sig123"),
            NS(type="tool_use", id="tu_1", name="get_weather", input={"city": "SF"}),
        ],
        stop_reason="tool_use",
    )
    turn2 = _response([NS(type="text", text="Sunny.")])
    client = _FakeCreateClient([turn1, turn2])
    model = AnthropicModel("claude-sonnet-4-5", client=client)

    history = [Message(role=Role.USER, content="weather?")]
    text, calls, _usage, stop = await model.complete(history, effort="high")
    assert stop == "tool_use"
    assert len(calls) == 1 and calls[0].id == "tu_1"

    # Agent loop appends the assistant turn + tool result and re-asks.
    history = [
        *history,
        Message(role=Role.ASSISTANT, content="", tool_calls=(calls[0],)),
        Message(role=Role.TOOL, content="sunny", tool_call_id="tu_1"),
    ]
    text, _calls, _usage, _stop = await model.complete(history, effort="high")
    assert text == "Sunny."

    sent = client.messages.calls[1]["messages"]
    assistant = next(m for m in sent if m["role"] == "assistant")
    blocks = assistant["content"]
    assert blocks[0] == {
        "type": "thinking",
        "thinking": "let me check",
        "signature": "sig123",
    }
    # tool_use follows the thinking block.
    assert any(b.get("type") == "tool_use" and b.get("id") == "tu_1" for b in blocks)


async def test_complete_redacted_thinking_passes_through_opaquely() -> None:
    turn1 = _response(
        [
            NS(type="redacted_thinking", data="opaque-payload=="),
            NS(type="tool_use", id="tu_r", name="t", input={}),
        ],
        stop_reason="tool_use",
    )
    turn2 = _response([NS(type="text", text="done")])
    client = _FakeCreateClient([turn1, turn2])
    model = AnthropicModel("claude-sonnet-4-5", client=client)

    history = [Message(role=Role.USER, content="go")]
    _text, calls, _u, _s = await model.complete(history, effort="low")
    history = [
        *history,
        Message(role=Role.ASSISTANT, content="", tool_calls=(calls[0],)),
        Message(role=Role.TOOL, content="ok", tool_call_id="tu_r"),
    ]
    await model.complete(history, effort="low")

    sent = client.messages.calls[1]["messages"]
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert assistant["content"][0] == {
        "type": "redacted_thinking",
        "data": "opaque-payload==",
    }


async def test_complete_thinking_without_tool_use_is_not_cached() -> None:
    """Turns without tool_use never need thinking replay — nothing
    should be remembered for them."""
    turn = _response(
        [
            NS(type="thinking", thinking="hm", signature="s"),
            NS(type="text", text="answer"),
        ]
    )
    client = _FakeCreateClient([turn])
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    text, calls, _u, _s = await model.complete(
        [Message(role=Role.USER, content="q")], effort="low"
    )
    assert text == "answer"
    assert calls == []
    assert model._thinking == {}  # noqa: SLF001


# ---------------------------------------------------------------------------
# 1b. Extended thinking — streaming path
# ---------------------------------------------------------------------------


def _thinking_stream_events() -> list[Any]:
    return [
        NS(type="message_start", message=NS(usage=NS(input_tokens=8, output_tokens=0))),
        NS(
            type="content_block_start",
            index=0,
            content_block=NS(type="thinking", thinking="", signature=""),
        ),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="thinking_delta", thinking="pondering "),
        ),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="thinking_delta", thinking="deeply"),
        ),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="signature_delta", signature="sig-xyz"),
        ),
        NS(type="content_block_stop", index=0),
        NS(
            type="content_block_start",
            index=1,
            content_block=NS(type="tool_use", id="tu_s", name="lookup"),
        ),
        NS(
            type="content_block_delta",
            index=1,
            delta=NS(type="input_json_delta", partial_json='{"q": "x"}'),
        ),
        NS(type="content_block_stop", index=1),
        NS(
            type="message_delta",
            delta=NS(stop_reason="tool_use"),
            usage=NS(output_tokens=4),
        ),
        NS(type="message_stop"),
    ]


async def test_stream_emits_thinking_chunks_and_caches_for_replay() -> None:
    events2 = [
        NS(type="message_start", message=NS(usage=NS(input_tokens=9, output_tokens=0))),
        NS(type="content_block_start", index=0, content_block=NS(type="text")),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="text_delta", text="done"),
        ),
        NS(type="content_block_stop", index=0),
        NS(
            type="message_delta",
            delta=NS(stop_reason="end_turn"),
            usage=NS(output_tokens=2),
        ),
        NS(type="message_stop"),
    ]
    client = _FakeStreamClient([_thinking_stream_events(), events2])
    model = AnthropicModel("claude-sonnet-4-5", client=client)

    history = [Message(role=Role.USER, content="q")]
    chunks = await _collect(model.stream(history, effort="high"))

    thinking = [c.text for c in chunks if c.kind == "thinking"]
    assert thinking == ["pondering ", "deeply"]
    # Thinking never leaks into text chunks.
    assert [c for c in chunks if c.kind == "text"] == []
    (tool,) = [c for c in chunks if c.kind == "tool_call"]
    assert tool.tool_call is not None and tool.tool_call.id == "tu_s"

    # Second turn: replay must lead the assistant turn.
    history = [
        *history,
        Message(role=Role.ASSISTANT, content="", tool_calls=(tool.tool_call,)),
        Message(role=Role.TOOL, content="found", tool_call_id="tu_s"),
    ]
    await _collect(model.stream(history, effort="high"))
    sent = client.messages.calls[1]["messages"]
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert assistant["content"][0] == {
        "type": "thinking",
        "thinking": "pondering deeply",
        "signature": "sig-xyz",
    }


# ---------------------------------------------------------------------------
# 2. budget_tokens vs max_tokens + temperature
# ---------------------------------------------------------------------------


async def test_thinking_budget_raises_max_tokens_and_drops_temperature() -> None:
    """effort="max" on a legacy model maps to budget_tokens=32768,
    which exceeds the 4096 default max_tokens. The adapter must lift
    max_tokens above the budget and drop temperature (the API
    rejects temperature != 1 with thinking enabled)."""
    client = _FakeCreateClient([_response([NS(type="text", text="ok")])])
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    await model.complete(
        [Message(role=Role.USER, content="q")],
        temperature=0.2,
        effort="max",
    )
    kwargs = client.messages.calls[0]
    assert kwargs["thinking"]["budget_tokens"] == 32768
    assert kwargs["max_tokens"] > 32768
    assert "temperature" not in kwargs


async def test_thinking_keeps_caller_supplied_larger_max_tokens() -> None:
    client = _FakeCreateClient([_response([NS(type="text", text="ok")])])
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    await model.complete(
        [Message(role=Role.USER, content="q")],
        max_tokens=100_000,
        effort="max",
    )
    kwargs = client.messages.calls[0]
    assert kwargs["max_tokens"] == 100_000


async def test_no_thinking_keeps_temperature_and_default_max_tokens() -> None:
    client = _FakeCreateClient([_response([NS(type="text", text="ok")])])
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    await model.complete(
        [Message(role=Role.USER, content="q")], temperature=0.2
    )
    kwargs = client.messages.calls[0]
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# 3. Prompt caching — breakpoint on the conversation tail
# ---------------------------------------------------------------------------


async def test_cache_breakpoint_lands_on_final_message_tail() -> None:
    client = _FakeCreateClient([_response([NS(type="text", text="ok")])])
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    await model.complete(
        [
            Message(role=Role.SYSTEM, content="instructions"),
            Message(role=Role.SYSTEM, content="memory"),
            Message(role=Role.SYSTEM, content="recall"),
            Message(role=Role.USER, content="turn 1"),
            Message(role=Role.ASSISTANT, content="reply 1"),
            Message(role=Role.USER, content="turn 2"),
        ],
        tools=None,
        prompt_caching=PromptCacheConfig(enabled=True),
    )
    kwargs = client.messages.calls[0]

    # Final message rewritten into block form with the tail marker.
    final = kwargs["messages"][-1]
    assert final["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert final["content"][-1]["text"] == "turn 2"

    # Total breakpoints across the request stay within the API's 4.
    n_system = sum(
        1 for b in kwargs["system"] if "cache_control" in b
    )
    n_messages = sum(
        1
        for m in kwargs["messages"]
        if isinstance(m["content"], list)
        for b in m["content"]
        if isinstance(b, dict) and "cache_control" in b
    )
    assert n_messages == 1
    assert n_system + n_messages <= 4


async def test_cache_breakpoints_with_tools_stay_within_four() -> None:
    from loomflow.core.types import ToolDef

    client = _FakeCreateClient([_response([NS(type="text", text="ok")])])
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    await model.complete(
        [
            Message(role=Role.SYSTEM, content="a"),
            Message(role=Role.SYSTEM, content="b"),
            Message(role=Role.SYSTEM, content="c"),
            Message(role=Role.USER, content="hi"),
        ],
        tools=[ToolDef(name="t", description="d")],
        prompt_caching=PromptCacheConfig(enabled=True),
    )
    kwargs = client.messages.calls[0]
    n_system = sum(1 for b in kwargs["system"] if "cache_control" in b)
    n_tools = sum(1 for t in kwargs["tools"] if "cache_control" in t)
    n_messages = sum(
        1
        for m in kwargs["messages"]
        if isinstance(m["content"], list)
        for b in m["content"]
        if isinstance(b, dict) and "cache_control" in b
    )
    assert n_tools == 1
    assert n_messages == 1
    assert n_system == 2  # rebalanced: 2 system + 1 tools + 1 tail
    assert n_system + n_tools + n_messages <= 4


# ---------------------------------------------------------------------------
# 4. complete() error handling — no silent stream fallback
# ---------------------------------------------------------------------------


async def test_anthropic_complete_propagates_real_sdk_errors() -> None:
    class _Boom(RuntimeError):
        pass

    class _Messages:
        def __init__(self) -> None:
            self.create_calls = 0
            self.stream_calls = 0

        async def create(self, **_kwargs: Any) -> Any:
            self.create_calls += 1
            raise _Boom("rate limited")

        def stream(self, **_kwargs: Any) -> Any:
            self.stream_calls += 1
            raise AssertionError("stream fallback must not fire")

    class _Client:
        def __init__(self) -> None:
            self.messages = _Messages()

    client = _Client()
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    with pytest.raises(_Boom):
        await model.complete([Message(role=Role.USER, content="q")])
    assert client.messages.create_calls == 1
    assert client.messages.stream_calls == 0


async def test_openai_complete_propagates_real_sdk_errors() -> None:
    class _Boom(RuntimeError):
        pass

    class _Completions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **_kwargs: Any) -> Any:
            self.calls += 1
            raise _Boom("server error")

    class _Client:
        def __init__(self) -> None:
            self.chat = NS(completions=_Completions())

    client = _Client()
    model = OpenAIModel("gpt-4o", client=client)
    with pytest.raises(_Boom):
        await model.complete([Message(role=Role.USER, content="q")])
    assert client.chat.completions.calls == 1  # exactly ONE api call


async def test_anthropic_complete_duck_typing_fallback_still_works() -> None:
    """Fake clients without ``messages.create`` (streaming-only)
    keep working through the AttributeError fallback, and
    prompt_caching now flows through to the stream call."""
    events = [
        NS(type="message_start", message=NS(usage=NS(input_tokens=3, output_tokens=0))),
        NS(type="content_block_start", index=0, content_block=NS(type="text")),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="text_delta", text="hi"),
        ),
        NS(type="content_block_stop", index=0),
        NS(
            type="message_delta",
            delta=NS(stop_reason="end_turn"),
            usage=NS(output_tokens=1),
        ),
        NS(type="message_stop"),
    ]
    client = _FakeStreamClient([events])
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    text, _c, _u, _s = await model.complete(
        [Message(role=Role.USER, content="q")],
        prompt_caching=PromptCacheConfig(enabled=True),
    )
    assert text == "hi"
    # The fallback stream call received the caching config: the
    # system-less request still marks the final message tail.
    kwargs = client.messages.calls[0]
    final = kwargs["messages"][-1]
    assert final["content"][-1]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# 5. RetryingModel — mid-stream errors are never retried
# ---------------------------------------------------------------------------


async def test_midstream_transient_error_is_not_retried() -> None:
    from loomflow.core import TransientModelError
    from loomflow.governance import RetryPolicy
    from loomflow.model.retrying import RetryingModel

    class _MidStreamFlaky:
        name = "flaky"

        def __init__(self) -> None:
            self.attempts = 0

        async def stream(self, messages: Any, **kwargs: Any) -> Any:
            self.attempts += 1
            yield ModelChunk(kind="text", text="one")
            yield ModelChunk(kind="text", text="two")
            raise TransientModelError("connection reset mid-stream")

        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            raise AssertionError("unused")

    inner = _MidStreamFlaky()
    wrapped = RetryingModel(
        inner, RetryPolicy(max_attempts=5, initial_delay_s=0.0, jitter=0.0)
    )
    seen: list[str] = []
    with pytest.raises(TransientModelError):
        async for chunk in wrapped.stream([]):
            seen.append(chunk.text or "")
    # The consumer saw each chunk exactly once and then the error —
    # no silent re-run of the stream, no duplicated chunks.
    assert seen == ["one", "two"]
    assert inner.attempts == 1


# ---------------------------------------------------------------------------
# 6. OpenAI — max_completion_tokens for reasoning models
# ---------------------------------------------------------------------------


class _CapturingOAICompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[
                NS(
                    message=NS(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )


def _capturing_oai_client() -> Any:
    completions = _CapturingOAICompletions()
    return NS(chat=NS(completions=completions)), completions


@pytest.mark.parametrize(
    "model_name,param",
    [
        ("o3-mini", "max_completion_tokens"),
        ("o1-preview", "max_completion_tokens"),
        ("gpt-5-mini", "max_completion_tokens"),
        ("gpt-4o", "max_tokens"),
        ("gpt-4.1-mini", "max_tokens"),
    ],
)
async def test_openai_output_cap_param_by_model_family(
    model_name: str, param: str
) -> None:
    client, completions = _capturing_oai_client()
    model = OpenAIModel(model_name, client=client)
    await model.complete(
        [Message(role=Role.USER, content="q")], max_tokens=128
    )
    assert completions.kwargs[param] == 128
    other = (
        "max_tokens" if param == "max_completion_tokens"
        else "max_completion_tokens"
    )
    assert other not in completions.kwargs


# ---------------------------------------------------------------------------
# 7. Pricing — cost_per_mtoken override + honest unknown-model warning
# ---------------------------------------------------------------------------


def test_estimate_cost_override_beats_table_and_skips_warning() -> None:
    from loomflow.model._pricing import _WARNED_UNKNOWN, estimate_cost

    _WARNED_UNKNOWN.discard("my-custom-finetune")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost = estimate_cost(
            "my-custom-finetune", 1_000_000, 1_000_000,
            override=(2.0, 6.0),
        )
    assert cost == pytest.approx(8.0)
    assert not caught  # override → no unknown-model warning


def test_unknown_model_warning_mentions_budget_not_enforced() -> None:
    from loomflow.model._pricing import _WARNED_UNKNOWN, estimate_cost

    _WARNED_UNKNOWN.discard("mystery-model-abc")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert estimate_cost("mystery-model-abc", 100, 100) == 0.0
    msgs = [str(w.message) for w in caught if "mystery-model-abc" in str(w.message)]
    assert len(msgs) == 1
    assert "max_cost_usd" in msgs[0]
    assert "cost_per_mtoken" in msgs[0]


async def test_openai_adapter_threads_cost_override_into_usage() -> None:
    client, _completions = _capturing_oai_client()
    model = OpenAIModel(
        "my-custom-finetune", client=client, cost_per_mtoken=(100.0, 200.0)
    )
    _t, _c, usage, _f = await model.complete(
        [Message(role=Role.USER, content="q")]
    )
    # 1 input + 1 output token at $100/$200 per MTok.
    assert usage.cost_usd == pytest.approx(300.0 / 1_000_000)


async def test_anthropic_adapter_threads_cost_override_into_usage() -> None:
    client = _FakeCreateClient([_response([NS(type="text", text="ok")])])
    model = AnthropicModel(
        "claude-sonnet-4-5", client=client, cost_per_mtoken=(10.0, 20.0)
    )
    _t, _c, usage, _f = await model.complete(
        [Message(role=Role.USER, content="q")]
    )
    # 10 in + 5 out (fake usage) at $10/$20 per MTok.
    assert usage.cost_usd == pytest.approx((10 * 10.0 + 5 * 20.0) / 1_000_000)


# ---------------------------------------------------------------------------
# 8. tool_result is_error
# ---------------------------------------------------------------------------


def test_tool_result_error_prefix_sets_is_error() -> None:
    msgs = [
        Message(role=Role.USER, content="go"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=(ToolCall(id="tc1", tool="t", args={}),),
        ),
        Message(role=Role.TOOL, content="ERROR: it broke", tool_call_id="tc1"),
    ]
    _, out = _to_anthropic_messages(msgs)
    result = out[-1]["content"][0]
    assert result["type"] == "tool_result"
    assert result["is_error"] is True


def test_tool_result_denied_prefix_sets_is_error() -> None:
    msgs = [
        Message(role=Role.USER, content="go"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=(ToolCall(id="tc1", tool="t", args={}),),
        ),
        Message(role=Role.TOOL, content="DENIED: nope", tool_call_id="tc1"),
    ]
    _, out = _to_anthropic_messages(msgs)
    assert out[-1]["content"][0]["is_error"] is True


def test_tool_result_success_has_no_is_error() -> None:
    msgs = [
        Message(role=Role.USER, content="go"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=(ToolCall(id="tc1", tool="t", args={}),),
        ),
        Message(role=Role.TOOL, content="all good", tool_call_id="tc1"),
    ]
    _, out = _to_anthropic_messages(msgs)
    assert "is_error" not in out[-1]["content"][0]


# ---------------------------------------------------------------------------
# 9. LiteLLM — no native structured output
# ---------------------------------------------------------------------------


def test_litellm_disables_native_structured_output() -> None:
    from loomflow.model.litellm import LiteLLMModel

    assert LiteLLMModel.supports_native_structured_output is False
    # And the request builder never emits response_format either.
    m = LiteLLMModel.__new__(LiteLLMModel)
    m.name = "mistral-large"  # type: ignore[attr-defined]

    from pydantic import BaseModel

    class _Out(BaseModel):
        x: int

    assert m._response_format(_Out) is None  # noqa: SLF001
    assert OpenAIModel.supports_native_structured_output is True


# ---------------------------------------------------------------------------
# ModelChunk — new "thinking" kind is a valid discriminator value
# ---------------------------------------------------------------------------


def test_model_chunk_accepts_thinking_kind() -> None:
    chunk = ModelChunk(kind="thinking", text="hmm")
    assert chunk.kind == "thinking"
    assert chunk.text == "hmm"
