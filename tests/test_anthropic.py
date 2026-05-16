"""AnthropicModel adapter — chunk normalization with a fake SDK client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any

import pytest

from loomflow import Agent
from loomflow.core.types import Message, ModelChunk, Role
from loomflow.model.anthropic import AnthropicModel, _to_anthropic_messages

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fake client mimicking anthropic.AsyncAnthropic.messages.stream(...)
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


class _FakeMessages:
    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.captured_kwargs: dict[str, Any] | None = None

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.captured_kwargs = kwargs
        return _FakeStream(self._events)


class _FakeAnthropicClient:
    def __init__(self, events: list[Any]) -> None:
        self.messages = _FakeMessages(events)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect(stream: AsyncIterator[ModelChunk]) -> list[ModelChunk]:
    return [c async for c in stream]


# ---------------------------------------------------------------------------
# Streaming behavior
# ---------------------------------------------------------------------------


async def test_text_deltas_normalize_to_text_chunks_and_finish() -> None:
    events = [
        NS(type="message_start", message=NS(usage=NS(input_tokens=10, output_tokens=0))),
        NS(type="content_block_start", index=0, content_block=NS(type="text")),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="text_delta", text="Hello "),
        ),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="text_delta", text="world"),
        ),
        NS(type="content_block_stop", index=0),
        NS(
            type="message_delta",
            delta=NS(stop_reason="end_turn"),
            usage=NS(output_tokens=5),
        ),
        NS(type="message_stop"),
    ]
    model = AnthropicModel("claude-opus-4-7", client=_FakeAnthropicClient(events))
    chunks = await _collect(model.stream([Message(role=Role.USER, content="hi")]))

    text_pieces = [c.text for c in chunks if c.kind == "text"]
    assert text_pieces == ["Hello ", "world"]

    (finish,) = [c for c in chunks if c.kind == "finish"]
    assert finish.finish_reason == "end_turn"
    assert finish.usage is not None
    assert finish.usage.input_tokens == 10
    assert finish.usage.output_tokens == 5


async def test_tool_use_block_aggregates_partial_json() -> None:
    events = [
        NS(type="message_start", message=NS(usage=NS(input_tokens=12, output_tokens=0))),
        NS(
            type="content_block_start",
            index=0,
            content_block=NS(type="tool_use", id="tu_abc", name="get_weather"),
        ),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="input_json_delta", partial_json='{"city":'),
        ),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="input_json_delta", partial_json=' "SF"}'),
        ),
        NS(type="content_block_stop", index=0),
        NS(
            type="message_delta",
            delta=NS(stop_reason="tool_use"),
            usage=NS(output_tokens=8),
        ),
        NS(type="message_stop"),
    ]
    model = AnthropicModel(client=_FakeAnthropicClient(events))
    chunks = await _collect(model.stream([Message(role=Role.USER, content="weather?")]))

    tool_chunks = [c for c in chunks if c.kind == "tool_call"]
    assert len(tool_chunks) == 1
    tc = tool_chunks[0].tool_call
    assert tc is not None
    assert tc.id == "tu_abc"
    assert tc.tool == "get_weather"
    assert tc.args == {"city": "SF"}

    (finish,) = [c for c in chunks if c.kind == "finish"]
    assert finish.finish_reason == "tool_use"


async def test_agent_run_with_anthropic_fake_returns_expected_text() -> None:
    events = [
        NS(type="message_start", message=NS(usage=NS(input_tokens=5, output_tokens=0))),
        NS(type="content_block_start", index=0, content_block=NS(type="text")),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="text_delta", text="The answer is 42."),
        ),
        NS(type="content_block_stop", index=0),
        NS(
            type="message_delta",
            delta=NS(stop_reason="end_turn"),
            usage=NS(output_tokens=6),
        ),
        NS(type="message_stop"),
    ]
    model = AnthropicModel("claude-opus-4-7", client=_FakeAnthropicClient(events))
    agent = Agent("be helpful", model=model)

    result = await agent.run("anything")
    assert result.output == "The answer is 42."
    assert result.tokens_in == 5
    assert result.tokens_out == 6


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def test_system_messages_kept_as_list_for_cache_block_emission() -> None:
    """As of 0.10.13, ``_to_anthropic_messages`` returns
    ``system_parts: list[str]`` (one entry per Role.SYSTEM message)
    so the cache-control helper can place independent ``cache_control``
    markers on the last N parts (instructions / memory / recall).
    Callers join with ``\\n\\n`` only when caching is off."""
    msgs = [
        Message(role=Role.SYSTEM, content="be terse"),
        Message(role=Role.SYSTEM, content="be polite"),
        Message(role=Role.USER, content="hi"),
    ]
    system_parts, out = _to_anthropic_messages(msgs)
    assert system_parts == ["be terse", "be polite"]
    assert "\n\n".join(system_parts) == "be terse\n\nbe polite"
    assert out == [{"role": "user", "content": "hi"}]


def test_assistant_with_tool_calls_emits_tool_use_blocks() -> None:
    from loomflow.core.types import ToolCall

    msgs = [
        Message(role=Role.USER, content="weather?"),
        Message(
            role=Role.ASSISTANT,
            content="checking",
            tool_calls=(ToolCall(id="tu1", tool="get_weather", args={"city": "SF"}),),
        ),
        Message(role=Role.TOOL, content="sunny", tool_call_id="tu1"),
        Message(role=Role.USER, content="thanks"),
    ]
    _, out = _to_anthropic_messages(msgs)

    # First: user "weather?"
    assert out[0] == {"role": "user", "content": "weather?"}
    # Second: assistant with text + tool_use
    assert out[1]["role"] == "assistant"
    blocks = out[1]["content"]
    assert {"type": "text", "text": "checking"} in blocks
    assert any(
        b.get("type") == "tool_use" and b.get("id") == "tu1" for b in blocks
    )
    # Third: user with tool_result block
    assert out[2]["role"] == "user"
    assert out[2]["content"][0]["type"] == "tool_result"
    assert out[2]["content"][0]["tool_use_id"] == "tu1"
    # Fourth: user "thanks" (a fresh user turn after tool_result was flushed)
    assert out[3] == {"role": "user", "content": "thanks"}
