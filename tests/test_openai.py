"""OpenAIModel adapter — chunk normalization with a fake SDK client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any

import pytest

from jeevesagent import Agent
from jeevesagent.core.types import Message, ModelChunk, Role, ToolCall
from jeevesagent.model.openai import OpenAIModel, _to_openai_messages

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fake client mimicking openai.AsyncOpenAI.chat.completions.create(stream=True)
# ---------------------------------------------------------------------------


class _FakeOAIStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> _FakeOAIStream:
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration as e:
            raise StopAsyncIteration from e


class _FakeCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.captured_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeOAIStream:
        self.captured_kwargs = kwargs
        return _FakeOAIStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks: list[Any]) -> None:
        self.completions = _FakeCompletions(chunks)


class _FakeOAIClient:
    def __init__(self, chunks: list[Any]) -> None:
        self.chat = _FakeChat(chunks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect(stream: AsyncIterator[ModelChunk]) -> list[ModelChunk]:
    return [c async for c in stream]


def _text_chunk(content: str) -> Any:
    return NS(
        usage=None,
        choices=[
            NS(
                delta=NS(content=content, tool_calls=None),
                finish_reason=None,
            )
        ],
    )


def _finish_chunk(reason: str = "stop") -> Any:
    return NS(
        usage=None,
        choices=[
            NS(
                delta=NS(content=None, tool_calls=None),
                finish_reason=reason,
            )
        ],
    )


def _usage_chunk(prompt_tokens: int, completion_tokens: int) -> Any:
    return NS(
        usage=NS(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        choices=[],
    )


# ---------------------------------------------------------------------------
# Streaming behavior
# ---------------------------------------------------------------------------


async def test_text_deltas_normalize_to_text_chunks() -> None:
    chunks = [
        _text_chunk("Hello "),
        _text_chunk("world"),
        _finish_chunk("stop"),
        _usage_chunk(7, 3),
    ]
    model = OpenAIModel("gpt-4o", client=_FakeOAIClient(chunks))
    out = await _collect(model.stream([Message(role=Role.USER, content="hi")]))

    text_pieces = [c.text for c in out if c.kind == "text"]
    assert text_pieces == ["Hello ", "world"]

    (finish,) = [c for c in out if c.kind == "finish"]
    assert finish.finish_reason == "stop"
    assert finish.usage is not None
    assert finish.usage.input_tokens == 7
    assert finish.usage.output_tokens == 3


async def test_tool_call_deltas_aggregate_by_index() -> None:
    chunks = [
        # Initial tool call delta with id and name, no args yet
        NS(
            usage=None,
            choices=[
                NS(
                    delta=NS(
                        content=None,
                        tool_calls=[
                            NS(
                                index=0,
                                id="call_42",
                                function=NS(name="get_weather", arguments=""),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        # Argument fragments
        NS(
            usage=None,
            choices=[
                NS(
                    delta=NS(
                        content=None,
                        tool_calls=[
                            NS(
                                index=0,
                                id=None,
                                function=NS(name=None, arguments='{"city":'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        NS(
            usage=None,
            choices=[
                NS(
                    delta=NS(
                        content=None,
                        tool_calls=[
                            NS(
                                index=0,
                                id=None,
                                function=NS(name=None, arguments=' "Paris"}'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        _finish_chunk("tool_calls"),
        _usage_chunk(20, 12),
    ]
    model = OpenAIModel(client=_FakeOAIClient(chunks))
    out = await _collect(model.stream([Message(role=Role.USER, content="weather?")]))

    tool_chunks = [c for c in out if c.kind == "tool_call"]
    assert len(tool_chunks) == 1
    tc = tool_chunks[0].tool_call
    assert tc is not None
    assert tc.id == "call_42"
    assert tc.tool == "get_weather"
    assert tc.args == {"city": "Paris"}

    (finish,) = [c for c in out if c.kind == "finish"]
    assert finish.finish_reason == "tool_calls"


async def test_agent_run_with_openai_fake_returns_text() -> None:
    chunks = [
        _text_chunk("forty-two"),
        _finish_chunk("stop"),
        _usage_chunk(5, 3),
    ]
    model = OpenAIModel("gpt-4o-mini", client=_FakeOAIClient(chunks))
    agent = Agent("ok", model=model)
    result = await agent.run("answer?")
    assert result.output == "forty-two"
    assert result.tokens_in == 5
    assert result.tokens_out == 3


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def test_assistant_message_with_tool_calls_serializes_arguments_as_json() -> None:
    msgs = [
        Message(role=Role.USER, content="weather?"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=(
                ToolCall(id="call_1", tool="get_weather", args={"city": "Tokyo"}),
            ),
        ),
        Message(role=Role.TOOL, content="cloudy", tool_call_id="call_1"),
    ]
    out = _to_openai_messages(msgs)

    assert out[0] == {"role": "user", "content": "weather?"}
    assert out[1]["role"] == "assistant"
    assert "tool_calls" in out[1]
    tc = out[1]["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "get_weather"
    # Arguments are JSON-serialised in OpenAI's format.
    import json as _json

    assert _json.loads(tc["function"]["arguments"]) == {"city": "Tokyo"}

    assert out[2] == {
        "role": "tool",
        "content": "cloudy",
        "tool_call_id": "call_1",
    }


def test_tool_definition_converts_to_openai_function_tool() -> None:
    from jeevesagent.core.types import ToolDef
    from jeevesagent.model.openai import _to_openai_tool

    td = ToolDef(
        name="lookup",
        description="Find a thing.",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    out = _to_openai_tool(td)
    assert out["type"] == "function"
    assert out["function"]["name"] == "lookup"
    assert out["function"]["description"] == "Find a thing."
    assert out["function"]["parameters"]["properties"]["q"]["type"] == "string"


# ---------------------------------------------------------------------------
# Resolver (string -> instance)
# ---------------------------------------------------------------------------


async def test_string_model_spec_resolves_to_correct_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing Agent with a string spec dispatches by prefix.

    We verify the resolver picks the right adapter and forwards the
    model-id string. Dummy API keys keep SDK construction quiet —
    no network calls happen here.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    a = Agent("hi", model="gpt-4o")
    assert type(a._model).__name__ == "OpenAIModel"
    assert a._model.name == "gpt-4o"

    b = Agent("hi", model="claude-opus-4-7")
    assert type(b._model).__name__ == "AnthropicModel"
    assert b._model.name == "claude-opus-4-7"

    c = Agent("hi", model="echo")
    assert type(c._model).__name__ == "EchoModel"


async def test_unknown_model_string_raises() -> None:
    """0.2.0 harmonised the resolver's error to ConfigError (was
    ValueError in 0.1.x). The message now also lists LiteLLM prefixes
    and the explicit ``litellm/`` opt-in."""
    from jeevesagent.core.errors import ConfigError

    with pytest.raises(ConfigError, match="unknown model spec"):
        Agent("hi", model="totally-unknown-prefix-xyz")
