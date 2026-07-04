"""OTel GenAI semantic-convention (``gen_ai.*``) attribute tests.

The framework keeps its legacy ``loom.*`` span names (pinned by
``test_telemetry.py``) and rides the draft-spec ``gen_ai.*``
attributes on the same spans:

* model-call spans carry ``gen_ai.operation.name="chat"``,
  ``gen_ai.provider.name``, ``gen_ai.request.model``, and — set
  post-hoc, once the call completes — ``gen_ai.usage.input_tokens``,
  ``gen_ai.usage.output_tokens``, ``gen_ai.response.finish_reasons``.
* tool spans carry ``gen_ai.operation.name="execute_tool"``,
  ``gen_ai.tool.name``, ``gen_ai.tool.call.id``.
"""

from __future__ import annotations

from typing import Any

import pytest

from loomflow import Agent, tool
from loomflow.core.types import ToolCall, Usage
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.observability import (
    InMemoryTelemetry,
    MultiTelemetry,
)
from loomflow.observability.semconv import (
    OPERATION_CHAT,
    OPERATION_EXECUTE_TOOL,
    PROVIDER_OTHER,
    chat_attrs,
    chat_span_name,
    provider_name,
    set_span_attributes,
    tool_attrs,
    tool_span_name,
    usage_attrs,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@tool
async def ping() -> str:
    """Return pong."""
    return "pong"


def _scripted_tool_flow() -> ScriptedModel:
    """Turn 1: call the ``ping`` tool. Turn 2: final answer."""
    return ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="ping", args={})],
                usage=Usage(input_tokens=11, output_tokens=7),
            ),
            ScriptedTurn(
                text="done",
                usage=Usage(input_tokens=23, output_tokens=5),
            ),
        ]
    )


def _spans_named(tel: InMemoryTelemetry, name: str) -> list[Any]:
    return [s for s in tel.spans() if s.name == name]


# ---------------------------------------------------------------------------
# Pure helper unit tests — semconv.py
# ---------------------------------------------------------------------------


def test_chat_and_tool_span_names_follow_the_spec() -> None:
    assert chat_span_name("gpt-4o") == "chat gpt-4o"
    assert chat_span_name("") == "chat"
    assert tool_span_name("ping") == "execute_tool ping"
    assert tool_span_name("") == "execute_tool"


def test_provider_name_from_adapter_class_name() -> None:
    class AnthropicModel:
        name = "whatever"

    class OpenAIModel:
        name = "whatever"

    assert provider_name(AnthropicModel()) == "anthropic"
    assert provider_name(OpenAIModel()) == "openai"


def test_provider_name_from_model_id_prefix() -> None:
    class M:
        def __init__(self, name: str) -> None:
            self.name = name

    assert provider_name(M("claude-sonnet-4-5")) == "anthropic"
    assert provider_name(M("gpt-4o-mini")) == "openai"
    assert provider_name(M("o3-mini")) == "openai"
    assert provider_name(M("gemini-2.0-flash")) == "gcp.gemini"
    assert provider_name(M("deepseek-chat")) == "deepseek"


def test_provider_name_from_litellm_style_route() -> None:
    class M:
        def __init__(self, name: str) -> None:
            self.name = name

    assert provider_name(M("anthropic/claude-sonnet-4-5")) == "anthropic"
    assert provider_name(M("groq/llama-3.3-70b")) == "groq"
    assert provider_name(M("mistral/mistral-large")) == "mistral_ai"


def test_provider_name_explicit_attribute_wins() -> None:
    class M:
        provider = "bedrock"
        name = "claude-sonnet-4-5"

    assert provider_name(M()) == "aws.bedrock"


def test_provider_name_falls_back_to_other() -> None:
    assert provider_name(ScriptedModel([])) == PROVIDER_OTHER


def test_chat_attrs_shape() -> None:
    attrs = chat_attrs(ScriptedModel([]))
    assert attrs["gen_ai.operation.name"] == OPERATION_CHAT
    assert attrs["gen_ai.provider.name"] == PROVIDER_OTHER
    assert attrs["gen_ai.request.model"] == "scripted"


def test_usage_attrs_shape() -> None:
    attrs = usage_attrs(10, 3, "stop")
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 3
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    # finish_reasons omitted when unknown
    assert "gen_ai.response.finish_reasons" not in usage_attrs(1, 2, None)


def test_tool_attrs_shape() -> None:
    attrs = tool_attrs("ping", call_id="c1")
    assert attrs["gen_ai.operation.name"] == OPERATION_EXECUTE_TOOL
    assert attrs["gen_ai.tool.name"] == "ping"
    assert attrs["gen_ai.tool.call.id"] == "c1"


def test_set_span_attributes_duck_types() -> None:
    # None span (fast-telemetry null context) is a no-op.
    set_span_attributes(None, {"a": 1})

    # ``set_attribute`` method wins when present.
    class FakeOtelSpan:
        def __init__(self) -> None:
            self.seen: dict[str, Any] = {}

        def set_attribute(self, key: str, value: Any) -> None:
            self.seen[key] = value

    fake = FakeOtelSpan()
    set_span_attributes(fake, {"a": 1, "b": None})
    assert fake.seen == {"a": 1}  # None values dropped

    # Fallback: mutable ``attributes`` dict updated in place.
    class ValueSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, Any] = {"x": 0}

    vs = ValueSpan()
    set_span_attributes(vs, {"y": 2, "z": None})
    assert vs.attributes == {"x": 0, "y": 2}


# ---------------------------------------------------------------------------
# End-to-end: agent run through InMemoryTelemetry
# ---------------------------------------------------------------------------


async def test_model_spans_carry_gen_ai_chat_attributes() -> None:
    tel = InMemoryTelemetry()
    agent = Agent(
        "hi", model=_scripted_tool_flow(), tools=[ping], telemetry=tel
    )
    await agent.run("ping please")

    model_spans = _spans_named(tel, "loom.model.complete")
    assert len(model_spans) == 2
    for s in model_spans:
        assert s.attributes["gen_ai.operation.name"] == "chat"
        assert s.attributes["gen_ai.provider.name"] == PROVIDER_OTHER
        assert s.attributes["gen_ai.request.model"] == "scripted"
        # Legacy attributes are still present (additive, no break).
        assert s.attributes["model"] == "scripted"
        assert "turn" in s.attributes


async def test_model_spans_carry_post_hoc_usage_tokens() -> None:
    tel = InMemoryTelemetry()
    agent = Agent(
        "hi", model=_scripted_tool_flow(), tools=[ping], telemetry=tel
    )
    await agent.run("ping please")

    by_turn = {
        s.attributes["turn"]: s
        for s in _spans_named(tel, "loom.model.complete")
    }
    turn1, turn2 = by_turn[1], by_turn[2]
    assert turn1.attributes["gen_ai.usage.input_tokens"] == 11
    assert turn1.attributes["gen_ai.usage.output_tokens"] == 7
    assert turn1.attributes["gen_ai.response.finish_reasons"] == (
        "tool_use",
    )
    assert turn2.attributes["gen_ai.usage.input_tokens"] == 23
    assert turn2.attributes["gen_ai.usage.output_tokens"] == 5
    assert turn2.attributes["gen_ai.response.finish_reasons"] == ("stop",)


async def test_tool_span_carries_gen_ai_execute_tool_attributes() -> None:
    tel = InMemoryTelemetry()
    agent = Agent(
        "hi", model=_scripted_tool_flow(), tools=[ping], telemetry=tel
    )
    await agent.run("ping please")

    tool_spans = _spans_named(tel, "loom.tool")
    assert len(tool_spans) == 1
    s = tool_spans[0]
    assert s.attributes["gen_ai.operation.name"] == "execute_tool"
    assert s.attributes["gen_ai.tool.name"] == "ping"
    assert s.attributes["gen_ai.tool.call.id"] == "c1"
    # Legacy attributes preserved.
    assert s.attributes["tool"] == "ping"
    assert s.attributes["call_id"] == "c1"


async def test_streaming_model_span_carries_gen_ai_attributes() -> None:
    tel = InMemoryTelemetry()
    agent = Agent(
        "hi", model=_scripted_tool_flow(), tools=[ping], telemetry=tel
    )
    async for _ in agent.stream("ping please"):
        pass

    stream_spans = _spans_named(tel, "loom.model.stream")
    assert len(stream_spans) == 2
    for s in stream_spans:
        assert s.attributes["gen_ai.operation.name"] == "chat"
        assert s.attributes["gen_ai.provider.name"] == PROVIDER_OTHER
        assert s.attributes["gen_ai.request.model"] == "scripted"
        assert "gen_ai.usage.input_tokens" in s.attributes
        assert "gen_ai.usage.output_tokens" in s.attributes
    first = min(stream_spans, key=lambda s: s.attributes["turn"])
    assert first.attributes["gen_ai.usage.input_tokens"] == 11
    assert first.attributes["gen_ai.response.finish_reasons"] == (
        "tool_use",
    )


async def test_multi_telemetry_propagates_post_hoc_attrs_to_sinks() -> None:
    """Post-hoc gen_ai.usage.* set on the composed span must land in
    every capture sink's record (the composer fans the additions out
    before the sinks close)."""
    in_mem = InMemoryTelemetry()
    tel = MultiTelemetry([in_mem])
    agent = Agent(
        "hi", model=_scripted_tool_flow(), tools=[ping], telemetry=tel
    )
    await agent.run("ping please")

    model_spans = _spans_named(in_mem, "loom.model.complete")
    assert len(model_spans) == 2
    for s in model_spans:
        assert s.attributes["gen_ai.operation.name"] == "chat"
        assert "gen_ai.usage.input_tokens" in s.attributes
        assert "gen_ai.usage.output_tokens" in s.attributes


# ---------------------------------------------------------------------------
# OTelTelemetry — post-hoc attributes reach the exported OTel span
# ---------------------------------------------------------------------------


async def test_otel_spans_carry_gen_ai_attributes_and_usage() -> None:
    otel_sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from loomflow.observability import OTelTelemetry

    exporter = InMemorySpanExporter()
    provider = otel_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tel = OTelTelemetry(tracer_provider=provider)

    agent = Agent(
        "hi", model=_scripted_tool_flow(), tools=[ping], telemetry=tel
    )
    await agent.run("ping please")

    finished = exporter.get_finished_spans()
    model_spans = [
        s for s in finished if s.name == "loom.model.complete"
    ]
    assert len(model_spans) == 2
    for s in model_spans:
        assert s.attributes is not None
        assert s.attributes["gen_ai.operation.name"] == "chat"
        assert s.attributes["gen_ai.provider.name"] == PROVIDER_OTHER
        assert s.attributes["gen_ai.request.model"] == "scripted"
        # Post-hoc usage made it onto the exported span.
        assert "gen_ai.usage.input_tokens" in s.attributes
        assert "gen_ai.usage.output_tokens" in s.attributes
        assert "gen_ai.response.finish_reasons" in s.attributes

    tool_spans = [s for s in finished if s.name == "loom.tool"]
    assert len(tool_spans) == 1
    t = tool_spans[0]
    assert t.attributes is not None
    assert t.attributes["gen_ai.operation.name"] == "execute_tool"
    assert t.attributes["gen_ai.tool.name"] == "ping"
    assert t.attributes["gen_ai.tool.call.id"] == "c1"
