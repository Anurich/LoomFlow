"""OpenTelemetry adapter tests.

Wires up an in-memory ``TracerProvider`` and ``MeterProvider`` so we can
assert on captured spans and metrics without any real exporters.
"""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from jeevesagent import Agent, OTelTelemetry, tool
from jeevesagent.core.types import ToolCall
from jeevesagent.governance.budget import BudgetConfig, StandardBudget
from jeevesagent.model.scripted import ScriptedModel, ScriptedTurn
from jeevesagent.observability.tracing import NoTelemetry

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_otel() -> tuple[
    OTelTelemetry, InMemorySpanExporter, InMemoryMetricReader
]:
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    telemetry = OTelTelemetry(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    return telemetry, span_exporter, metric_reader


def _metrics_by_name(reader: InMemoryMetricReader) -> dict[str, list[Any]]:
    """Flatten the OTel metric tree into ``{name: [data_point, ...]}``."""
    out: dict[str, list[Any]] = {}
    data = reader.get_metrics_data()
    if data is None:
        return out
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                out.setdefault(metric.name, []).extend(
                    metric.data.data_points
                )
    return out


# ---------------------------------------------------------------------------
# NoTelemetry — confirms the default is a true no-op
# ---------------------------------------------------------------------------


async def test_no_telemetry_trace_yields_a_span_object() -> None:
    tel = NoTelemetry()
    async with tel.trace("anything", x=1) as span:
        assert span.name == "anything"
        assert span.attributes == {"x": 1}


async def test_no_telemetry_emit_metric_is_a_noop() -> None:
    tel = NoTelemetry()
    # Just confirms it doesn't raise; nothing to assert on.
    await tel.emit_metric("jeeves.foo", 1)
    await tel.emit_metric("jeeves.foo_ms", 100)


# ---------------------------------------------------------------------------
# OTelTelemetry — direct usage (no agent loop)
# ---------------------------------------------------------------------------


async def test_otel_trace_records_a_span_with_attributes() -> None:
    tel, exporter, _ = _setup_otel()

    async with tel.trace("test.span", color="blue", count=3) as span:
        assert span.name == "test.span"
        assert len(span.trace_id) == 32  # hex of 128-bit trace_id

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "test.span"
    assert s.attributes["color"] == "blue"
    assert s.attributes["count"] == 3


async def test_otel_trace_filters_none_attributes() -> None:
    """OTel rejects None attribute values; the adapter must strip them."""
    tel, exporter, _ = _setup_otel()
    async with tel.trace("test", real="value", nothing=None):
        pass
    s = exporter.get_finished_spans()[0]
    assert "real" in s.attributes
    assert "nothing" not in s.attributes


async def test_otel_trace_records_exception_and_marks_status_error() -> None:
    tel, exporter, _ = _setup_otel()
    with pytest.raises(RuntimeError):
        async with tel.trace("flaky"):
            raise RuntimeError("kaboom")

    s = exporter.get_finished_spans()[0]
    assert s.status.status_code.name == "ERROR"
    # The exception is recorded as an event on the span.
    assert any("exception" in e.name for e in s.events)


async def test_otel_emit_metric_routes_to_counter_or_histogram() -> None:
    tel, _, reader = _setup_otel()
    await tel.emit_metric("widget.count", 5)  # counter
    await tel.emit_metric("widget.duration_ms", 42.0)  # histogram

    found = _metrics_by_name(reader)
    assert "widget.count" in found
    assert "widget.duration_ms" in found
    # Counter sum
    sum_pt = found["widget.count"][0]
    assert sum_pt.value == 5
    # Histogram sum/count
    histo_pt = found["widget.duration_ms"][0]
    assert histo_pt.sum == 42.0
    assert histo_pt.count == 1


async def test_otel_metric_attributes_are_propagated() -> None:
    tel, _, reader = _setup_otel()
    await tel.emit_metric("foo.count", 1, kind="alpha")
    await tel.emit_metric("foo.count", 2, kind="beta")

    found = _metrics_by_name(reader)
    pts = found["foo.count"]
    by_kind = {dict(pt.attributes)["kind"]: pt.value for pt in pts}
    assert by_kind == {"alpha": 1, "beta": 2}


# ---------------------------------------------------------------------------
# Wired into Agent: per-run / turn / model.stream / tool spans
# ---------------------------------------------------------------------------


async def test_agent_run_emits_run_turn_and_model_spans() -> None:
    tel, exporter, reader = _setup_otel()
    agent = Agent("hi", model="echo", telemetry=tel)
    await agent.run("hello world")

    names = [s.name for s in exporter.get_finished_spans()]
    assert "jeeves.run" in names
    assert "jeeves.turn" in names
    assert "jeeves.model.stream" in names

    metrics = _metrics_by_name(reader)
    assert "jeeves.tokens.input" in metrics
    assert "jeeves.tokens.output" in metrics
    assert "jeeves.session.duration_ms" in metrics


async def test_run_span_has_session_id_and_model_attributes() -> None:
    tel, exporter, _ = _setup_otel()
    agent = Agent("hi", model="echo", telemetry=tel)
    result = await agent.run("anything")

    run_span = next(
        s for s in exporter.get_finished_spans() if s.name == "jeeves.run"
    )
    assert run_span.attributes["session_id"] == result.session_id
    assert run_span.attributes["model"] == "echo"


async def test_turn_span_is_a_child_of_run_span() -> None:
    tel, exporter, _ = _setup_otel()
    agent = Agent("hi", model="echo", telemetry=tel)
    await agent.run("hello")

    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    run = by_name["jeeves.run"]
    turn = by_name["jeeves.turn"]
    # The turn span's parent must be the run span.
    assert turn.parent is not None
    assert turn.parent.span_id == run.context.span_id


async def test_tool_span_emitted_with_tool_attribute_and_duration_metric() -> None:
    @tool
    async def ping() -> str:
        """Return pong."""
        return "pong"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="ping", args={})]
            ),
            ScriptedTurn(text="ok"),
        ]
    )
    tel, exporter, reader = _setup_otel()
    agent = Agent("hi", model=model, tools=[ping], telemetry=tel)
    await agent.run("ping?")

    tool_spans = [
        s for s in exporter.get_finished_spans() if s.name == "jeeves.tool"
    ]
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes["tool"] == "ping"
    assert tool_spans[0].attributes["call_id"] == "c1"

    metrics = _metrics_by_name(reader)
    assert "jeeves.tool.duration_ms" in metrics
    pt = metrics["jeeves.tool.duration_ms"][0]
    # Histogram entry exists with expected attributes.
    assert pt.count == 1
    attrs = dict(pt.attributes)
    assert attrs["tool"] == "ping"
    assert attrs["ok"] is True


async def test_parallel_tool_calls_emit_independent_tool_spans() -> None:
    @tool
    async def alpha() -> str:
        return "a"

    @tool
    async def beta() -> str:
        return "b"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="ca", tool="alpha", args={}),
                    ToolCall(id="cb", tool="beta", args={}),
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    tel, exporter, _ = _setup_otel()
    agent = Agent("hi", model=model, tools=[alpha, beta], telemetry=tel)
    await agent.run("...")

    tool_spans = [
        s for s in exporter.get_finished_spans() if s.name == "jeeves.tool"
    ]
    tools = {s.attributes["tool"] for s in tool_spans}
    assert tools == {"alpha", "beta"}


async def test_budget_exceeded_increments_metric_and_completes_run() -> None:
    tel, _, reader = _setup_otel()
    budget = StandardBudget(BudgetConfig(max_tokens=0))
    agent = Agent("hi", model="echo", budget=budget, telemetry=tel)

    result = await agent.run("anything")
    assert result.interrupted

    metrics = _metrics_by_name(reader)
    assert "jeeves.budget.exceeded" in metrics
    pt = metrics["jeeves.budget.exceeded"][0]
    assert pt.value == 1


async def test_session_duration_metric_recorded_once_per_run() -> None:
    tel, _, reader = _setup_otel()
    agent = Agent("hi", model="echo", telemetry=tel)
    await agent.run("first")
    await agent.run("second")

    metrics = _metrics_by_name(reader)
    histo_pts = metrics["jeeves.session.duration_ms"]
    assert len(histo_pts) >= 1
    total_count = sum(pt.count for pt in histo_pts)
    assert total_count == 2
