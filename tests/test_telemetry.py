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

from loomflow import Agent, tool
from loomflow.core.types import ToolCall
from loomflow.governance.budget import BudgetConfig, StandardBudget
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.observability import OTelTelemetry
from loomflow.observability.tracing import NoTelemetry

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
    await tel.emit_metric("loom.foo", 1)
    await tel.emit_metric("loom.foo_ms", 100)


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
    assert "loom.run" in names
    assert "loom.turn" in names
    # ReAct emits ``jeeves.model.complete`` on the non-streaming hot
    # path (agent.run) and ``jeeves.model.stream`` when consuming
    # via agent.stream. Either span name is correct.
    assert (
        "loom.model.complete" in names
        or "loom.model.stream" in names
    )

    metrics = _metrics_by_name(reader)
    assert "loom.tokens.input" in metrics
    assert "loom.tokens.output" in metrics
    assert "loom.session.duration_ms" in metrics


async def test_run_span_has_session_id_and_model_attributes() -> None:
    tel, exporter, _ = _setup_otel()
    agent = Agent("hi", model="echo", telemetry=tel)
    result = await agent.run("anything")

    run_span = next(
        s for s in exporter.get_finished_spans() if s.name == "loom.run"
    )
    assert run_span.attributes["session_id"] == result.session_id
    assert run_span.attributes["model"] == "echo"


async def test_turn_span_is_a_child_of_run_span() -> None:
    tel, exporter, _ = _setup_otel()
    agent = Agent("hi", model="echo", telemetry=tel)
    await agent.run("hello")

    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    run = by_name["loom.run"]
    turn = by_name["loom.turn"]
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
        s for s in exporter.get_finished_spans() if s.name == "loom.tool"
    ]
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes["tool"] == "ping"
    assert tool_spans[0].attributes["call_id"] == "c1"

    metrics = _metrics_by_name(reader)
    assert "loom.tool.duration_ms" in metrics
    pt = metrics["loom.tool.duration_ms"][0]
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
        s for s in exporter.get_finished_spans() if s.name == "loom.tool"
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
    assert "loom.budget.exceeded" in metrics
    pt = metrics["loom.budget.exceeded"][0]
    assert pt.value == 1


async def test_session_duration_metric_recorded_once_per_run() -> None:
    tel, _, reader = _setup_otel()
    agent = Agent("hi", model="echo", telemetry=tel)
    await agent.run("first")
    await agent.run("second")

    metrics = _metrics_by_name(reader)
    histo_pts = metrics["loom.session.duration_ms"]
    assert len(histo_pts) >= 1
    total_count = sum(pt.count for pt in histo_pts)
    assert total_count == 2


# ---------------------------------------------------------------------------
# InMemoryTelemetry / ConsoleTelemetry / MultiTelemetry — no OTel SDK
# ---------------------------------------------------------------------------


import io  # noqa: E402

from loomflow.observability import (  # noqa: E402
    ConsoleTelemetry,
    InMemoryTelemetry,
    MultiTelemetry,
)


async def test_in_memory_telemetry_captures_spans_with_attrs() -> None:
    """``InMemoryTelemetry`` records every span the agent loop
    opens. Spans carry their attributes intact so tests can
    assert on what the framework instrumented."""
    tel = InMemoryTelemetry()
    agent = Agent("hi", model="echo", telemetry=tel)
    await agent.run("hello", user_id="alice")

    span_names = {s.name for s in tel.spans()}
    assert "loom.run" in span_names
    assert "loom.turn" in span_names

    run_span = next(s for s in tel.spans() if s.name == "loom.run")
    assert run_span.duration_ms > 0
    # session_id and model should be on the run span (framework
    # adds them in the trace() call).
    assert "session_id" in run_span.attributes


async def test_in_memory_telemetry_records_parent_span_hierarchy() -> None:
    """Nested spans should record the correct parent_span_id —
    proves contextvar-based parent tracking works across the
    agent loop's async stack. This is the test that catches
    silent regressions if anyone refactors task-spawning to use
    a primitive that doesn't propagate contextvars."""
    tel = InMemoryTelemetry()
    agent = Agent("hi", model="echo", telemetry=tel)
    await agent.run("test")

    spans = tel.spans()
    run_span = next(s for s in spans if s.name == "loom.run")
    # The run span is the root — no parent.
    assert run_span.parent_span_id is None
    # turn spans should be children of run.
    turn_spans = [s for s in spans if s.name == "loom.turn"]
    assert turn_spans
    for ts in turn_spans:
        assert ts.parent_span_id == run_span.span_id


async def test_in_memory_telemetry_captures_metrics_with_kind() -> None:
    """Every emit_metric call lands as a CapturedMetric tagged
    with counter-vs-histogram based on the name suffix."""
    tel = InMemoryTelemetry()
    await tel.emit_metric("loom.tokens.input", 42)
    await tel.emit_metric("loom.session.duration_ms", 100.5)

    metrics = tel.metrics()
    assert len(metrics) == 2
    by_name = {m.name: m for m in metrics}
    assert by_name["loom.tokens.input"].instrument_kind == "counter"
    assert by_name["loom.session.duration_ms"].instrument_kind == "histogram"


async def test_in_memory_telemetry_clear_resets_state() -> None:
    """``.clear()`` between test cases lets the accumulator be
    reused without bleeding state."""
    tel = InMemoryTelemetry()
    await tel.emit_metric("loom.x", 1)
    async with tel.trace("test"):
        pass
    assert tel.spans() and tel.metrics()
    tel.clear()
    assert tel.spans() == [] and tel.metrics() == []


async def test_in_memory_telemetry_records_exceptions() -> None:
    """A span that raises should be recorded with the
    exception repr — useful for assertions like
    ``assert any(s.exception for s in tel.spans())``."""
    tel = InMemoryTelemetry()
    with pytest.raises(RuntimeError, match="boom"):
        async with tel.trace("danger"):
            raise RuntimeError("boom")
    span = tel.spans()[0]
    assert span.name == "danger"
    assert span.exception is not None
    assert "boom" in span.exception


async def test_console_telemetry_writes_span_completions_to_stream() -> None:
    """``ConsoleTelemetry`` emits one line per span completion
    with name, duration, and attributes. Default stream is
    stderr; tests pass a StringIO."""
    buf = io.StringIO()
    tel = ConsoleTelemetry(stream=buf)
    async with tel.trace("outer", user_id="alice"):
        async with tel.trace("inner"):
            pass

    output = buf.getvalue()
    assert "outer" in output
    assert "inner" in output
    assert "user_id=alice" in output
    # "inner" closes before "outer", so it appears first in the stream.
    assert output.index("inner") < output.index("outer")


async def test_console_telemetry_shows_metrics_with_kind() -> None:
    """Metric emits print with the auto-detected instrument
    kind so the dev viewer can tell counters from histograms
    at a glance."""
    buf = io.StringIO()
    tel = ConsoleTelemetry(stream=buf)
    await tel.emit_metric("loom.tokens.input", 100)
    await tel.emit_metric("loom.session.duration_ms", 250.5)

    output = buf.getvalue()
    assert "(counter)" in output
    assert "(histogram)" in output


async def test_console_telemetry_show_metrics_false_suppresses_metric_lines() -> None:
    """``show_metrics=False`` is for users who only want span
    traces and find per-token metric chatter distracting."""
    buf = io.StringIO()
    tel = ConsoleTelemetry(stream=buf, show_metrics=False)
    await tel.emit_metric("loom.tokens.input", 100)
    assert buf.getvalue() == ""


async def test_multi_telemetry_fans_spans_and_metrics_to_every_sink() -> None:
    """``MultiTelemetry`` forwards each ``trace`` /
    ``emit_metric`` call to every sink. Watch live in stderr
    AND assert in tests — the canonical fan-out pattern."""
    in_mem = InMemoryTelemetry()
    buf = io.StringIO()
    console = ConsoleTelemetry(stream=buf)
    tel = MultiTelemetry([in_mem, console])

    async with tel.trace("span_a", user_id="bob"):
        pass
    await tel.emit_metric("loom.tokens.input", 7)

    # In-memory side has the span recorded.
    in_mem_spans = in_mem.spans()
    assert len(in_mem_spans) == 1
    assert in_mem_spans[0].name == "span_a"
    assert in_mem_spans[0].attributes["user_id"] == "bob"
    # Metric also captured.
    assert any(m.name == "loom.tokens.input" for m in in_mem.metrics())
    # Console stream has both.
    out = buf.getvalue()
    assert "span_a" in out
    assert "loom.tokens.input" in out


async def test_multi_telemetry_empty_sinks_rejected_with_clear_error() -> None:
    """``MultiTelemetry([])`` is meaningless — use ``NoTelemetry``
    instead. Fail at construction with a message that names the
    fix."""
    with pytest.raises(ValueError, match="at least one sink"):
        MultiTelemetry([])
