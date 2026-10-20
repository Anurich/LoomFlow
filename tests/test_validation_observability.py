"""Telemetry follows the OpenTelemetry GenAI semantic conventions.

Model spans were named ``loom.model.complete`` with kind INTERNAL,
input tokens left out cached ones, failed tools had status UNSET,
exceptions were recorded twice, and metrics had no units and carried
the session id (unbounded cardinality). The serve ``/stream`` endpoint
also reused the ``error`` event name for its own failure frame.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

otel_sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from loomflow import Agent, ScriptedModel, ScriptedTurn, tool  # noqa: E402
from loomflow.core.types import ToolCall, Usage  # noqa: E402
from loomflow.observability import OTelTelemetry  # noqa: E402

pytestmark = pytest.mark.anyio


def _telemetry() -> tuple[OTelTelemetry, InMemorySpanExporter, InMemoryMetricReader]:
    exporter = InMemorySpanExporter()
    provider = otel_sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    tel = OTelTelemetry(tracer_provider=provider, meter_provider=MeterProvider(metric_readers=[reader]))
    return tel, exporter, reader


def _metrics(reader: InMemoryMetricReader) -> dict[str, Any]:
    data = reader.get_metrics_data()
    found: dict[str, Any] = {}
    for rm in data.resource_metrics if data else ():
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                found[metric.name] = metric
    return found


def _sdk_honours_bucket_advisory() -> bool:
    """Older opentelemetry-sdk releases accept but ignore the advisory."""
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("probe")
    try:
        histogram = meter.create_histogram("probe", explicit_bucket_boundaries_advisory=[1.0, 2.0])
    except TypeError:
        return False
    histogram.record(1)
    data = reader.get_metrics_data()
    point = data.resource_metrics[0].scope_metrics[0].metrics[0].data.data_points[0]
    return list(point.explicit_bounds) == [1.0, 2.0]


async def test_spans_and_metrics_follow_the_genai_semconv() -> None:
    @tool
    async def boom() -> str:
        """Always fails."""
        raise ValueError("nope")

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="boom", args={})],
                usage=Usage(input_tokens=10, cached_input_tokens=90, output_tokens=5),
            ),
            ScriptedTurn(text="done", usage=Usage(input_tokens=12, output_tokens=3)),
        ]
    )
    tel, exporter, reader = _telemetry()
    await Agent("x", model=model, tools=[boom], telemetry=tel).run("go")
    spans = exporter.get_finished_spans()
    chat = [s for s in spans if s.name == "chat scripted"]
    assert len(chat) == 2 and all(s.kind.name == "CLIENT" for s in chat)
    first = chat[0].attributes or {}
    assert first["loom.span.name"] == "loom.model.complete"
    # Input tokens include the cached ones; the cache count is its own attribute.
    assert first["gen_ai.usage.input_tokens"] == 100
    assert first["gen_ai.usage.cache_read.input_tokens"] == 90
    [failed_tool] = [s for s in spans if s.name == "execute_tool boom"]
    assert failed_tool.status.status_code.name == "ERROR"
    assert (failed_tool.attributes or {})["error.type"] == "tool_error"
    assert any(s.name == "invoke_agent" for s in spans)

    metrics = _metrics(reader)
    duration = metrics["gen_ai.client.inference.duration"]
    assert duration.unit == "s"
    point = duration.data.data_points[0]
    if _sdk_honours_bucket_advisory():
        assert list(point.explicit_bounds)[:3] == [0.01, 0.02, 0.04]
    usage = metrics["gen_ai.client.inference.usage.input_tokens"]
    assert usage.unit == "{token}"
    assert sum(p.value for p in usage.data.data_points) == 112
    assert metrics["gen_ai.execute_tool.duration"].unit == "s"
    assert metrics["loom.tool.duration_ms"].unit == "ms"
    for metric in metrics.values():
        for p in metric.data.data_points:
            assert "session_id" not in (p.attributes or {}), metric.name


async def test_an_exception_is_recorded_once_with_error_type() -> None:
    tel, exporter, _ = _telemetry()
    with pytest.raises(RuntimeError):
        async with tel.trace("work"):
            raise RuntimeError("boom")
    [span] = exporter.get_finished_spans()
    assert sum(1 for e in span.events if e.name == "exception") == 1
    assert (span.attributes or {})["error.type"] == "RuntimeError"
    assert span.status.status_code.name == "ERROR"


async def test_serve_stream_failure_uses_its_own_event_name() -> None:
    import anyio

    from loomflow.serve import create_app

    class Failing(ScriptedModel):
        async def stream(self, messages: Any, **kwargs: Any) -> Any:
            raise RuntimeError("provider down")
            yield  # pragma: no cover

    app = create_app(Agent("x", model=Failing([])))
    sent: list[dict[str, Any]] = []
    body = json.dumps({"prompt": "hi"}).encode()
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    never = anyio.Event()

    async def receive() -> dict[str, Any]:
        if messages:
            return messages.pop(0)
        await never.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/stream", "headers": [],
             "query_string": b"", "scheme": "http", "http_version": "1.1"}
    await app(scope, receive, send)
    text = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body").decode()
    names = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("event: ")]
    assert names[-2:] == ["stream_error", "done"]
    frame = text.split("event: stream_error\ndata: ", 1)[1].split("\n", 1)[0]
    assert json.loads(frame) == {"error": "RuntimeError", "message": "provider down"}
