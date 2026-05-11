"""Example 13 — Telemetry (OpenTelemetry spans + metrics).

Every Loom primitive emits typed spans and metrics when a
:class:`~loomflow.Telemetry` adapter is configured. The default
:class:`~loomflow.NoTelemetry` is zero-cost; in production you
wire :class:`~loomflow.observability.OTelTelemetry` to whatever
OTLP collector / Jaeger / Honeycomb your org uses, and every
``Agent.run`` produces a fully-attributed trace tree.

This example uses an **in-memory** OTel SpanExporter +
MetricReader so the example runs without a collector — you can
inspect the captured spans / metrics directly in Python.

Three things this example demonstrates:

* The trace tree the framework emits — ``loom.run`` →
  ``loom.turn`` → ``loom.model.stream`` and ``loom.tool``.
* Per-call metrics — token counts and cost per model call,
  session duration on completion.
* The histogram-vs-counter dispatch — names ending in ``_ms`` /
  ``_seconds`` / ``_bytes`` go to histograms; everything else
  becomes a counter. One ``emit_metric`` call, the right OTel
  instrument under the hood.

No API keys required — uses :class:`ScriptedModel`.

Requires ``pip install opentelemetry-sdk`` (already in the
``[otel]`` extra). Without it, the example raises a clear
``ImportError`` from the OTel adapter's constructor.

Run::

    python examples/13_telemetry.py
"""

from __future__ import annotations

import asyncio

from loomflow import (
    Agent,
    ScriptedModel,
    ScriptedTurn,
    ToolCall,
    tool,
)
from loomflow.observability import OTelTelemetry


@tool
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main() -> None:
    # ---- OTel setup — in-memory exporters so we can inspect ------------
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    # Spans go through a SimpleSpanProcessor → InMemoryExporter.
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    # Metrics use a periodic reader → in-memory snapshot.
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    telemetry = OTelTelemetry(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    # ---- Run an agent that calls a tool --------------------------------
    # ScriptedModel returns canned turns so the example is
    # deterministic; no network call. First turn calls add(2, 3);
    # second turn emits a final answer.
    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", tool="add", args={"a": 2, "b": 3})
                ]
            ),
            ScriptedTurn(text="The sum is 5."),
        ]
    )

    agent = Agent(
        "You are a precise arithmetic assistant.",
        model=model,
        tools=[add],
        telemetry=telemetry,
    )

    result = await agent.run("What is 2 + 3?", user_id="alice")
    print(f"Agent output: {result.output!r}\n")

    # ---- Inspect captured spans ----------------------------------------
    print("=" * 60)
    print("Span tree (one trace, parent → children)")
    print("=" * 60)

    spans = span_exporter.get_finished_spans()
    # Reorder oldest-first so the parents print before children.
    spans = sorted(spans, key=lambda s: s.start_time)
    for s in spans:
        # Indent children under their parent visually. The OTel
        # SDK exposes the parent's span_id on ``parent``; we walk
        # up to compute depth quickly.
        depth = 0
        cur = s
        while cur.parent is not None:
            depth += 1
            parent_id = cur.parent.span_id
            parent = next(
                (p for p in spans if p.context.span_id == parent_id),
                None,
            )
            if parent is None:
                break
            cur = parent
        indent = "  " * depth
        duration_ms = (s.end_time - s.start_time) / 1_000_000
        attrs = ", ".join(
            f"{k}={v}" for k, v in s.attributes.items() if v is not None
        )
        print(f"{indent}• {s.name}  ({duration_ms:.1f}ms)  [{attrs}]")

    # ---- Inspect captured metrics --------------------------------------
    print()
    print("=" * 60)
    print("Metrics emitted")
    print("=" * 60)

    metrics_data = metric_reader.get_metrics_data()
    if metrics_data is None:
        print("  (no metrics emitted — fast_telemetry=True elsewhere?)")
    else:
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    instrument_kind = (
                        "histogram"
                        if metric.name.endswith(("_ms", "_seconds", "_bytes"))
                        else "counter"
                    )
                    points = list(metric.data.data_points)
                    if not points:
                        continue
                    if instrument_kind == "histogram":
                        sums = sum(p.sum for p in points)
                        counts = sum(p.count for p in points)
                        print(
                            f"  {metric.name:<28} "
                            f"({instrument_kind}): "
                            f"sum={sums:.3f}, n={counts}"
                        )
                    else:
                        total = sum(p.value for p in points)
                        print(
                            f"  {metric.name:<28} "
                            f"({instrument_kind}): {total}"
                        )

    print()
    print("Notes:")
    print("  • `loom.session.duration_ms` → histogram (auto-detected by suffix)")
    print("  • `loom.tokens.input` / `.output` → counters")
    print("  • Same `emit_metric()` API; right OTel instrument under the hood")
    print()
    print("In production, swap the in-memory exporters for OTLP:")
    print(
        "  from opentelemetry.exporter.otlp.proto.grpc.trace_exporter "
        "import OTLPSpanExporter"
    )


if __name__ == "__main__":
    asyncio.run(main())
