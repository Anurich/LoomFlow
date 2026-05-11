"""Example 13 — Telemetry (spans + metrics, no collector required).

Every Loom primitive emits typed spans and metrics when a
:class:`~loomflow.Telemetry` adapter is configured. The default
:class:`~loomflow.NoTelemetry` is zero-cost; this example uses
the convenience sinks that ship with Loom so no OTel collector
deploy is required to see what's happening.

Four sinks demonstrated here:

* :class:`InMemoryTelemetry` — accumulates spans + metrics in
  lists; introspect via ``.spans()`` / ``.metrics()``. Best for
  unit tests and exploration.
* :class:`ConsoleTelemetry` — print spans + metrics to stderr
  as they happen. Best for "tail my agent in dev".
* :class:`FileTelemetry` — JSONL append-only on disk. One
  structured JSON line per span / metric — parseable by ``jq``,
  Splunk, Datadog log pipelines. Pairs with ``FileAuditLog``.
* :class:`MultiTelemetry` — fan out to multiple sinks. Watch
  live in stderr AND assert in tests.

For production, swap to :class:`OTelTelemetry` (OpenTelemetry
SDK with OTLP exporter); the agent code doesn't change.

No API keys required — uses :class:`ScriptedModel`.

Run::

    python examples/13_telemetry.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import anyio

from loomflow import (
    Agent,
    ScriptedModel,
    ScriptedTurn,
    ToolCall,
    tool,
)
from loomflow.observability import (
    ConsoleTelemetry,
    FileTelemetry,
    InMemoryTelemetry,
    MultiTelemetry,
)


@tool
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def _scripted_agent(telemetry: Any) -> Agent:
    """Build the same canned agent for every demo so the only
    thing that varies between parts is the telemetry sink."""
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
    return Agent(
        "You are a precise arithmetic assistant.",
        model=model,
        tools=[add],
        telemetry=telemetry,
    )


async def main() -> None:
    # ---- Part 1 — InMemoryTelemetry --------------------------------------
    print("=" * 60)
    print("Part 1 — InMemoryTelemetry (assert in tests)")
    print("=" * 60)

    in_mem = InMemoryTelemetry()
    agent = _scripted_agent(in_mem)
    result = await agent.run("What is 2 + 3?", user_id="alice")
    print(f"  Agent output: {result.output!r}\n")

    # Inspect what got recorded — no OTel SDK objects involved.
    print("  Spans (sorted by completion time):")
    for s in in_mem.spans():
        attrs = ", ".join(f"{k}={v}" for k, v in s.attributes.items())
        print(f"    • {s.name:<22}  ({s.duration_ms:5.1f}ms)  [{attrs}]")

    print("\n  Metrics:")
    for m in in_mem.metrics():
        print(
            f"    • {m.name:<28}  {m.value:<8}  ({m.instrument_kind})"
        )

    # ---- Part 2 — ConsoleTelemetry ---------------------------------------
    print()
    print("=" * 60)
    print("Part 2 — ConsoleTelemetry (live in stderr)")
    print("=" * 60)
    print("(span lines printed to stderr below as the agent runs)")

    console = ConsoleTelemetry(stream=sys.stderr, show_metrics=False)
    agent = _scripted_agent(console)
    await agent.run("What is 2 + 3?", user_id="bob")

    # ---- Part 3 — MultiTelemetry — both at once --------------------------
    print()
    print("=" * 60)
    print("Part 3 — MultiTelemetry (watch live AND inspect after)")
    print("=" * 60)

    in_mem2 = InMemoryTelemetry()
    console2 = ConsoleTelemetry(stream=sys.stderr, show_metrics=False)
    combo = MultiTelemetry([console2, in_mem2])

    agent = _scripted_agent(combo)
    await agent.run("What is 2 + 3?", user_id="carol")

    print()
    print("  After the run, the in-memory side has everything too:")
    print(f"    • {len(in_mem2.spans())} spans captured")
    print(f"    • {len(in_mem2.metrics())} metrics captured")
    tool_spans = [s for s in in_mem2.spans() if s.name == "loom.tool"]
    if tool_spans:
        print(
            f"    • Tool span attributes: {dict(tool_spans[0].attributes)}"
        )

    # ---- Part 4 — FileTelemetry (JSONL on disk) --------------------------
    print()
    print("=" * 60)
    print("Part 4 — FileTelemetry (JSONL append-only)")
    print("=" * 60)

    log_path = Path("./_telemetry_demo.jsonl")
    if await anyio.to_thread.run_sync(log_path.exists):
        await anyio.to_thread.run_sync(log_path.unlink)

    file_tel = FileTelemetry(log_path)
    agent = _scripted_agent(file_tel)
    await agent.run("What is 2 + 3?", user_id="dave")

    # Read back the JSONL — each line is a structured record.
    lines = (await anyio.to_thread.run_sync(log_path.read_text)).splitlines()
    print(f"  Wrote {len(lines)} JSON lines to {log_path}\n")
    spans = [json.loads(line) for line in lines if json.loads(line)["kind"] == "span"]
    metrics = [json.loads(line) for line in lines if json.loads(line)["kind"] == "metric"]
    print(f"    {len(spans)} span records (with parent_span_id linkage)")
    print(f"    {len(metrics)} metric records (counter / histogram tagged)")
    print()
    print("  Sample (first span record):")
    print(f"    {json.dumps(spans[0], indent=2)[:300]}...")
    print()
    print("  Query offline with jq:")
    print('    jq -c \'select(.kind=="span" and .duration_ms > 1)\' \\')
    print(f"        {log_path}")
    print('    jq -c \'select(.attributes.user_id=="dave")\' \\')
    print(f"        {log_path}")

    # Cleanup so the example is idempotent.
    await anyio.to_thread.run_sync(log_path.unlink)

    # ---- Production-path reminder ----------------------------------------
    print()
    print("=" * 60)
    print("Production path")
    print("=" * 60)
    print("Swap the sink for OTelTelemetry — agent code doesn't change.\n")
    print("  from loomflow.observability import OTelTelemetry")
    print("  # Configure your OTel SDK once at startup with an OTLP")
    print("  # exporter pointing at your collector / Jaeger /")
    print("  # Honeycomb / Datadog.")
    print("  telemetry = OTelTelemetry()")
    print("  agent = Agent(..., telemetry=telemetry)")


if __name__ == "__main__":
    asyncio.run(main())
