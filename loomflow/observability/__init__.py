"""Observability adapters: tracing, metrics, evals.

Telemetry sinks shipped today:

* :class:`NoTelemetry` — no-op default. Zero overhead; safe on
  every loop step.
* :class:`InMemoryTelemetry` — collects spans + metrics in lists
  for tests and exploration. Introspect via ``.spans()`` and
  ``.metrics()``.
* :class:`ConsoleTelemetry` — print to stderr as spans complete
  / metrics emit. "Tail my agent in dev" without a collector.
* :class:`FileTelemetry` — JSONL append-only on disk. Each span
  / metric becomes a structured line, parseable by ``jq``,
  Splunk, Datadog log pipelines. Pairs with
  :class:`~loomflow.security.FileAuditLog` for full offline
  forensics.
* :class:`MultiTelemetry` — fan-out across multiple sinks. Watch
  live in stderr AND assert in tests.
* :class:`OTelTelemetry` — OpenTelemetry-backed for production
  (Jaeger, Honeycomb, Datadog, any OTLP collector).

:class:`CapturedSpan` / :class:`CapturedMetric` are the records
returned by :class:`InMemoryTelemetry`; useful for type
annotations in test code.
"""

from .resolver import resolve_telemetry
from .tracing import (
    CapturedMetric,
    CapturedSpan,
    ConsoleTelemetry,
    FileTelemetry,
    InMemoryTelemetry,
    MultiTelemetry,
    NoTelemetry,
    OTelTelemetry,
)

__all__ = [
    "CapturedMetric",
    "CapturedSpan",
    "ConsoleTelemetry",
    "FileTelemetry",
    "InMemoryTelemetry",
    "MultiTelemetry",
    "NoTelemetry",
    "OTelTelemetry",
    "resolve_telemetry",
]
