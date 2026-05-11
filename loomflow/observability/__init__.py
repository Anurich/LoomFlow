"""Observability adapters: tracing, metrics, evals.

Telemetry sinks shipped today:

* :class:`NoTelemetry` — no-op default. Zero overhead; safe on
  every loop step.
* :class:`InMemoryTelemetry` — collects spans + metrics in lists
  for tests and exploration. Introspect via ``.spans()`` and
  ``.metrics()``.
* :class:`ConsoleTelemetry` — print to stderr as spans complete
  / metrics emit. "Tail my agent in dev" without a collector.
* :class:`MultiTelemetry` — fan-out across multiple sinks. Watch
  live in stderr AND assert in tests.
* :class:`OTelTelemetry` — OpenTelemetry-backed for production
  (Jaeger, Honeycomb, Datadog, any OTLP collector).

:class:`CapturedSpan` / :class:`CapturedMetric` are the records
returned by :class:`InMemoryTelemetry`; useful for type
annotations in test code.
"""

from .tracing import (
    CapturedMetric,
    CapturedSpan,
    ConsoleTelemetry,
    InMemoryTelemetry,
    MultiTelemetry,
    NoTelemetry,
    OTelTelemetry,
)

__all__ = [
    "CapturedMetric",
    "CapturedSpan",
    "ConsoleTelemetry",
    "InMemoryTelemetry",
    "MultiTelemetry",
    "NoTelemetry",
    "OTelTelemetry",
]
