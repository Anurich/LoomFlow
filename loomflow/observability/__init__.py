"""Observability adapters: tracing, metrics, evals.

Today: :class:`NoTelemetry` (no-op default) and :class:`OTelTelemetry`
(OpenTelemetry-backed). Phase 6 may add inline-evals signals.
"""

from .tracing import NoTelemetry, OTelTelemetry

__all__ = ["NoTelemetry", "OTelTelemetry"]
