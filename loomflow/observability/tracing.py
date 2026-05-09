"""Telemetry adapters.

* :class:`NoTelemetry` — no-op default. Both methods do as little work as
  possible so wrapping every loop step in ``async with telemetry.trace(...)``
  has effectively zero cost when telemetry isn't configured.
* :class:`OTelTelemetry` — OpenTelemetry-backed. Lazy SDK imports inside
  ``__init__``. Spans go to whatever ``TracerProvider`` is configured;
  metrics go to whatever ``MeterProvider`` is configured. Tests pass
  in-memory providers; production users wire up their exporters once at
  startup and the adapter inherits.

Metric type dispatch is by suffix:

* names ending in ``_ms``, ``_seconds``, or ``_bytes`` -> histogram
* everything else -> counter

That keeps the public interface a single :meth:`emit_metric` while still
producing the right OTel instrument under the hood.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ..core.types import Span

_HISTOGRAM_SUFFIXES = ("_ms", "_seconds", "_bytes")


def _clean(attrs: dict[str, Any]) -> dict[str, Any]:
    """Drop None values; OTel attribute APIs reject them."""
    return {k: v for k, v in attrs.items() if v is not None}


class NoTelemetry:
    """No-op telemetry. Very cheap; safe to call on every loop step."""

    @asynccontextmanager
    async def trace(self, name: str, **attrs: Any) -> AsyncIterator[Span]:
        yield Span(name=name, trace_id="", span_id="", attributes=dict(attrs))

    async def emit_metric(self, name: str, value: float, **attrs: Any) -> None:
        return None


class OTelTelemetry:
    """OpenTelemetry-backed :class:`~loomflow.core.protocols.Telemetry`."""

    def __init__(
        self,
        *,
        tracer_provider: Any | None = None,
        meter_provider: Any | None = None,
        instrumentation_name: str = "loomflow",
    ) -> None:
        try:
            from opentelemetry import metrics, trace
        except ImportError as exc:  # pragma: no cover — depends on user env
            raise ImportError(
                "opentelemetry is not installed. "
                "Install with: pip install 'loomflow[otel]'"
            ) from exc

        self._tracer = trace.get_tracer(
            instrumentation_name,
            tracer_provider=tracer_provider,
        )
        if meter_provider is not None:
            self._meter = meter_provider.get_meter(instrumentation_name)
        else:
            self._meter = metrics.get_meter(instrumentation_name)

        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}

    @asynccontextmanager
    async def trace(self, name: str, **attrs: Any) -> AsyncIterator[Span]:
        clean_attrs = _clean(attrs)
        with self._tracer.start_as_current_span(
            name, attributes=clean_attrs
        ) as otel_span:
            ctx = otel_span.get_span_context()
            our_span = Span(
                name=name,
                trace_id=format(ctx.trace_id, "032x"),
                span_id=format(ctx.span_id, "016x"),
                attributes=dict(clean_attrs),
            )
            try:
                yield our_span
            except Exception as exc:
                otel_span.record_exception(exc)
                from opentelemetry.trace import Status, StatusCode

                otel_span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    async def emit_metric(
        self, name: str, value: float, **attrs: Any
    ) -> None:
        clean_attrs = _clean(attrs)
        if name.endswith(_HISTOGRAM_SUFFIXES):
            histo = self._histograms.get(name)
            if histo is None:
                histo = self._meter.create_histogram(name)
                self._histograms[name] = histo
            histo.record(value, attributes=clean_attrs)
        else:
            counter = self._counters.get(name)
            if counter is None:
                counter = self._meter.create_counter(name)
                self._counters[name] = counter
            counter.add(value, attributes=clean_attrs)
