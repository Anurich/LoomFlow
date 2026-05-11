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

import sys
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ..core.types import Span

_HISTOGRAM_SUFFIXES = ("_ms", "_seconds", "_bytes")


def _clean(attrs: dict[str, Any]) -> dict[str, Any]:
    """Drop None values; OTel attribute APIs reject them."""
    return {k: v for k, v in attrs.items() if v is not None}


def _instrument_kind(name: str) -> Literal["counter", "histogram"]:
    """Mirror the OTel adapter's suffix-based dispatch so the
    in-memory / console / multi sinks tag each captured metric
    with the same instrument kind users see in OTLP output."""
    return "histogram" if name.endswith(_HISTOGRAM_SUFFIXES) else "counter"


# ---------------------------------------------------------------------------
# Captured records — used by InMemoryTelemetry and friends so tests
# can assert on span structure / metrics without depending on the
# OTel SDK at all.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapturedSpan:
    """One span recorded by an :class:`InMemoryTelemetry` /
    :class:`ConsoleTelemetry`. Richer than :class:`Span` (the
    minimal value-object) because we also record timing,
    parent-span linkage, and exception status — the fields a
    test or a console viewer actually wants to see."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    attributes: Mapping[str, Any] = field(default_factory=dict)
    exception: str | None = None  # ``repr(exc)`` if the body raised


@dataclass(frozen=True)
class CapturedMetric:
    """One metric emit recorded by an in-memory / console telemetry."""

    name: str
    value: float
    instrument_kind: Literal["counter", "histogram"]
    attributes: Mapping[str, Any]
    emitted_at: datetime


# Parent-span tracking shared by InMemoryTelemetry and
# ConsoleTelemetry. Stored as a contextvar so nested spans —
# including those started inside ``anyio.create_task_group()`` —
# pick the right parent automatically.
_parent_span_id: ContextVar[str | None] = ContextVar(
    "loomflow_telemetry_parent_span_id", default=None
)
_trace_id: ContextVar[str | None] = ContextVar(
    "loomflow_telemetry_trace_id", default=None
)


def _new_hex_id(bytes_: int = 8) -> str:
    """Generate a short hex id for spans / traces. 16-char span id,
    32-char trace id by convention (matches OTel)."""
    return uuid.uuid4().hex[: bytes_ * 2]


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


# ---------------------------------------------------------------------------
# Convenience telemetry sinks — no OTel collector required
# ---------------------------------------------------------------------------


class InMemoryTelemetry:
    """Collect spans + metrics in lists so tests and tinkerers can
    introspect them directly. **Not for production** — unbounded
    growth.

    The :meth:`spans` and :meth:`metrics` accessors return the
    accumulated records oldest-first. Each :class:`CapturedSpan`
    carries timing, parent linkage, and exception status; each
    :class:`CapturedMetric` carries the auto-detected instrument
    kind (counter vs histogram) so assertions match what
    :class:`OTelTelemetry` would have emitted.

    Use :meth:`clear` between test cases so the accumulator
    doesn't bleed state.

    Parent-span linkage uses a :class:`ContextVar` so spans
    started inside ``anyio.create_task_group()`` (the framework's
    parallel-tool-dispatch path) inherit the correct parent.
    """

    def __init__(self) -> None:
        self._spans: list[CapturedSpan] = []
        self._metrics: list[CapturedMetric] = []

    @asynccontextmanager
    async def trace(self, name: str, **attrs: Any) -> AsyncIterator[Span]:
        clean_attrs = _clean(attrs)
        span_id = _new_hex_id(8)
        # Inherit the active trace_id when one's already running;
        # mint a new one when this is the root.
        trace_id = _trace_id.get() or _new_hex_id(16)
        parent_id = _parent_span_id.get()
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()

        # Install this span as the active parent for children
        # spawned inside the contextmanager.
        trace_token = _trace_id.set(trace_id)
        parent_token = _parent_span_id.set(span_id)
        exc_repr: str | None = None
        try:
            yield Span(
                name=name,
                trace_id=trace_id,
                span_id=span_id,
                attributes=dict(clean_attrs),
            )
        except Exception as exc:
            exc_repr = repr(exc)
            raise
        finally:
            _parent_span_id.reset(parent_token)
            _trace_id.reset(trace_token)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._spans.append(
                CapturedSpan(
                    name=name,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_id,
                    started_at=started_at,
                    ended_at=datetime.now(UTC),
                    duration_ms=elapsed_ms,
                    attributes=clean_attrs,
                    exception=exc_repr,
                )
            )

    async def emit_metric(
        self, name: str, value: float, **attrs: Any
    ) -> None:
        self._metrics.append(
            CapturedMetric(
                name=name,
                value=value,
                instrument_kind=_instrument_kind(name),
                attributes=_clean(attrs),
                emitted_at=datetime.now(UTC),
            )
        )

    def spans(self) -> list[CapturedSpan]:
        """All captured spans, oldest-first. Sorted by ``ended_at``
        since spans complete in reverse-open order (children
        before parents) — sorting on end-time gives a stable
        timeline that's easier to read."""
        return sorted(self._spans, key=lambda s: s.ended_at)

    def metrics(self) -> list[CapturedMetric]:
        """All captured metrics, in emit order."""
        return list(self._metrics)

    def clear(self) -> None:
        """Reset the accumulators. Call between test cases."""
        self._spans.clear()
        self._metrics.clear()


class ConsoleTelemetry:
    """Print spans + metrics to a stream as they happen. Default
    target is :obj:`sys.stderr` so the output doesn't mix with the
    application's stdout.

    Each span emits a single line on completion containing name,
    duration (ms), and any non-None attributes. Metrics emit one
    line per call with the auto-detected instrument kind. Nested
    spans are indented by their parent depth so the trace tree is
    visible at a glance.

    Useful for "tail my agent in dev" without deploying an OTel
    collector. **Not for production** — synchronous writes to a
    text stream are not appropriate at scale.
    """

    def __init__(
        self,
        *,
        stream: Any = None,
        show_metrics: bool = True,
    ) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._show_metrics = show_metrics

    @asynccontextmanager
    async def trace(self, name: str, **attrs: Any) -> AsyncIterator[Span]:
        clean_attrs = _clean(attrs)
        span_id = _new_hex_id(8)
        trace_id = _trace_id.get() or _new_hex_id(16)
        parent_id = _parent_span_id.get()
        # Compute depth for indentation: count the chain of
        # active parents. We approximate by counting active spans
        # — good enough for visual hierarchy without maintaining
        # a separate depth contextvar.
        depth = 0
        # Walk the contextvar lineage isn't supported by ContextVar;
        # instead, store depth alongside the parent id when we set
        # the contextvar. Cheapest impl: use a separate depth var.
        depth = _console_depth.get() if parent_id is not None else 0

        t0 = time.perf_counter()
        trace_token = _trace_id.set(trace_id)
        parent_token = _parent_span_id.set(span_id)
        depth_token = _console_depth.set(depth + 1)
        exc_repr: str | None = None
        try:
            yield Span(
                name=name,
                trace_id=trace_id,
                span_id=span_id,
                attributes=dict(clean_attrs),
            )
        except Exception as exc:
            exc_repr = repr(exc)
            raise
        finally:
            _console_depth.reset(depth_token)
            _parent_span_id.reset(parent_token)
            _trace_id.reset(trace_token)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            indent = "  " * depth
            attrs_str = (
                "  " + ", ".join(f"{k}={v}" for k, v in clean_attrs.items())
                if clean_attrs
                else ""
            )
            err = f"  [ERROR: {exc_repr}]" if exc_repr else ""
            print(
                f"{indent}• {name}  ({elapsed_ms:.1f}ms){attrs_str}{err}",
                file=self._stream,
                flush=True,
            )

    async def emit_metric(
        self, name: str, value: float, **attrs: Any
    ) -> None:
        if not self._show_metrics:
            return
        clean_attrs = _clean(attrs)
        kind = _instrument_kind(name)
        attrs_str = (
            "  " + ", ".join(f"{k}={v}" for k, v in clean_attrs.items())
            if clean_attrs
            else ""
        )
        print(
            f"  metric  {name}={value} ({kind}){attrs_str}",
            file=self._stream,
            flush=True,
        )


# Depth tracking for ConsoleTelemetry's indented output. Stored
# separately from ``_parent_span_id`` so a non-Console parent
# telemetry (e.g. NoTelemetry inside a MultiTelemetry) doesn't
# break the indentation.
_console_depth: ContextVar[int] = ContextVar(
    "loomflow_console_telemetry_depth", default=0
)


class MultiTelemetry:
    """Fan-out :class:`Telemetry` — every span and metric is
    forwarded to *every* configured sink in declaration order.

    Composes the sinks ship above. Common pattern:

    .. code-block:: python

        from loomflow.observability import (
            ConsoleTelemetry, InMemoryTelemetry, MultiTelemetry,
        )

        in_mem = InMemoryTelemetry()
        telemetry = MultiTelemetry([ConsoleTelemetry(), in_mem])

        agent = Agent(..., telemetry=telemetry)
        await agent.run("...")

        # Watch live in stderr AND inspect afterwards
        assert any(s.name == "loom.tool" for s in in_mem.spans())

    Span IDs are *generated by the first sink* and shared with the
    others — keeps trace hierarchy consistent across captures.
    Exceptions raised inside a sink's ``trace`` propagate after
    every other sink has had a chance to record the cleanup
    (``finally`` blocks fire even on exceptional exit thanks to
    :class:`AsyncExitStack`).
    """

    def __init__(self, sinks: Iterable[Any]) -> None:
        self._sinks: list[Any] = list(sinks)
        if not self._sinks:
            raise ValueError(
                "MultiTelemetry requires at least one sink. "
                "Use NoTelemetry() if you want a no-op."
            )

    @asynccontextmanager
    async def trace(self, name: str, **attrs: Any) -> AsyncIterator[Span]:
        # Enter every sink's contextmanager. AsyncExitStack ensures
        # they all get cleaned up even if one raises mid-enter,
        # and exits them in reverse order at the end.
        async with AsyncExitStack() as stack:
            spans: list[Span] = []
            for sink in self._sinks:
                span = await stack.enter_async_context(
                    sink.trace(name, **attrs)
                )
                spans.append(span)
            # Yield the first sink's Span — that's the one whose
            # trace_id / span_id we treat as canonical. Other
            # sinks' spans are still active in the background.
            yield spans[0]

    async def emit_metric(
        self, name: str, value: float, **attrs: Any
    ) -> None:
        # Fan out. If one sink errors, the rest still see the
        # emit — collect exceptions, raise the first at the end.
        errors: list[BaseException] = []
        for sink in self._sinks:
            try:
                await sink.emit_metric(name, value, **attrs)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]
