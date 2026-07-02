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

import json
import sys
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import anyio

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


# Internal seam: :class:`MultiTelemetry` mints one (trace_id,
# span_id, parent_span_id) tuple per composed span and hands it to
# each capture sink under this reserved attrs key. Sinks that
# advertise ``_accepts_span_ctx = True`` pop it and (a) record the
# shared ids instead of minting their own, (b) leave the module
# contextvars alone — the composer manages them exactly once, so
# two capture sinks in one MultiTelemetry can't stomp each other's
# parent linkage. Never set this key from user code.
_SPAN_CTX_KEY = "_loomflow_span_ctx"


def _resolve_span_ctx(
    attrs: dict[str, Any],
) -> tuple[str, str, str | None, bool]:
    """Pop the internal span-context seam from ``attrs`` (mutates)
    and return ``(trace_id, span_id, parent_id, externally_managed)``.
    When no seam is present the ids are minted / inherited from the
    module contextvars, and the caller must install this span as the
    active parent itself (``externally_managed=False``)."""
    ctx = attrs.pop(_SPAN_CTX_KEY, None)
    if ctx is not None:
        trace_id, span_id, parent_id = ctx
        return str(trace_id), str(span_id), parent_id, True
    span_id = _new_hex_id(8)
    trace_id = _trace_id.get() or _new_hex_id(16)
    return trace_id, span_id, _parent_span_id.get(), False


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

    # Opt-in to MultiTelemetry's shared span-context seam.
    _accepts_span_ctx: bool = True

    def __init__(self) -> None:
        self._spans: list[CapturedSpan] = []
        self._metrics: list[CapturedMetric] = []

    @asynccontextmanager
    async def trace(self, name: str, **attrs: Any) -> AsyncIterator[Span]:
        # Inherit the active trace_id when one's already running;
        # mint a new one when this is the root. A MultiTelemetry
        # composer supplies the ids instead (managed=True) and
        # owns the contextvars.
        trace_id, span_id, parent_id, managed = _resolve_span_ctx(attrs)
        clean_attrs = _clean(attrs)
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()

        # Install this span as the active parent for children
        # spawned inside the contextmanager (composer does this
        # once for all sinks when managed).
        trace_token: Token[str | None] | None = None
        parent_token: Token[str | None] | None = None
        if not managed:
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
            if parent_token is not None:
                _parent_span_id.reset(parent_token)
            if trace_token is not None:
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

    # Opt-in to MultiTelemetry's shared span-context seam.
    _accepts_span_ctx: bool = True

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
        # Ids minted here, or supplied by a MultiTelemetry composer
        # (managed=True) that also owns the parent/trace contextvars.
        trace_id, span_id, parent_id, managed = _resolve_span_ctx(attrs)
        clean_attrs = _clean(attrs)
        # Walking the contextvar lineage isn't supported by ContextVar;
        # instead, store depth alongside the parent id when we set
        # the contextvar. Cheapest impl: use a separate depth var.
        # (Console-private, so it's always managed here even when
        # the span ids come from a composer.)
        depth = _console_depth.get() if parent_id is not None else 0

        t0 = time.perf_counter()
        trace_token: Token[str | None] | None = None
        parent_token: Token[str | None] | None = None
        if not managed:
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
            if parent_token is not None:
                _parent_span_id.reset(parent_token)
            if trace_token is not None:
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

    Span identity (trace_id / span_id / parent linkage) is minted
    ONCE per composed span by MultiTelemetry itself and shared with
    every capture-style sink (in-memory / console / file) via an
    internal seam, and the parent-tracking contextvars are set
    exactly once around the composed enter. Sinks left to manage
    the shared contextvars themselves would stomp each other —
    the second sink's ``set`` lands before the first sink's span
    body runs, so nested spans would link to the wrong parent and
    the sinks would disagree about the hierarchy. Sinks without
    the seam (OTel, custom) are entered unchanged. Exceptions
    raised inside a sink's ``trace`` propagate after every other
    sink has had a chance to record the cleanup (``finally``
    blocks fire even on exceptional exit thanks to
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
        # Never forward a caller-supplied value under the reserved
        # seam key — the composed identity below is authoritative.
        attrs.pop(_SPAN_CTX_KEY, None)
        # Mint the composed span's identity once; every ctx-aware
        # sink records these same ids, so all captures agree on
        # trace_id / span_id / parent linkage.
        span_id = _new_hex_id(8)
        trace_id = _trace_id.get() or _new_hex_id(16)
        parent_id = _parent_span_id.get()
        span_ctx = (trace_id, span_id, parent_id)
        # Set the shared contextvars exactly once for the whole
        # composition (sinks skip their own set when handed a ctx).
        trace_token = _trace_id.set(trace_id)
        parent_token = _parent_span_id.set(span_id)
        try:
            # Enter every sink's contextmanager. AsyncExitStack
            # ensures they all get cleaned up even if one raises
            # mid-enter, and exits them in reverse order at the end.
            async with AsyncExitStack() as stack:
                for sink in self._sinks:
                    if getattr(sink, "_accepts_span_ctx", False):
                        cm = sink.trace(
                            name, **{_SPAN_CTX_KEY: span_ctx}, **attrs
                        )
                    else:
                        cm = sink.trace(name, **attrs)
                    await stack.enter_async_context(cm)
                # Yield the composed Span — the canonical identity
                # every ctx-aware sink recorded.
                yield Span(
                    name=name,
                    trace_id=trace_id,
                    span_id=span_id,
                    attributes=_clean(attrs),
                )
        finally:
            _parent_span_id.reset(parent_token)
            _trace_id.reset(trace_token)

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


class FileTelemetry:
    """Append-only JSONL telemetry sink.

    Each span or metric is serialised to one line of structured
    JSON in the configured file, so the output is parseable by
    ``jq``, Splunk, Datadog Log Pipelines, etc. Mirrors the
    :class:`~loomflow.security.FileAuditLog` pattern: parent
    directory is created if missing, writes go through
    ``anyio.to_thread.run_sync`` so the event loop never blocks
    on disk I/O, and an internal lock serialises concurrent
    writes from parallel tool dispatches.

    Each line has a ``"kind"`` field discriminating span vs
    metric records. Span lines additionally carry the parent
    linkage (``parent_span_id``) needed to reconstruct the trace
    tree offline.

    .. code-block:: python

        from loomflow.observability import FileTelemetry

        agent = Agent(..., telemetry=FileTelemetry("./traces.jsonl"))
        await agent.run("...")

    Querying offline with ``jq``:

    .. code-block:: shell

        # Spans that took longer than 1s
        jq -c 'select(.kind=="span" and .duration_ms > 1000)' \\
            traces.jsonl

        # One user's session
        jq -c 'select(.attributes.session_id=="sess_xyz")' \\
            traces.jsonl

    **Not** a replacement for :class:`~loomflow.security.FileAuditLog`
    — they capture different things. Audit log = business events
    for compliance ("did Alice's refund go through?"); telemetry
    = performance / diagnostic spans ("why was this run slow?").
    Run both together in production.

    No rotation built in — use ``logrotate`` / ``journald`` /
    your platform's log management to cap file size. The
    framework deliberately stays out of that policy decision.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = anyio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    # Opt-in to MultiTelemetry's shared span-context seam.
    _accepts_span_ctx: bool = True

    @asynccontextmanager
    async def trace(self, name: str, **attrs: Any) -> AsyncIterator[Span]:
        # Ids minted here, or supplied by a MultiTelemetry composer
        # (managed=True) that also owns the parent/trace contextvars.
        trace_id, span_id, parent_id, managed = _resolve_span_ctx(attrs)
        clean_attrs = _clean(attrs)
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()

        trace_token: Token[str | None] | None = None
        parent_token: Token[str | None] | None = None
        if not managed:
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
            if parent_token is not None:
                _parent_span_id.reset(parent_token)
            if trace_token is not None:
                _trace_id.reset(trace_token)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            await self._write(
                {
                    "kind": "span",
                    "name": name,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "parent_span_id": parent_id,
                    "started_at": started_at.isoformat(),
                    "ended_at": datetime.now(UTC).isoformat(),
                    "duration_ms": elapsed_ms,
                    "attributes": dict(clean_attrs),
                    "exception": exc_repr,
                }
            )

    async def emit_metric(
        self, name: str, value: float, **attrs: Any
    ) -> None:
        await self._write(
            {
                "kind": "metric",
                "name": name,
                "value": value,
                "instrument_kind": _instrument_kind(name),
                "attributes": _clean(attrs),
                "emitted_at": datetime.now(UTC).isoformat(),
            }
        )

    async def _write(self, record: dict[str, Any]) -> None:
        # Lock-protected so parallel tool dispatches don't
        # interleave their write() calls mid-line. The actual
        # disk write happens in a worker thread.
        async with self._lock:
            await anyio.to_thread.run_sync(self._sync_write, record)

    def _sync_write(self, record: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str))
            fh.write("\n")
