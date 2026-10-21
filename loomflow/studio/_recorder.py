"""Recording: turn a run's spans, events and model calls into a trace.

Three inputs build a trace:

* **Spans** from loomflow's telemetry. :class:`StudioTelemetry` is a
  telemetry sink; for a recorded run the agent's telemetry is
  composed with it (``MultiTelemetry``), so the run, its turns, model
  calls, tool calls and workflow steps arrive with their timing and
  attributes, and nesting comes from the context.
* **Events** the run emits (:meth:`Recorder.on_event`): the prompt,
  the final result, tool arguments and results, guardrail hits,
  architecture progress, errors.
* **Model calls** (:meth:`Recorder.model_call`): the messages sent and
  the reply. These never travel on spans, so they never reach an
  OpenTelemetry exporter.

A trace is kept in memory while it runs and written when it ends, by
one background thread (a slow disk never slows a run). Its row is
also written as ``running`` when it starts, so long runs show up.
Text passes through the configured ``redact`` function and a size cap
before it is kept; with ``capture_content=False`` no prompt, reply or
tool payload is kept at all.
"""

from __future__ import annotations

import atexit
import logging
import queue
import random
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from ..core.types import Event, EventKind, Span
from ..observability.tracing import _SPAN_CTX_KEY
from . import _state
from ._store import StudioStore

_LOG = logging.getLogger("loomflow.studio")

MAX_TEXT = 32_000
"""Characters kept of any one string (a prompt, a reply, a tool result)."""
MAX_MESSAGES = 60
"""Messages kept per model call (the most recent)."""
MAX_SPANS = 5_000
"""Spans kept per trace; later ones are counted, not stored."""
QUEUE_SIZE = 10_000
PRUNE_EVERY = 3600.0
"""Seconds between deletions of old runs, when ``keep_days`` is set."""

_KINDS = {
    "loom.run": "agent",
    "loom.turn": "turn",
    "loom.model.complete": "model",
    "loom.model.stream": "model",
    "loom.tool": "tool",
    "loom.workflow.step": "step",
}


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class SpanRecord:
    span_id: str
    parent_id: str | None
    name: str
    kind: str
    started_at: float
    ended_at: float | None = None
    duration_ms: float | None = None
    status: str = "ok"
    error: str | None = None
    run_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    content: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def close(self, *, at: float | None = None) -> None:
        self.ended_at = at if at is not None else time.time()
        self.duration_ms = (time.perf_counter() - self._t0) * 1000.0

    def fail(self, error: str) -> None:
        self.status = "error"
        self.error = error

    def row(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
            "content": self.content,
        }


@dataclass
class TraceState:
    """One trace being recorded."""

    recorder: Recorder
    trace_id: str
    project: str
    tags: list[str]
    metadata: dict[str, Any]
    experiment_id: str | None = None
    case_key: str | None = None
    spans: dict[str, SpanRecord] = field(default_factory=dict)
    root_id: str | None = None
    tool_spans: dict[str, str] = field(default_factory=dict)
    step_spans: dict[str, list[str]] = field(default_factory=dict)
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None
    dropped: int = 0

    def add(self, span: SpanRecord) -> SpanRecord | None:
        if len(self.spans) >= MAX_SPANS:
            self.dropped += 1
            return None
        self.spans[span.span_id] = span
        if span.parent_id is None and self.root_id is None:
            self.root_id = span.span_id
        return span

    def event(self, name: str, *, at: float, content: dict[str, Any], status: str = "ok") -> None:
        """A point-in-time step (a guardrail hit, an error, ...) under
        the current span."""
        span = SpanRecord(
            span_id=_new_id()[:16],
            parent_id=_state.parent_var.get(),
            name=name,
            kind="event",
            started_at=at,
            ended_at=at,
            duration_ms=0.0,
            status=status,
            content=content,
        )
        self.add(span)


class StudioTelemetry:
    """The telemetry sink that feeds spans into the current trace.

    Outside a recorded run it does nothing. It keeps its own parent
    linkage (``parent_var``), so it agrees with itself whether it is
    used alone or composed with other sinks.
    """

    # MultiTelemetry hands composed spans their shared ids.
    _accepts_span_ctx: bool = True

    def __init__(self, recorder: Recorder) -> None:
        self._recorder = recorder

    @asynccontextmanager
    async def trace(self, name: str, **attrs: Any) -> AsyncIterator[Span]:
        span_ctx = attrs.pop(_SPAN_CTX_KEY, None)
        span_id = str(span_ctx[1]) if span_ctx else _new_id()[:16]
        clean = {k: v for k, v in attrs.items() if v is not None}
        state = _state.trace_var.get()
        handle = Span(
            name=name,
            trace_id=state.trace_id if state is not None else _new_id(),
            span_id=span_id,
            attributes=dict(clean),
        )
        record = None
        if state is not None:
            record = state.add(_open_span(state, name, span_id, clean))
        if record is None:
            yield handle
            return
        token = _state.parent_var.set(span_id)
        try:
            yield handle
        except Exception as exc:
            record.fail(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _state.parent_var.reset(token)
            record.attributes.update(
                {k: v for k, v in handle.attributes.items() if v is not None}
            )
            error_type = record.attributes.get("error.type")
            if error_type and record.status == "ok":
                record.fail(str(error_type))
            record.close()

    async def emit_metric(self, name: str, value: float, **attrs: Any) -> None:
        return None


def _open_span(state: TraceState, name: str, span_id: str, attrs: dict[str, Any]) -> SpanRecord:
    kind = _KINDS.get(name, "span")
    label = name
    if kind == "agent":
        label = str(attrs.get("gen_ai.agent.name") or "agent")
    elif kind == "turn":
        label = f"turn {attrs.get('turn', '')}".strip()
    elif kind == "model":
        label = f"chat {attrs.get('model', '')}".strip()
    elif kind == "tool":
        label = str(attrs.get("tool") or "tool")
        if attrs.get("call_id"):
            state.tool_spans[str(attrs["call_id"])] = span_id
    elif kind == "step":
        label = str(attrs.get("step") or "step")
        state.step_spans.setdefault(label, []).append(span_id)
    return SpanRecord(
        span_id=span_id,
        parent_id=_state.parent_var.get(),
        name=label,
        kind=kind,
        started_at=time.time(),
        attributes=dict(attrs),
    )


class Recorder:
    """Records traces into one Studio database (one per file per
    process; see :func:`loomflow.studio.enable`)."""

    def __init__(self, settings: _state.Settings) -> None:
        self.settings = settings
        self.store = StudioStore(settings.path)
        self.sink = StudioTelemetry(self)
        self._queue: queue.Queue[tuple[Any, ...] | None] = queue.Queue(maxsize=QUEUE_SIZE)
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self._dropped = 0
        self._warned: set[str] = set()

    def configure(self, settings: _state.Settings) -> None:
        self.settings = settings

    @property
    def path(self) -> Any:
        return self.settings.path

    # ---- traces --------------------------------------------------------

    def sampled(self) -> bool:
        rate = self.settings.sample_rate
        return rate >= 1.0 or (rate > 0.0 and random.random() < rate)

    def new_trace(self) -> TraceState:
        scope = _state.scope_var.get()
        project = (
            (scope.project if scope is not None else None)
            or self.settings.project
            or _state.default_project()
        )
        return TraceState(
            recorder=self,
            trace_id=_new_id(),
            project=project,
            tags=list(scope.tags) if scope is not None else [],
            metadata=dict(scope.metadata) if scope is not None else {},
            experiment_id=scope.experiment_id if scope is not None else None,
            case_key=scope.case_key if scope is not None else None,
        )

    def start(self, state: TraceState) -> None:
        """Write the trace's ``running`` row."""
        self.submit(("trace", self._trace_row(state, running=True), [], list(state.agents.values())))

    def finish(self, state: TraceState) -> None:
        """Queue the finished trace for writing."""
        spans = [s.row() for s in state.spans.values()]
        row = self._trace_row(state, running=False)
        if state.dropped:
            row["metadata"] = {**row["metadata"], "studio.dropped_spans": state.dropped}
        self.submit(("trace", row, spans, list(state.agents.values())))

    def _trace_row(self, state: TraceState, *, running: bool) -> dict[str, Any]:
        root = state.spans.get(state.root_id or "")
        attrs = root.attributes if root is not None else {}
        content = root.content if root is not None else {}
        if running:
            status = "running"
        elif root is None:
            status = "error" if state.error else "ok"
        elif root.status == "error" or state.error:
            status = "error"
        elif attrs.get("interrupted"):
            status = "interrupted"
        else:
            status = "ok"
        tokens_in, tokens_out, cost = self._totals(state, root)
        model = attrs.get("model")
        if model is None:
            model = next(
                (s.attributes.get("model") for s in state.spans.values() if s.kind == "model"), None
            )
        return {
            "trace_id": state.trace_id,
            "project": state.project,
            "kind": root.kind if root is not None else "agent",
            "name": root.name if root is not None else None,
            "agent_key": attrs.get("agent_key"),
            "agent_version": None,
            "run_id": root.run_id if root is not None else None,
            "user_id": attrs.get("user_id"),
            "session_id": attrs.get("session_id"),
            "input": _preview(content.get("input")),
            "output": _preview(content.get("output")),
            "status": status,
            "error": (root.error if root is not None and root.error else None) or state.error,
            "interruption_reason": attrs.get("interruption_reason"),
            "model": model,
            "started_at": root.started_at if root is not None else time.time(),
            "ended_at": None if running or root is None else root.ended_at,
            "duration_ms": None if running or root is None else root.duration_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost,
            "tags": state.tags,
            "metadata": state.metadata,
            "experiment_id": state.experiment_id,
            "case_key": state.case_key,
        }

    @staticmethod
    def _totals(state: TraceState, root: SpanRecord | None) -> tuple[int, int, float]:
        """Tokens and cost: the root agent's own totals (they include its
        sub-runs' model calls), else the sum over top-level agents."""
        if root is not None and root.kind == "agent" and "tokens_in" in root.attributes:
            a = root.attributes
            return int(a.get("tokens_in") or 0), int(a.get("tokens_out") or 0), float(a.get("cost_usd") or 0.0)
        agents = [s for s in state.spans.values() if s.kind == "agent" and "tokens_in" in s.attributes]
        top = [s for s in agents if not any(
            _is_ancestor(state, other.span_id, s) for other in agents if other is not s
        )]
        return (
            sum(int(s.attributes.get("tokens_in") or 0) for s in top),
            sum(int(s.attributes.get("tokens_out") or 0) for s in top),
            sum(float(s.attributes.get("cost_usd") or 0.0) for s in top),
        )

    # ---- what runs report ----------------------------------------------

    async def agent_started(self, agent: Any, run_ctx: Any, prompt: str) -> None:
        from ._catalog import describe_agent

        state = _state.trace_var.get()
        span = state.spans.get(_state.parent_var.get() or "") if state is not None else None
        if state is None or span is None or span.kind != "agent":
            return
        record = await describe_agent(agent)
        state.agents[record["key"] + record["fingerprint"]] = record
        span.name = record["name"]
        span.run_id = run_ctx.run_id
        span.attributes.update({
            "agent_key": record["key"],
            "fingerprint": record["fingerprint"],
            "user_id": run_ctx.user_id,
            "session_id": run_ctx.session_id,
            "run_id": run_ctx.run_id,
        })
        span.content["input"] = self.text(prompt)
        if span.span_id == state.root_id:
            self.start(state)

    def on_event(self, event: Event) -> None:
        state = _state.trace_var.get()
        if state is None:
            return
        try:
            self._on_event(state, event)
        except Exception as exc:  # noqa: BLE001 — recording must never break a run
            self._warn_once("event", f"Studio skipped an event it couldn't record: {exc!r}")

    def _on_event(self, state: TraceState, event: Event) -> None:
        kind, payload = event.kind, event.payload
        current = state.spans.get(_state.parent_var.get() or "")
        at = event.at.timestamp()
        if kind == EventKind.MODEL_CHUNK or kind.value.startswith("workflow"):
            return
        if kind == EventKind.STARTED:
            if current is not None and current.kind == "agent":
                current.content["input"] = self.text(payload.get("prompt"))
        elif kind == EventKind.COMPLETED:
            result = payload.get("result")
            if current is not None and current.kind == "agent" and isinstance(result, Mapping):
                current.content["output"] = self.text(result.get("output"))
                if result.get("parsed") is not None:
                    current.content["parsed"] = self.data(result.get("parsed"))
                current.attributes.update({
                    "turns": result.get("turns"),
                    "tokens_in": result.get("tokens_in"),
                    "tokens_out": result.get("tokens_out"),
                    "cached_tokens_in": result.get("cached_tokens_in"),
                    "cost_usd": result.get("cost_usd"),
                    "interrupted": result.get("interrupted"),
                    "interruption_reason": result.get("interruption_reason"),
                })
        elif kind == EventKind.TOOL_CALL:
            call = payload.get("call") or {}
            target = state.spans.get(state.tool_spans.get(str(call.get("id")), ""))
            if target is not None:
                target.content["args"] = self.data(call.get("args"))
            else:
                state.event(f"tool call {call.get('tool')}", at=at, content={"args": self.data(call.get("args"))})
        elif kind == EventKind.TOOL_RESULT:
            result = payload.get("result") or {}
            target = state.spans.get(state.tool_spans.get(str(result.get("call_id")), ""))
            body = {
                "result": self.data(result.get("output")),
                "error": self.text(result.get("error")),
                "denied": bool(result.get("denied")),
                "reason": self.text(result.get("reason")),
            }
            if target is not None:
                target.content.update(body)
                if result.get("denied"):
                    target.status = "denied"
            else:
                state.event("tool result", at=at, content=body)
        elif kind == EventKind.ERROR:
            state.event(
                f"error: {payload.get('type', 'Error')}",
                at=at,
                content={"message": self.text(payload.get("message"))},
                status="error",
            )
        elif kind == EventKind.ARCHITECTURE_EVENT:
            name = str(payload.get("name") or "event")
            data = {k: v for k, v in payload.items() if k != "name"}
            if name == "guardrail.triggered":
                name = f"guardrail {data.get('guard') or ''}".strip()
            state.event(name, at=at, content={"data": self.data(data)})
        else:
            state.event(kind.value.replace("_", " "), at=at, content={"data": self.data(payload)})

    def model_call(
        self,
        span: Any,
        *,
        model: str | None,
        messages: Sequence[Any],
        text: str | None,
        tool_calls: Sequence[Any] = (),
        usage: Any = None,
        started_at: float | None = None,
    ) -> None:
        state = _state.trace_var.get()
        if state is None:
            return
        try:
            target = state.spans.get(getattr(span, "span_id", "") or "")
            if target is None:
                # A model call no span covers (ReWOO's planner, a judge,
                # an output-validation retry): record it as its own step.
                now = time.time()
                target = SpanRecord(
                    span_id=_new_id()[:16],
                    parent_id=_state.parent_var.get(),
                    name=f"chat {model or ''}".strip(),
                    kind="model",
                    started_at=started_at if started_at is not None else now,
                    ended_at=now,
                    duration_ms=(now - started_at) * 1000.0 if started_at is not None else None,
                    attributes={"model": model},
                )
                if state.add(target) is None:
                    return
                if usage is not None:
                    target.attributes.update({
                        "gen_ai.usage.input_tokens": getattr(usage, "input_tokens", None),
                        "gen_ai.usage.output_tokens": getattr(usage, "output_tokens", None),
                        "cost_usd": getattr(usage, "cost_usd", None),
                    })
            target.content["messages"] = [self.message(m) for m in list(messages)[-MAX_MESSAGES:]]
            target.content["reply"] = self.text(text)
            if tool_calls:
                target.content["tool_calls"] = [
                    {"tool": getattr(c, "tool", None), "args": self.data(getattr(c, "args", None))}
                    for c in tool_calls
                ]
            if usage is not None and getattr(usage, "cost_usd", None):
                target.attributes.setdefault("cost_usd", usage.cost_usd)
        except Exception as exc:  # noqa: BLE001
            self._warn_once("model", f"Studio skipped a model call it couldn't record: {exc!r}")

    # ---- content -------------------------------------------------------

    def text(self, value: Any) -> str | None:
        """``value`` as kept text: redacted, capped, or ``None`` when
        content capture is off."""
        if value is None or not self.settings.capture_content:
            return None
        text = value if isinstance(value, str) else str(value)
        if self.settings.redact is not None:
            text = self.settings.redact(text)
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT] + f"… [{len(text) - MAX_TEXT} more characters]"
        return text

    def data(self, value: Any, _depth: int = 0) -> Any:
        """A JSON-safe copy of ``value`` with its strings kept as
        :meth:`text` keeps them."""
        if not self.settings.capture_content:
            return None
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            return self.text(value)
        if _depth >= 6:
            return self.text(repr(value))
        if isinstance(value, Mapping):
            items = list(value.items())[:200]
            return {str(k): self.data(v, _depth + 1) for k, v in items}
        if isinstance(value, list | tuple | set | frozenset):
            return [self.data(v, _depth + 1) for v in list(value)[:200]]
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            try:
                return self.data(dump(mode="json"), _depth + 1)
            except Exception:  # noqa: BLE001
                pass
        return self.text(value)

    def message(self, message: Any) -> dict[str, Any]:
        role = getattr(message, "role", None)
        out: dict[str, Any] = {
            "role": getattr(role, "value", role),
            "content": self.text(getattr(message, "content", None)),
        }
        if getattr(message, "tool_call_id", None):
            out["tool_call_id"] = message.tool_call_id
        if getattr(message, "name", None):
            out["name"] = message.name
        calls = getattr(message, "tool_calls", ()) or ()
        if calls:
            out["tool_calls"] = [
                {"tool": getattr(c, "tool", None), "args": self.data(getattr(c, "args", None))}
                for c in calls
            ]
        images = getattr(message, "images", ()) or ()
        if images:
            out["images"] = len(images)
        return out

    # ---- writing -------------------------------------------------------

    def submit(self, op: tuple[Any, ...]) -> None:
        """Queue a write. Never blocks: when the queue is full the write
        is dropped (and counted, with one warning)."""
        self._ensure_thread()
        try:
            self._queue.put_nowait(op)
        except queue.Full:
            self._dropped += 1
            self._warn_once(
                "full",
                f"Studio's write queue is full ({QUEUE_SIZE} items); dropping records."
                " The disk may be too slow, or the writer stopped.",
            )

    def flush(self, timeout: float = 5.0) -> bool:  # noqa: ASYNC109 — blocking by design
        """Wait until everything queued so far is written. Returns
        ``False`` if ``timeout`` seconds pass first."""
        deadline = time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
        return True

    def close(self, timeout: float = 5.0) -> None:  # noqa: ASYNC109 — blocking by design
        """Write what's queued, then stop the writer and close its
        connection (which checkpoints the WAL into the main file)."""
        if self._thread is None:
            return
        self.flush(timeout)
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            return
        self._thread.join(timeout)
        self._thread = None

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        with self._thread_lock:
            if self._thread is None:
                thread = threading.Thread(
                    target=self._write_loop, name="loomflow-studio-writer", daemon=True
                )
                thread.start()
                self._thread = thread
                _register_exit_flush()

    def _write_loop(self) -> None:
        conn = self.store.writer()
        try:
            StudioStore.prepare_search(conn)
        except Exception as exc:  # noqa: BLE001 — search falls back to a scan
            self._warn_once("search", f"Studio couldn't build its search index: {exc!r}")
        last_prune = self._prune(conn)
        stop = False
        while not stop:
            op = self._queue.get()
            batch = [op]
            while len(batch) < 200 and op is not None:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            stop = any(item is None for item in batch)
            try:
                with conn:
                    for item in batch:
                        if item is not None:
                            self._apply(conn, item)
            except Exception as exc:  # noqa: BLE001 — keep the writer alive
                self._warn_once("write", f"Studio couldn't write to {self.settings.path}: {exc!r}")
            finally:
                for _ in batch:
                    self._queue.task_done()
            if time.monotonic() - last_prune >= PRUNE_EVERY:
                last_prune = self._prune(conn)
        conn.close()

    def _prune(self, conn: Any) -> float:
        """Delete runs older than ``keep_days``, if set; returns when."""
        days = self.settings.keep_days
        if days is not None:
            try:
                StudioStore.prune(conn, time.time() - days * 86400)
            except Exception as exc:  # noqa: BLE001 — keep the writer alive
                self._warn_once("prune", f"Studio couldn't delete old runs from {self.settings.path}: {exc!r}")
        return time.monotonic()

    @staticmethod
    def _apply(conn: Any, op: tuple[Any, ...]) -> None:
        kind = op[0]
        if kind == "trace":
            StudioStore.write_trace(conn, op[1], op[2], op[3])
        elif kind == "score":
            StudioStore.add_score(conn, op[1])
        elif kind == "experiment_start":
            StudioStore.start_experiment(conn, op[1])
        elif kind == "experiment_case":
            StudioStore.add_experiment_case(conn, op[1])
        elif kind == "experiment_finish":
            StudioStore.finish_experiment(conn, op[1], op[2], op[3])

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _LOG.warning(message)


def _is_ancestor(state: TraceState, ancestor_id: str, span: SpanRecord) -> bool:
    parent = span.parent_id
    seen = 0
    while parent is not None and seen < 1000:
        if parent == ancestor_id:
            return True
        node = state.spans.get(parent)
        parent = node.parent_id if node is not None else None
        seen += 1
    return False


def _preview(value: Any) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text[:2000]


_exit_registered = False


def _register_exit_flush() -> None:
    global _exit_registered
    if not _exit_registered:
        _exit_registered = True
        atexit.register(_flush_all_at_exit)


def _flush_all_at_exit() -> None:
    for recorder in _state.all_recorders():
        recorder.close(timeout=5.0)


def tap(recorder: Recorder, emit: Callable[[Event], Awaitable[None]]) -> Callable[[Event], Awaitable[None]]:
    """``emit`` that also records each event."""

    async def recording_emit(event: Event) -> None:
        recorder.on_event(event)
        await emit(event)

    return recording_emit
