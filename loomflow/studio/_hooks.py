"""The seams loomflow's core calls. Each is a no-op unless Studio is
recording the current run, at the cost of one context-variable read.

* :func:`agent_run` / :func:`workflow_run` wrap a run: the first one
  in a call chain starts a trace (when Studio is on and the trace is
  sampled); runs inside it become part of it.
* :func:`compose` adds Studio's telemetry sink to a run's telemetry.
* :func:`tap` records the events a run emits.
* :func:`agent_started` and :func:`model_call` add what spans and
  events don't carry: the agent's definition and model messages.
* The ``experiment_*`` functions record an eval harness run.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from ..core.types import Event, EventKind
from . import _state

Emit = Callable[[Event], Awaitable[None]]


def recording() -> bool:
    """True inside a run Studio is recording."""
    return _state.trace_var.get() is not None


@asynccontextmanager
async def _trace_root(start: Callable[[Any], None] | None = None) -> AsyncIterator[Any]:
    """Start a trace unless one is running (or suppressed) already;
    yield its state when this call owns it, else ``None``."""
    if _state.trace_var.get() is not None or _state.suppress_var.get():
        yield None
        return
    recorder = _state.active_recorder()
    if recorder is None:
        yield None
        return
    if not recorder.sampled():
        token = _state.suppress_var.set(True)
        try:
            yield None
        finally:
            _state.suppress_var.reset(token)
        return
    state = recorder.new_trace()
    trace_token = _state.trace_var.set(state)
    parent_token = _state.parent_var.set(None)
    try:
        if start is not None:
            start(state)
        yield state
    except BaseException as exc:
        # A timeout or cancellation ends the run outside its own span.
        state.error = state.error or f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _state.parent_var.reset(parent_token)
        _state.trace_var.reset(trace_token)
        recorder.finish(state)


_NOTHING: contextlib.AbstractAsyncContextManager[Any] = contextlib.nullcontext()


def agent_run() -> contextlib.AbstractAsyncContextManager[Any]:
    """Around one ``Agent`` run (``_loop_guarded``). With nothing to
    record (the default) it's a shared no-op context: the hot path."""
    if (
        _state.trace_var.get() is None
        and not _state.suppress_var.get()
        and _state.current_settings() is None
    ):
        return _NOTHING
    return _trace_root()


def compose(telemetry: Any) -> Any:
    """``telemetry`` with Studio's sink added, while recording."""
    state = _state.trace_var.get()
    if state is None:
        return telemetry
    from ..observability.tracing import MultiTelemetry, NoTelemetry

    sink = state.recorder.sink
    if telemetry is None or isinstance(telemetry, NoTelemetry):
        return sink
    return MultiTelemetry([telemetry, sink])


def tap(emit: Emit) -> Emit:
    state = _state.trace_var.get()
    if state is None:
        return emit
    from ._recorder import tap as _tap

    return _tap(state.recorder, emit)


async def agent_started(agent: Any, run_ctx: Any, prompt: str) -> None:
    state = _state.trace_var.get()
    if state is not None:
        await state.recorder.agent_started(agent, run_ctx, prompt)


def model_call(
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
    if state is not None:
        state.recorder.model_call(
            span,
            model=model,
            messages=messages,
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            started_at=started_at,
        )


def now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


@dataclass
class WorkflowHandle:
    state: Any
    span: Any


@asynccontextmanager
async def workflow_run(
    workflow: Any,
    input: Any,
    *,
    user_id: str | None,
    session_id: str,
) -> AsyncIterator[WorkflowHandle | None]:
    """Around one ``Workflow`` run: a ``workflow`` span (the trace's
    root, or a child of the run that called it)."""
    async with _trace_root() as owned:
        state = _state.trace_var.get()
        if state is None:
            yield None
            return
        from ._catalog import describe_workflow
        from ._recorder import SpanRecord, _new_id

        record = describe_workflow(workflow)
        state.agents[record["key"] + record["fingerprint"]] = record
        span = SpanRecord(
            span_id=_new_id()[:16],
            parent_id=_state.parent_var.get(),
            name=record["name"],
            kind="workflow",
            started_at=time.time(),
            attributes={
                "agent_key": record["key"],
                "fingerprint": record["fingerprint"],
                "user_id": user_id,
                "session_id": session_id,
            },
            content={"input": state.recorder.data(input)},
        )
        if state.add(span) is None:
            yield None
            return
        if owned is not None:
            state.recorder.start(state)
        token = _state.parent_var.set(span.span_id)
        try:
            yield WorkflowHandle(state=state, span=span)
        except Exception as exc:
            span.fail(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _state.parent_var.reset(token)
            span.close()


def workflow_event(handle: WorkflowHandle | None, event: Event) -> None:
    if handle is None:
        return
    state, payload = handle.state, event.payload
    recorder = state.recorder
    if event.kind == EventKind.WORKFLOW_STEP_COMPLETED:
        span = _latest_step(state, str(payload.get("node")))
        if span is not None:
            span.content["output"] = recorder.data(payload.get("output"))
            if payload.get("interrupted"):
                span.status = "interrupted"
                span.attributes["interruption_reason"] = payload.get("interruption_reason")
    elif event.kind == EventKind.WORKFLOW_STEP_FAILED:
        span = _latest_step(state, str(payload.get("node")))
        if span is not None and span.status == "ok":
            span.fail(str(payload.get("error")))
    elif event.kind == EventKind.WORKFLOW_COMPLETED:
        handle.span.content["output"] = recorder.data(payload.get("output"))
        if payload.get("interrupted"):
            handle.span.attributes["interrupted"] = True
            handle.span.attributes["interruption_reason"] = payload.get("interruption_reason")


def _latest_step(state: Any, node: str) -> Any:
    for span_id in reversed(state.step_spans.get(node, [])):
        span = state.spans.get(span_id)
        if span is not None and "output" not in span.content:
            return span
    return None


# ---------------------------------------------------------------------------
# Experiments (the eval harness)
# ---------------------------------------------------------------------------


@dataclass
class ExperimentHandle:
    recorder: Any
    id: str
    project: str
    metrics: list[str] = field(default_factory=list)


def case_key(case: Any) -> str:
    """How a case is matched across experiments: its ``id`` when one was
    given, else a hash of its input and expected output (generated ids
    differ every time a dataset is loaded)."""
    if "id" in getattr(case, "model_fields_set", set()):
        return str(case.id)
    raw = f"{getattr(case, 'input', '')}\x00{getattr(case, 'expected', '') or ''}"
    return "input:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def dataset_name(dataset: Any, cases: Sequence[Any]) -> str:
    name = getattr(dataset, "name", None)
    if isinstance(name, str) and name:
        return name
    digest = hashlib.sha1("\x00".join(case_key(c) for c in cases).encode("utf-8")).hexdigest()[:8]
    return f"dataset {digest}"


async def experiment_start(
    agent: Any,
    dataset: Any,
    cases: Sequence[Any],
    metrics: Sequence[str],
    *,
    name: str | None = None,
) -> ExperimentHandle | None:
    if _state.trace_var.get() is not None:
        return None  # an eval inside a recorded run: just record the runs
    recorder = _state.active_recorder()
    if recorder is None:
        return None
    from ._catalog import describe_agent
    from ._recorder import _new_id

    agent_record: Mapping[str, Any] | None = None
    if hasattr(agent, "_instructions"):
        agent_record = await describe_agent(agent)
    scope = _state.scope_var.get()
    project = (scope.project if scope else None) or recorder.settings.project or _state.default_project()
    exp_id = "exp_" + _new_id()[:20]
    label = name or (agent_record["name"] if agent_record else "experiment")
    recorder.submit((
        "experiment_start",
        {
            "id": exp_id,
            "project": project,
            "name": label,
            "dataset": dataset_name(dataset, cases),
            "agent": agent_record,
            "started_at": time.time(),
            "metadata": {"metrics": list(metrics), "cases": len(cases)},
        },
    ))
    return ExperimentHandle(recorder=recorder, id=exp_id, project=project, metrics=list(metrics))


@contextlib.contextmanager
def experiment_case(handle: ExperimentHandle | None, key: str) -> Iterator[None]:
    """Runs inside are recorded as this experiment's case ``key``."""
    if handle is None:
        yield
        return
    outer = _state.scope_var.get()
    scope = _state.Scope(
        enabled=True,
        project=handle.project,
        tags=outer.tags if outer else (),
        metadata=dict(outer.metadata) if outer else {},
        experiment_id=handle.id,
        case_key=key,
    )
    token = _state.scope_var.set(scope)
    try:
        yield
    finally:
        _state.scope_var.reset(token)


def experiment_record_case(
    handle: ExperimentHandle | None,
    *,
    position: int,
    key: str,
    case: Any,
    run_id: str | None,
    output: str | None,
    scores: Mapping[str, float],
    error: str | None,
) -> None:
    if handle is None:
        return
    recorder = handle.recorder
    recorder.submit((
        "experiment_case",
        {
            "experiment_id": handle.id,
            "case_key": key,
            "position": position,
            "input": recorder.text(getattr(case, "input", None)),
            "expected": recorder.text(getattr(case, "expected", None)),
            "output": recorder.text(output),
            "run_id": run_id,
            "scores": dict(scores),
            "error": error,
        },
    ))
    for metric, value in scores.items():
        recorder.submit(("score", _score_row(run_id, metric, value, None, "eval")))


def experiment_finish(handle: ExperimentHandle | None, summary: Any) -> None:
    if handle is not None:
        handle.recorder.submit(("experiment_finish", handle.id, time.time(), summary))


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def _score_row(
    run_id: str | None, key: str, value: Any, comment: str | None, source: str
) -> dict[str, Any]:
    score: float | None
    text: str | None
    if isinstance(value, bool):
        score, text = (1.0 if value else 0.0), ("yes" if value else "no")
    elif isinstance(value, int | float):
        score, text = float(value), None
    else:
        score, text = None, str(value)
    return {
        "run_id": run_id,
        "key": key,
        "score": score,
        "value": text,
        "comment": comment,
        "source": source,
        "created_at": time.time(),
    }


def record_scores(result: Any, scores: Mapping[str, Any], *, source: str) -> None:
    """Record metric scores for a finished run (the online scorer)."""
    run_id = getattr(result, "run_id", None)
    if not run_id or not scores:
        return
    recorder = _state.active_recorder()
    if recorder is None:
        return
    for key, value in scores.items():
        recorder.submit(("score", _score_row(run_id, key, value, None, source)))
