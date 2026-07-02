"""Workflow — developer-controlled step graphs (directed, cycles
allowed with visit caps).

Loom ships **two peer primitives** for building LLM systems:

* :class:`~loomflow.Agent` — the LLM controls the loop. Open-ended
  reasoning, dynamic tool selection, ReAct-style think/act/observe.
* :class:`Workflow` (this module) — *the developer* controls the
  graph. Predictable steps, deterministic branching, audit-friendly.

Picking between them is an engineering decision, not a stylistic one
(see ``docs/workflow_vs_agent.md``). Production systems usually need
both, composed together. This module is the answer to "I have parts
of both."

Three ways to use it, ordered by ceremony:

1. **Plain Python with ``@step``.** Decorate any ``async def`` and
   it picks up telemetry + audit + (optional) journal hooks
   automatically. Control flow stays as ``if/await/return``::

       @step
       async def classify(text: str) -> str: ...

       @step
       async def respond(text: str, label: str) -> str: ...

       async def my_workflow(text: str, user_id: str) -> str:
           label = await classify(text)
           return await respond(text, label)

2. **Sugar constructors for common shapes.** No graph builder
   needed; one call makes a fully wired Workflow::

       wf = Workflow.chain([classify, respond])
       wf = Workflow.route(classifier, {"a": fn_a, "b": fn_b})
       wf = Workflow.parallel([fn_a, fn_b, fn_c], merge=combine)

3. **Explicit graph builder.** For cases where the graph IS the
   artifact (compliance flows, BPMN-like approval chains)::

       wf = Workflow("triage")
       wf.add_node("classify", classify)
       wf.add_node("billing", billing_agent)
       wf.add_node("tech", tech_agent)
       wf.set_start("classify")
       wf.add_router("classify", lambda r: r.lower(),
                     {"billing": "billing", "tech": "tech"})
       wf.add_edge("billing", END)
       wf.add_edge("tech", END)

Composition with :class:`~loomflow.Agent`:

* **Agent inside a Workflow.** Pass an ``Agent`` instance as a node;
  the framework calls ``.run(input)`` automatically and threads the
  live :class:`~loomflow.RunContext` (user_id / session_id /
  metadata) through to the inner agent run.
* **Workflow inside an Agent.** Call ``wf.as_tool()`` to get a
  :class:`~loomflow.Tool` that an Agent can invoke. The whole
  workflow runs as one tool call from the agent's perspective.

Both directions reuse the same observability spine — telemetry
spans, audit-log entries, ``user_id`` partition. A trace shows
exactly which decisions were workflow-deterministic and which
were LLM-driven, tagged via the ``pattern`` span attribute.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio

from ..core.context import (
    RunContext,
    _ambient_memory_var,
    _ambient_response_tone_var,
    _ambient_workspace_var,
    _ctx_var,
    get_run_context,
)
from ..core.ids import new_id
from ..core.protocols import Memory, Telemetry
from ..core.types import Event, EventKind
from ..observability.tracing import NoTelemetry
from ..tools.registry import Tool

if TYPE_CHECKING:
    # ``Agent`` isn't referenced in annotations here (we import it
    # lazily at runtime inside ``_coerce_step`` to avoid a circular
    # import). ``AuditLog`` IS used in a string annotation on the
    # ``Workflow`` constructor, so it stays.
    from pathlib import Path  # noqa: F401

    from ..security.audit import AuditLog  # noqa: F401

__all__ = [
    "END",
    "START",
    "Workflow",
    "WorkflowResult",
    "step",
]


# ---------------------------------------------------------------------------
# Sentinels — graph entry / exit markers
# ---------------------------------------------------------------------------


class _Sentinel:
    """Distinct token for ``START`` / ``END``. Compared by identity."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name


START = _Sentinel("START")
"""Sentinel source for ``add_edge(START, node)`` — alias for
``set_start(node)``. Lets graphs read symmetrically with ``END``:
``add_edge(START, "first")`` and ``add_edge("last", END)``."""

END = _Sentinel("END")
"""Sentinel target for ``add_edge(node, END)`` — terminates the run."""


# ---------------------------------------------------------------------------
# Result + event types
# ---------------------------------------------------------------------------


@dataclass
class WorkflowResult:
    """Outcome of a :meth:`Workflow.run` call.

    * ``output`` — the final node's return value. Type matches the
      last step's return type; users who want stronger typing can
      use a Pydantic model as the per-step value.
    * ``visited`` — list of node names in execution order, with
      repeats preserved. A linear flow visits each node once; a
      cyclic flow (``A → B → classify → C → B → classify → END``)
      shows the full trace so callers can see how many iterations
      ran. Use ``set(result.visited)`` for "which nodes were
      touched at all," and ``Counter(result.visited)`` for per-node
      visit counts.
    * ``per_step`` — mapping of node name -> the *last* value that
      node produced. With cycles, intermediate values for revisits
      are NOT preserved here (the event stream from ``stream()``
      has the full per-iteration history if you need it).
    """

    output: Any
    visited: list[str] = field(default_factory=list)
    per_step: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step normalization — turn Agent / callable / Workflow into an awaitable
# ---------------------------------------------------------------------------


# A "step" in user terms can be:
#   * an async function     (await fn(input))
#   * a sync function       (run on a worker thread)
#   * an Agent instance     (call .run(input).output)
#   * a Workflow instance   (call .run(input).output)
#
# ``_StepFn`` is the normalized internal type; everything user-facing
# accepts ``StepLike`` and we coerce.
_StepFn = Callable[[Any], Awaitable[Any]]
StepLike = Any  # exposed permissively; runtime-validated in _coerce_step


def _coerce_step(s: StepLike) -> _StepFn:
    """Normalize anything the user passes as a step into the
    internal ``_StepFn`` shape.

    Accepts:
      * ``Agent`` — calls ``await s.run(input, **ctx).output``
      * ``Workflow`` — calls ``await s.run(input, **ctx).output``
      * ``async def`` — awaited directly with the previous output
      * ``def`` — dispatched to a worker thread via anyio
    """
    # Avoid circular import: Agent lives in loomflow.agent.api
    from ..agent.api import Agent

    if isinstance(s, Workflow):
        async def _run_wf(inp: Any) -> Any:
            res = await s.run(inp)
            return res.output
        return _run_wf

    if isinstance(s, Agent):
        async def _run_agent(inp: Any) -> Any:
            ctx = get_run_context()
            res = await s.run(
                str(inp) if not isinstance(inp, str) else inp,
                user_id=ctx.user_id,
                session_id=ctx.session_id,
            )
            return res.output
        return _run_agent

    if inspect.iscoroutinefunction(s):
        async def _run_async(inp: Any) -> Any:
            return await s(inp)
        return _run_async

    if callable(s):
        async def _run_sync(inp: Any) -> Any:
            return await anyio.to_thread.run_sync(lambda: s(inp))
        return _run_sync

    raise TypeError(
        f"step must be a callable, Agent, or Workflow; got {type(s).__name__}"
    )


def _step_name(s: StepLike, fallback: str) -> str:
    """Best-effort name extraction for an unannotated step. Used by
    sugar constructors (``chain`` / ``route`` / ``parallel``) so the
    auto-generated graph has stable, debuggable node names."""
    name = getattr(s, "name", None)
    if isinstance(name, str) and name:
        return name
    name = getattr(s, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return fallback


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


@dataclass
class _Router:
    """Conditional edge: classify the previous node's output, look
    up the destination in ``routes``, fall through to ``default``
    if no key matches."""

    fn: Callable[[Any], Any]
    routes: dict[str, str]
    default: str | _Sentinel | None = None


# Edge target: either a literal next-node name, an END sentinel, or
# a router that picks based on the previous node's output.
_EdgeTarget = str | _Sentinel | _Router


# ---------------------------------------------------------------------------
# @step decorator — observability for plain Python workflows
# ---------------------------------------------------------------------------


def step(
    fn: Callable[..., Awaitable[Any]] | None = None,
    *,
    name: str | None = None,
) -> Any:
    """Decorator that adds telemetry + audit hooks to an async step.

    When called inside a live :class:`~loomflow.RunContext` (set
    by :meth:`Workflow.run` or by an enclosing
    :meth:`~loomflow.Agent.run`), the step opens a
    ``jeeves.workflow.step`` span tagged with the step name and
    writes ``step_started`` / ``step_completed`` audit entries
    attributed to the current ``user_id``. Outside any context the
    decorator is transparent — the function runs with no overhead.

    Use either as ``@step`` (uses the function's name) or as
    ``@step(name="custom-step-name")``.

    Example::

        @step
        async def classify(text: str) -> str:
            return (await classifier.run(text)).output

        # Plain Python control flow + free observability:
        async def my_workflow(text: str) -> str:
            label = await classify(text)
            return await respond(text, label)
    """

    def _wrap(f: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        # Fail loudly at decoration time when ``@step`` is applied
        # to a sync ``def``. Otherwise the failure surfaces deep in
        # the workflow runner as ``'str' can't be used in 'await'
        # expression``, which gives the user no hint that the issue
        # is on their function. Workflow steps run on the event
        # loop and *must* be awaitable.
        if not inspect.iscoroutinefunction(f):
            qualname = getattr(f, "__qualname__", getattr(f, "__name__", "step"))
            fname = getattr(f, "__name__", "step")
            raise TypeError(
                f"@step requires an async function, but "
                f"{qualname} is synchronous. Either:\n"
                f"  • Add 'async' to the def: "
                f"`async def {fname}(...)` — gives this step "
                f"telemetry / audit / journaling.\n"
                f"  • Drop @step and pass the plain function "
                f"directly (Workflow.chain / .route accept "
                f"sync callables; they're dispatched to a "
                f"worker thread)."
            )

        raw_name: Any = name or getattr(f, "__name__", "step")
        step_name: str = raw_name if isinstance(raw_name, str) else "step"

        async def _wrapped(*args: Any, **kwargs: Any) -> Any:
            ctx = get_run_context()
            telemetry: Telemetry | None = ctx.metadata.get(
                "_workflow_telemetry"
            ) if ctx.metadata else None
            if telemetry is None or isinstance(telemetry, NoTelemetry):
                # Outside a live workflow / no telemetry wired —
                # run the function without overhead. This is the
                # graceful-fallback property: ``@step`` is safe to
                # leave on functions that may be called from a unit
                # test or directly by a user.
                return await f(*args, **kwargs)

            async with telemetry.trace(
                "loom.workflow.step",
                step=step_name,
                user_id=ctx.user_id,
                session_id=ctx.session_id,
                pattern="workflow",
            ):
                return await f(*args, **kwargs)

        # Preserve the original name for introspection helpers.
        # ``__qualname__`` may be missing on partial / lambdas; coerce
        # any non-str fallback through ``str()`` to satisfy the type.
        _wrapped.__name__ = step_name
        qual = getattr(f, "__qualname__", step_name)
        _wrapped.__qualname__ = qual if isinstance(qual, str) else step_name
        _wrapped.__wrapped__ = f  # type: ignore[attr-defined]
        return _wrapped

    if fn is not None:
        return _wrap(fn)
    return _wrap


# ---------------------------------------------------------------------------
# audit_log= resolver — accepts an AuditLog instance, a path string /
# Path (auto-wrapped as ``FileAuditLog``), or ``None``. Validates at
# construction time so a wrong-shape value (e.g. a bare ``list``) fails
# loudly here instead of deep inside ``Workflow._audit`` with the
# cryptic ``list.append() takes no keyword arguments``.
# ---------------------------------------------------------------------------


async def _eval_classifier(fn: Callable[[Any], Any], value: Any) -> Any:
    """Run a router classifier against ``value`` and return the
    routing key. Tolerates both sync and async classifier functions
    so callers can write either:

    .. code-block:: python

        def classify(q: str) -> str: ...                # sync
        async def classify(q: str) -> str: ...          # async — calls a model

    Async classifiers are awaited; sync ones are called directly.
    Without this, an ``async def`` classifier would surface as a
    coroutine object in the routing dict and the router would
    raise ``no matching route``.
    """
    if inspect.iscoroutinefunction(fn):
        return await fn(value)
    result = fn(value)
    # Defensive: a sync wrapper may still return a coroutine
    # (e.g. ``lambda v: some_async_fn(v)``). Await it transparently
    # rather than stringifying the coroutine repr.
    if inspect.iscoroutine(result):
        return await result
    return result


def _resolve_audit_log(spec: Any) -> AuditLog | None:
    """Delegate to the public :func:`resolve_audit_log` resolver.

    Kept as a thin module-level wrapper so existing call sites
    (Workflow constructors) stay unchanged. See
    :func:`loomflow.security.resolve_audit_log` for the accepted
    forms — instance, path, dict, or None.
    """
    from ..security.audit import resolve_audit_log

    return resolve_audit_log(spec)


# ---------------------------------------------------------------------------
# Workflow — the main primitive
# ---------------------------------------------------------------------------


class Workflow:
    """Developer-controlled directed graph. Peer of
    :class:`~loomflow.Agent`.

    Not strictly a DAG — cycles are first-class (feedback /
    refinement loops), bounded by the ``max_steps`` and
    ``max_visits_per_node`` safety caps.

    Which primitive when:

    * **Deterministic routing / fixed steps** (the developer
      decides the control flow) → this class. ``Workflow.route``
      covers classify-then-dispatch with *your* classifier
      function.
    * **LLM-driven routing to one specialist** →
      :meth:`loomflow.Team.router` (Router architecture).
    * **LLM-driven delegation across many agents** →
      :meth:`loomflow.Team.supervisor` and friends, or
      ``Agent(architecture=...)``.
    * **Drawing a picture of an agent tree** → ``loomflow.graph``
      (visualization only — it never executes anything).

    Construct with the explicit graph builder (``add_node`` /
    ``add_edge`` / ``set_start``) for full control, or use one of
    the sugar classmethods for common shapes:

    * :meth:`chain` — linear sequence
    * :meth:`route` — classify, then dispatch
    * :meth:`parallel` — fan out, run, merge

    Run with :meth:`run` (collects everything, returns a
    :class:`WorkflowResult`) or :meth:`stream` (yields
    :class:`~loomflow.Event` per step).

    Compose with :class:`~loomflow.Agent`:

    * **Agent as a step** — pass an ``Agent`` instance to
      :meth:`add_node`; the framework calls ``.run`` and threads
      the live :class:`RunContext` through.
    * **Workflow as a tool** — call :meth:`as_tool` to get a
      :class:`Tool` an Agent can invoke.

    Both share the framework's observability spine: ``user_id`` is
    forwarded to nested agent runs, telemetry spans nest correctly,
    audit-log entries carry per-step attribution.
    """

    def __init__(
        self,
        name: str = "workflow",
        *,
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | str | Path | dict[str, Any] | None = None,
        memory: Memory | None = None,
        response_tone: str | None = None,
        max_steps: int = 100,
        max_visits_per_node: int = 25,
        workspace: Any | str | Mapping[str, Any] | None = None,
    ) -> None:
        """Construct an empty workflow.

        Cycles are supported — feedback loops (``A → B → classify
        → (C|D|END) → B`` and similar refinement patterns) are
        first-class. ``max_steps`` and ``max_visits_per_node`` are
        the safety caps that keep a buggy router (always picking
        the same branch) from looping forever.

        * ``max_steps`` — total steps executed in one ``run`` /
          ``stream`` call. Linear workflows visit each node once;
          cyclic flows pay this budget per iteration.
        * ``max_visits_per_node`` — any single node can be entered
          this many times. Tighter than ``max_steps`` because most
          runaways are one node looping on itself.

        ``memory`` is the **shared agent memory** for this run.
        Any :class:`~loomflow.Agent` step that did *not* receive
        an explicit ``memory=`` at construction picks this up at
        run time, so episodes / facts written by one agent are
        visible to the next without per-agent wiring. Agents that
        DID specify their own memory keep using it — explicit
        always wins. Pass an instance once and reuse it across
        ``wf.run()`` calls to keep memory across "conversations".

        Hit either cap and the workflow raises ``RuntimeError`` with
        the offending node named.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if max_visits_per_node < 1:
            raise ValueError("max_visits_per_node must be >= 1")
        self.name = name
        self._nodes: dict[str, _StepFn] = {}
        self._raw_nodes: dict[str, StepLike] = {}  # for ``as_tool`` schema
        self._edges: dict[str, _EdgeTarget] = {}
        self._start: str | None = None
        # Set when ``add_router(START, ...)`` is called; ``stream``
        # evaluates this on the input to pick the first node when
        # ``_start`` is ``None``. See ``add_router`` for semantics.
        self._entry_router: _Router | None = None
        self._telemetry = telemetry
        self._audit_log = _resolve_audit_log(audit_log)
        self._memory = memory
        # Workflow-level default response tone. Nested ``Agent`` steps
        # that didn't set their own ``response_tone=`` inherit this via
        # an ambient contextvar installed in ``stream``. None = no
        # propagation. See ``loomflow.core.tone`` for preset semantics.
        self._response_tone = response_tone
        # Workflow-level shared notebook. Nested ``Agent`` steps that
        # didn't set their own ``workspace=`` pick this up at run
        # start via :data:`_ambient_workspace_var` and auto-wire the
        # five notebook tools onto themselves. One Workflow + one
        # workspace gives every nested agent a shared scratchpad
        # without per-agent wiring.
        from ..workspace.resolver import resolve_workspace as _resolve_ws
        self._workspace: Any = _resolve_ws(workspace)
        self._max_steps = max_steps
        self._max_visits_per_node = max_visits_per_node

    # ---- graph builder ----------------------------------------------------

    def add_node(self, name: str, fn: StepLike) -> Workflow:
        """Register a node. ``fn`` can be an ``async def``, a sync
        function, an ``Agent`` instance, or a nested ``Workflow``."""
        if name in self._nodes:
            raise ValueError(f"node {name!r} already registered")
        self._nodes[name] = _coerce_step(fn)
        self._raw_nodes[name] = fn
        return self

    def add_edge(
        self, source: str | _Sentinel, target: str | _Sentinel
    ) -> Workflow:
        """Add an unconditional edge from ``source`` to ``target``.

        ``source`` can be ``START`` as an alias for
        :meth:`set_start` — ``add_edge(START, "first")`` reads
        symmetrically with ``add_edge("last", END)`` and matches
        the pattern users coming from LangGraph expect. The
        ``target`` of an ``add_edge(START, ...)`` call must be a
        registered node name (not another sentinel).
        """
        if isinstance(source, _Sentinel):
            if source is not START:
                raise ValueError(
                    f"add_edge source must be a node name or START; "
                    f"got sentinel {source!r}. Use add_edge(node, END) "
                    f"to terminate, set_start(name) / "
                    f"add_edge(START, name) to mark entry."
                )
            if isinstance(target, _Sentinel):
                raise ValueError(
                    "add_edge(START, target): target must be a "
                    "registered node name, not a sentinel."
                )
            return self.set_start(target)
        self._validate_source(source)
        self._edges[source] = target
        return self

    def add_router(
        self,
        source: str | _Sentinel,
        fn: Callable[[Any], Any],
        routes: Mapping[str, str | _Sentinel],
        *,
        default: str | _Sentinel | None = None,
    ) -> Workflow:
        """Attach a conditional edge to ``source``.

        At runtime ``fn(previous_output)`` is called; the returned
        key is looked up in ``routes`` to pick the next node. Keys
        not in ``routes`` fall through to ``default`` (or raise if
        no default is set).

        ``source`` can be ``START`` to **branch at the entry of the
        graph** — ``fn(input)`` is evaluated against the workflow's
        input value and the matching node becomes the first node
        executed. Mirrors LangGraph's ``add_conditional_edges(START,
        ...)`` and avoids the no-op "passthrough entry" pattern.
        ``set_start`` and ``add_router(START, ...)`` are mutually
        exclusive — calling one resets the other.
        """
        # Normalize routes once for both code paths.
        normalized: dict[str, str] = {}
        for k, v in routes.items():
            if isinstance(v, _Sentinel):
                normalized[k] = "__END__"
            else:
                normalized[k] = v

        if isinstance(source, _Sentinel):
            if source is not START:
                raise ValueError(
                    f"add_router source must be a node name or START; "
                    f"got sentinel {source!r}. Use add_router(START, ...) "
                    f"to branch at the entry, or add_router(node, ...) "
                    f"for a mid-graph branch."
                )
            # Validate target nodes exist (catches typos at build
            # time rather than at first run). END targets are fine.
            for k, v in normalized.items():
                if v != "__END__" and v not in self._nodes:
                    raise ValueError(
                        f"add_router(START, ...): route {k!r} → {v!r} "
                        f"is not a registered node. Call "
                        f"add_node({v!r}, ...) first."
                    )
            self._entry_router = _Router(
                fn=fn, routes=normalized, default=default
            )
            # Entry router supersedes any explicit set_start — keep
            # _start in sync so introspection stays consistent.
            self._start = None
            return self

        self._validate_source(source)
        self._edges[source] = _Router(fn=fn, routes=normalized, default=default)
        return self

    def set_start(self, node: str) -> Workflow:
        """Mark ``node`` as the graph's entry point.

        Mutually exclusive with :meth:`add_router` ``(START, ...)`` —
        whichever is called last wins.
        """
        if node not in self._nodes:
            raise ValueError(f"start node {node!r} is not registered")
        self._start = node
        # Clear any prior entry router so the two never coexist.
        self._entry_router = None
        return self

    # ---- run + stream -----------------------------------------------------

    async def run(
        self,
        input: Any = None,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Execute the graph. Each node receives the previous node's
        return value; the final node's return is ``result.output``.

        ``user_id`` / ``session_id`` / ``metadata`` populate the
        live :class:`RunContext` for the duration of the run. Any
        nested :class:`Agent` runs (when a node is an Agent) inherit
        this context automatically.
        """
        events: list[Event] = []
        async for ev in self.stream(
            input,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        ):
            events.append(ev)

        # Reconstruct WorkflowResult from the event stream so
        # ``run`` and ``stream`` share execution.
        visited: list[str] = []
        per_step: dict[str, Any] = {}
        output: Any = None
        for ev in events:
            if ev.kind == EventKind.WORKFLOW_STEP_COMPLETED:
                node = ev.payload["node"]
                visited.append(node)
                per_step[node] = ev.payload["output"]
                output = ev.payload["output"]
            elif ev.kind == EventKind.WORKFLOW_COMPLETED:
                # Cover the zero-step case (e.g. ``add_router(START,
                # ...)`` with ``default=END`` taking the unmatched
                # path): no STEP_COMPLETED ever fires, so the
                # workflow's output is whatever ``stream`` carries
                # forward — the original input.
                if not visited:
                    output = ev.payload.get("output")
        return WorkflowResult(
            output=output, visited=visited, per_step=per_step
        )

    async def stream(
        self,
        input: Any = None,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:  # AsyncIterator[Event] — typed as Any for sphinx-autoapi
        """Execute the graph as an async generator of
        :class:`~loomflow.Event` instances.

        Yields ``WORKFLOW_STARTED``, one ``WORKFLOW_STEP_STARTED`` /
        ``WORKFLOW_STEP_COMPLETED`` pair per visited node, and
        finally ``WORKFLOW_COMPLETED`` (or ``ERROR`` on failure).
        Consumers can break out of the iterator early to cancel.
        """
        if self._start is None and self._entry_router is None:
            raise RuntimeError(
                f"workflow {self.name!r} has no start node; "
                "call set_start(name) / add_edge(START, name) / "
                "add_router(START, ...) or use one of the sugar "
                "constructors (chain / route / parallel)"
            )

        # Auto-generate a session_id if the caller didn't supply
        # one. Every Event needs one (for trace correlation), and
        # downstream consumers expect a stable id per workflow run.
        sid = session_id if session_id is not None else new_id("wf_session")

        # Install a RunContext so @step decorators and nested agent
        # runs can pick up user_id / session_id / telemetry hookup.
        run_meta = dict(metadata or {})
        if self._telemetry is not None:
            run_meta["_workflow_telemetry"] = self._telemetry
        ctx = RunContext(
            user_id=user_id,
            session_id=sid,
            metadata=run_meta,
        )
        token = _ctx_var.set(ctx)
        # Install the workflow's ``memory=`` (if any) as the
        # ambient memory for this run. Nested ``Agent`` steps that
        # didn't get their own ``memory=`` at construction read
        # this contextvar and use it instead of their default.
        # Explicit-on-Agent always wins; we only fill the gap.
        memory_token = _ambient_memory_var.set(self._memory)
        # Same pattern for ``response_tone``: nested agents that
        # didn't set their own pick up the workflow-level tone.
        tone_token = _ambient_response_tone_var.set(self._response_tone)
        # Workspace ambient — nested ``Agent`` steps that didn't set
        # their own ``workspace=`` see the workflow's notebook here
        # and auto-wire the five notebook tools at run start.
        workspace_token = _ambient_workspace_var.set(self._workspace)

        try:
            yield Event(
                kind=EventKind.WORKFLOW_STARTED,
                session_id=sid,
                payload={"workflow": self.name, "input": input},
            )

            # Resolve the first node. When ``add_router(START, ...)``
            # was used, evaluate the entry router on the input to
            # pick the entry node. Otherwise use the explicit start.
            if self._entry_router is not None:
                key = await _eval_classifier(self._entry_router.fn, input)
                target = self._entry_router.routes.get(str(key))
                if target is None:
                    if self._entry_router.default is None:
                        raise RuntimeError(
                            f"entry router on {self.name!r} produced "
                            f"key {key!r} with no matching route and "
                            f"no default"
                        )
                    if isinstance(self._entry_router.default, _Sentinel):
                        current = None
                    else:
                        current = self._entry_router.default
                elif target == "__END__":
                    current = None
                else:
                    current = target
            else:
                current = self._start
            value: Any = input
            # Cycles are supported up to the per-workflow caps.
            # ``visit_counts`` tracks how many times each node has
            # been entered; ``total_steps`` tracks the global step
            # count so a long zig-zag (no single node loops, but
            # many do) still terminates eventually.
            visit_counts: dict[str, int] = {}
            total_steps = 0

            while current is not None:
                total_steps += 1
                if total_steps > self._max_steps:
                    raise RuntimeError(
                        f"workflow {self.name!r} exceeded max_steps="
                        f"{self._max_steps} at node {current!r}; "
                        "raise the cap or fix the routing logic that "
                        "keeps re-entering nodes"
                    )
                visit_counts[current] = visit_counts.get(current, 0) + 1
                if visit_counts[current] > self._max_visits_per_node:
                    raise RuntimeError(
                        f"workflow {self.name!r} re-entered {current!r} "
                        f"more than max_visits_per_node="
                        f"{self._max_visits_per_node} times; the router "
                        "controlling the loop probably never picks the "
                        "termination branch"
                    )

                fn = self._nodes[current]
                yield Event(
                    kind=EventKind.WORKFLOW_STEP_STARTED,
                    session_id=sid,
                    payload={"workflow": self.name, "node": current},
                )

                # Per-step telemetry span. Nested agent runs land
                # under this span automatically.
                tel = self._telemetry or NoTelemetry()
                async with tel.trace(
                    "loom.workflow.step",
                    step=current,
                    user_id=user_id,
                    session_id=sid,
                    pattern="workflow",
                ):
                    await self._audit(
                        ctx, "step_started", {"node": current}
                    )
                    try:
                        value = await fn(value)
                    except Exception as exc:  # noqa: BLE001
                        yield Event(
                            kind=EventKind.WORKFLOW_STEP_FAILED,
                            session_id=sid,
                            payload={
                                "workflow": self.name,
                                "node": current,
                                "error": str(exc),
                            },
                        )
                        await self._audit(
                            ctx,
                            "step_failed",
                            {"node": current, "error": str(exc)},
                        )
                        raise
                    await self._audit(
                        ctx, "step_completed", {"node": current}
                    )

                yield Event(
                    kind=EventKind.WORKFLOW_STEP_COMPLETED,
                    session_id=sid,
                    payload={
                        "workflow": self.name,
                        "node": current,
                        "output": value,
                    },
                )

                # Pick the next node from the outgoing edge.
                current = await self._next_node(current, value)

            yield Event(
                kind=EventKind.WORKFLOW_COMPLETED,
                session_id=sid,
                payload={"workflow": self.name, "output": value},
            )
        finally:
            _ambient_workspace_var.reset(workspace_token)
            _ambient_response_tone_var.reset(tone_token)
            _ambient_memory_var.reset(memory_token)
            _ctx_var.reset(token)

    # ---- visualisation ----------------------------------------------------

    def to_mermaid(self) -> str:
        """Render the workflow graph as a Mermaid ``flowchart TD``
        diagram.

        Returns a string suitable for:

        * Pasting into Markdown (GitHub renders Mermaid natively).
        * https://mermaid.live for online editing / PNG / SVG export.
        * Jupyter via ``IPython.display.Markdown(...)`` — or just
          type ``wf`` into a cell; ``_repr_markdown_`` calls this
          method for you.

        Conventions:
        * Solid arrows are unconditional edges.
        * Labelled solid arrows are explicit router branches.
        * Dotted arrows are router *default* branches.
        * ``START`` and ``END`` are stadium-shaped nodes; user
          steps are rounded rectangles.
        """
        lines = ["flowchart TD"]
        if not self._nodes:
            lines.append('    empty["(empty workflow)"]')
            return "\n".join(lines)

        # Mermaid node IDs are restricted; sanitise to a safe alias
        # while keeping the user's name as the visible label.
        def _alias(name: str) -> str:
            safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
            return f"n_{safe}"

        # Declare each node with its display label first so later
        # edge lines can reference the alias without re-declaring.
        for name in self._nodes:
            lines.append(f'    {_alias(name)}["{name}"]')

        if self._entry_router is not None:
            # START itself branches conditionally.
            for label, dst in self._entry_router.routes.items():
                if dst == "__END__":
                    lines.append(
                        f"    START([START]) -->|{label}| END([END])"
                    )
                else:
                    lines.append(
                        f"    START([START]) -->|{label}| {_alias(dst)}"
                    )
            if self._entry_router.default is not None:
                if isinstance(self._entry_router.default, _Sentinel):
                    lines.append(
                        "    START([START]) -.->|default| END([END])"
                    )
                else:
                    lines.append(
                        f"    START([START]) -.->|default| "
                        f"{_alias(self._entry_router.default)}"
                    )
        elif self._start is not None:
            lines.append(
                f"    START([START]) --> {_alias(self._start)}"
            )

        for src, target in self._edges.items():
            src_id = _alias(src)
            if isinstance(target, _Router):
                for label, dst in target.routes.items():
                    if dst == "__END__":
                        lines.append(
                            f"    {src_id} -->|{label}| END([END])"
                        )
                    else:
                        lines.append(
                            f"    {src_id} -->|{label}| {_alias(dst)}"
                        )
                # Default branch — dotted to distinguish from
                # explicit-key branches at a glance.
                if target.default is not None:
                    if isinstance(target.default, _Sentinel):
                        lines.append(
                            f"    {src_id} -.->|default| END([END])"
                        )
                    else:
                        lines.append(
                            f"    {src_id} -.->|default| "
                            f"{_alias(target.default)}"
                        )
            elif isinstance(target, _Sentinel):
                lines.append(f"    {src_id} --> END([END])")
            else:
                lines.append(f"    {src_id} --> {_alias(target)}")

        return "\n".join(lines)

    def to_dot(self) -> str:
        """Render the workflow as a Graphviz DOT digraph.

        Pipe through ``dot -Tpng -o graph.png`` or paste into
        https://dreampuf.github.io/GraphvizOnline. Use
        :meth:`to_mermaid` if you don't want a Graphviz install —
        Mermaid renders inline on GitHub and in Jupyter without
        any external tool.
        """
        lines = [f'digraph "{self.name}" {{', "    rankdir=TB;"]
        if not self._nodes:
            lines.append('    empty [label="(empty workflow)"];')
            lines.append("}")
            return "\n".join(lines)

        for name in self._nodes:
            lines.append(f'    "{name}" [shape=box, style=rounded];')

        end_seen_start = False
        if self._entry_router is not None or self._start is not None:
            lines.append(
                '    "__start__" [label="START", shape=oval];'
            )
        if self._entry_router is not None:
            for label, dst in self._entry_router.routes.items():
                if dst == "__END__":
                    lines.append(
                        f'    "__start__" -> "__end__" '
                        f'[label="{label}"];'
                    )
                    end_seen_start = True
                else:
                    lines.append(
                        f'    "__start__" -> "{dst}" '
                        f'[label="{label}"];'
                    )
            if self._entry_router.default is not None:
                if isinstance(self._entry_router.default, _Sentinel):
                    lines.append(
                        '    "__start__" -> "__end__" '
                        '[label="default", style=dashed];'
                    )
                    end_seen_start = True
                else:
                    lines.append(
                        f'    "__start__" -> '
                        f'"{self._entry_router.default}" '
                        f'[label="default", style=dashed];'
                    )
        elif self._start is not None:
            lines.append(
                f'    "__start__" -> "{self._start}";'
            )

        end_seen = False
        for src, target in self._edges.items():
            if isinstance(target, _Router):
                for label, dst in target.routes.items():
                    if dst == "__END__":
                        lines.append(
                            f'    "{src}" -> "__end__" '
                            f'[label="{label}"];'
                        )
                        end_seen = True
                    else:
                        lines.append(
                            f'    "{src}" -> "{dst}" '
                            f'[label="{label}"];'
                        )
                if target.default is not None:
                    if isinstance(target.default, _Sentinel):
                        lines.append(
                            f'    "{src}" -> "__end__" '
                            f'[label="default", style=dashed];'
                        )
                        end_seen = True
                    else:
                        lines.append(
                            f'    "{src}" -> "{target.default}" '
                            f'[label="default", style=dashed];'
                        )
            elif isinstance(target, _Sentinel):
                lines.append(f'    "{src}" -> "__end__";')
                end_seen = True
            else:
                lines.append(f'    "{src}" -> "{target}";')

        if end_seen or end_seen_start:
            lines.append('    "__end__" [label="END", shape=oval];')
        lines.append("}")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        """Auto-render in Jupyter when the user types ``wf`` in a
        cell. Wraps :meth:`to_mermaid` in a fenced Mermaid block;
        Jupyter (and JupyterLab >= 4 / VS Code) renders it inline."""
        return f"```mermaid\n{self.to_mermaid()}\n```"

    # ---- composition ------------------------------------------------------

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        input_arg: str = "input",
    ) -> Tool:
        """Expose this workflow as a :class:`~loomflow.Tool` an
        Agent can call.

        From the agent's perspective the whole workflow runs as one
        tool invocation. ``input_arg`` names the single string
        parameter the tool accepts (default ``"input"``); the
        framework forwards that value to :meth:`run` and returns
        ``result.output``.
        """
        tool_name = name or self.name
        tool_desc = description or f"Run the {self.name!r} workflow."

        async def _call(**kwargs: Any) -> Any:
            value = kwargs.get(input_arg, "")
            ctx = get_run_context()
            res = await self.run(
                value, user_id=ctx.user_id, session_id=ctx.session_id
            )
            return res.output

        return Tool(
            name=tool_name,
            description=tool_desc,
            fn=_call,
            input_schema={
                "type": "object",
                "properties": {input_arg: {"type": "string"}},
                "required": [input_arg],
            },
        )

    # ---- sugar constructors (most users start here) ----------------------

    @classmethod
    def chain(
        cls,
        steps: list[StepLike],
        *,
        name: str = "chain",
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | str | Path | dict[str, Any] | None = None,
        memory: Memory | None = None,
        response_tone: str | None = None,
        max_steps: int = 100,
        max_visits_per_node: int = 25,
    ) -> Workflow:
        """Linear sequence: ``steps[0] → steps[1] → ... → steps[-1]``.

        Each step receives the previous step's return value. The
        final step's return is the workflow's output.

        Pass ``telemetry`` / ``audit_log`` / ``memory`` /
        ``max_steps`` / ``max_visits_per_node`` here for the sugar
        constructor too — otherwise these kwargs would only be
        reachable via the explicit ``Workflow(...)`` constructor.
        """
        if not steps:
            raise ValueError("chain requires at least one step")
        wf = cls(
            name,
            telemetry=telemetry,
            audit_log=audit_log,
            memory=memory,
            response_tone=response_tone,
            max_steps=max_steps,
            max_visits_per_node=max_visits_per_node,
        )
        names: list[str] = []
        for i, s in enumerate(steps):
            n = _step_name(s, f"step_{i}")
            # Disambiguate duplicates so chain([fn, fn]) works.
            base = n
            j = 1
            while n in wf._nodes:
                n = f"{base}_{j}"
                j += 1
            wf.add_node(n, s)
            names.append(n)
        wf.set_start(names[0])
        for cur, nxt in zip(names, names[1:], strict=False):
            wf.add_edge(cur, nxt)
        wf.add_edge(names[-1], END)
        return wf

    @classmethod
    def route(
        cls,
        classifier: StepLike,
        routes: Mapping[str, StepLike],
        *,
        default: StepLike | None = None,
        name: str = "route",
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | str | Path | dict[str, Any] | None = None,
        memory: Memory | None = None,
        response_tone: str | None = None,
        max_steps: int = 100,
        max_visits_per_node: int = 25,
    ) -> Workflow:
        """Classify-then-dispatch.

        The ``classifier`` step's output is converted to ``str`` and
        used as a key into ``routes``. The matching step runs with
        the *original* input (not the classifier's output), so
        handlers see what the user asked, not the routing label.
        Pass ``default`` for a fallback when no key matches; without
        one, an unmatched key raises.
        """
        if not routes:
            raise ValueError("route requires at least one entry in routes")
        wf = cls(
            name,
            telemetry=telemetry,
            audit_log=audit_log,
            memory=memory,
            response_tone=response_tone,
            max_steps=max_steps,
            max_visits_per_node=max_visits_per_node,
        )
        wf.add_node("classify", classifier)

        # Wire each route as a node with edge → END.
        keys = list(routes.keys())
        for k, s in routes.items():
            n = f"route_{k}"
            wf.add_node(n, s)
            wf.add_edge(n, END)
        if default is not None:
            wf.add_node("route_default", default)
            wf.add_edge("route_default", END)

        # The classifier produces a routing key; the router picks
        # the matching node, but each handler needs the ORIGINAL
        # input. Carry it through PER-RUN state — ``stream()``
        # installs a fresh ``RunContext`` (with a fresh ``metadata``
        # dict) for every run, so concurrent ``run()`` calls on this
        # one Workflow instance each see their own captured input.
        # (A construction-time closure dict here would be shared
        # across runs: request B's classifier could overwrite the
        # captured value before request A's handler reads it — a
        # cross-request data leak under the normal build-once,
        # run-per-request server pattern.)
        input_key = "_route_original_input"

        async def _capture_classifier(value: Any) -> Any:
            md = get_run_context().metadata
            if isinstance(md, dict):
                md[input_key] = value
            # Run the user's classifier with the original input.
            return await _coerce_step(classifier)(value)

        # Replace the classify node with the capturing wrapper.
        wf._nodes["classify"] = _capture_classifier

        wf.set_start("classify")

        target_map: dict[str, str | _Sentinel] = {
            k: f"route_{k}" for k in keys
        }
        wf.add_router(
            "classify",
            lambda v: str(v).strip(),
            target_map,
            default="route_default" if default is not None else None,
        )

        # Each handler needs the original input, not the classifier
        # output. Wrap the registered handlers to substitute in the
        # per-run captured value from RunContext.metadata.
        def _make_shim(_inner: _StepFn) -> _StepFn:
            async def _shim(_value: Any) -> Any:
                md = get_run_context().metadata
                return await _inner(md.get(input_key))
            return _shim

        for k in keys:
            handler_name = f"route_{k}"
            wf._nodes[handler_name] = _make_shim(wf._nodes[handler_name])
        if default is not None:
            wf._nodes["route_default"] = _make_shim(
                wf._nodes["route_default"]
            )

        return wf

    @classmethod
    def parallel(
        cls,
        steps: list[StepLike],
        *,
        merge: Callable[[list[Any]], Any] | None = None,
        return_exceptions: bool = False,
        name: str = "parallel",
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | str | Path | dict[str, Any] | None = None,
        memory: Memory | None = None,
        response_tone: str | None = None,
        max_steps: int = 100,
        max_visits_per_node: int = 25,
        workspace: Any | str | Mapping[str, Any] | None = None,
    ) -> Workflow:
        """Fan-out, run all steps with the *same* input, then merge.

        ``merge(results)`` produces the workflow's final output;
        defaults to returning the list of results unchanged. Steps
        run concurrently via :mod:`anyio` task groups.

        ``return_exceptions=True`` gives gather-style error
        handling: a failing branch no longer cancels its siblings —
        the raised exception *object* is placed at that branch's
        index in the results list handed to ``merge``, and the
        workflow completes normally with the partial results. With
        the default ``False``, the first branch exception cancels
        the remaining branches and propagates (grouped by the anyio
        task group).

        Each branch runs under its own ``loom.workflow.step``
        telemetry span (``step="fan_out.<branch>"``, named after the
        step function) nested inside the ``fan_out`` node's span, so
        per-branch latency and failures show up in traces even
        though the event stream reports the fan-out as one node.

        ``workspace`` flows through to every nested Agent step via
        the same ambient-contextvar mechanism as the regular
        :class:`Workflow` constructor — one notebook shared across
        all parallel branches.
        """
        if not steps:
            raise ValueError("parallel requires at least one step")
        coerced = [_coerce_step(s) for s in steps]
        merge_fn = merge if merge is not None else (lambda xs: xs)
        # Stable, disambiguated branch names for per-branch spans.
        branch_names: list[str] = []
        for i, s in enumerate(steps):
            n = _step_name(s, f"branch_{i}")
            base, j = n, 1
            while n in branch_names:
                n = f"{base}_{j}"
                j += 1
            branch_names.append(n)

        async def _fan_out(value: Any) -> Any:
            results: list[Any] = [None] * len(coerced)
            # Per-branch telemetry — same ambient hookup as @step:
            # ``stream()`` stashes the workflow's telemetry in the
            # per-run RunContext metadata.
            ctx = get_run_context()
            tel: Telemetry = (
                ctx.metadata.get("_workflow_telemetry") or NoTelemetry()
            )

            async def _one(i: int, fn: _StepFn) -> None:
                try:
                    async with tel.trace(
                        "loom.workflow.step",
                        step=f"fan_out.{branch_names[i]}",
                        user_id=ctx.user_id,
                        session_id=ctx.session_id,
                        pattern="workflow",
                    ):
                        results[i] = await fn(value)
                except Exception as exc:  # noqa: BLE001
                    if not return_exceptions:
                        raise
                    # Gather-style: record the failure at this
                    # branch's slot; siblings keep running. (The
                    # span above already recorded the exception.)
                    results[i] = exc

            async with anyio.create_task_group() as tg:
                for i, fn in enumerate(coerced):
                    tg.start_soon(_one, i, fn)
            return merge_fn(results)

        wf = cls(
            name,
            telemetry=telemetry,
            audit_log=audit_log,
            memory=memory,
            response_tone=response_tone,
            max_steps=max_steps,
            max_visits_per_node=max_visits_per_node,
            workspace=workspace,
        )
        wf.add_node("fan_out", _fan_out)
        wf.set_start("fan_out")
        wf.add_edge("fan_out", END)
        return wf

    # ---- internals --------------------------------------------------------

    def _validate_source(self, source: str) -> None:
        if source not in self._nodes:
            raise ValueError(
                f"source node {source!r} is not registered; "
                f"call add_node({source!r}, ...) first"
            )

    async def _next_node(self, current: str, value: Any) -> str | None:
        edge = self._edges.get(current)
        if edge is None:
            # No outgoing edge — terminal.
            return None
        if isinstance(edge, _Sentinel):
            return None
        if isinstance(edge, _Router):
            key = await _eval_classifier(edge.fn, value)
            target = edge.routes.get(str(key))
            if target is None:
                if edge.default is None:
                    raise RuntimeError(
                        f"router on {current!r} produced key {key!r} "
                        f"with no matching route and no default"
                    )
                if isinstance(edge.default, _Sentinel):
                    return None
                return edge.default
            if target == "__END__":
                return None
            return target
        # Plain string target.
        return edge

    async def _audit(
        self, ctx: RunContext, action: str, payload: dict[str, Any]
    ) -> None:
        if self._audit_log is None or ctx.session_id is None:
            return
        entry_payload = {"workflow": self.name, **payload}
        try:
            await self._audit_log.append(
                session_id=ctx.session_id,
                actor="workflow",
                action=action,
                payload=entry_payload,
                user_id=ctx.user_id,
            )
        except TypeError:
            # Legacy AuditLog without the user_id kwarg.
            await self._audit_log.append(
                session_id=ctx.session_id,
                actor="workflow",
                action=action,
                payload=entry_payload,
            )
