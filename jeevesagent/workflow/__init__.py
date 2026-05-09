"""Workflow — developer-controlled DAGs.

JeevesAgent ships **two peer primitives** for building LLM systems:

* :class:`~jeevesagent.Agent` — the LLM controls the loop. Open-ended
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

Composition with :class:`~jeevesagent.Agent`:

* **Agent inside a Workflow.** Pass an ``Agent`` instance as a node;
  the framework calls ``.run(input)`` automatically and threads the
  live :class:`~jeevesagent.RunContext` (user_id / session_id /
  metadata) through to the inner agent run.
* **Workflow inside an Agent.** Call ``wf.as_tool()`` to get a
  :class:`~jeevesagent.Tool` that an Agent can invoke. The whole
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

from ..core.context import RunContext, _ctx_var, get_run_context
from ..core.ids import new_id
from ..core.protocols import Telemetry
from ..core.types import Event, EventKind
from ..observability.tracing import NoTelemetry
from ..tools.registry import Tool

if TYPE_CHECKING:
    # ``Agent`` isn't referenced in annotations here (we import it
    # lazily at runtime inside ``_coerce_step`` to avoid a circular
    # import). ``AuditLog`` IS used in a string annotation on the
    # ``Workflow`` constructor, so it stays.
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
"""Placeholder for the conceptual entry of a graph."""

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
    # Avoid circular import: Agent lives in jeevesagent.agent.api
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

    When called inside a live :class:`~jeevesagent.RunContext` (set
    by :meth:`Workflow.run` or by an enclosing
    :meth:`~jeevesagent.Agent.run`), the step opens a
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
                "jeeves.workflow.step",
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
# Workflow — the main primitive
# ---------------------------------------------------------------------------


class Workflow:
    """Developer-controlled DAG. Peer of :class:`~jeevesagent.Agent`.

    Construct with the explicit graph builder (``add_node`` /
    ``add_edge`` / ``set_start``) for full control, or use one of
    the sugar classmethods for common shapes:

    * :meth:`chain` — linear sequence
    * :meth:`route` — classify, then dispatch
    * :meth:`parallel` — fan out, run, merge

    Run with :meth:`run` (collects everything, returns a
    :class:`WorkflowResult`) or :meth:`stream` (yields
    :class:`~jeevesagent.Event` per step).

    Compose with :class:`~jeevesagent.Agent`:

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
        audit_log: AuditLog | None = None,
        max_steps: int = 100,
        max_visits_per_node: int = 25,
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
        self._telemetry = telemetry
        self._audit_log = audit_log
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

    def add_edge(self, source: str, target: str | _Sentinel) -> Workflow:
        """Add an unconditional edge from ``source`` to ``target``."""
        self._validate_source(source)
        self._edges[source] = target
        return self

    def add_router(
        self,
        source: str,
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
        """
        self._validate_source(source)
        # Normalize values to either str or sentinel.
        normalized: dict[str, str] = {}
        for k, v in routes.items():
            if isinstance(v, _Sentinel):
                normalized[k] = "__END__"
            else:
                normalized[k] = v
        self._edges[source] = _Router(fn=fn, routes=normalized, default=default)
        return self

    def set_start(self, node: str) -> Workflow:
        """Mark ``node`` as the graph's entry point."""
        if node not in self._nodes:
            raise ValueError(f"start node {node!r} is not registered")
        self._start = node
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
        :class:`~jeevesagent.Event` instances.

        Yields ``WORKFLOW_STARTED``, one ``WORKFLOW_STEP_STARTED`` /
        ``WORKFLOW_STEP_COMPLETED`` pair per visited node, and
        finally ``WORKFLOW_COMPLETED`` (or ``ERROR`` on failure).
        Consumers can break out of the iterator early to cancel.
        """
        if self._start is None:
            raise RuntimeError(
                f"workflow {self.name!r} has no start node; "
                "call set_start() or use one of the sugar constructors"
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

        try:
            yield Event(
                kind=EventKind.WORKFLOW_STARTED,
                session_id=sid,
                payload={"workflow": self.name, "input": input},
            )

            current: str | None = self._start
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
                    "jeeves.workflow.step",
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
                current = self._next_node(current, value)

            yield Event(
                kind=EventKind.WORKFLOW_COMPLETED,
                session_id=sid,
                payload={"workflow": self.name, "output": value},
            )
        finally:
            _ctx_var.reset(token)

    # ---- composition ------------------------------------------------------

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        input_arg: str = "input",
    ) -> Tool:
        """Expose this workflow as a :class:`~jeevesagent.Tool` an
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
        audit_log: AuditLog | None = None,
        max_steps: int = 100,
        max_visits_per_node: int = 25,
    ) -> Workflow:
        """Linear sequence: ``steps[0] → steps[1] → ... → steps[-1]``.

        Each step receives the previous step's return value. The
        final step's return is the workflow's output.

        Pass ``telemetry`` / ``audit_log`` / ``max_steps`` /
        ``max_visits_per_node`` here for the sugar constructor too —
        otherwise these kwargs would only be reachable via the
        explicit ``Workflow(...)`` constructor.
        """
        if not steps:
            raise ValueError("chain requires at least one step")
        wf = cls(
            name,
            telemetry=telemetry,
            audit_log=audit_log,
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
        audit_log: AuditLog | None = None,
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
        # input. We capture it in the route shim below.
        original_input: dict[str, Any] = {}

        async def _capture_classifier(value: Any) -> Any:
            original_input["v"] = value
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
        # output. Wrap the registered handlers to substitute it in.
        for k in keys:
            handler_name = f"route_{k}"
            inner = wf._nodes[handler_name]

            def _make_shim(_inner: _StepFn) -> _StepFn:
                async def _shim(_value: Any) -> Any:
                    return await _inner(original_input.get("v"))
                return _shim

            wf._nodes[handler_name] = _make_shim(inner)
        if default is not None:
            inner_d = wf._nodes["route_default"]

            def _make_shim_default(_inner: _StepFn) -> _StepFn:
                async def _shim(_value: Any) -> Any:
                    return await _inner(original_input.get("v"))
                return _shim
            wf._nodes["route_default"] = _make_shim_default(inner_d)

        return wf

    @classmethod
    def parallel(
        cls,
        steps: list[StepLike],
        *,
        merge: Callable[[list[Any]], Any] | None = None,
        name: str = "parallel",
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | None = None,
        max_steps: int = 100,
        max_visits_per_node: int = 25,
    ) -> Workflow:
        """Fan-out, run all steps with the *same* input, then merge.

        ``merge(results)`` produces the workflow's final output;
        defaults to returning the list of results unchanged. Steps
        run concurrently via :mod:`anyio` task groups.
        """
        if not steps:
            raise ValueError("parallel requires at least one step")
        coerced = [_coerce_step(s) for s in steps]
        merge_fn = merge if merge is not None else (lambda xs: xs)

        async def _fan_out(value: Any) -> Any:
            results: list[Any] = [None] * len(coerced)

            async def _one(i: int, fn: _StepFn) -> None:
                results[i] = await fn(value)

            async with anyio.create_task_group() as tg:
                for i, fn in enumerate(coerced):
                    tg.start_soon(_one, i, fn)
            return merge_fn(results)

        wf = cls(
            name,
            telemetry=telemetry,
            audit_log=audit_log,
            max_steps=max_steps,
            max_visits_per_node=max_visits_per_node,
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

    def _next_node(self, current: str, value: Any) -> str | None:
        edge = self._edges.get(current)
        if edge is None:
            # No outgoing edge — terminal.
            return None
        if isinstance(edge, _Sentinel):
            return None
        if isinstance(edge, _Router):
            key = edge.fn(value)
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
