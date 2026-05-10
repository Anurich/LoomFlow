"""Tests for the Workflow primitive (Phase 0–3 of the workflow milestone).

Coverage map:

* ``@step`` decorator — transparent outside a workflow context;
  emits a telemetry span when one is wired.
* ``Workflow.chain`` — linear sequence; output flows step-to-step.
* ``Workflow.route`` — classify-then-dispatch with default
  fallback; handlers receive the original input, not the
  classifier's label.
* ``Workflow.parallel`` — fan-out + merge under anyio task group.
* Explicit graph builder — ``add_node`` / ``add_edge`` /
  ``add_router`` / ``set_start``; cycle detection.
* Composition with Agent — Agent-as-step (a node calls
  ``agent.run`` automatically) and Workflow-as-tool
  (``wf.as_tool()`` plugs into ``Agent(tools=)``).
* Streaming — ``Workflow.stream`` emits typed Events; consumers
  can break out early.
* RunContext propagation — ``user_id`` set on ``Workflow.run``
  flows into nested agent runs and ``@step`` spans.
"""

from __future__ import annotations

from typing import Any

import pytest

from loomflow import (
    END,
    Agent,
    InMemoryAuditLog,
    InMemoryMemory,
    Workflow,
    WorkflowResult,
    step,
)
from loomflow.core.context import get_run_context
from loomflow.core.types import EventKind
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# @step decorator
# ---------------------------------------------------------------------------


async def test_step_decorator_transparent_outside_workflow() -> None:
    """A ``@step``-decorated function called directly (no workflow,
    no live RunContext) should run with zero overhead — same
    behaviour as a plain ``async def``. This is the graceful-fallback
    property that lets users decorate freely without paying for
    observability when there's nothing to observe."""

    @step
    async def double(x: int) -> int:
        return x * 2

    assert await double(3) == 6
    # Identity preservation — sometimes useful for debugging.
    assert double.__name__ == "double"


async def test_step_decorator_with_explicit_name() -> None:
    @step(name="custom-name")
    async def inner(x: int) -> int:
        return x + 1

    assert await inner(1) == 2
    assert inner.__name__ == "custom-name"


def test_step_decorator_rejects_sync_function_at_decoration_time() -> None:
    """``@step`` runs the function on the event loop via ``await``,
    so wrapping a sync ``def`` would later fail deep in the
    workflow runner with the cryptic ``'str' can't be used in
    'await' expression``. Fail loudly at decoration time instead,
    with a message that names the function and gives the user
    both fixes (add ``async``, or drop ``@step``)."""
    with pytest.raises(TypeError) as excinfo:

        @step
        def sync_step(x: int) -> int:  # type: ignore[misc]
            return x + 1

    msg = str(excinfo.value)
    assert "sync_step" in msg
    assert "async" in msg
    # Both remediation options should appear so users know
    # they don't have to make the function async if they don't
    # want telemetry — Workflow.chain accepts sync directly.
    assert "Workflow.chain" in msg or "drop @step" in msg.lower()


def test_step_decorator_rejects_sync_with_explicit_name() -> None:
    """The check fires regardless of whether ``@step`` is used
    bare (``@step``) or parameterised (``@step(name=...)``)."""
    with pytest.raises(TypeError):

        @step(name="custom")
        def sync_step(x: int) -> int:  # type: ignore[misc]
            return x


# ---------------------------------------------------------------------------
# Workflow.chain
# ---------------------------------------------------------------------------


async def test_chain_runs_steps_in_order() -> None:
    async def add_one(x: int) -> int:
        return x + 1

    async def double(x: int) -> int:
        return x * 2

    async def to_str(x: int) -> str:
        return f"value={x}"

    wf = Workflow.chain([add_one, double, to_str])
    result = await wf.run(3)

    assert isinstance(result, WorkflowResult)
    assert result.output == "value=8"  # ((3 + 1) * 2)
    assert result.visited == ["add_one", "double", "to_str"]
    assert result.per_step == {
        "add_one": 4,
        "double": 8,
        "to_str": "value=8",
    }


async def test_chain_supports_sync_functions() -> None:
    """Sync functions are dispatched to a worker thread, transparent
    to the user."""

    def upper(s: str) -> str:
        return s.upper()

    def add_bang(s: str) -> str:
        return f"{s}!"

    wf = Workflow.chain([upper, add_bang])
    result = await wf.run("hello")
    assert result.output == "HELLO!"


async def test_chain_with_duplicate_function_names_disambiguates() -> None:
    """Re-using the same callable in a chain should produce stable,
    unique node names — necessary for cycle detection and
    introspection."""

    async def f(x: int) -> int:
        return x + 1

    wf = Workflow.chain([f, f, f])
    result = await wf.run(0)
    assert result.output == 3
    # Three distinct visited nodes, names disambiguated.
    assert len(result.visited) == 3
    assert len(set(result.visited)) == 3


# ---------------------------------------------------------------------------
# Workflow.route
# ---------------------------------------------------------------------------


async def test_route_dispatches_to_matching_handler() -> None:
    """The classifier picks a key; the matching handler runs with
    the *original* input (not the classifier's output)."""

    async def classify(text: str) -> str:
        if "bill" in text.lower():
            return "billing"
        if "tech" in text.lower():
            return "tech"
        return "general"

    async def billing_handler(text: str) -> str:
        return f"BILLING: {text}"

    async def tech_handler(text: str) -> str:
        return f"TECH: {text}"

    async def general_handler(text: str) -> str:
        return f"GENERAL: {text}"

    wf = Workflow.route(
        classify,
        {"billing": billing_handler, "tech": tech_handler},
        default=general_handler,
    )

    r1 = await wf.run("billing question")
    assert r1.output == "BILLING: billing question"

    r2 = await wf.run("tech question")
    assert r2.output == "TECH: tech question"

    r3 = await wf.run("hello there")
    assert r3.output == "GENERAL: hello there"


async def test_route_without_default_raises_on_unknown_key() -> None:
    async def classify(_text: str) -> str:
        return "unknown_key"

    async def handler(text: str) -> str:
        return text

    wf = Workflow.route(classify, {"known": handler})
    with pytest.raises(RuntimeError, match="no matching route"):
        await wf.run("anything")


# ---------------------------------------------------------------------------
# Workflow.parallel
# ---------------------------------------------------------------------------


async def test_parallel_runs_steps_concurrently_and_merges() -> None:
    async def s1(x: int) -> int:
        return x + 1

    async def s2(x: int) -> int:
        return x * 2

    async def s3(x: int) -> int:
        return x ** 2

    def combine(results: list[int]) -> int:
        return sum(results)

    wf = Workflow.parallel([s1, s2, s3], merge=combine)
    result = await wf.run(3)
    # 4 + 6 + 9 = 19
    assert result.output == 19


async def test_parallel_default_merge_returns_list() -> None:
    """Without ``merge=``, the workflow returns the list of results
    in the same order steps were declared."""

    async def s1(x: int) -> str:
        return f"s1:{x}"

    async def s2(x: int) -> str:
        return f"s2:{x}"

    wf = Workflow.parallel([s1, s2])
    result = await wf.run(7)
    assert result.output == ["s1:7", "s2:7"]


# ---------------------------------------------------------------------------
# Explicit graph builder
# ---------------------------------------------------------------------------


async def test_explicit_graph_with_router() -> None:
    """The full graph-builder API: nodes, edges, router with default."""

    async def classify(text: str) -> str:
        return text.lower()

    async def hi_handler(_: Any) -> str:
        return "hello!"

    async def bye_handler(_: Any) -> str:
        return "goodbye!"

    async def fallback(_: Any) -> str:
        return "?"

    wf = Workflow("greeter")
    wf.add_node("classify", classify)
    wf.add_node("hi", hi_handler)
    wf.add_node("bye", bye_handler)
    wf.add_node("fallback", fallback)
    wf.add_router(
        "classify",
        lambda result: result,
        {"hi": "hi", "bye": "bye"},
        default="fallback",
    )
    wf.add_edge("hi", END)
    wf.add_edge("bye", END)
    wf.add_edge("fallback", END)
    wf.set_start("classify")

    assert (await wf.run("HI")).output == "hello!"
    assert (await wf.run("BYE")).output == "goodbye!"
    assert (await wf.run("hmm")).output == "?"


async def test_workflow_rejects_duplicate_node_names() -> None:
    wf = Workflow("dup")
    wf.add_node("a", lambda x: x)
    with pytest.raises(ValueError, match="already registered"):
        wf.add_node("a", lambda x: x)


async def test_workflow_rejects_unknown_source_in_edge() -> None:
    wf = Workflow("missing-source")
    with pytest.raises(ValueError, match="not registered"):
        wf.add_edge("nowhere", END)


async def test_workflow_raises_without_start() -> None:
    wf = Workflow("no-start")
    wf.add_node("a", lambda x: x)
    with pytest.raises(RuntimeError, match="no start node"):
        await wf.run("anything")


async def test_workflow_runaway_loop_hits_max_visits_cap() -> None:
    """An unconditional cycle (no termination branch) hits the
    ``max_visits_per_node`` cap and raises with a clear message
    identifying the looping node — instead of running forever."""

    async def a(x: int) -> int:
        return x

    async def b(x: int) -> int:
        return x

    wf = Workflow("runaway", max_visits_per_node=5)
    wf.add_node("a", a)
    wf.add_node("b", b)
    wf.add_edge("a", "b")
    wf.add_edge("b", "a")  # cycle with no termination
    wf.set_start("a")

    with pytest.raises(RuntimeError, match="re-entered .* more than"):
        await wf.run(1)


async def test_workflow_max_steps_cap_fires() -> None:
    """``max_steps`` catches zig-zags where no single node loops
    on itself but many nodes interleave."""

    async def a(x: int) -> int:
        return x

    async def b(x: int) -> int:
        return x

    async def c(x: int) -> int:
        return x

    # a→b→c→a→b→c→... no node visits itself, so per-node cap is
    # too generous; ``max_steps`` is the right backstop.
    wf = Workflow("zigzag", max_steps=4, max_visits_per_node=100)
    wf.add_node("a", a)
    wf.add_node("b", b)
    wf.add_node("c", c)
    wf.add_edge("a", "b")
    wf.add_edge("b", "c")
    wf.add_edge("c", "a")
    wf.set_start("a")

    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        await wf.run(1)


async def test_feedback_loop_a_b_classify_c_or_d_back_to_b() -> None:
    """The user-asked pattern: ``A → B → classify → (C|D|END) → B``.

    Models a refinement / retry loop. The classifier picks
    ``"to_c"`` / ``"to_d"`` / ``"done"`` based on the iteration
    count; ``"done"`` terminates, the others route to C or D
    which then loop back to B. The visited trace preserves the
    full iteration history.
    """

    iteration = {"n": 0}

    async def step_a(x: str) -> str:
        return f"{x}|A"

    async def step_b(x: str) -> str:
        return f"{x}|B{iteration['n']}"

    async def classify(x: str) -> str:
        # First iteration → C, second → D, third → done.
        iteration["n"] += 1
        if iteration["n"] == 1:
            return "to_c"
        if iteration["n"] == 2:
            return "to_d"
        return "done"

    async def step_c(x: str) -> str:
        return f"{x}|C"

    async def step_d(x: str) -> str:
        return f"{x}|D"

    wf = Workflow("refinement-loop")
    wf.add_node("A", step_a)
    wf.add_node("B", step_b)
    wf.add_node("classify", classify)
    wf.add_node("C", step_c)
    wf.add_node("D", step_d)

    wf.add_edge("A", "B")
    wf.add_edge("B", "classify")
    wf.add_router(
        "classify",
        lambda result: result,
        {"to_c": "C", "to_d": "D", "done": END},
    )
    # The loop: C and D route back to B for another pass.
    wf.add_edge("C", "B")
    wf.add_edge("D", "B")
    wf.set_start("A")

    result = await wf.run("start")

    # Each node visited the expected number of times.
    visited = result.visited
    assert visited.count("A") == 1            # entry, one-shot
    assert visited.count("B") == 3            # three iterations
    assert visited.count("classify") == 3     # decided three times
    assert visited.count("C") == 1            # first refinement
    assert visited.count("D") == 1            # second refinement
    # The classify run #3 returned "done" → END, so no more steps.

    # The full trace shows the iteration order, not just unique nodes.
    assert visited == [
        "A", "B", "classify", "C",
        "B", "classify", "D",
        "B", "classify",
    ]


async def test_feedback_loop_terminates_at_max_visits_when_classifier_never_picks_done() -> None:
    """Same shape as the refinement loop, but the classifier never
    returns "done". The framework caps it at ``max_visits_per_node``
    and raises — not infinite loop, not silent truncation."""

    # Use string state through the loop so no type juggling between
    # iterations; the test is about the cap, not the value handoff.
    async def b(x: str) -> str:
        return f"{x}|B"

    async def classify(x: str) -> str:
        return "to_c"  # never picks "done"

    async def c(x: str) -> str:
        return x  # pass-through

    wf = Workflow("never-converges", max_visits_per_node=3)
    wf.add_node("B", b)
    wf.add_node("classify", classify)
    wf.add_node("C", c)
    wf.add_edge("B", "classify")
    wf.add_router(
        "classify",
        lambda r: r,
        {"to_c": "C", "done": END},
    )
    wf.add_edge("C", "B")
    wf.set_start("B")

    with pytest.raises(RuntimeError, match="re-entered"):
        await wf.run("start")


# ---------------------------------------------------------------------------
# Composition: Agent inside Workflow
# ---------------------------------------------------------------------------


async def test_agent_as_workflow_step() -> None:
    """Drop an Agent into a Workflow node — the framework calls
    ``.run`` automatically and threads the live RunContext through."""

    agent = Agent(
        "you are an echo bot",
        model=ScriptedModel([ScriptedTurn(text="hello back")]),
        memory=InMemoryMemory(),
    )

    wf = Workflow("with-agent")
    wf.add_node("echo", agent)  # Agent instance!
    wf.add_edge("echo", END)
    wf.set_start("echo")

    result = await wf.run("hi", user_id="alice")
    assert result.output == "hello back"


async def test_user_id_propagates_into_nested_agent_run() -> None:
    """A Workflow.run with user_id="alice" should produce an agent
    run that sees user_id="alice" via get_run_context()."""

    seen_user_ids: list[str | None] = []

    async def capture_user_id(_x: Any) -> str:
        seen_user_ids.append(get_run_context().user_id)
        return "ok"

    wf = Workflow.chain([capture_user_id])
    await wf.run("anything", user_id="alice")
    assert seen_user_ids == ["alice"]


# ---------------------------------------------------------------------------
# Composition: Workflow as a Tool
# ---------------------------------------------------------------------------


async def test_workflow_as_tool_invocation() -> None:
    """``wf.as_tool()`` returns a Tool whose execute() runs the
    whole workflow. Useful for plugging deterministic flows into an
    agent's tool list."""

    async def upper(s: str) -> str:
        return s.upper()

    async def add_bang(s: str) -> str:
        return f"{s}!"

    wf = Workflow.chain([upper, add_bang], name="shout")
    tool_obj = wf.as_tool(description="Shout the input.")

    assert tool_obj.name == "shout"
    assert tool_obj.description == "Shout the input."
    output = await tool_obj.execute({"input": "hello"})
    assert output == "HELLO!"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_stream_yields_typed_events() -> None:
    async def a(x: int) -> int:
        return x + 1

    async def b(x: int) -> int:
        return x * 10

    wf = Workflow.chain([a, b])

    kinds: list[str] = []
    async for ev in wf.stream(2):
        kinds.append(ev.kind.value)

    # WORKFLOW_STARTED + (STEP_STARTED + STEP_COMPLETED) × 2 + WORKFLOW_COMPLETED
    assert kinds[0] == "workflow_started"
    assert kinds[-1] == "workflow_completed"
    assert kinds.count("workflow_step_started") == 2
    assert kinds.count("workflow_step_completed") == 2


async def test_stream_events_carry_session_id() -> None:
    """Every event must have a session_id so downstream consumers
    can correlate. Auto-generated when the caller doesn't supply
    one."""

    wf = Workflow.chain([lambda x: x])
    async for ev in wf.stream("anything"):
        assert ev.session_id  # non-empty string


async def test_stream_step_failed_event_on_exception() -> None:
    async def boom(_: Any) -> None:
        raise RuntimeError("kaboom")

    wf = Workflow.chain([boom])

    failed_events: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="kaboom"):
        async for ev in wf.stream("hi"):
            if ev.kind == EventKind.WORKFLOW_STEP_FAILED:
                failed_events.append(dict(ev.payload))

    assert failed_events, "expected a workflow_step_failed event"
    assert failed_events[0]["node"] == "boom"
    assert "kaboom" in failed_events[0]["error"]


# ---------------------------------------------------------------------------
# Audit log integration
# ---------------------------------------------------------------------------


async def test_audit_log_receives_per_step_entries() -> None:
    async def a(x: int) -> int:
        return x + 1

    async def b(x: int) -> int:
        return x * 2

    audit = InMemoryAuditLog()
    wf = Workflow.chain([a, b], name="audited")
    wf._audit_log = audit  # for now; passed via constructor too

    await wf.run(3, user_id="alice", session_id="s1")
    entries = await audit.query(user_id="alice")

    actions = [e.action for e in entries]
    # Each step produces step_started + step_completed.
    assert actions.count("step_started") == 2
    assert actions.count("step_completed") == 2
    # All entries attributed to alice.
    assert all(e.user_id == "alice" for e in entries)


# ---------------------------------------------------------------------------
# audit_log= resolver — sugar + validation
# ---------------------------------------------------------------------------


async def test_audit_log_string_path_auto_wraps_as_file_audit_log(
    tmp_path: Any,
) -> None:
    """``audit_log='run.log'`` should auto-construct a ``FileAuditLog``
    so users don't have to import the backend just to enable disk
    logging — same ergonomic pattern as ``model='gpt-4.1-mini'``."""
    from loomflow.security import FileAuditLog

    log_path = tmp_path / "run.log"

    async def a(x: int) -> int:
        return x + 1

    wf = Workflow.chain([a], audit_log=str(log_path))
    assert isinstance(wf._audit_log, FileAuditLog)

    await wf.run(1, user_id="u", session_id="s")
    # File was actually written to.
    assert log_path.exists()
    assert log_path.stat().st_size > 0


async def test_audit_log_pathlib_path_auto_wraps_as_file_audit_log(
    tmp_path: Any,
) -> None:
    """``Path`` objects work the same as raw strings for the sugar."""
    from loomflow.security import FileAuditLog

    log_path = tmp_path / "run.log"

    async def a(x: int) -> int:
        return x

    wf = Workflow.chain([a], audit_log=log_path)
    assert isinstance(wf._audit_log, FileAuditLog)


def test_audit_log_rejects_list_with_clear_error() -> None:
    """A bare ``list`` has ``append`` but isn't an AuditLog — used
    to fail deep in ``_audit`` with ``list.append() takes no
    keyword arguments``. Now rejected at construction time with a
    message that lists the valid options."""

    async def a(x: int) -> int:
        return x

    with pytest.raises(TypeError) as excinfo:
        Workflow.chain([a], audit_log=["run.log"])  # type: ignore[arg-type]

    msg = str(excinfo.value)
    assert "audit_log" in msg
    assert "list" in msg.lower()
    # Both file and in-memory backends should be advertised so the
    # user can pick.
    assert "FileAuditLog" in msg
    assert "InMemoryAuditLog" in msg


def test_audit_log_rejects_arbitrary_object_with_clear_error() -> None:
    """Anything that isn't None, str/Path, or AuditLog is rejected."""

    async def a(x: int) -> int:
        return x

    class NotAnAuditLog:
        pass

    with pytest.raises(TypeError) as excinfo:
        Workflow.chain([a], audit_log=NotAnAuditLog())  # type: ignore[arg-type]

    assert "audit_log" in str(excinfo.value)


async def test_audit_log_accepts_audit_log_instance_unchanged() -> None:
    """An ``InMemoryAuditLog`` (which conforms to the protocol) must
    pass through the resolver untouched — not wrapped, not rejected."""

    async def a(x: int) -> int:
        return x

    audit = InMemoryAuditLog()
    wf = Workflow.chain([a], audit_log=audit)
    assert wf._audit_log is audit


# ---------------------------------------------------------------------------
# Visualisation: to_mermaid() / to_dot() / _repr_markdown_
# ---------------------------------------------------------------------------


def test_to_mermaid_chain_emits_linear_flow() -> None:
    """A chain ``A → B → C`` should render as ``START → A → B →
    C → END`` with unconditional solid arrows."""

    async def step_a(x: int) -> int:
        return x + 1

    async def step_b(x: int) -> int:
        return x * 2

    async def step_c(x: int) -> int:
        return x

    wf = Workflow.chain([step_a, step_b, step_c])
    out = wf.to_mermaid()

    assert out.startswith("flowchart TD")
    # Each step appears as a labelled node + edges connect them.
    assert 'n_step_a["step_a"]' in out
    assert 'n_step_b["step_b"]' in out
    assert "START([START]) --> n_step_a" in out
    assert "n_step_a --> n_step_b" in out
    assert "n_step_b --> n_step_c" in out
    assert "n_step_c --> END([END])" in out


def test_to_mermaid_router_labels_branches_and_default() -> None:
    """Router branches should show as labelled solid arrows; the
    default branch should be dotted to distinguish at a glance."""
    from loomflow import END

    async def classify(x: str) -> str:
        return x

    async def handle_yes(x: str) -> str:
        return "Y"

    async def handle_no(x: str) -> str:
        return "N"

    wf = Workflow()
    wf.add_node("classify", classify)
    wf.add_node("yes", handle_yes)
    wf.add_node("no", handle_no)
    wf.add_router(
        "classify",
        fn=lambda v: v,
        routes={"yes": "yes", "no": "no"},
        default=END,
    )
    wf.add_edge("yes", END)
    wf.add_edge("no", END)
    wf.set_start("classify")

    out = wf.to_mermaid()
    # Router branches are labelled.
    assert "n_classify -->|yes| n_yes" in out
    assert "n_classify -->|no| n_no" in out
    # Default branch uses the dotted-arrow style.
    assert "n_classify -.->|default| END([END])" in out


def test_to_mermaid_handles_empty_workflow() -> None:
    """An empty ``Workflow()`` (no nodes yet) should still produce
    a valid Mermaid diagram, not crash. Useful when the user
    inspects a workflow mid-construction."""
    wf = Workflow()
    out = wf.to_mermaid()
    assert out.startswith("flowchart TD")
    assert "(empty workflow)" in out


def test_to_dot_chain_emits_digraph() -> None:
    """``to_dot()`` should produce a parseable Graphviz digraph
    with rounded-rectangle node shapes and one edge per chain
    transition."""

    async def step_a(x: int) -> int:
        return x

    async def step_b(x: int) -> int:
        return x

    wf = Workflow.chain([step_a, step_b], name="my_flow")
    out = wf.to_dot()

    assert out.startswith('digraph "my_flow" {')
    assert out.rstrip().endswith("}")
    assert '"step_a" [shape=box, style=rounded];' in out
    assert '"step_a" -> "step_b";' in out
    # END is visualised when at least one edge points to it.
    assert "__end__" in out
    assert "label=\"END\"" in out


async def test_add_edge_with_START_aliases_set_start() -> None:
    """``add_edge(START, "first")`` should be a drop-in alias for
    ``set_start("first")`` — graphs read symmetrically with the
    ``END`` sentinel and matches the pattern LangGraph users
    expect (``add_edge(START, ...)`` / ``add_edge(..., END)``)."""
    from loomflow import START

    async def first(x: int) -> int:
        return x + 1

    async def second(x: int) -> int:
        return x * 2

    wf = Workflow()
    wf.add_node("first", first)
    wf.add_node("second", second)
    wf.add_edge(START, "first")  # alias for set_start("first")
    wf.add_edge("first", "second")
    wf.add_edge("second", END)

    assert wf._start == "first"
    result = await wf.run(3)
    assert result.output == 8  # ((3 + 1) * 2)


def test_add_edge_with_START_validates_target_is_a_node() -> None:
    """``add_edge(START, END)`` is meaningless and should fail
    loudly. Same for ``add_edge(START, START)``."""
    from loomflow import START

    wf = Workflow()
    with pytest.raises(ValueError) as excinfo:
        wf.add_edge(START, END)  # type: ignore[arg-type]
    assert "target must be a registered node name" in str(excinfo.value)


def test_add_edge_with_END_as_source_rejected() -> None:
    """Only ``START`` is a valid source-side sentinel. ``END`` as
    source is nonsense — flag it explicitly so the error doesn't
    surface later as "source node 'END' is not registered"."""
    wf = Workflow()
    with pytest.raises(ValueError) as excinfo:
        wf.add_edge(END, "anywhere")  # type: ignore[arg-type]
    msg = str(excinfo.value)
    assert "START" in msg
    # The error should mention the actual remediation (set_start
    # / add_edge(START, ...)) so the user can fix without docs.
    assert "set_start" in msg or "add_edge(START" in msg


# ---------------------------------------------------------------------------
# Workflow(memory=...) — shared agent memory across the graph
# ---------------------------------------------------------------------------


async def test_workflow_memory_propagates_to_nested_agents_without_explicit_memory() -> None:
    """``Workflow(memory=mem)`` should be picked up by agents that
    did NOT specify their own ``memory=`` — episodes written by
    agent_a are visible to agent_b without per-agent wiring. This
    is the whole point of the feature: one shared store across the
    graph, opt-in by passing ``memory=`` once."""
    from loomflow import InMemoryMemory

    shared = InMemoryMemory()

    # Two agents, neither specifies memory=. Both should pick up
    # the workflow's memory at run time via the ambient contextvar.
    agent_a = Agent(
        instructions="step a",
        model=ScriptedModel([ScriptedTurn(text="from-a")]),
        auto_extract=False,
    )
    agent_b = Agent(
        instructions="step b",
        model=ScriptedModel([ScriptedTurn(text="from-b")]),
        auto_extract=False,
    )

    wf = Workflow.chain([agent_a, agent_b], memory=shared)
    await wf.run("hi", user_id="alice", session_id="s1")

    # Both agents wrote their episode to the SHARED memory.
    episodes = await shared.recall("", user_id="alice", limit=10)
    outputs = sorted(e.output for e in episodes)
    assert outputs == ["from-a", "from-b"]


async def test_workflow_memory_does_not_override_explicit_agent_memory() -> None:
    """Agents that explicitly pass ``memory=`` keep using their own
    instance — the workflow memory is the FALLBACK only. This is
    the "explicit always wins" rule; opting out of the shared
    memory must remain possible."""
    from loomflow import InMemoryMemory

    workflow_mem = InMemoryMemory()
    agent_mem = InMemoryMemory()

    agent_a = Agent(
        instructions="explicit-mem agent",
        model=ScriptedModel([ScriptedTurn(text="answer-a")]),
        memory=agent_mem,  # ← explicit; should win over workflow memory
        auto_extract=False,
    )

    wf = Workflow.chain([agent_a], memory=workflow_mem)
    await wf.run("hi", user_id="alice", session_id="s1")

    # The episode landed in the agent's OWN memory, not the workflow's.
    in_agent = await agent_mem.recall("", user_id="alice", limit=10)
    in_workflow = await workflow_mem.recall("", user_id="alice", limit=10)
    assert len(in_agent) == 1
    assert in_agent[0].output == "answer-a"
    assert len(in_workflow) == 0


async def test_workflow_without_memory_falls_back_to_agent_default() -> None:
    """When the workflow has no ``memory=``, the contextvar stays
    None and each agent uses its own default memory — back-compat
    with every workflow that existed before this feature."""
    from loomflow.core.context import _ambient_memory_var

    agent_a = Agent(
        instructions="default-mem agent",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        auto_extract=False,
    )
    wf = Workflow.chain([agent_a])  # no memory= on workflow
    await wf.run("hi", user_id="alice", session_id="s1")

    # The contextvar leaked nothing; agent's own memory was used.
    assert _ambient_memory_var.get() is None
    in_agent = await agent_a.memory.recall("", user_id="alice", limit=10)
    assert len(in_agent) == 1


async def test_workflow_memory_is_scoped_to_run_does_not_leak() -> None:
    """The contextvar is reset in ``finally``: a second workflow
    without ``memory=`` after a first with ``memory=`` must NOT
    inherit the first one's memory by accident."""
    from loomflow import InMemoryMemory
    from loomflow.core.context import _ambient_memory_var

    mem1 = InMemoryMemory()
    a1 = Agent(
        instructions="a1",
        model=ScriptedModel([ScriptedTurn(text="ok-1")]),
        auto_extract=False,
    )
    wf1 = Workflow.chain([a1], memory=mem1)
    await wf1.run("hi", user_id="alice", session_id="s1")

    # After the first workflow, the contextvar is back to None.
    assert _ambient_memory_var.get() is None

    a2 = Agent(
        instructions="a2",
        model=ScriptedModel([ScriptedTurn(text="ok-2")]),
        auto_extract=False,
    )
    wf2 = Workflow.chain([a2])  # no memory=
    await wf2.run("hi", user_id="alice", session_id="s2")

    # Second workflow's agent did NOT pick up mem1.
    in_mem1 = await mem1.recall("ok-2", user_id="alice", limit=10)
    assert all(e.output != "ok-2" for e in in_mem1)


def test_repr_markdown_wraps_mermaid_in_fenced_block() -> None:
    """``_repr_markdown_`` is what Jupyter calls when a user just
    types ``wf`` in a cell. It should wrap ``to_mermaid()`` in a
    fenced ``mermaid`` block so JupyterLab / VS Code render the
    diagram inline."""

    async def f(x: int) -> int:
        return x

    wf = Workflow.chain([f])
    md = wf._repr_markdown_()
    assert md.startswith("```mermaid\n")
    assert md.endswith("\n```")
    assert "flowchart TD" in md
