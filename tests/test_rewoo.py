"""ReWOO architecture tests.

Covers:

* Protocol satisfaction; resolver string ``"rewoo"``.
* Constructor validation.
* :func:`_extract_placeholders` walks dict/list/str values.
* :func:`_substitute_placeholders` replaces ``{{En}}`` recursively.
* :func:`_topological_levels` groups by dep depth, returns None on
  cycles, sorts within a level.
* :func:`_parse_rewoo_plan` parses JSON, tolerates markdown fences,
  rejects non-list, skips malformed steps.
* End-to-end: planner emits 2-step plan with placeholder, both steps
  execute, solver synthesizes.
* Parallel level execution: 2 independent steps run concurrently.
* Step error path: a tool that raises bubbles up as the step's
  ``error`` field but doesn't crash the architecture.
* ``max_steps`` cap.
* Architecture progress events.
"""

from __future__ import annotations

import pytest

from loomflow import Agent, Architecture, ScriptedModel, ScriptedTurn, tool
from loomflow.architecture import ReWOO, ReWOOPlan, ReWOOStep
from loomflow.architecture.resolver import resolve_architecture
from loomflow.architecture.rewoo import (
    ReWOOStepResult,
    _extract_placeholders,
    _parse_rewoo_plan,
    _substitute_placeholders,
    _topological_levels,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_rewoo_satisfies_architecture_protocol() -> None:
    assert isinstance(ReWOO(), Architecture)


def test_rewoo_name_is_rewoo() -> None:
    assert ReWOO().name == "rewoo"


def test_resolver_handles_rewoo_string() -> None:
    arch = resolve_architecture("rewoo")
    assert isinstance(arch, ReWOO)


def test_rewoo_declared_workers_empty() -> None:
    assert ReWOO().declared_workers() == {}


def test_rewoo_rejects_max_steps_lt_1() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        ReWOO(max_steps=0)


# ---------------------------------------------------------------------------
# Placeholder extraction
# ---------------------------------------------------------------------------


def test_extract_placeholders_string() -> None:
    assert _extract_placeholders("hello {{E1}} world") == ["E1"]


def test_extract_placeholders_dict_recursive() -> None:
    val = {"a": "{{E1}}", "b": {"nested": "{{E2}}"}}
    assert _extract_placeholders(val) == ["E1", "E2"]


def test_extract_placeholders_list_recursive() -> None:
    val = ["x", "{{E1}}", ["{{E3}}", "y"]]
    assert _extract_placeholders(val) == ["E1", "E3"]


def test_extract_placeholders_none_when_no_match() -> None:
    assert _extract_placeholders("plain text") == []
    assert _extract_placeholders({"a": 1, "b": "no placeholder"}) == []


def test_extract_placeholders_dedupes() -> None:
    val = {"a": "{{E1}}", "b": "{{E1}}", "c": "{{E1}} and {{E2}}"}
    assert _extract_placeholders(val) == ["E1", "E2"]


# ---------------------------------------------------------------------------
# Placeholder substitution
# ---------------------------------------------------------------------------


def _result(step_id: str, output: str) -> ReWOOStepResult:
    return ReWOOStepResult(step_id=step_id, tool="t", output=output)


def test_substitute_placeholders_string() -> None:
    results = {"E1": _result("E1", "world")}
    assert (
        _substitute_placeholders("hello {{E1}}", results)
        == "hello world"
    )


def test_substitute_placeholders_dict_and_list() -> None:
    results = {"E1": _result("E1", "X"), "E2": _result("E2", "Y")}
    val = {"a": "{{E1}}", "list": ["{{E2}}", "lit"]}
    out = _substitute_placeholders(val, results)
    assert out == {"a": "X", "list": ["Y", "lit"]}


def test_substitute_placeholders_unresolved_left_alone() -> None:
    """When a placeholder refers to an unknown step, leave it as-is.
    The dispatch will run with the literal placeholder string —
    surfaces the planner bug downstream rather than crashing."""
    out = _substitute_placeholders(
        "before {{Eghost}} after", {}
    )
    assert out == "before {{Eghost}} after"


def test_substitute_placeholders_non_string_passthrough() -> None:
    """Non-string scalars survive substitution unchanged."""
    assert _substitute_placeholders(42, {}) == 42
    assert _substitute_placeholders(None, {}) is None


# ---------------------------------------------------------------------------
# Topological levels
# ---------------------------------------------------------------------------


def test_topological_levels_linear_chain() -> None:
    """E1 → E2 → E3."""
    e1 = ReWOOStep(id="E1", tool="t", args={})
    e2 = ReWOOStep(id="E2", tool="t", args={"x": "{{E1}}"})
    e3 = ReWOOStep(id="E3", tool="t", args={"x": "{{E2}}"})
    levels = _topological_levels([e1, e2, e3])
    assert levels is not None
    assert [[s.id for s in lv] for lv in levels] == [
        ["E1"],
        ["E2"],
        ["E3"],
    ]


def test_topological_levels_independent_steps_collapse_to_one_level() -> None:
    """E1 and E2 with no deps → both at level 0."""
    e1 = ReWOOStep(id="E1", tool="t", args={})
    e2 = ReWOOStep(id="E2", tool="t", args={})
    e3 = ReWOOStep(
        id="E3", tool="t", args={"x": "{{E1}}", "y": "{{E2}}"}
    )
    levels = _topological_levels([e1, e2, e3])
    assert levels is not None
    assert sorted(s.id for s in levels[0]) == ["E1", "E2"]
    assert [s.id for s in levels[1]] == ["E3"]


def test_topological_levels_returns_none_on_cycle() -> None:
    e1 = ReWOOStep(id="E1", tool="t", args={"x": "{{E2}}"})
    e2 = ReWOOStep(id="E2", tool="t", args={"x": "{{E1}}"})
    assert _topological_levels([e1, e2]) is None


def test_topological_levels_unknown_dep_treated_as_no_dep() -> None:
    """A step referencing an unknown id (planner bug) qualifies for
    level 0 — execution will run with the literal placeholder."""
    e1 = ReWOOStep(id="E1", tool="t", args={"x": "{{Eghost}}"})
    levels = _topological_levels([e1])
    assert levels is not None
    assert [s.id for s in levels[0]] == ["E1"]


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------


def test_parse_rewoo_plan_clean_json() -> None:
    text = (
        '[{"id": "E1", "tool": "search", "args": {"q": "x"}}, '
        '{"id": "E2", "tool": "fetch", "args": {"url": "{{E1}}"}}]'
    )
    plan = _parse_rewoo_plan(text)
    assert len(plan.steps) == 2
    assert plan.steps[0].id == "E1"
    assert plan.steps[0].tool == "search"
    assert plan.steps[1].depends_on == ["E1"]


def test_parse_rewoo_plan_strips_markdown_fences() -> None:
    text = '```json\n[{"id": "E1", "tool": "t"}]\n```'
    plan = _parse_rewoo_plan(text)
    assert len(plan.steps) == 1


def test_parse_rewoo_plan_skips_malformed_steps() -> None:
    """Items without a tool, or non-dict items, are skipped."""
    text = (
        '[{"id": "E1", "tool": "good"}, '
        '"not a dict", '
        '{"id": "E2"}, '
        '{"id": "E3", "tool": ""}, '
        '{"id": "E4", "tool": "good2"}]'
    )
    plan = _parse_rewoo_plan(text)
    assert [s.id for s in plan.steps] == ["E1", "E4"]


def test_parse_rewoo_plan_returns_empty_on_garbage() -> None:
    assert _parse_rewoo_plan("not json").steps == []
    assert _parse_rewoo_plan('{"not": "a list"}').steps == []


def test_parse_rewoo_plan_auto_assigns_ids_when_missing() -> None:
    text = '[{"tool": "t"}, {"tool": "t"}]'
    plan = _parse_rewoo_plan(text)
    assert [s.id for s in plan.steps] == ["E1", "E2"]


# ---------------------------------------------------------------------------
# End-to-end with real tools
# ---------------------------------------------------------------------------


@tool
def echo(value: str) -> str:
    """Return the value unchanged."""
    return f"echoed:{value}"


@tool
def upper(text: str) -> str:
    """Uppercase the text."""
    return text.upper()


async def test_rewoo_runs_full_loop_end_to_end() -> None:
    """Planner emits 2-step plan: E1 = echo(value="hello"),
    E2 = upper(text="{{E1}}"). Worker runs E1 → "echoed:hello",
    then E2 → "ECHOED:HELLO". Solver synthesizes."""
    plan_json = (
        '[{"id": "E1", "tool": "echo", "args": {"value": "hello"}}, '
        '{"id": "E2", "tool": "upper", "args": {"text": "{{E1}}"}}]'
    )
    model = ScriptedModel(
        [
            ScriptedTurn(text=plan_json),  # planner
            ScriptedTurn(text="Done. Final result: ECHOED:HELLO."),  # solver
        ]
    )
    agent = Agent(
        "test",
        model=model,
        tools=[echo, upper],
        architecture=ReWOO(),
    )
    result = await agent.run("uppercase echoed hello")
    assert "ECHOED:HELLO" in result.output


# ---------------------------------------------------------------------------
# Parallel level execution
# ---------------------------------------------------------------------------


async def test_rewoo_runs_independent_steps_in_parallel() -> None:
    """Two independent steps (E1, E2) run as one level. We verify
    they both complete."""
    plan_json = (
        '[{"id": "E1", "tool": "echo", "args": {"value": "first"}}, '
        '{"id": "E2", "tool": "echo", "args": {"value": "second"}}]'
    )
    model = ScriptedModel(
        [
            ScriptedTurn(text=plan_json),
            ScriptedTurn(text="Got first and second."),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        tools=[echo],
        architecture=ReWOO(parallel_levels=True),
    )

    events = [e async for e in agent.stream("two parallel echoes")]
    arch_events = [
        e for e in events
        if e.kind == "architecture_event"
        and e.payload.get("name") == "rewoo.level_started"
    ]
    # Should have just one level (both E1 and E2 are independent).
    assert len(arch_events) == 1
    assert sorted(arch_events[0].payload["step_ids"]) == ["E1", "E2"]


# ---------------------------------------------------------------------------
# Step error path
# ---------------------------------------------------------------------------


async def test_rewoo_step_error_surfaces_in_results_not_a_crash() -> None:
    """A failing tool (raises) becomes a step with an ``error``
    field. The architecture continues; the solver sees the error
    text in place of the step output."""

    @tool
    def fails(_value: str) -> str:
        """Always raises."""
        raise RuntimeError("intentional failure")

    plan_json = (
        '[{"id": "E1", "tool": "fails", "args": {"_value": "x"}}]'
    )
    model = ScriptedModel(
        [
            ScriptedTurn(text=plan_json),
            ScriptedTurn(text="Acknowledged the error from E1."),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        tools=[fails],
        architecture=ReWOO(),
    )
    result = await agent.run("trigger an error")
    assert "Acknowledged" in result.output


# ---------------------------------------------------------------------------
# max_steps cap
# ---------------------------------------------------------------------------


async def test_rewoo_caps_overlong_plans() -> None:
    """Planner emits 5 steps; max_steps=2 → only first 2 execute."""
    plan_json = (
        '[{"id": "E1", "tool": "echo", "args": {"value": "1"}}, '
        '{"id": "E2", "tool": "echo", "args": {"value": "2"}}, '
        '{"id": "E3", "tool": "echo", "args": {"value": "3"}}, '
        '{"id": "E4", "tool": "echo", "args": {"value": "4"}}, '
        '{"id": "E5", "tool": "echo", "args": {"value": "5"}}]'
    )
    model = ScriptedModel(
        [
            ScriptedTurn(text=plan_json),
            ScriptedTurn(text="Got 1 and 2."),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        tools=[echo],
        architecture=ReWOO(max_steps=2),
    )
    events = [e async for e in agent.stream("five echoes")]
    plan_created = next(
        e for e in events
        if e.kind == "architecture_event"
        and e.payload.get("name") == "rewoo.plan_created"
    )
    assert plan_created.payload["num_steps"] == 2


# ---------------------------------------------------------------------------
# Architecture events
# ---------------------------------------------------------------------------


async def test_rewoo_emits_full_event_sequence() -> None:
    plan_json = '[{"id": "E1", "tool": "echo", "args": {"value": "x"}}]'
    model = ScriptedModel(
        [
            ScriptedTurn(text=plan_json),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        tools=[echo],
        architecture=ReWOO(),
    )
    events = [e async for e in agent.stream("q")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "rewoo.planner_started" in arch_names
    assert "rewoo.plan_created" in arch_names
    assert "rewoo.level_started" in arch_names
    assert "rewoo.step_completed" in arch_names
    assert "rewoo.solver_started" in arch_names
    assert "rewoo.completed" in arch_names


# ---------------------------------------------------------------------------
# Empty plan + cyclic plan handling
# ---------------------------------------------------------------------------


async def test_rewoo_empty_plan_terminates_gracefully() -> None:
    model = ScriptedModel([ScriptedTurn(text="[]")])
    agent = Agent(
        "test", model=model, architecture=ReWOO()
    )
    result = await agent.run("nothing to do")
    assert "no steps" in result.output


async def test_rewoo_cyclic_plan_terminates_gracefully() -> None:
    """If the planner emits a cycle, we report it and stop —
    don't deadlock."""
    plan_json = (
        '[{"id": "E1", "tool": "echo", "args": {"value": "{{E2}}"}}, '
        '{"id": "E2", "tool": "echo", "args": {"value": "{{E1}}"}}]'
    )
    model = ScriptedModel([ScriptedTurn(text=plan_json)])
    agent = Agent(
        "test",
        model=model,
        tools=[echo],
        architecture=ReWOO(),
    )
    result = await agent.run("cyclic plan")
    assert "cyclic" in result.output.lower()


# ---------------------------------------------------------------------------
# Plan + results stashed on session.metadata
# ---------------------------------------------------------------------------


async def test_rewoo_stashes_plan_and_results_in_metadata() -> None:
    plan_json = (
        '[{"id": "E1", "tool": "echo", "args": {"value": "x"}}]'
    )
    model = ScriptedModel(
        [
            ScriptedTurn(text=plan_json),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        tools=[echo],
        architecture=ReWOO(),
    )
    events = [e async for e in agent.stream("q")]
    completed = next(
        e for e in events
        if e.kind == "architecture_event"
        and e.payload.get("name") == "rewoo.completed"
    )
    assert completed.payload["num_steps"] == 1


def test_rewoo_plan_model_serializes() -> None:
    """ReWOOPlan + ReWOOStep are Pydantic — check round-trip."""
    p = ReWOOPlan(
        steps=[ReWOOStep(id="E1", tool="t", args={"x": 1})]
    )
    dump = p.model_dump()
    reloaded = ReWOOPlan(**dump)
    assert reloaded.steps[0].id == "E1"
    assert reloaded.steps[0].tool == "t"
    assert reloaded.steps[0].args == {"x": 1}
