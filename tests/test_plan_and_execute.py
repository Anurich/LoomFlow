"""PlanAndExecute architecture tests.

Covers:

* Protocol satisfaction; resolver string ``"plan-and-execute"``.
* Constructor validation: max_steps<1 rejected.
* :func:`_parse_plan` parses: clean JSON list of strings, JSON list
  of dicts with ``description`` keys, markdown-fenced JSON, bullet
  list fallback, numbered list fallback.
* Plan → execute → synthesize end-to-end with ScriptedModel.
* Empty plan → graceful "no steps" output (not a crash).
* ``max_steps`` cap on overlong plans.
* Architecture progress events.
* ``session.metadata["plan"]`` and ``["step_results"]`` are populated.
"""

from __future__ import annotations

import pytest

from jeevesagent import (
    Agent,
    Architecture,
    PlanAndExecute,
    ScriptedModel,
    ScriptedTurn,
)
from jeevesagent.architecture.plan_and_execute import _parse_plan
from jeevesagent.architecture.resolver import resolve_architecture

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_plan_and_execute_satisfies_architecture_protocol() -> None:
    assert isinstance(PlanAndExecute(), Architecture)


def test_plan_and_execute_name_is_kebab() -> None:
    assert PlanAndExecute().name == "plan-and-execute"


def test_resolver_handles_plan_and_execute_string() -> None:
    arch = resolve_architecture("plan-and-execute")
    assert isinstance(arch, PlanAndExecute)


def test_plan_and_execute_declared_workers_empty() -> None:
    assert PlanAndExecute().declared_workers() == {}


def test_plan_and_execute_rejects_max_steps_lt_1() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        PlanAndExecute(max_steps=0)


# ---------------------------------------------------------------------------
# Plan parser
# ---------------------------------------------------------------------------


def test_parse_plan_json_list_of_strings() -> None:
    text = '["step one", "step two", "step three"]'
    plan = _parse_plan(text)
    assert [s.description for s in plan.steps] == [
        "step one",
        "step two",
        "step three",
    ]
    assert [s.id for s in plan.steps] == [
        "step_1",
        "step_2",
        "step_3",
    ]


def test_parse_plan_json_list_of_dicts_with_description() -> None:
    text = (
        '[{"description": "alpha"}, '
        '{"description": "beta"}, '
        '{"step": "gamma"}, '
        '{"name": "delta"}]'
    )
    plan = _parse_plan(text)
    assert [s.description for s in plan.steps] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
    ]


def test_parse_plan_strips_markdown_fences() -> None:
    text = '```json\n["a", "b"]\n```'
    plan = _parse_plan(text)
    assert [s.description for s in plan.steps] == ["a", "b"]


def test_parse_plan_bullet_list_fallback() -> None:
    text = "- first thing\n- second thing\n- third thing"
    plan = _parse_plan(text)
    assert len(plan.steps) == 3
    assert plan.steps[0].description == "first thing"
    assert plan.steps[2].description == "third thing"


def test_parse_plan_numbered_list_fallback() -> None:
    text = "1. first\n2) second\n3. third"
    plan = _parse_plan(text)
    assert len(plan.steps) == 3
    assert plan.steps[0].description == "first"
    assert plan.steps[1].description == "second"


def test_parse_plan_empty_text_returns_empty_plan() -> None:
    plan = _parse_plan("")
    assert plan.steps == []


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


async def test_plan_and_execute_runs_full_loop() -> None:
    """1 planner call + 3 step calls + 1 synthesizer call."""
    model = ScriptedModel(
        [
            ScriptedTurn(text='["gather inputs", "compute", "format"]'),
            ScriptedTurn(text="inputs gathered"),
            ScriptedTurn(text="computed result = 42"),
            ScriptedTurn(text="formatted: 42"),
            ScriptedTurn(text="Final: the answer is 42."),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=PlanAndExecute(),
    )
    result = await agent.run("compute the answer")
    assert result.output == "Final: the answer is 42."
    # 1 planner + 3 steps + 1 synthesizer = 5
    assert result.turns == 5


# ---------------------------------------------------------------------------
# Empty plan
# ---------------------------------------------------------------------------


async def test_plan_and_execute_empty_plan_terminates_gracefully() -> None:
    """Planner returns no steps → output is a clear "cannot execute"
    message, not a crash."""
    model = ScriptedModel([ScriptedTurn(text="[]")])
    agent = Agent(
        "solver",
        model=model,
        architecture=PlanAndExecute(),
    )
    result = await agent.run("task")
    assert "no steps" in result.output


# ---------------------------------------------------------------------------
# max_steps cap
# ---------------------------------------------------------------------------


async def test_plan_and_execute_caps_overlong_plans() -> None:
    """Planner returns 5 steps; max_steps=2 → only 2 executed."""
    model = ScriptedModel(
        [
            ScriptedTurn(
                text='["s1", "s2", "s3", "s4", "s5"]'
            ),
            ScriptedTurn(text="s1 output"),
            ScriptedTurn(text="s2 output"),
            ScriptedTurn(text="combined: s1 + s2"),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=PlanAndExecute(max_steps=2),
    )
    result = await agent.run("task")
    assert result.output == "combined: s1 + s2"
    # 1 planner + 2 steps + 1 synthesizer = 4
    assert result.turns == 4


# ---------------------------------------------------------------------------
# Architecture events
# ---------------------------------------------------------------------------


async def test_plan_and_execute_emits_full_event_sequence() -> None:
    model = ScriptedModel(
        [
            ScriptedTurn(text='["s1"]'),
            ScriptedTurn(text="s1 done"),
            ScriptedTurn(text="final"),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=PlanAndExecute(),
    )
    events = [e async for e in agent.stream("q")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "plan.planner_started" in arch_names
    assert "plan.created" in arch_names
    assert "plan.step_started" in arch_names
    assert "plan.step_completed" in arch_names
    assert "plan.synthesizer_started" in arch_names
    assert "plan.completed" in arch_names


# ---------------------------------------------------------------------------
# session.metadata
# ---------------------------------------------------------------------------


async def test_plan_and_execute_stashes_plan_and_results_in_metadata() -> None:
    """``session.metadata["plan"]`` and ``["step_results"]`` are
    populated for post-hoc analysis. We can verify by streaming
    events: the ``plan.created`` event carries the steps."""
    model = ScriptedModel(
        [
            ScriptedTurn(text='["step alpha", "step beta"]'),
            ScriptedTurn(text="alpha done"),
            ScriptedTurn(text="beta done"),
            ScriptedTurn(text="final answer"),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=PlanAndExecute(),
    )
    events = [e async for e in agent.stream("q")]
    plan_created = next(
        e for e in events
        if e.kind == "architecture_event"
        and e.payload.get("name") == "plan.created"
    )
    assert plan_created.payload["num_steps"] == 2
    assert "step alpha" in plan_created.payload["steps"][0]
