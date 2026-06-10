"""run_until= / GoalStopHook — the /goal run-until-done loop.

Two levels, same idioms as ``test_stop_hooks.py``:

* **Normaliser** — pure ``_normalize_run_until_spec`` dispatch +
  error cases (no Agent needed).
* **Agent integration** — a ``ScriptedModel`` worker plus a separate
  ``ScriptedModel`` checker (wired via ``run_until={"checker": ...}``)
  so the DONE / NOT_DONE decision is deterministic, then assert on
  ``Agent.run`` outcomes and the ``run_until.exit`` metadata.

The checker model is asked once per architecture pass; scripting its
turns to emit ``DONE`` / ``NOT_DONE`` drives the loop predictably.
"""

from __future__ import annotations

import pytest

from loomflow import (
    Agent,
    EchoModel,
    GoalStopHook,
    ScriptedModel,
    ScriptedTurn,
)
from loomflow.agent.api import _normalize_run_until_spec
from loomflow.core.errors import ConfigError

pytestmark = pytest.mark.anyio


# ---- Normaliser -------------------------------------------------------


def test_normalize_none_returns_none() -> None:
    assert _normalize_run_until_spec(None) is None


def test_normalize_str_wraps_condition() -> None:
    assert _normalize_run_until_spec("all tests pass") == {
        "condition": "all tests pass"
    }


def test_normalize_str_strips_whitespace() -> None:
    assert _normalize_run_until_spec("  done  ") == {"condition": "done"}


def test_normalize_empty_str_rejected() -> None:
    with pytest.raises(ConfigError):
        _normalize_run_until_spec("   ")


def test_normalize_dict_passes_recognised_keys() -> None:
    spec = _normalize_run_until_spec(
        {
            "condition": "x",
            "checker": "claude-haiku-4-5",
            "max_iterations": 5,
            "max_no_progress": 2,
            "max_cost_usd": 1.5,
        }
    )
    assert spec == {
        "condition": "x",
        "checker": "claude-haiku-4-5",
        "max_iterations": 5,
        "max_no_progress": 2,
        "max_cost_usd": 1.5,
    }


def test_normalize_dict_missing_condition_rejected() -> None:
    with pytest.raises(ConfigError):
        _normalize_run_until_spec({"max_iterations": 5})


def test_normalize_dict_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError):
        _normalize_run_until_spec({"condition": "x", "bogus": 1})


def test_normalize_bad_type_rejected() -> None:
    with pytest.raises(ConfigError):
        _normalize_run_until_spec(123)  # type: ignore[arg-type]


# ---- GoalStopHook construction validation -----------------------------


def test_goal_hook_empty_condition_rejected() -> None:
    with pytest.raises(ValueError):
        GoalStopHook("   ")


def test_goal_hook_bad_max_iterations_rejected() -> None:
    with pytest.raises(ValueError):
        GoalStopHook("x", max_iterations=0)


def test_goal_hook_bad_max_no_progress_rejected() -> None:
    with pytest.raises(ValueError):
        GoalStopHook("x", max_no_progress=0)


def test_goal_hook_bad_cost_rejected() -> None:
    with pytest.raises(ValueError):
        GoalStopHook("x", max_cost_usd=0)


# ---- Auto-registration ------------------------------------------------


def test_run_until_str_registers_hook() -> None:
    a = Agent("hi", model=EchoModel(), run_until="all tests pass")
    names = [getattr(h, "name", type(h).__name__) for h in a._stop_hooks]
    assert "run_until" in names


def test_run_until_none_registers_nothing() -> None:
    a = Agent("hi", model=EchoModel())
    names = [getattr(h, "name", type(h).__name__) for h in a._stop_hooks]
    assert "run_until" not in names


def test_run_until_after_living_plan() -> None:
    """When both are set, living_plan runs first (finish the plan),
    run_until second."""
    a = Agent(
        "hi",
        model=EchoModel(),
        living_plan=True,
        run_until="done",
    )
    names = [getattr(h, "name", type(h).__name__) for h in a._stop_hooks]
    assert names.index("living_plan") < names.index("run_until")


# ---- Agent integration: checker says DONE / NOT_DONE ------------------


async def test_checker_done_stops_first_pass() -> None:
    """Checker says DONE after the first pass → no continuation."""
    worker = ScriptedModel(turns=[ScriptedTurn(text="did the work")])
    checker = ScriptedModel(turns=[ScriptedTurn(text="DONE — looks complete")])
    a = Agent("hi", model=worker, run_until={"condition": "x", "checker": checker})
    r = await a.run("go")
    assert r.turns == 1
    assert r.interrupted is False


async def test_checker_not_done_then_done_continues_once() -> None:
    """NOT_DONE on pass 1 → re-prompt; DONE on pass 2 → stop. Two
    worker passes."""
    worker = ScriptedModel(
        turns=[ScriptedTurn(text="attempt 1"), ScriptedTurn(text="attempt 2")]
    )
    checker = ScriptedModel(
        turns=[
            ScriptedTurn(text="NOT_DONE — missing piece"),
            ScriptedTurn(text="DONE — now complete"),
        ]
    )
    a = Agent("hi", model=worker, run_until={"condition": "x", "checker": checker})
    r = await a.run("go")
    assert r.turns == 2
    assert r.interrupted is False


async def test_not_done_substring_not_treated_as_done() -> None:
    """'NOT_DONE' contains 'DONE' — must NOT be read as done. The
    checker keeps saying NOT_DONE; the loop should re-prompt until a
    guardrail stops it (here max_iterations)."""
    worker = ScriptedModel(turns=[ScriptedTurn(text=f"t{i}") for i in range(10)])
    checker = ScriptedModel(
        turns=[ScriptedTurn(text="NOT_DONE") for _ in range(10)]
    )
    a = Agent(
        "hi",
        model=worker,
        run_until={"condition": "x", "checker": checker, "max_iterations": 3},
    )
    r = await a.run("go")
    # Never falsely stopped on "DONE" substring; ran multiple passes.
    assert r.turns >= 2


# ---- Guardrail: max_iterations ---------------------------------------


async def test_max_iterations_caps_the_loop() -> None:
    """Checker always NOT_DONE → loop stops at max_iterations with the
    run_until.exit reason recorded."""
    worker = ScriptedModel(turns=[ScriptedTurn(text=f"t{i}") for i in range(10)])
    checker = ScriptedModel(
        turns=[ScriptedTurn(text="NOT_DONE") for _ in range(10)]
    )
    a = Agent(
        "hi",
        model=worker,
        run_until={"condition": "x", "checker": checker, "max_iterations": 3},
    )
    r = await a.run("go")
    # initial pass + 2 continuations = 3 passes (iteration+1 >= 3 stops).
    assert r.turns == 3
    # A guardrail cut the loop short before the goal was met → surfaced
    # as an interruption so a /goal UI can tell it apart from success.
    assert r.interrupted is True
    assert r.interruption_reason == "run_until:max_iterations"


# ---- Guardrail: no-progress ------------------------------------------


async def test_no_progress_stops_when_output_unchanged() -> None:
    """Worker emits the SAME output every pass and checker keeps
    saying NOT_DONE → no-progress detection stops the loop before
    max_iterations."""
    worker = ScriptedModel(
        turns=[ScriptedTurn(text="same") for _ in range(10)]
    )
    checker = ScriptedModel(
        turns=[ScriptedTurn(text="NOT_DONE") for _ in range(10)]
    )
    a = Agent(
        "hi",
        model=worker,
        run_until={
            "condition": "x",
            "checker": checker,
            "max_iterations": 20,
            "max_no_progress": 2,
        },
    )
    r = await a.run("go")
    # Stalls trip well before the iteration cap of 20.
    assert r.turns < 20
    # No-progress is a guardrail stop → interrupted with the reason.
    assert r.interrupted is True
    assert r.interruption_reason == "run_until:no_progress"
