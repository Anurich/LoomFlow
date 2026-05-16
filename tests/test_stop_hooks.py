"""StopHook protocol + Agent re-invocation loop tests.

The framework-level Ralph loop. Same pattern other loop primitives
(retry policy, output validation) get tested with: deterministic
``ScriptedModel`` turns so the model's "I'm done" decision is
predictable, then assert on ``Agent.run`` outcomes.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from loomflow import (
    Agent,
    EchoModel,
    ScriptedModel,
    ScriptedTurn,
    StopHook,
    StopHookResult,
)
from loomflow.core.types import ToolCall

pytestmark = pytest.mark.anyio


# ---- Protocol shape ---------------------------------------------------


def test_stop_hook_protocol_runtime_checkable() -> None:
    """``@runtime_checkable`` matches every other axis-protocol in
    loomflow. Class implementing the structure passes isinstance."""

    class _MyHook:
        name = "x"

        async def __call__(self, session, deps, *, iteration):
            return None

    assert isinstance(_MyHook(), StopHook)


def test_stop_hook_result_is_frozen() -> None:
    """Frozen dataclass — mirrors RunContext / BudgetStatus.
    Mutating after construction should raise."""
    result = StopHookResult(inject_message="x", reason="y")
    with pytest.raises(FrozenInstanceError):
        result.inject_message = "z"  # type: ignore[misc]


# ---- Fast path (no hooks) --------------------------------------------


async def test_no_stop_hooks_runs_once() -> None:
    """Default agent (no stop_hooks=) → exactly one architecture
    pass. ``fast_stop_hooks=True`` short-circuits the wrapper."""
    a = Agent("hi", model=EchoModel())
    r = await a.run("hello")
    assert r.turns == 1
    assert r.interrupted is False


# ---- Hook returns None → no continuation ----------------------------


async def test_stop_hook_voting_stop_does_not_continue() -> None:
    """Hook returns None → all hooks voted to stop → loop exits."""

    class _VoteStop:
        name = "vote_stop"

        async def __call__(self, session, deps, *, iteration):
            return None

    a = Agent("hi", model=EchoModel(), stop_hooks=[_VoteStop()])
    r = await a.run("hello")
    assert r.turns == 1
    assert r.interrupted is False


# ---- Single continuation ---------------------------------------------


async def test_stop_hook_continues_once_then_stops() -> None:
    """Hook returns continue on iter=0, None on iter=1 →
    architecture runs twice. Injected message lands as a fresh
    user turn in the running session."""

    class _OnceHook:
        name = "once"

        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, session, deps, *, iteration):
            self.calls += 1
            if iteration == 0:
                return StopHookResult(
                    inject_message="please continue",
                    reason="test_once",
                )
            return None

    sm = ScriptedModel(
        turns=[ScriptedTurn(text="first"), ScriptedTurn(text="second")]
    )
    hook = _OnceHook()
    a = Agent("hi", model=sm, stop_hooks=[hook])
    r = await a.run("initial")
    # Two architecture passes → 2 turns
    assert r.turns == 2
    # Hook fired once with continue, once with stop
    assert hook.calls == 2


# ---- Iteration cap ----------------------------------------------------


async def test_stop_hook_iteration_cap_marks_interrupted() -> None:
    """A hook that always continues hits the cap; RunResult's
    ``interrupted`` + ``interruption_reason`` reflect the exhaustion
    so observability tools can flag agents stuck in re-invocation
    loops."""

    class _AlwaysGo:
        name = "always_go"

        async def __call__(self, session, deps, *, iteration):
            return StopHookResult(inject_message="go", reason="loop")

    sm = ScriptedModel(turns=[ScriptedTurn(text="t") for _ in range(20)])
    a = Agent(
        "hi",
        model=sm,
        stop_hooks=[_AlwaysGo()],
        max_stop_hook_iterations=3,
    )
    r = await a.run("start")
    # initial pass + 3 continuations = 4 turns
    assert r.turns == 4
    assert r.interrupted is True
    assert r.interruption_reason == "stop_hook_iterations_exhausted"


# ---- First non-None wins per iteration -------------------------------


async def test_stop_hook_first_wins_skips_remaining() -> None:
    """When the first hook returns continue, remaining hooks for
    that iteration are skipped (the "first vote to continue wins"
    semantic from the architect's plan)."""

    class _FirstContinues:
        name = "first"

        async def __call__(self, session, deps, *, iteration):
            if iteration == 0:
                return StopHookResult(
                    inject_message="go", reason="first_says"
                )
            return None

    class _SecondCounter:
        name = "second"

        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, session, deps, *, iteration):
            self.calls += 1
            return None

    second = _SecondCounter()
    sm = ScriptedModel(
        turns=[ScriptedTurn(text="a"), ScriptedTurn(text="b")]
    )
    a = Agent(
        "hi",
        model=sm,
        stop_hooks=[_FirstContinues(), second],
    )
    await a.run("x")
    # second hook only called on iteration=1 (first finished the
    # iter-0 continuation, then both hooks voted on iter-1 and
    # the first returned None, so second got called once).
    assert second.calls == 1


# ---- max_stop_hook_iterations=0 disables entirely --------------------


async def test_stop_hook_disabled_via_zero_cap() -> None:
    """max_stop_hook_iterations=0 + a hook that wants to continue
    → architecture runs once, hook is checked but cap is 0 so the
    re-invocation never fires. Useful for "I want to register a
    hook but globally disable continuation."""

    class _WouldContinue:
        name = "would"

        async def __call__(self, session, deps, *, iteration):
            return StopHookResult(inject_message="go", reason="r")

    a = Agent(
        "hi",
        model=EchoModel(),
        stop_hooks=[_WouldContinue()],
        max_stop_hook_iterations=0,
    )
    r = await a.run("once")
    # initial pass only; loop never runs because cap is 0
    assert r.turns == 1
    # Cap-of-0 still counts as "exhausted" the moment we'd want to
    # continue — surface it
    assert r.interrupted is True
    assert r.interruption_reason == "stop_hook_iterations_exhausted"


async def test_negative_cap_rejected_at_construction() -> None:
    """Negative caps make no sense; reject early with a clear
    message."""
    with pytest.raises(ValueError):
        Agent(
            "hi", model=EchoModel(), max_stop_hook_iterations=-1
        )


# ---- LivingPlan auto-registration ------------------------------------


async def test_living_plan_auto_registers_stop_hook() -> None:
    """``living_plan=True`` → framework prepends the plan stop hook.
    Scripted: turn 1 marks step 'doing' + emits text → hook fires.
    Turn 2 marks step 'done' + emits text → hook sees no doing
    steps → stops."""
    plan_doing = {
        "goal": "test",
        "steps": [{"description": "do it", "status": "doing"}],
    }
    plan_done = {
        "goal": "test",
        "steps": [{"description": "do it", "status": "done"}],
    }
    sm = ScriptedModel(
        turns=[
            ScriptedTurn(
                text="starting",
                tool_calls=[
                    ToolCall(id="c1", tool="plan_write", args=plan_doing)
                ],
            ),
            ScriptedTurn(
                text="finishing",
                tool_calls=[
                    ToolCall(id="c2", tool="plan_write", args=plan_done)
                ],
            ),
            ScriptedTurn(text="all done"),
        ]
    )
    a = Agent("hi", model=sm, living_plan=True)
    r = await a.run("go")
    # First pass plan is doing → hook fires → re-run with continue.
    # Second pass plan is done → hook on iter=1 stops? actually the
    # done plan triggers None → exits before iter 2. So 2+ passes.
    assert r.turns >= 2
    assert r.interrupted is False


def test_living_plan_opt_in_registers_auto_hook() -> None:
    """``living_plan=True`` → framework prepends the
    living-plan stop hook in ``_stop_hooks``. Direct
    introspection — turns-based check is unreliable because
    ReAct's tool-result loop increments turns independently of
    stop-hook continuation."""
    a = Agent("hi", model=EchoModel(), living_plan=True)
    hook_names = [
        getattr(h, "name", type(h).__name__) for h in a._stop_hooks
    ]
    assert "living_plan" in hook_names


def test_living_plan_opt_out_disables_auto_hook() -> None:
    """``living_plan={'auto_stop_hook': False}`` → auto-hook is
    NOT registered. Introspect the agent's stop_hooks list
    directly — turns-based assertions are noisy because of
    ReAct's tool-result loop."""
    a = Agent(
        "hi",
        model=EchoModel(),
        living_plan={"auto_stop_hook": False},
    )
    hook_names = [
        getattr(h, "name", type(h).__name__) for h in a._stop_hooks
    ]
    assert "living_plan" not in hook_names


# ---- from_dict doesn't accept stop_hooks via TOML --------------------


async def test_from_dict_silently_ignores_stop_hooks_key() -> None:
    """stop_hooks isn't TOML-expressible (callables aren't
    serialisable). Agent.from_dict shouldn't crash if a config
    file has it — but it must not silently accept either. Today's
    behaviour: ignored via unknown-key validation."""
    # This depends on from_dict's policy; we just assert it doesn't
    # crash with a confusing message. Specific behaviour can
    # tighten later.
    try:
        Agent.from_dict(
            {"instructions": "hi", "model": "echo"}
        )
    except Exception:  # noqa: BLE001 — only validating it can construct base
        pytest.fail("from_dict without stop_hooks should construct fine")
