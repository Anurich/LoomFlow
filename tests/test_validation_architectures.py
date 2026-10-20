"""Regression tests for the architecture validation findings.

Each test reproduces a finding from the validation pass and fails on
the code before its fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn, tool
from loomflow.architecture import MultiAgentDebate
from loomflow.core.types import EventKind, ToolCall, Usage
from loomflow.governance.budget import BudgetConfig, StandardBudget
from loomflow.runtime import SqliteRuntime

pytestmark = pytest.mark.anyio


class _Slow(ScriptedModel):
    def __init__(self, turns: list[ScriptedTurn], delay: float) -> None:
        super().__init__(turns)
        self.delay = delay

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        await anyio.sleep(self.delay)
        return await super().complete(*args, **kwargs)

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        await anyio.sleep(self.delay)
        async for chunk in super().stream(*args, **kwargs):
            yield chunk


def _debater(text: str, delay: float = 0.0, usage: Usage | None = None) -> Agent:
    turn = ScriptedTurn(text=text, usage=usage or Usage())
    return Agent("d", model=_Slow([turn] * 4, delay))


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delays", [(0.0, 0.05, 0.1), (0.1, 0.0, 0.05)])
async def test_debate_vote_counts_final_answers_not_completion_order(
    delays: tuple[float, float, float],
) -> None:
    debaters = [
        _debater("I think it is 6+1. Final answer: 7", delays[0]),
        _debater("6 times 7. Final answer: 42", delays[1]),
        _debater("Multiplying gives it. **Final answer:** 42.", delays[2]),
    ]
    host = Agent(
        "h",
        model="echo",
        architecture=MultiAgentDebate(debaters=debaters, rounds=1, convergence_check=False),
    )
    assert (await host.run("What is 6*7?")).output == "6 times 7. Final answer: 42"


async def test_debate_similar_arguments_for_different_answers_do_not_converge() -> None:
    reasoning = (
        "We start by noting the problem asks for the product of six and seven so we "
        "multiply the two numbers carefully step by step and double check the arithmetic."
    )
    judge = Agent("j", model=ScriptedModel([ScriptedTurn(text="judged")]))
    host = Agent(
        "h",
        model="echo",
        architecture=MultiAgentDebate(
            debaters=[_debater(f"{reasoning} Final answer: 42"), _debater(f"{reasoning} Final answer: 48")],
            judge=judge,
            rounds=1,
        ),
    )
    names = [
        ev.payload.get("name")
        async for ev in host.stream("6*7?")
        if ev.kind == EventKind.ARCHITECTURE_EVENT
    ]
    assert "debate.converged" not in names
    assert "debate.judging" in names


class _Boom(ScriptedModel):
    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("provider 500")

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("provider 500")
        yield  # pragma: no cover


async def test_debate_goes_on_without_a_failing_debater() -> None:
    host = Agent(
        "h",
        model="echo",
        architecture=MultiAgentDebate(
            debaters=[Agent("bad", model=_Boom([])), _debater("Final answer: fine")], rounds=1
        ),
    )
    names: list[Any] = []
    async for ev in host.stream("q"):
        if ev.kind == EventKind.ARCHITECTURE_EVENT:
            names.append(ev.payload.get("name"))
        if ev.kind == EventKind.COMPLETED:
            assert ev.payload["result"]["output"] == "Final answer: fine"
    assert "debate.debater_failed" in names


async def test_debate_fails_plainly_when_no_debater_answers() -> None:
    host = Agent(
        "h",
        model="echo",
        architecture=MultiAgentDebate(
            debaters=[Agent("a", model=_Boom([])), Agent("b", model=_Boom([]))], rounds=1
        ),
    )
    with pytest.raises(RuntimeError, match="no debater answered"):
        await host.run("q")


async def test_debate_interrupted_debater_is_not_counted() -> None:
    @tool
    async def noop() -> str:
        """Do nothing."""
        return "ok"

    looping = ScriptedModel(
        [ScriptedTurn(tool_calls=[ToolCall(tool="noop", args={})]) for _ in range(6)]
    )
    stuck = Agent("stuck", model=looping, tools=[noop], max_turns=1)
    host = Agent(
        "h",
        model="echo",
        architecture=MultiAgentDebate(
            debaters=[stuck, _debater("Final answer: 4"), _debater("Final answer: 5")],
            rounds=1,
            convergence_check=False,
        ),
    )
    assert (await host.run("q")).output == "Final answer: 4"


async def test_debate_charges_debater_spend_to_the_coordinator_budget() -> None:
    big = Usage(input_tokens=5000, output_tokens=5000)
    budget = StandardBudget(BudgetConfig(max_tokens=1000))
    host = Agent(
        "h",
        model="echo",
        architecture=MultiAgentDebate(
            debaters=[_debater("Final answer: a", usage=big), _debater("Final answer: b", usage=big)],
            rounds=3,
            convergence_check=False,
        ),
        budget=budget,
    )
    result = await host.run("q")
    assert result.interrupted and result.interruption_reason == "budget:max_tokens"
    assert budget.usage_for(None)["tokens_total"] >= 20_000


async def test_resumed_debate_does_not_repeat_debater_side_effects(tmp_path: Path) -> None:
    charges: list[int] = []

    @tool
    async def charge(n: int) -> str:
        """Charge."""
        charges.append(n)
        return "charged"

    def debaters(runtime: SqliteRuntime) -> list[Agent]:
        return [
            Agent(
                "d",
                model=ScriptedModel(
                    [
                        ScriptedTurn(tool_calls=[ToolCall(tool="charge", args={"n": i})]),
                        ScriptedTurn(text=f"Final answer: {i}"),
                    ]
                    * 2  # round 0 and round 1
                ),
                tools=[charge],
                runtime=runtime,
            )
            for i in range(2)
        ]

    def host(runtime: SqliteRuntime, judge_model: ScriptedModel) -> Agent:
        return Agent(
            "h",
            model="echo",
            runtime=runtime,
            architecture=MultiAgentDebate(
                debaters=debaters(runtime),
                judge=Agent("j", model=judge_model, runtime=runtime),
                rounds=1,
                convergence_check=False,
            ),
        )

    db = tmp_path / "j.db"
    with pytest.raises(RuntimeError):
        await host(SqliteRuntime(db), _Boom([])).run("go", session_id="debate")
    first = list(charges)
    assert sorted(first) == [0, 0, 1, 1]  # round 0 and round 1

    out = await host(SqliteRuntime(db), ScriptedModel([ScriptedTurn(text="verdict")])).run(
        "go", session_id="debate"
    )
    assert out.output == "verdict"
    assert charges == first  # nothing charged twice
