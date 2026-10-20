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


class _Spy(ScriptedModel):
    """Records the last message of every model call."""

    def __init__(self, seen: list[str], turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.seen = seen

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.seen.append(str(messages[-1].content))
        return await super().complete(messages, **kwargs)

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        self.seen.append(str(messages[-1].content))
        async for chunk in super().stream(messages, **kwargs):
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


# ---------------------------------------------------------------------------
# Shared parsing: JSON in prose, scores on other scales
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        'Here is my review:\n```json\n{"score": 0.9}\n```',
        '{"score": 0.9}\nLet me know if you need more.',
        'Sure. {"score": 0.9} Hope that helps.',
    ],
)
def test_parse_fenced_json_finds_json_in_prose(text: str) -> None:
    from loomflow.architecture.helpers import parse_fenced_json

    assert parse_fenced_json(text) == {"score": 0.9}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Score: 7/10", 0.7),
        ("score: 3/10\nwrong direction", 0.3),
        ("Score: 40%", 0.4),
        ("**Score:** 0.3", 0.3),
        ("Score (0-1): 0.3", 0.3),
        ("score: -0.5", 0.0),
        ("8/10", 0.8),
        ("Rating: 2 out of 10, nothing works", 0.2),
        ("score: 7", 0.7),
        ("score: 1.5", 1.0),
        ('{"score": 0.4}', 0.4),
        ("0 errors found", 0.0),
    ],
)
def test_parse_score_reads_the_scale(text: str, expected: float) -> None:
    from loomflow.architecture.helpers import parse_score

    assert parse_score(text) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ActorCritic
# ---------------------------------------------------------------------------


async def test_actor_critic_approves_json_wrapped_in_prose() -> None:
    from loomflow.architecture import ActorCritic

    actor = Agent("a", model=ScriptedModel([ScriptedTurn(text="draft")]))
    critic = Agent(
        "c",
        model=ScriptedModel(
            [ScriptedTurn(text='Here is my review:\n```json\n{"issues": [], "score": 0.95}\n```')]
        ),
    )
    host = Agent("h", model="echo", architecture=ActorCritic(actor=actor, critic=critic, max_rounds=2))
    names = [
        ev.payload.get("name")
        async for ev in host.stream("task")
        if ev.kind == EventKind.ARCHITECTURE_EVENT
    ]
    assert "actor_critic.approved" in names


async def test_actor_critic_charges_worker_spend_to_the_coordinator_budget() -> None:
    from loomflow.architecture import ActorCritic

    use = Usage(input_tokens=100, output_tokens=100)
    actor = Agent("a", model=ScriptedModel([ScriptedTurn(text=f"v{i}", usage=use) for i in range(3)]))
    critic = Agent(
        "c", model=ScriptedModel([ScriptedTurn(text='{"issues": ["x"], "score": 0.1}', usage=use)] * 3)
    )
    budget = StandardBudget(BudgetConfig(max_tokens=50))
    host = Agent(
        "h", model="echo", budget=budget,
        architecture=ActorCritic(actor=actor, critic=critic, max_rounds=3),
    )
    result = await host.run("task")
    assert result.interrupted and result.interruption_reason == "budget:max_tokens"


# ---------------------------------------------------------------------------
# Tree of Thoughts
# ---------------------------------------------------------------------------


async def test_tot_winner_comes_from_the_deepest_level() -> None:
    from loomflow.architecture import TreeOfThoughts

    seen: list[str] = []

    model = _Spy(
        seen,
        [
            ScriptedTurn(text="1. step A\n2. step B"),
            ScriptedTurn(text="score: 0.9"),
            ScriptedTurn(text="score: 0.1"),
            ScriptedTurn(text="1. A then A1\n2. A then A2"),
            ScriptedTurn(text="score: 0.8"),
            ScriptedTurn(text="score: 0.7"),
            ScriptedTurn(text="FINAL"),
        ]
    )
    agent = Agent(
        "x", model=model,
        architecture=TreeOfThoughts(branch_factor=2, beam_width=1, max_depth=2, parallel=False),
    )
    await agent.run("solve it")
    assert "Step 2: A then A1" in seen[-1]


async def test_tot_resume_replays_every_call(tmp_path: Path) -> None:
    from loomflow.architecture import TreeOfThoughts

    script = [
        ScriptedTurn(text="1. A\n2. B"), ScriptedTurn(text="score: 0.6"),
        ScriptedTurn(text="score: 0.4"), ScriptedTurn(text="FINAL"),
    ]
    model = ScriptedModel(script * 2)
    runtime = SqliteRuntime(tmp_path / "j.db")

    def agent() -> Agent:
        arch = TreeOfThoughts(branch_factor=2, beam_width=1, max_depth=1, parallel=False)
        return Agent("x", model=model, runtime=runtime, architecture=arch)

    await agent().run("p", session_id="s")
    before = model.remaining
    await agent().resume("p", session_id="s")
    assert model.remaining == before


# ---------------------------------------------------------------------------
# ReWOO
# ---------------------------------------------------------------------------


def _rewoo_tools(calls: list[tuple[str, str]]) -> list[Any]:
    @tool
    async def fetch(url: str) -> str:
        """Fetch a URL."""
        calls.append(("fetch", url))
        if url == "bad":
            raise RuntimeError("404 not found")
        return "X" * 300_000 if url == "big" else f"page<{url}>"

    @tool
    async def summarize(text: str) -> str:
        """Summarize text."""
        calls.append(("summarize", text[:40]))
        return f"summary({len(text)} chars)"

    return [fetch, summarize]


async def test_rewoo_skips_steps_whose_inputs_failed_or_do_not_exist() -> None:
    import json

    from loomflow.architecture import ReWOO

    plan = [
        {"id": "E1", "tool": "fetch", "args": {"url": "bad"}},
        {"id": "E2", "tool": "summarize", "args": {"text": "{{E1}}"}},
        {"id": "E3", "tool": "summarize", "args": {"text": "see {{E9}}"}},
        {"id": "E4", "tool": "fetch", "args": {"url": "first"}},
        {"id": "E4", "tool": "fetch", "args": {"url": "second"}},
        {"id": "E5", "tool": "fetch", "args": {"url": "big"}},
    ]
    seen: list[str] = []

    calls: list[tuple[str, str]] = []
    agent = Agent(
        "x",
        model=_Spy(seen, [ScriptedTurn(text=f"Plan:\n```json\n{json.dumps(plan)}\n```"), ScriptedTurn(text="ANSWER")]),
        tools=_rewoo_tools(calls),
        architecture=ReWOO(),
    )
    kinds = [ev.kind async for ev in agent.stream("task")]
    assert sorted(calls) == [("fetch", "bad"), ("fetch", "big"), ("fetch", "first"), ("fetch", "second")]
    assert kinds.count(EventKind.TOOL_CALL) == 4 and kinds.count(EventKind.TOOL_RESULT) == 4
    solver = seen[-1]
    assert "skipped: input step(s) E1 failed" in solver
    assert "skipped: references unknown step(s) E9" in solver
    assert 'fetch({"url": "big"})' in solver
    assert len(solver) < 60_000  # the 300k output is capped


def test_rewoo_reads_plans_in_other_shapes() -> None:
    from loomflow.architecture.rewoo import _parse_rewoo_plan

    wrapped = _parse_rewoo_plan('{"steps": [{"id": "E1", "tool": "fetch", "args": {"url": "a"}}]}')
    assert [s.id for s in wrapped.steps] == ["E1"]
    paper = _parse_rewoo_plan('[{"id": "E2", "tool": "fetch", "args": {"url": "#E1"}}]')
    assert paper.steps[0].depends_on == ["E1"]


async def test_rewoo_custom_planner_prompt_with_json_braces() -> None:
    from loomflow.architecture import ReWOO

    agent = Agent(
        "x",
        model=ScriptedModel([ScriptedTurn(text="[]")]),
        tools=_rewoo_tools([]),
        architecture=ReWOO(planner_prompt='Steps look like {"id": "E1"}. Tools: {tool_descriptions}'),
    )
    assert (await agent.run("t")).output.startswith("Planner produced no steps")


@pytest.mark.parametrize("arch_name", ["ReWOO", "PlanAndExecute"])
async def test_planner_call_is_budget_gated(arch_name: str) -> None:
    import loomflow.architecture as archs

    budget = StandardBudget(BudgetConfig(max_tokens=10))
    await budget.consume(tokens_in=100, tokens_out=0, cost_usd=0.0)
    model = ScriptedModel([ScriptedTurn(text='["a"]'), ScriptedTurn(text="x"), ScriptedTurn(text="y")])
    agent = Agent("x", model=model, tools=_rewoo_tools([]), budget=budget, architecture=getattr(archs, arch_name)())
    result = await agent.run("t")
    assert result.interrupted and model.remaining == 3


# ---------------------------------------------------------------------------
# Plan-and-Execute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"steps": ["Search docs", "Write answer"]}',
        "Here is the plan:\n1. Search docs\n2. Write answer",
        'Sure! Plan below.\n```json\n["Search docs", "Write answer"]\n```',
        '["Search docs", "", "Write answer"]',
    ],
)
def test_plan_and_execute_parses_plans(text: str) -> None:
    from loomflow.architecture.plan_and_execute import _parse_plan

    plan = _parse_plan(text)
    assert [(s.id, s.description) for s in plan.steps] == [
        ("step_1", "Search docs"),
        ("step_2", "Write answer"),
    ]


# ---------------------------------------------------------------------------
# Self-Refine
# ---------------------------------------------------------------------------


async def test_self_refine_stop_phrase_must_be_the_critique() -> None:
    from loomflow.architecture import SelfRefine

    model = ScriptedModel(
        [
            ScriptedTurn(text="draft with an error"),
            ScriptedTurn(text="- Grammar: no issues.\n- Fact: the date is wrong."),
            ScriptedTurn(text="fixed draft"),
            ScriptedTurn(text="No issues."),
        ]
    )
    result = await Agent("x", model=model, architecture=SelfRefine(max_rounds=3)).run("write")
    assert result.output == "fixed draft"
    assert model.remaining == 0


async def test_self_refine_refiner_sees_the_whole_history() -> None:
    from loomflow.architecture import SelfRefine

    seen: list[str] = []

    model = _Spy(
        seen,
        [
            ScriptedTurn(text="y0"), ScriptedTurn(text="fb0: issue X"), ScriptedTurn(text="y1"),
            ScriptedTurn(text="fb1: issue Y"), ScriptedTurn(text="y2"),
        ]
    )
    await Agent("x", model=model, architecture=SelfRefine(max_rounds=2)).run("write")
    assert "y0" in seen[-1] and "fb0" in seen[-1] and "fb1" in seen[-1]


# ---------------------------------------------------------------------------
# Role models and secrets
# ---------------------------------------------------------------------------


def test_role_models_use_the_agent_secrets() -> None:
    pytest.importorskip("anthropic")
    from loomflow.architecture import TreeOfThoughts
    from loomflow.security import DictSecrets

    arch = TreeOfThoughts(evaluator_model="claude-haiku-4-5")
    Agent(
        "x",
        model="claude-haiku-4-5",
        secrets=DictSecrets({"ANTHROPIC_API_KEY": "sk-ant-from-secrets"}),
        architecture=arch,
    )
    assert arch._evaluator_model._client.api_key == "sk-ant-from-secrets"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Reflexion
# ---------------------------------------------------------------------------


async def test_reflexion_retries_an_attempt_that_ran_out_of_turns() -> None:
    from loomflow.architecture import ReAct, Reflexion

    @tool
    async def look(x: str) -> str:
        """Look around."""
        return "nothing here"

    stuck = [ScriptedTurn(tool_calls=[ToolCall(tool="look", args={"x": "room"})]) for _ in range(2)]
    seen: list[str] = []

    model = _Spy(
        seen,
        stuck
        + [
            ScriptedTurn(text="Stop looking in the room; try the kitchen."),
            ScriptedTurn(text="found it in the kitchen"),
            ScriptedTurn(text="score: 0.9"),
        ]
    )
    agent = Agent(
        "x", model=model, tools=[look],
        architecture=Reflexion(base=ReAct(max_turns=2), max_attempts=2),
    )
    result = await agent.run("find the key")
    assert result.output == "found it in the kitchen" and not result.interrupted
    reflector_prompt = seen[2]
    assert "Action: look" in reflector_prompt and "nothing here" in reflector_prompt


async def test_reflexion_lesson_block_is_bounded() -> None:
    from loomflow.architecture import Reflexion

    turns: list[ScriptedTurn] = []
    for i in range(4):
        turns += [ScriptedTurn(text=f"answer {i}"), ScriptedTurn(text="score: 0.1"), ScriptedTurn(text=f"lesson {i}")]
    agent = Agent("x", model=ScriptedModel(turns), architecture=Reflexion(max_attempts=5, max_lessons=2))
    await agent.run("task")
    blocks = await agent.memory.working()
    lessons = next(b.content for b in blocks if b.name == "reflexion_lessons")
    assert lessons.splitlines() == ["- lesson 2", "- lesson 3"]


# ---------------------------------------------------------------------------
# Persistent workers survive a restart
# ---------------------------------------------------------------------------


async def test_router_with_persistent_workers_resumes_after_a_restart(tmp_path: Path) -> None:
    from loomflow.architecture import RouterRoute
    from loomflow.core.types import Event
    from loomflow.team import Team

    charges: list[int] = []

    @tool
    async def charge_card(amount: int) -> str:
        """Charge a card."""
        charges.append(amount)
        return "charged"

    def team(runtime: SqliteRuntime) -> Agent:
        specialist = Agent(
            "billing",
            model=ScriptedModel(
                [
                    ScriptedTurn(tool_calls=[ToolCall(tool="charge_card", args={"amount": 5})]),
                    ScriptedTurn(text="refunded"),
                ]
            ),
            tools=[charge_card],
            runtime=runtime,
        )
        return Team.router(
            [RouterRoute("billing", specialist, "money")],
            model=ScriptedModel([ScriptedTurn(text="route: billing\nconfidence: 0.9")]),
            runtime=runtime,
            persistent_subagents=True,
        )

    async def crash_when_routed(event: Event) -> None:
        if event.payload.get("name") == "router.completed":
            raise RuntimeError("process died")

    db = tmp_path / "j.db"
    with pytest.raises(RuntimeError, match="process died"):
        await team(SqliteRuntime(db)).run("refund me", session_id="p1", emit=crash_when_routed)
    assert charges == [5]
    result = await team(SqliteRuntime(db)).run("refund me", session_id="p1")
    assert result.output == "refunded"
    assert charges == [5]


# ---------------------------------------------------------------------------
# Router, Supervisor, Blackboard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**route:** billing\n**confidence:** 0.9", ("billing", 0.9)),
        ("route: `billing`\nconfidence: 0.9", ("billing", 0.9)),
        ("route: tech support\nconfidence: 0.9", ("tech support", 0.9)),
        ("route: billing\nconfidence: 85%", ("billing", 0.85)),
    ],
)
def test_router_reads_formatted_classifications(text: str, expected: tuple[str, float]) -> None:
    from loomflow.architecture.router import _parse_classification

    assert _parse_classification(text) == pytest.approx(expected)


async def test_router_matches_route_names_loosely() -> None:
    from loomflow.architecture import Router, RouterRoute

    specialist = Agent("s", model=ScriptedModel([ScriptedTurn(text="TECH ANSWER")]))
    host = Agent(
        "h",
        model=ScriptedModel([ScriptedTurn(text="**Route:** Tech Support\nConfidence: 0.9")]),
        architecture=Router(routes=[RouterRoute("tech_support", specialist, "tech")]),
    )
    assert (await host.run("my wifi")).output == "TECH ANSWER"


def _call(tool_name: str, **args: Any) -> ScriptedTurn:
    return ScriptedTurn(tool_calls=[ToolCall(tool=tool_name, args=args)])


async def test_supervisor_forward_ends_the_turn() -> None:
    from loomflow.team import Team

    worker = Agent("w", model=ScriptedModel([ScriptedTurn(text="WORKER RAW")]))
    sup = ScriptedModel(
        [
            _call("delegate", worker="w", instructions="do X"),
            _call("forward_message", worker="w"),
            ScriptedTurn(text="my own synthesis"),
        ]
    )
    result = await Team.supervisor({"w": worker}, model=sup, persistent_subagents=False).run("task")
    assert result.output == "WORKER RAW"
    assert sup.remaining == 1  # no model call after the forward


async def test_supervisor_forward_before_delegate_lets_the_model_retry() -> None:
    from loomflow.team import Team

    worker = Agent("w", model=ScriptedModel([ScriptedTurn(text="WORKER RAW")]))
    sup = ScriptedModel(
        [
            _call("forward_message", worker="w"),
            _call("delegate", worker="w", instructions="do X"),
            ScriptedTurn(text="final"),
        ]
    )
    result = await Team.supervisor({"w": worker}, model=sup, persistent_subagents=False).run("task")
    assert result.output == "final"


async def test_supervisor_prompt_names_the_configured_tools() -> None:
    from loomflow.team import Team

    seen: list[str] = []

    class Capture(ScriptedModel):
        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            seen.append(str(messages[0].content))
            return await super().complete(messages, **kwargs)

    worker = Agent("w", model=ScriptedModel([ScriptedTurn(text="x")]))
    team = Team.supervisor(
        {"w": worker},
        model=Capture([ScriptedTurn(text="ok")]),
        delegate_tool_name="ask_worker",
        forward_tool_name="passthrough",
        persistent_subagents=False,
    )
    await team.run("task")
    assert "`ask_worker(" in seen[0] and "`passthrough(" in seen[0]
    assert "`delegate(" not in seen[0] and "forward_message(" not in seen[0]


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("no", False), ("true", True), (False, False)])
def test_blackboard_reads_terminate_strings(value: Any, expected: bool) -> None:
    import json

    from loomflow.architecture.blackboard import _parse_coordinator_decision

    decision = _parse_coordinator_decision(json.dumps({"terminate": value, "next_agent": "null"}))
    assert decision.terminate is expected
    assert decision.next_agent is None


async def test_blackboard_keeps_the_board_when_the_budget_runs_out() -> None:
    from loomflow.architecture import BlackboardArchitecture

    big = Usage(input_tokens=600, output_tokens=600)
    contributor = Agent(
        "a", model=ScriptedModel([ScriptedTurn(text=t, usage=big) for t in ("first", "second", "third")])
    )
    host = Agent(
        "h",
        model="echo",
        budget=StandardBudget(BudgetConfig(max_tokens=1000)),
        architecture=BlackboardArchitecture(agents={"a": contributor}, max_rounds=3),
    )
    result = await host.run("p")
    assert result.interrupted and result.output == "first"
