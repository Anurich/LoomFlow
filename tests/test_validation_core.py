"""Regression tests for the agent-core validation findings.

Each test reproduces a finding from the validation pass and fails on
the code before its fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from loomflow import Agent, ScriptedModel, ScriptedTurn, Tuning, tool
from loomflow.core.types import Role, ToolCall

pytestmark = pytest.mark.anyio


class _Recording(ScriptedModel):
    def __init__(self, turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.seen: list[list[Any]] = []
        self.calls = 0

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.seen.append(list(messages))
        return await super().complete(messages, **kwargs)

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.seen.append(list(messages))
        async for chunk in super().stream(messages, **kwargs):
            yield chunk


# ---------------------------------------------------------------------------
# run_until
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_iterations", [1, 2])
async def test_run_until_checks_the_last_pass(max_iterations: int) -> None:
    main = ScriptedModel([ScriptedTurn(text=f"pass-{i} all tests pass") for i in range(5)])
    verdicts = ["NOT_DONE\nno"] * (max_iterations - 1) + ["DONE\nok"]
    checker = _Recording([ScriptedTurn(text=v) for v in verdicts])
    agent = Agent(
        "x",
        model=main,
        run_until={"condition": "all tests pass", "checker": checker, "max_iterations": max_iterations},
    )
    result = await agent.run("go")
    assert not result.interrupted
    assert checker.calls == max_iterations


async def test_run_until_still_caps_iterations_when_unmet() -> None:
    main = ScriptedModel([ScriptedTurn(text=f"pass-{i}") for i in range(5)])
    checker = ScriptedModel([ScriptedTurn(text="NOT_DONE\nno")] * 5)
    agent = Agent("x", model=main, run_until={"condition": "c", "checker": checker, "max_iterations": 2})
    result = await agent.run("go")
    assert result.interrupted and result.interruption_reason == "run_until:max_iterations"


# ---------------------------------------------------------------------------
# Tool transcripts
# ---------------------------------------------------------------------------


async def test_persisted_tool_transcripts_are_not_duplicated_across_runs() -> None:
    @tool
    async def lookup(n: int) -> str:
        """Look up."""
        return f"r{n}"

    turns: list[ScriptedTurn] = []
    for i in range(3):
        turns += [
            ScriptedTurn(tool_calls=[ToolCall(id=f"call-{i}", tool="lookup", args={"n": i})]),
            ScriptedTurn(text=f"answer-{i}"),
        ]
    agent = Agent("x", model=ScriptedModel(turns), tools=[lookup], persist_tool_transcripts=True)
    for i in range(3):
        await agent.run(f"prompt-{i}", session_id="s")
    history = await agent.memory.session_messages("s")
    tool_ids = [m.tool_call_id for m in history if m.role is Role.TOOL]
    assert tool_ids == ["call-0", "call-1", "call-2"]


# ---------------------------------------------------------------------------
# Structured output and schemas
# ---------------------------------------------------------------------------


class _Summary(BaseModel):
    title: str


class _Detailed(BaseModel):
    title: str
    body: str
    tags: list[str]


async def test_union_output_picks_the_best_matching_member() -> None:
    payload = '{"title": "T", "body": "B", "tags": ["x"]}'
    agent = Agent("x", model=ScriptedModel([ScriptedTurn(text=payload)]))
    result = await agent.run("go", output_schema=_Summary | _Detailed)
    assert isinstance(result.parsed, _Detailed)


async def test_union_output_schema_is_described_to_native_adapters() -> None:
    class Native(_Recording):
        supports_native_structured_output = True

    model = Native([ScriptedTurn(text='{"title": "T"}')])
    await Agent("BASE", model=model).run("go", output_schema=_Summary | _Detailed)
    system = next(m.content for m in model.seen[0] if m.role is Role.SYSTEM)
    assert "STRUCTURED OUTPUT REQUIRED" in system and "_Detailed" in system


def _two_item_models() -> tuple[type[BaseModel], type[BaseModel]]:
    def make_a() -> type[BaseModel]:
        class Item(BaseModel):
            sku: str

        class Order(BaseModel):
            items: list[Item]

        return Order

    def make_b() -> type[BaseModel]:
        class Item(BaseModel):
            weight_kg: float

        class Shipment(BaseModel):
            items: list[Item]

        return Shipment

    return make_a(), make_b()


def test_tool_schema_keeps_same_named_nested_models_apart() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    order_t, shipment_t = _two_item_models()

    def ship(order: Any, shipment: Any) -> str:
        """Ship."""
        return "ok"

    # Real classes, not the strings postponed annotations would give.
    ship.__annotations__ = {"order": order_t, "shipment": shipment_t, "return": str}
    schema = tool(ship).input_schema
    validator = jsonschema.Draft202012Validator(schema)
    good = {"order": {"items": [{"sku": "A1"}]}, "shipment": {"items": [{"weight_kg": 2.0}]}}
    assert list(validator.iter_errors(good)) == []
    swapped = {"order": {"items": [{"weight_kg": 2.0}]}, "shipment": {"items": [{"sku": "A1"}]}}
    assert list(validator.iter_errors(swapped))


def test_union_output_schema_keeps_same_named_nested_models_apart() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    from loomflow.agent.api import _union_json_schema

    order_t, shipment_t = _two_item_models()
    validator = jsonschema.Draft202012Validator(_union_json_schema([order_t, shipment_t]))
    assert list(validator.iter_errors({"items": [{"sku": "A1"}]})) == []


def test_tool_descriptions_are_not_cut_to_one_line() -> None:
    @tool(description="Retrieves the price.\nOnly NYSE/NASDAQ symbols.\nReturns USD.")
    def price(ticker: Annotated[str, Field(description="e.g. AAPL")]) -> str:
        return "1"

    def search(query: str, limit: int = 5) -> str:
        """Search the corpus.

        Use for factual lookups; never for math.

        Args:
            query: full-text query.
            limit: max hits.
        """
        return ""

    assert price.description == "Retrieves the price.\nOnly NYSE/NASDAQ symbols.\nReturns USD."
    searched = tool(search)
    assert searched.description == "Search the corpus.\n\nUse for factual lookups; never for math."
    assert searched.input_schema["properties"]["query"]["description"] == "full-text query."
    assert searched.input_schema["properties"]["limit"]["description"] == "max hits."


# ---------------------------------------------------------------------------
# Configuration typos
# ---------------------------------------------------------------------------


def test_from_config_rejects_unknown_keys(tmp_path: Path) -> None:
    from loomflow.core.errors import ConfigError

    path = tmp_path / "agent.toml"
    path.write_text('instructions = "x"\nmodel = "echo"\nmax_turn = 3\n')
    with pytest.raises(ConfigError, match="did you mean 'max_turns'"):
        Agent.from_config(path)
    path.write_text('instructions = "x"\nmodel = "echo"\n[budget]\nmax_cost = 0.01\n')
    with pytest.raises(ConfigError, match="max_cost_usd"):
        Agent.from_config(path)


def test_from_config_reads_snip_window_and_prompt_caching(tmp_path: Path) -> None:
    path = tmp_path / "agent.toml"
    path.write_text('instructions = "x"\nmodel = "echo"\nsnip_window = 4\nprompt_caching = true\n')
    agent = Agent.from_config(path)
    assert agent._snip_window == 4 and agent._prompt_caching.enabled


def test_prompt_caching_rejects_unknown_keys() -> None:
    from loomflow.agent.api import _resolve_prompt_caching
    from loomflow.core.errors import ConfigError

    with pytest.raises(ConfigError, match="ttl"):
        _resolve_prompt_caching({"enabled": True, "tll": "1h"})


async def test_negative_output_validation_retries_is_a_value_error() -> None:
    agent = Agent("x", model=ScriptedModel([ScriptedTurn(text="bad")]))
    with pytest.raises(ValueError, match="output_validation_retries"):
        await agent.run("hi", output_schema=_Summary, output_validation_retries=-1)


# ---------------------------------------------------------------------------
# Evals
# ---------------------------------------------------------------------------


async def test_eval_gate_fails_when_a_metric_raises() -> None:
    from loomflow.eval import EvalHarness
    from loomflow.eval.dataset import Case

    class Flaky:
        name = "judge"

        async def score(self, case: Any, result: Any, events: Any) -> float:
            if case.id != "c0":
                raise RuntimeError("judge API 529 overloaded")
            return 1.0

    cases = [Case(id=f"c{i}", input=f"q{i}") for i in range(4)]
    with pytest.warns(UserWarning):
        report = await EvalHarness(Agent("x", model="echo"), [Flaky()]).run(cases)
    assert not report.passed
    with pytest.raises(AssertionError, match="raised on 3/4"):
        report.assert_thresholds({"judge": 0.9})


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Score: **0.8**", 0.8),
        ("score: 0.8.", 0.8),
        ("Final score: 0.8", 0.8),
        ("score: 4/5", 0.8),
        ("score: 80%", 0.8),
        ("score: 1.7", None),
        ("the score was high", None),
    ],
)
def test_judge_reads_common_score_lines(reply: str, expected: float | None) -> None:
    from loomflow.eval.judge import _parse_score_line

    assert _parse_score_line(reply) == (pytest.approx(expected) if expected is not None else None)


# ---------------------------------------------------------------------------
# Auto-compact
# ---------------------------------------------------------------------------


async def test_auto_compact_compacts_inside_a_long_tool_loop() -> None:
    @tool
    def big(i: int) -> str:
        """Big output."""
        return "X" * 4000

    turns = [ScriptedTurn(tool_calls=[ToolCall(id=f"c{i}", tool="big", args={"i": i})]) for i in range(8)]
    model = _Recording([*turns, ScriptedTurn(text="final")])
    summariser = ScriptedModel([ScriptedTurn(text="SUMMARY")] * 10)
    agent = Agent(
        "sys",
        model=model,
        tools=[big],
        auto_compact_at_tokens=2000,
        tuning=Tuning(auto_compact_summariser=summariser, auto_compact_keep_recent_turns=1),
    )
    result = await agent.run("do the big reads")
    assert result.output == "final"
    assert max(len(call) for call in model.seen) < 10
    assert any("auto-compacted" in (m.content or "") for call in model.seen for m in call)


async def test_auto_compact_applies_to_long_rehydrated_history() -> None:
    model = _Recording([ScriptedTurn(text=f"answer-{i} " + "Y" * 3000) for i in range(5)])
    summariser = ScriptedModel([ScriptedTurn(text="SUMMARY")] * 10)
    agent = Agent(
        "sys",
        model=model,
        auto_compact_at_tokens=1500,
        tuning=Tuning(auto_compact_summariser=summariser, auto_compact_keep_recent_turns=1),
    )
    for i in range(5):
        await agent.run(f"q{i}", session_id="S")
    sizes = [sum(len(m.content or "") for m in call) for call in model.seen]
    # Uncompacted, the last call would carry four 3000-char answers.
    assert sizes[-1] < 7000
    assert any("auto-compacted" in (m.content or "") for m in model.seen[-1])


# ---------------------------------------------------------------------------
# Ambient workspace
# ---------------------------------------------------------------------------


async def test_ambient_workspace_reaches_the_system_prompt() -> None:
    from loomflow.core.context import _ambient_workspace_var
    from loomflow.workspace.resolver import resolve_workspace

    model = _Recording([ScriptedTurn(text="ok")])
    agent = Agent("BASE", model=model)
    token = _ambient_workspace_var.set(resolve_workspace("memory"))
    try:
        await agent.run("hi")
    finally:
        _ambient_workspace_var.reset(token)
    system = next(m.content for m in model.seen[0] if m.role is Role.SYSTEM)
    assert "notebook" in system.lower()
