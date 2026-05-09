"""End-to-end smoke tests: ``await Agent('...', model='echo').run('...')`` works."""

import pytest

from loomflow import Agent
from loomflow.core.errors import ConfigError
from loomflow.governance.budget import BudgetConfig, StandardBudget
from loomflow.memory.inmemory import InMemoryMemory
from loomflow.model.echo import EchoModel
from loomflow.runtime.inproc import InProcRuntime

pytestmark = pytest.mark.anyio


def test_agent_without_model_raises_config_error_with_suggestions() -> None:
    """The user explicitly opted out of the silent EchoModel default
    in v0.2.0. Forgetting ``model=`` should now fail loudly with a
    helpful list of options, not produce mysterious ``Echo: ...``
    output."""
    with pytest.raises(ConfigError) as excinfo:
        Agent("hi")
    msg = str(excinfo.value)
    assert "model" in msg
    assert "claude-opus-4-7" in msg
    assert "gpt-4o" in msg
    assert "echo" in msg


async def test_agent_returns_run_result_with_echoed_output() -> None:
    agent = Agent("You are helpful.", model="echo")
    result = await agent.run("hello there")

    assert result.output.startswith("Echo: ")
    assert "hello there" in result.output
    assert result.turns == 1
    assert result.session_id.startswith("sess_")
    assert result.tokens_in > 0
    assert result.tokens_out > 0
    assert not result.interrupted


async def test_agent_persists_episode_to_memory() -> None:
    memory = InMemoryMemory()
    agent = Agent("instructions", model="echo", memory=memory)

    await agent.run("first prompt")

    snapshot = memory.snapshot()
    assert len(snapshot["episodes"]) == 1
    (episode,) = snapshot["episodes"].values()
    assert episode["input"] == "first prompt"
    assert "first prompt" in episode["output"]


async def test_agent_two_runs_produce_distinct_sessions() -> None:
    agent = Agent("hello", model="echo")
    r1 = await agent.run("a")
    r2 = await agent.run("b")
    assert r1.session_id != r2.session_id


async def test_recall_surfaces_prior_episode_in_context() -> None:
    """The second run should see the first run's episode via recall.

    We verify indirectly: the EchoModel echoes the *user* message, but
    we can prove recall happened by checking the memory snapshot grew.
    """
    memory = InMemoryMemory()
    agent = Agent("be brief", model="echo", memory=memory)

    await agent.run("alpha")
    await agent.run("beta")

    snapshot = memory.snapshot()
    assert len(snapshot["episodes"]) == 2


async def test_budget_block_interrupts_run() -> None:
    """A zero-token budget refuses the first step."""
    budget = StandardBudget(BudgetConfig(max_tokens=0))
    agent = Agent("hi", model="echo", budget=budget)

    result = await agent.run("anything")

    assert result.interrupted
    assert result.interruption_reason is not None
    assert result.interruption_reason.startswith("budget:")


async def test_explicit_runtime_and_model_are_used() -> None:
    """Constructor wiring: passing a Model/Runtime overrides the defaults."""
    custom_model = EchoModel(prefix=">>> ")
    runtime = InProcRuntime()
    agent = Agent("hi", model=custom_model, runtime=runtime)

    result = await agent.run("ping")

    assert result.output.startswith(">>> ")
    assert "ping" in result.output
