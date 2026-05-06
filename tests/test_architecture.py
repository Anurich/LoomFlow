"""Architecture layer surface tests.

Verifies:

* The :class:`Architecture` protocol is satisfied by :class:`ReAct`.
* :class:`AgentSession` and :class:`Dependencies` are constructable.
* :func:`resolve_architecture` accepts ``None``, ``"react"``, and an
  instance; rejects unknown strings with :class:`ConfigError`.
* :class:`Agent` defaults to :class:`ReAct` when ``architecture=`` is
  omitted, and forwards ``architecture=`` to the public
  ``.architecture`` property.
* A minimal custom architecture (single text response, no tools)
  drives an end-to-end ``Agent.run()`` correctly — proves the
  Protocol shape is what the framework expects.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from jeevesagent import Agent, ScriptedModel, ScriptedTurn
from jeevesagent.architecture import (
    AgentSession,
    Architecture,
    Dependencies,
    ReAct,
    resolve_architecture,
)
from jeevesagent.core.errors import ConfigError
from jeevesagent.core.types import Event

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_react_satisfies_architecture_protocol() -> None:
    """``isinstance(ReAct(), Architecture)`` must be True — the
    protocol is ``runtime_checkable`` precisely so users can verify
    custom architectures against it."""
    assert isinstance(ReAct(), Architecture)


def test_react_declares_no_workers() -> None:
    """ReAct is single-agent — ``declared_workers`` returns empty."""
    assert ReAct().declared_workers() == {}


def test_react_has_name() -> None:
    assert ReAct().name == "react"


# ---------------------------------------------------------------------------
# resolve_architecture
# ---------------------------------------------------------------------------


def test_resolve_architecture_none_returns_react_default() -> None:
    arch = resolve_architecture(None)
    assert isinstance(arch, ReAct)


def test_resolve_architecture_string_react() -> None:
    arch = resolve_architecture("react")
    assert isinstance(arch, ReAct)


def test_resolve_architecture_passes_through_instance() -> None:
    instance = ReAct(max_turns=7)
    assert resolve_architecture(instance) is instance


def test_resolve_architecture_unknown_string_raises_configerror() -> None:
    with pytest.raises(ConfigError, match="unknown architecture"):
        resolve_architecture("nonexistent-arch")


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def test_agent_default_architecture_is_react() -> None:
    agent = Agent("hi", model="echo")
    assert isinstance(agent.architecture, ReAct)


def test_agent_accepts_architecture_string() -> None:
    agent = Agent("hi", model="echo", architecture="react")
    assert isinstance(agent.architecture, ReAct)


def test_agent_accepts_architecture_instance() -> None:
    instance = ReAct(max_turns=3)
    agent = Agent("hi", model="echo", architecture=instance)
    assert agent.architecture is instance


def test_agent_unknown_architecture_string_raises_configerror() -> None:
    with pytest.raises(ConfigError, match="unknown architecture"):
        Agent("hi", model="echo", architecture="nope")


# ---------------------------------------------------------------------------
# Custom architecture — end-to-end
# ---------------------------------------------------------------------------


class _NoopArchitecture:
    """Minimal architecture that emits one synthetic event and stops.

    Proves the Protocol contract: implementations don't need to
    inherit from anything, just implement ``name``, ``run``, and
    ``declared_workers``.
    """

    name = "noop"

    def declared_workers(self) -> dict[str, Agent]:
        return {}

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        session.output = f"[noop processed: {prompt}]"
        session.turns = 1
        # No model / tool calls; just structurally yield one event.
        yield Event.budget_warning(
            session.id,
            __import__("jeevesagent.core.types", fromlist=["BudgetStatus"])
            .BudgetStatus.warn_("noop arch warns once"),
        )


def test_noop_arch_satisfies_protocol() -> None:
    assert isinstance(_NoopArchitecture(), Architecture)


async def test_custom_architecture_drives_agent_run_to_completion() -> None:
    """A custom architecture without a model still lets the Agent
    emit STARTED / COMPLETED, persist an episode, and return a
    populated :class:`RunResult`."""
    agent = Agent(
        "hi",
        model="echo",  # not actually called by NoopArchitecture
        architecture=_NoopArchitecture(),
    )
    result = await agent.run("hello world")
    assert "[noop processed: hello world]" in result.output
    assert result.turns == 1
    assert result.session_id.startswith("sess_")


async def test_custom_architecture_events_visible_via_stream() -> None:
    """Events the architecture yields show up in the public stream
    surface, sandwiched between STARTED and COMPLETED."""
    agent = Agent(
        "hi", model="echo", architecture=_NoopArchitecture()
    )
    events = [event async for event in agent.stream("ping")]
    kinds = [e.kind for e in events]
    assert kinds[0] == "started"
    assert kinds[-1] == "completed"
    assert "budget_warning" in kinds


# ---------------------------------------------------------------------------
# ReAct end-to-end via Agent — sanity check the refactored loop
# ---------------------------------------------------------------------------


async def test_react_drives_a_scripted_two_turn_run() -> None:
    """One tool call, one final text response — the canonical ReAct
    shape. Ensures the extracted iteration matches v0.1.x behaviour."""
    from jeevesagent import tool
    from jeevesagent.core.types import ToolCall

    @tool
    async def echo_back(msg: str) -> str:
        """Echo back the message."""
        return f"echoed:{msg}"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", tool="echo_back", args={"msg": "hi"})
                ]
            ),
            ScriptedTurn(text="all done"),
        ]
    )
    agent = Agent("test", model=model, tools=[echo_back])
    result = await agent.run("go")
    assert "all done" in result.output
    assert result.turns == 2
