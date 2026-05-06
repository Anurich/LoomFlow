"""v0.2.0 polish: Agent.__repr__, RunResult convenience properties,
agent.consolidate() return value, single-callable tool ergonomics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from jeevesagent import (
    Agent,
    Consolidator,
    InMemoryMemory,
    ScriptedModel,
    ScriptedTurn,
    Tool,
    tool,
)
from jeevesagent.core.types import RunResult, ToolCall

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# RunResult convenience
# ---------------------------------------------------------------------------


def test_run_result_total_tokens_sums_in_and_out() -> None:
    started = datetime.now(UTC)
    r = RunResult(
        session_id="s1",
        output="x",
        turns=2,
        tokens_in=12,
        tokens_out=34,
        cost_usd=0.0,
        started_at=started,
        finished_at=started + timedelta(seconds=2),
    )
    assert r.total_tokens == 46


def test_run_result_duration_is_finished_minus_started() -> None:
    started = datetime.now(UTC)
    finished = started + timedelta(milliseconds=750)
    r = RunResult(
        session_id="s1",
        output="x",
        turns=1,
        started_at=started,
        finished_at=finished,
    )
    assert r.duration == timedelta(milliseconds=750)
    assert r.duration.total_seconds() == 0.75


# ---------------------------------------------------------------------------
# Agent.__repr__
# ---------------------------------------------------------------------------


def test_agent_repr_includes_model_memory_runtime_max_turns() -> None:
    agent = Agent("hi", model="echo")
    rep = repr(agent)
    assert rep.startswith("Agent(")
    assert "model='echo'" in rep
    assert "memory=InMemoryMemory" in rep
    assert "runtime=InProcRuntime" in rep
    assert "tools=InProcessToolHost" in rep
    assert "max_turns=" in rep


# ---------------------------------------------------------------------------
# agent.consolidate() returns count
# ---------------------------------------------------------------------------


async def test_consolidate_returns_zero_when_no_consolidator() -> None:
    agent = Agent("hi", model="echo")
    n = await agent.consolidate()
    assert n == 0


async def test_consolidate_returns_zero_when_memory_has_no_facts_attr() -> None:
    """Backends without a ``.facts`` attribute (custom Memory impls)
    return 0 — consolidate is a no-op there."""

    class _BareMemory:
        async def working(self) -> list:  # noqa: ANN001
            return []

        async def update_block(self, name: str, content: str) -> None:
            pass

        async def append_block(self, name: str, content: str) -> None:
            pass

        async def remember(self, episode) -> str:  # noqa: ANN001
            return episode.id

        async def recall(self, query, **kwargs):  # noqa: ANN001
            return []

        async def consolidate(self) -> None:
            return None

    agent = Agent("hi", model="echo", memory=_BareMemory())  # type: ignore[arg-type]
    assert await agent.consolidate() == 0


async def test_consolidate_returns_count_of_new_facts() -> None:
    extracted = (
        '[{"subject":"u","predicate":"p1","object":"o1","confidence":0.9},'
        '{"subject":"u","predicate":"p2","object":"o2","confidence":0.9}]'
    )
    consolidator_model = ScriptedModel([ScriptedTurn(text=extracted)])
    memory = InMemoryMemory(
        consolidator=Consolidator(model=consolidator_model)
    )
    agent_model = ScriptedModel([ScriptedTurn(text="ack")])
    agent = Agent("hi", model=agent_model, memory=memory)

    await agent.run("hello")
    count = await agent.consolidate()
    assert count == 2  # two facts extracted


# ---------------------------------------------------------------------------
# Tool coercion: single callable / single Tool
# ---------------------------------------------------------------------------


async def test_tools_accepts_single_callable() -> None:
    """``tools=my_fn`` (no list) should work and auto-wrap into a one-tool host."""

    @tool
    async def echo(msg: str) -> str:
        """Echo back."""
        return msg

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", tool="echo", args={"msg": "hi"})
                ]
            ),
            ScriptedTurn(text="ok"),
        ]
    )
    agent = Agent("hi", model=model, tools=echo)  # ← single Tool, no list
    result = await agent.run("...")
    assert "ok" in result.output


async def test_tools_accepts_single_undecorated_function() -> None:
    """A bare ``def`` that hasn't been ``@tool``-decorated still works."""

    async def lookup(name: str) -> str:
        """Look up a thing."""
        return f"found:{name}"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", tool="lookup", args={"name": "x"})
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent("hi", model=model, tools=lookup)
    result = await agent.run("...")
    assert "done" in result.output


async def test_tools_accepts_single_explicit_Tool_object() -> None:
    """``Tool(...)`` instance passed directly (not in a list) also works."""

    async def ping() -> str:
        return "pong"

    t = Tool(name="ping", description="ping", fn=ping)
    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="ping", args={})]
            ),
            ScriptedTurn(text="ok"),
        ]
    )
    agent = Agent("hi", model=model, tools=t)
    result = await agent.run("...")
    assert "ok" in result.output


# ---------------------------------------------------------------------------
# agent.add_tool — register tools after construction
# ---------------------------------------------------------------------------


async def test_add_tool_registers_with_inprocess_host() -> None:
    @tool
    async def ping() -> str:
        """Return pong."""
        return "pong"

    agent = Agent("hi", model="echo")
    registered = agent.add_tool(ping)
    assert registered.name == "ping"

    # Tool is now visible in the host.
    defs = await agent._tool_host.list_tools()
    assert "ping" in {d.name for d in defs}


async def test_add_tool_accepts_undecorated_function() -> None:
    async def shout(msg: str) -> str:
        """Uppercase the message."""
        return msg.upper()

    agent = Agent("hi", model="echo")
    registered = agent.add_tool(shout)
    assert registered.name == "shout"


async def test_add_tool_then_use_in_run() -> None:
    """Round-trip: register a tool, then run an agent that uses it."""

    @tool
    async def lookup(name: str) -> str:
        """Look up name."""
        return f"found:{name}"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", tool="lookup", args={"name": "x"})
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent("hi", model=model)
    agent.add_tool(lookup)
    result = await agent.run("...")
    assert "done" in result.output


async def test_with_tool_decorator_registers_function() -> None:
    """``@agent.with_tool`` registers the function and returns it
    unchanged (so the function stays directly callable)."""
    agent = Agent("hi", model="echo")

    @agent.with_tool
    async def echo(msg: str) -> str:
        """Echo a message."""
        return msg

    # Returned object is the original function, not wrapped.
    assert callable(echo)
    # And it's registered as a tool.
    assert "echo" in await agent.tools_list()
    # Direct callability preserved.
    assert await echo("ping") == "ping"


async def test_with_tool_works_with_sync_function() -> None:
    agent = Agent("hi", model="echo")

    @agent.with_tool
    def upper(s: str) -> str:
        """Uppercase ``s``."""
        return s.upper()

    assert "upper" in await agent.tools_list()
    assert upper("hi") == "HI"


async def test_remove_tool_unregisters_by_name() -> None:
    @tool
    async def ping() -> str:
        return "pong"

    agent = Agent("hi", model="echo")
    agent.add_tool(ping)
    assert "ping" in await agent.tools_list()

    removed = agent.remove_tool("ping")
    assert removed is True
    assert "ping" not in await agent.tools_list()


async def test_remove_tool_returns_false_when_not_registered() -> None:
    agent = Agent("hi", model="echo")
    assert agent.remove_tool("nonexistent") is False


async def test_tools_list_returns_names() -> None:
    @tool
    async def alpha() -> str:
        return "a"

    @tool
    async def beta() -> str:
        return "b"

    agent = Agent("hi", model="echo", tools=[alpha, beta])
    names = sorted(await agent.tools_list())
    assert names == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Public introspection properties
# ---------------------------------------------------------------------------


def test_agent_exposes_model_memory_runtime_as_properties() -> None:
    agent = Agent("hi", model="echo")
    assert agent.model is agent._model  # noqa: SLF001
    assert agent.memory is agent._memory  # noqa: SLF001
    assert agent.runtime is agent._runtime  # noqa: SLF001
    assert agent.tool_host is agent._tool_host  # noqa: SLF001
    assert agent.budget is agent._budget  # noqa: SLF001
    assert agent.permissions is agent._permissions  # noqa: SLF001


# ---------------------------------------------------------------------------
# agent.recall() shortcut
# ---------------------------------------------------------------------------


async def test_recall_returns_episodes_from_memory() -> None:
    agent = Agent("hi", model="echo")
    # Persist a couple of runs so memory has episodes.
    await agent.run("first prompt about apples")
    await agent.run("second prompt about oranges")

    matches = await agent.recall("apples", limit=1)
    assert len(matches) == 1
    assert "apples" in matches[0].input


async def test_recall_kind_arg_is_propagated() -> None:
    """recall(kind='semantic') should still work even when memory has
    no fact store — backends fall back to recency."""
    agent = Agent("hi", model="echo")
    await agent.run("hello")
    out = await agent.recall("anything", kind="semantic", limit=5)
    # Just verify no exception; shape is up to backend.
    assert isinstance(out, list)


async def test_add_tool_rejects_when_host_is_not_inprocess() -> None:
    """If the user passed a custom ToolHost (e.g. an MCPRegistry),
    add_tool can't extend it — raise a clear error."""
    from collections.abc import AsyncIterator

    from jeevesagent.core.errors import ConfigError
    from jeevesagent.core.types import ToolDef, ToolEvent, ToolResult

    class _CustomHost:
        async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
            return []

        async def call(
            self, tool: str, args: object, *, call_id: str = ""
        ) -> ToolResult:
            return ToolResult.error_(call_id, "custom host")

        async def watch(self) -> AsyncIterator[ToolEvent]:
            empty: tuple[ToolEvent, ...] = ()
            for ev in empty:
                yield ev

    @tool
    async def x() -> str:
        return "x"

    agent = Agent("hi", model="echo", tools=_CustomHost())
    with pytest.raises(ConfigError, match="add_tool requires InProcessToolHost"):
        agent.add_tool(x)


async def test_tools_list_still_works() -> None:
    """Backward-compat: lists still work after adding single-fn shorthand."""

    @tool
    async def a() -> str:
        return "a"

    @tool
    async def b() -> str:
        return "b"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="ca", tool="a", args={}),
                    ToolCall(id="cb", tool="b", args={}),
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent("hi", model=model, tools=[a, b])
    result = await agent.run("...")
    assert "done" in result.output
