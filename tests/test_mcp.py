"""MCP client + registry tests using fake sessions (no mcp SDK needed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from loomflow import Agent
from loomflow.core.types import ToolCall
from loomflow.mcp import MCPClient, MCPRegistry, MCPServerSpec
from loomflow.mcp.spec import MCPServerSpec as Spec
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fake MCP session — mimics what mcp.ClientSession exposes
# ---------------------------------------------------------------------------


@dataclass
class _FakeMcpTool:
    name: str
    description: str = ""
    inputSchema: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeContent:
    type: str = "text"
    text: str = ""


@dataclass
class _FakeListResult:
    tools: list[_FakeMcpTool]


@dataclass
class _FakeCallResult:
    content: list[_FakeContent] = field(default_factory=list)
    isError: bool = False
    structuredContent: Any = None


class _FakeMcpSession:
    """Reproduces just enough of ``mcp.ClientSession`` for our adapter."""

    def __init__(
        self,
        tools: list[_FakeMcpTool],
        call_handler: Any | None = None,
    ) -> None:
        self._tools = tools
        self._handler = call_handler
        self.initialized = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> _FakeListResult:
        return _FakeListResult(tools=list(self._tools))

    async def call_tool(self, name: str, args: dict[str, Any]) -> _FakeCallResult:
        self.calls.append((name, dict(args)))
        if self._handler is not None:
            result = await self._handler(name, args)
            if isinstance(result, _FakeCallResult):
                return result
            return _FakeCallResult(content=[_FakeContent(text=str(result))])
        return _FakeCallResult(content=[_FakeContent(text=f"called {name}")])


# ---------------------------------------------------------------------------
# Spec tests
# ---------------------------------------------------------------------------


def test_stdio_spec_construction() -> None:
    s = Spec.stdio("git", "uvx", ["mcp-server-git", "--repo", "/tmp/r"])
    assert s.transport == "stdio"
    assert s.command == "uvx"
    assert s.args == ("mcp-server-git", "--repo", "/tmp/r")
    assert s.url is None


def test_http_spec_construction_with_headers() -> None:
    s = Spec.http(
        "remote",
        "https://example.com/mcp",
        headers={"X-Custom": "1"},
    )
    assert s.transport == "http"
    assert s.url == "https://example.com/mcp"
    assert ("X-Custom", "1") in s.headers


# ---------------------------------------------------------------------------
# Client tests (fake session bypasses the real SDK)
# ---------------------------------------------------------------------------


async def test_client_with_injected_session_skips_real_connect() -> None:
    session = _FakeMcpSession([_FakeMcpTool(name="ping")])
    client = MCPClient(MCPServerSpec.stdio("test", "noop"), session=session)
    assert client.is_connected
    tools = await client.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "ping"
    # initialize() shouldn't be called because we bypassed connect()
    assert session.initialized is False


async def test_client_call_tool_passes_through_to_session() -> None:
    session = _FakeMcpSession([_FakeMcpTool(name="echo")])
    client = MCPClient(MCPServerSpec.stdio("test", "noop"), session=session)
    result = await client.call_tool("echo", {"msg": "hi"})
    assert result.content[0].text == "called echo"
    assert session.calls == [("echo", {"msg": "hi"})]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


async def test_registry_aggregates_tools_from_multiple_servers() -> None:
    s1 = _FakeMcpSession(
        [
            _FakeMcpTool(name="get_weather", description="weather"),
            _FakeMcpTool(name="forecast"),
        ]
    )
    s2 = _FakeMcpSession([_FakeMcpTool(name="lookup")])

    c1 = MCPClient(MCPServerSpec.stdio("city_api", "noop"), session=s1)
    c2 = MCPClient(MCPServerSpec.stdio("dictionary", "noop"), session=s2)
    reg = MCPRegistry([c1, c2])

    defs = await reg.list_tools()
    names = {d.name for d in defs}
    # Three tools, all unique names => bare names
    assert names == {"get_weather", "forecast", "lookup"}

    # Server attribute is populated
    by_name = {d.name: d for d in defs}
    assert by_name["get_weather"].server == "city_api"
    assert by_name["lookup"].server == "dictionary"


async def test_name_collision_auto_disambiguates() -> None:
    s1 = _FakeMcpSession([_FakeMcpTool(name="search")])
    s2 = _FakeMcpSession([_FakeMcpTool(name="search")])
    c1 = MCPClient(MCPServerSpec.stdio("alpha", "noop"), session=s1)
    c2 = MCPClient(MCPServerSpec.stdio("beta", "noop"), session=s2)
    reg = MCPRegistry([c1, c2])

    defs = await reg.list_tools()
    names = {d.name for d in defs}
    assert names == {"alpha.search", "beta.search"}


async def test_registry_routes_calls_to_correct_server() -> None:
    s1 = _FakeMcpSession([_FakeMcpTool(name="alpha_tool")])
    s2 = _FakeMcpSession([_FakeMcpTool(name="beta_tool")])
    c1 = MCPClient(MCPServerSpec.stdio("alpha", "noop"), session=s1)
    c2 = MCPClient(MCPServerSpec.stdio("beta", "noop"), session=s2)
    reg = MCPRegistry([c1, c2])

    r = await reg.call("beta_tool", {"x": 1}, call_id="c1")
    assert r.ok
    assert r.call_id == "c1"
    # Only s2 should have been called
    assert s1.calls == []
    assert s2.calls == [("beta_tool", {"x": 1})]


async def test_qualified_name_routes_when_disambiguated() -> None:
    s1 = _FakeMcpSession([_FakeMcpTool(name="search")])
    s2 = _FakeMcpSession([_FakeMcpTool(name="search")])
    c1 = MCPClient(MCPServerSpec.stdio("alpha", "noop"), session=s1)
    c2 = MCPClient(MCPServerSpec.stdio("beta", "noop"), session=s2)
    reg = MCPRegistry([c1, c2])

    r = await reg.call("beta.search", {"q": "x"}, call_id="c1")
    assert r.ok
    # The bare name was forwarded to the underlying session
    assert s1.calls == []
    assert s2.calls == [("search", {"q": "x"})]


async def test_unknown_tool_returns_error_result() -> None:
    s = _FakeMcpSession([_FakeMcpTool(name="known")])
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=s)]
    )

    r = await reg.call("ghost", {}, call_id="c1")
    assert not r.ok
    assert r.error is not None
    assert "unknown" in r.error.lower()


async def test_session_error_surfaces_as_tool_result_error() -> None:
    async def boom(name: str, args: dict[str, Any]) -> Any:
        raise RuntimeError("upstream timeout")

    s = _FakeMcpSession([_FakeMcpTool(name="flaky")], call_handler=boom)
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=s)]
    )

    r = await reg.call("flaky", {}, call_id="c1")
    assert not r.ok
    assert r.error is not None
    assert "upstream timeout" in r.error


async def test_is_error_flag_marks_tool_result_failed() -> None:
    async def returns_error(name: str, args: dict[str, Any]) -> _FakeCallResult:
        return _FakeCallResult(
            content=[_FakeContent(text="server-side oops")],
            isError=True,
        )

    s = _FakeMcpSession(
        [_FakeMcpTool(name="grumpy")],
        call_handler=returns_error,
    )
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=s)]
    )
    r = await reg.call("grumpy", {}, call_id="c1")
    assert not r.ok
    assert r.error == "server-side oops"


async def test_structured_content_preferred_over_text_blocks() -> None:
    async def structured(name: str, args: dict[str, Any]) -> _FakeCallResult:
        return _FakeCallResult(
            content=[_FakeContent(text="ignored")],
            structuredContent={"answer": 42},
        )

    s = _FakeMcpSession(
        [_FakeMcpTool(name="solver")],
        call_handler=structured,
    )
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=s)]
    )
    r = await reg.call("solver", {}, call_id="c1")
    assert r.ok
    assert r.output == {"answer": 42}


async def test_query_filter_in_list_tools() -> None:
    s = _FakeMcpSession(
        [
            _FakeMcpTool(name="get_weather", description="get current weather"),
            _FakeMcpTool(name="forecast", description="future weather"),
            _FakeMcpTool(name="email_send", description="send an email"),
        ]
    )
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=s)]
    )
    weather_tools = await reg.list_tools(query="weather")
    names = {d.name for d in weather_tools}
    assert names == {"get_weather", "forecast"}


# ---------------------------------------------------------------------------
# End-to-end: Agent using MCPRegistry as its tool host
# ---------------------------------------------------------------------------


async def test_agent_run_dispatches_to_mcp_tool() -> None:
    """The full Agent loop: model emits a tool call, registry routes it,
    result feeds back to the model, model finishes."""

    async def weather_handler(
        name: str, args: dict[str, Any]
    ) -> _FakeCallResult:
        city = args.get("city", "?")
        return _FakeCallResult(content=[_FakeContent(text=f"sunny in {city}")])

    s = _FakeMcpSession(
        [_FakeMcpTool(name="get_weather", description="weather")],
        call_handler=weather_handler,
    )
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("city_api", "noop"), session=s)]
    )

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", tool="get_weather", args={"city": "Tokyo"})
                ]
            ),
            ScriptedTurn(text="It's sunny."),
        ]
    )

    agent = Agent("weather assistant", model=model, tools=reg)
    result = await agent.run("what's the weather in tokyo?")

    assert "sunny" in result.output.lower()
    assert s.calls == [("get_weather", {"city": "Tokyo"})]
    assert not result.interrupted


# ---------------------------------------------------------------------------
# Reconnect-and-retry — one-shot self-heal on transport errors
# ---------------------------------------------------------------------------


async def test_call_reconnects_and_retries_after_transport_error() -> None:
    """When the first ``call_tool`` raises a transport-style error,
    the registry should tear down the broken client, swap in a
    fresh one (provided here by the test via ``_make_client``), and
    retry the call once. The second attempt succeeds, the caller
    sees a successful ToolResult."""

    async def boom_once(name: str, args: dict[str, Any]) -> Any:
        raise ConnectionError("upstream pipe closed")

    broken_session = _FakeMcpSession(
        [_FakeMcpTool(name="probe")], call_handler=boom_once
    )

    async def healed(name: str, args: dict[str, Any]) -> _FakeCallResult:
        return _FakeCallResult(content=[_FakeContent(text="healed")])

    healed_session = _FakeMcpSession(
        [_FakeMcpTool(name="probe")], call_handler=healed
    )

    class _ReconnectRegistry(MCPRegistry):
        """Replaces ``_make_client`` so the reconnect attempt yields
        a fresh client backed by our pre-baked healed session."""

        def _make_client(self, spec: MCPServerSpec) -> MCPClient:
            return MCPClient(spec, session=healed_session)

    reg = _ReconnectRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=broken_session)]
    )

    r = await reg.call("probe", {"x": 1}, call_id="c1")
    assert r.ok, r.error
    assert r.output == "healed"
    # Broken session saw the first attempt; healed session saw the retry.
    assert broken_session.calls == [("probe", {"x": 1})]
    assert healed_session.calls == [("probe", {"x": 1})]


async def test_retry_failure_surfaces_error_not_silently_succeeds() -> None:
    """If the reconnect succeeds but the retried call ALSO fails,
    the registry must surface the second error — not pretend the
    call succeeded just because reconnect worked."""

    async def always_boom(name: str, args: dict[str, Any]) -> Any:
        raise RuntimeError("still broken")

    s1 = _FakeMcpSession([_FakeMcpTool(name="t")], call_handler=always_boom)
    s2 = _FakeMcpSession([_FakeMcpTool(name="t")], call_handler=always_boom)

    class _ReconnectRegistry(MCPRegistry):
        def _make_client(self, spec: MCPServerSpec) -> MCPClient:
            return MCPClient(spec, session=s2)

    reg = _ReconnectRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=s1)]
    )
    r = await reg.call("t", {}, call_id="c1")
    assert not r.ok
    assert r.error is not None
    assert "still broken" in r.error


async def test_reconnect_failure_surfaces_first_error() -> None:
    """When reconnection itself can't establish a fresh session,
    the registry returns the ORIGINAL exception (not the reconnect
    failure) — that's what tells the caller why their call failed."""

    async def boom(name: str, args: dict[str, Any]) -> Any:
        raise RuntimeError("original cause")

    s = _FakeMcpSession([_FakeMcpTool(name="t")], call_handler=boom)

    class _ReconnectRegistry(MCPRegistry):
        def _make_client(self, spec: MCPServerSpec) -> MCPClient:
            # Build a client whose connect() will raise (no session,
            # bogus command so the real transport path errors out).
            return MCPClient(spec)

    reg = _ReconnectRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=s)]
    )
    r = await reg.call("t", {}, call_id="c1")
    assert not r.ok
    assert r.error is not None
    assert "original cause" in r.error
