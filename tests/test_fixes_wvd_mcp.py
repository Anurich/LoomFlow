"""Regression tests for the WVD review fixes in ``loomflow/mcp/registry.py``.

Covers:

* ``aclose()`` fault isolation — one client raising during close must
  not cancel the sibling closes; errors are aggregated into a single
  ``ExceptionGroup`` after every client has been closed.
* ``refresh()`` fault isolation — one server whose ``list_tools``
  raises is skipped (recorded in ``registry.unavailable``) while the
  other servers stay fully usable.
* ``connect()`` fault isolation — a client whose connect raises does
  not take the sibling connects down with it.
* Targeted refresh after a reconnect — ``_reset_client`` re-lists
  ONLY the reset server; healthy servers' cached tool lists are
  reused (list_tools call counts on the fakes prove it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from loomflow.mcp import MCPClient, MCPRegistry, MCPServerSpec

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fakes (mirroring tests/test_mcp.py, plus list_tools call counting)
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
    content: list[Any] = field(default_factory=list)
    isError: bool = False
    structuredContent: Any = None


class _FakeMcpSession:
    def __init__(
        self,
        tools: list[_FakeMcpTool],
        call_handler: Any | None = None,
    ) -> None:
        self._tools = tools
        self._handler = call_handler
        self.initialized = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.list_tools_calls = 0

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> _FakeListResult:
        self.list_tools_calls += 1
        return _FakeListResult(tools=list(self._tools))

    async def call_tool(self, name: str, args: dict[str, Any]) -> _FakeCallResult:
        self.calls.append((name, dict(args)))
        if self._handler is not None:
            result = await self._handler(name, args)
            if isinstance(result, _FakeCallResult):
                return result
            return _FakeCallResult(content=[_FakeContent(text=str(result))])
        return _FakeCallResult(content=[_FakeContent(text=f"called {name}")])


class _FlakyListSession(_FakeMcpSession):
    """``list_tools`` raises while ``broken`` is True."""

    def __init__(self, tools: list[_FakeMcpTool]) -> None:
        super().__init__(tools)
        self.broken = True

    async def list_tools(self) -> _FakeListResult:
        self.list_tools_calls += 1
        if self.broken:
            raise RuntimeError("server exploded during list_tools")
        return _FakeListResult(tools=list(self._tools))


class _ExplodingCloseClient(MCPClient):
    """A client whose ``aclose`` raises after recording the attempt."""

    def __init__(self, spec: MCPServerSpec, session: _FakeMcpSession) -> None:
        super().__init__(spec, session=session)
        self.close_attempts = 0

    async def aclose(self) -> None:
        self.close_attempts += 1
        raise RuntimeError(f"boom closing {self.name}")


class _TrackingCloseClient(MCPClient):
    """A client that records whether its ``aclose`` ran to completion."""

    def __init__(self, spec: MCPServerSpec, session: _FakeMcpSession) -> None:
        super().__init__(spec, session=session)
        self.closed = False

    async def aclose(self) -> None:
        await super().aclose()
        self.closed = True


class _ExplodingConnectClient(MCPClient):
    """A client whose ``connect`` always raises (dead server)."""

    async def connect(self) -> None:
        raise ConnectionError(f"cannot reach {self.name}")


# ---------------------------------------------------------------------------
# Fix 1 — aclose() fault isolation
# ---------------------------------------------------------------------------


async def test_aclose_with_one_failing_client_still_closes_others() -> None:
    bad = _ExplodingCloseClient(
        MCPServerSpec.stdio("bad", "noop"),
        _FakeMcpSession([_FakeMcpTool(name="bad_tool")]),
    )
    good = _TrackingCloseClient(
        MCPServerSpec.stdio("good", "noop"),
        _FakeMcpSession([_FakeMcpTool(name="good_tool")]),
    )
    # Exploding client FIRST so, pre-fix, its exception would cancel
    # the sibling close mid-teardown.
    reg = MCPRegistry([bad, good])
    await reg.connect()

    with pytest.raises(ExceptionGroup) as excinfo:
        await reg.aclose()

    # The good client was still closed, the error was collected.
    assert good.closed
    assert bad.close_attempts == 1
    assert len(excinfo.value.exceptions) == 1
    assert "boom closing bad" in str(excinfo.value.exceptions[0])
    # Registry state is reset even on a failing close.
    assert reg._tool_index == {}
    assert reg._connected is False


async def test_aclose_aggregates_multiple_failures() -> None:
    bad1 = _ExplodingCloseClient(
        MCPServerSpec.stdio("bad1", "noop"),
        _FakeMcpSession([_FakeMcpTool(name="t1")]),
    )
    bad2 = _ExplodingCloseClient(
        MCPServerSpec.stdio("bad2", "noop"),
        _FakeMcpSession([_FakeMcpTool(name="t2")]),
    )
    good = _TrackingCloseClient(
        MCPServerSpec.stdio("good", "noop"),
        _FakeMcpSession([_FakeMcpTool(name="t3")]),
    )
    reg = MCPRegistry([bad1, good, bad2])
    await reg.connect()

    with pytest.raises(ExceptionGroup) as excinfo:
        await reg.aclose()

    assert good.closed
    assert bad1.close_attempts == 1
    assert bad2.close_attempts == 1
    assert len(excinfo.value.exceptions) == 2


async def test_aclose_without_failures_raises_nothing() -> None:
    good = _TrackingCloseClient(
        MCPServerSpec.stdio("good", "noop"),
        _FakeMcpSession([_FakeMcpTool(name="t")]),
    )
    reg = MCPRegistry([good])
    await reg.connect()
    await reg.aclose()
    assert good.closed


# ---------------------------------------------------------------------------
# Fix 2 — refresh() fault isolation + registry.unavailable
# ---------------------------------------------------------------------------


async def test_refresh_with_one_failing_server_keeps_others_usable() -> None:
    good_session = _FakeMcpSession([_FakeMcpTool(name="good_tool")])
    bad_session = _FlakyListSession([_FakeMcpTool(name="bad_tool")])
    reg = MCPRegistry(
        [
            MCPClient(MCPServerSpec.stdio("bad", "noop"), session=bad_session),
            MCPClient(MCPServerSpec.stdio("good", "noop"), session=good_session),
        ]
    )

    # Pre-fix: connect() (via refresh) raised and made EVERYTHING
    # unusable. Post-fix: the healthy server's tools are listed.
    defs = await reg.list_tools()
    assert {d.name for d in defs} == {"good_tool"}
    assert reg.unavailable == {"bad"}

    # ...and calls to the healthy server work.
    r = await reg.call("good_tool", {"x": 1}, call_id="c1")
    assert r.ok, r.error
    assert good_session.calls == [("good_tool", {"x": 1})]


async def test_failed_server_recovers_on_next_refresh() -> None:
    good_session = _FakeMcpSession([_FakeMcpTool(name="good_tool")])
    bad_session = _FlakyListSession([_FakeMcpTool(name="bad_tool")])
    reg = MCPRegistry(
        [
            MCPClient(MCPServerSpec.stdio("bad", "noop"), session=bad_session),
            MCPClient(MCPServerSpec.stdio("good", "noop"), session=good_session),
        ]
    )
    await reg.connect()
    assert reg.unavailable == {"bad"}

    bad_session.broken = False
    await reg.refresh()
    assert reg.unavailable == set()
    names = {d.name for d in await reg.list_tools()}
    assert names == {"good_tool", "bad_tool"}
    r = await reg.call("bad_tool", {}, call_id="c1")
    assert r.ok


async def test_connect_with_one_dead_server_keeps_others_usable() -> None:
    good_session = _FakeMcpSession([_FakeMcpTool(name="good_tool")])
    reg = MCPRegistry(
        [
            _ExplodingConnectClient(MCPServerSpec.stdio("dead", "noop")),
            MCPClient(MCPServerSpec.stdio("good", "noop"), session=good_session),
        ]
    )
    defs = await reg.list_tools()
    assert {d.name for d in defs} == {"good_tool"}
    # The dead server's list_tools pull (which re-attempts connect)
    # failed, so it lands in unavailable.
    assert reg.unavailable == {"dead"}
    r = await reg.call("good_tool", {}, call_id="c1")
    assert r.ok


# ---------------------------------------------------------------------------
# Fix 3 — targeted refresh after a client reset
# ---------------------------------------------------------------------------


async def test_reset_refreshes_only_the_reset_server() -> None:
    async def pipe_broke(name: str, args: dict[str, Any]) -> Any:
        raise ConnectionError("pipe closed")

    broken = _FakeMcpSession(
        [_FakeMcpTool(name="probe")], call_handler=pipe_broke
    )
    healed = _FakeMcpSession(
        [_FakeMcpTool(name="probe"), _FakeMcpTool(name="brand_new")]
    )
    bystander = _FakeMcpSession([_FakeMcpTool(name="beta_tool")])

    class _Reg(MCPRegistry):
        def _make_client(self, spec: MCPServerSpec) -> MCPClient:
            return MCPClient(spec, session=healed)

    reg = _Reg(
        [
            MCPClient(MCPServerSpec.stdio("alpha", "noop"), session=broken),
            MCPClient(MCPServerSpec.stdio("beta", "noop"), session=bystander),
        ]
    )

    # Initial connect: one pull per server.
    await reg.connect()
    assert broken.list_tools_calls == 1
    assert bystander.list_tools_calls == 1

    # Transport error → reset → targeted refresh of ONLY "alpha".
    r = await reg.call("probe", {}, call_id="c1")
    assert r.ok, r.error
    assert healed.list_tools_calls == 1  # the reset server re-listed
    assert bystander.list_tools_calls == 1  # the healthy one was NOT

    # The rebuilt index includes the reset server's new tool AND the
    # healthy server's cached tools; both are callable.
    names = {d.name for d in await reg.list_tools()}
    assert "brand_new" in names
    assert "beta_tool" in names
    r2 = await reg.call("brand_new", {}, call_id="c2")
    assert r2.ok
    r3 = await reg.call("beta_tool", {}, call_id="c3")
    assert r3.ok
    # Still no extra pull on the healthy server.
    assert bystander.list_tools_calls == 1
