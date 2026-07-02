"""Regression tests for the WSF1 review fixes in ``loomflow/mcp/``.

Covers:

* Cancel-scope task affinity: the transport/session context managers
  are entered AND exited by one dedicated owning task, so ``connect``
  and ``aclose`` may run in arbitrary (different) tasks.
* MCP tool annotations (``destructiveHint`` / ``readOnlyHint``) map
  onto :attr:`ToolDef.destructive`.
* Reconnect-and-retry fires only on connection-phase errors, never on
  tool-execution errors (which may have already caused a side effect).
* ``refresh()`` runs after a successful reconnect so the index isn't
  stale.
* Both the bare and the ``server.tool``-qualified names are accepted
  at call time, without duplicating ``list_tools`` output.
* Image / binary content blocks are represented in the output rather
  than silently dropped.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from loomflow.mcp import MCPClient, MCPRegistry, MCPServerSpec

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fakes (mirroring tests/test_mcp.py)
# ---------------------------------------------------------------------------


@dataclass
class _FakeMcpTool:
    name: str
    description: str = ""
    inputSchema: dict[str, Any] = field(default_factory=dict)
    annotations: Any = None


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


class _ScopedTransportClient(MCPClient):
    """A client whose fake 'transport' wraps an anyio ``CancelScope``,
    reproducing the task-affinity constraint of the real MCP SDK
    transports: if the context manager is exited by a different task
    than the one that entered it, anyio raises
    ``RuntimeError: Attempted to exit cancel scope in a different task``.
    """

    def __init__(self, spec: MCPServerSpec, fake_session: _FakeMcpSession) -> None:
        super().__init__(spec)
        self._fake = fake_session
        self.scope_exited = False

    async def _open_session(self, stack: AsyncExitStack) -> Any:
        @asynccontextmanager
        async def scoped() -> AsyncIterator[_FakeMcpSession]:
            with anyio.CancelScope():
                try:
                    yield self._fake
                finally:
                    self.scope_exited = True

        session = await stack.enter_async_context(scoped())
        await session.initialize()
        return session


# ---------------------------------------------------------------------------
# Fix 1 — lifecycle task affinity
# ---------------------------------------------------------------------------


async def test_client_connect_and_close_from_different_tasks() -> None:
    """connect() in a task-group child, use + aclose() from the test
    task — exactly the topology MCPRegistry produces. With cancel
    scopes inside the transport CM this only works when ONE owning
    task enters and exits the stack."""
    fake = _FakeMcpSession([_FakeMcpTool(name="ping")])
    client = _ScopedTransportClient(MCPServerSpec.stdio("s", "noop"), fake)

    async with anyio.create_task_group() as tg:
        tg.start_soon(client.connect)  # child task

    assert client.is_connected
    assert fake.initialized

    tools = await client.list_tools()
    assert [t.name for t in tools] == ["ping"]
    result = await client.call_tool("ping", {"a": 1})
    assert result.content[0].text == "called ping"

    await client.aclose()  # different task than the connecting child
    assert not client.is_connected
    assert client.scope_exited


async def test_registry_lifecycle_across_tasks_with_scoped_transports() -> None:
    """The registry connects clients in task-group CHILD tasks and
    closes them later from the caller's task; with cancel-scoped
    transports the whole cycle must still unwind cleanly."""
    f1 = _FakeMcpSession([_FakeMcpTool(name="alpha_tool")])
    f2 = _FakeMcpSession([_FakeMcpTool(name="beta_tool")])
    c1 = _ScopedTransportClient(MCPServerSpec.stdio("alpha", "noop"), f1)
    c2 = _ScopedTransportClient(MCPServerSpec.stdio("beta", "noop"), f2)
    reg = MCPRegistry([c1, c2])

    async with reg:
        r = await reg.call("beta_tool", {"x": 1}, call_id="c1")
        assert r.ok, r.error
        assert f2.calls == [("beta_tool", {"x": 1})]

    assert c1.scope_exited
    assert c2.scope_exited
    assert not c1.is_connected
    assert not c2.is_connected


async def test_client_reconnect_after_close() -> None:
    """A closed client can connect again with a fresh lifecycle."""
    fake = _FakeMcpSession([_FakeMcpTool(name="ping")])
    client = _ScopedTransportClient(MCPServerSpec.stdio("s", "noop"), fake)
    await client.connect()
    await client.aclose()
    assert not client.is_connected
    await client.connect()
    assert client.is_connected
    result = await client.call_tool("ping", {})
    assert result.content[0].text == "called ping"
    await client.aclose()


# ---------------------------------------------------------------------------
# Fix 2 — annotations → ToolDef.destructive
# ---------------------------------------------------------------------------


async def test_destructive_annotations_map_to_tooldef() -> None:
    tools = [
        _FakeMcpTool(
            name="rm",
            annotations=SimpleNamespace(destructiveHint=True, readOnlyHint=False),
        ),
        # destructiveHint takes precedence over readOnlyHint=False
        _FakeMcpTool(
            name="append",
            annotations=SimpleNamespace(destructiveHint=False, readOnlyHint=False),
        ),
        # weaker signal: no destructiveHint, but explicitly not read-only
        _FakeMcpTool(
            name="mutate",
            annotations=SimpleNamespace(destructiveHint=None, readOnlyHint=False),
        ),
        _FakeMcpTool(
            name="peek",
            annotations=SimpleNamespace(destructiveHint=None, readOnlyHint=True),
        ),
        _FakeMcpTool(name="plain"),  # no annotations at all
    ]
    session = _FakeMcpSession(tools)
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=session)]
    )
    by_name = {d.name: d for d in await reg.list_tools()}
    assert by_name["rm"].destructive is True
    assert by_name["append"].destructive is False
    assert by_name["mutate"].destructive is True
    assert by_name["peek"].destructive is False
    assert by_name["plain"].destructive is False


async def test_destructive_annotations_as_mapping() -> None:
    """Some SDK versions hand annotations back as plain dicts."""
    session = _FakeMcpSession(
        [_FakeMcpTool(name="rm", annotations={"destructiveHint": True})]
    )
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=session)]
    )
    (tool_def,) = await reg.list_tools()
    assert tool_def.destructive is True


# ---------------------------------------------------------------------------
# Fix 3a — retry only on connection-phase errors
# ---------------------------------------------------------------------------


async def test_execution_error_does_not_trigger_reconnect_retry() -> None:
    """A tool-execution error (the server may already have run the
    side effect) must surface immediately — no reconnect, no silent
    second execution."""

    async def boom(name: str, args: dict[str, Any]) -> Any:
        raise RuntimeError("tool blew up")

    session = _FakeMcpSession([_FakeMcpTool(name="t")], call_handler=boom)
    resets: list[str] = []

    class _Reg(MCPRegistry):
        def _make_client(self, spec: MCPServerSpec) -> MCPClient:
            resets.append(spec.name)
            return MCPClient(spec, session=session)

    reg = _Reg([MCPClient(MCPServerSpec.stdio("only", "noop"), session=session)])
    r = await reg.call("t", {}, call_id="c1")
    assert not r.ok
    assert r.error is not None
    assert "tool blew up" in r.error
    assert resets == []  # reconnect path never engaged
    assert len(session.calls) == 1  # executed exactly once


async def test_connection_error_still_triggers_one_retry() -> None:
    async def pipe_broke(name: str, args: dict[str, Any]) -> Any:
        raise ConnectionError("pipe closed")

    broken = _FakeMcpSession([_FakeMcpTool(name="probe")], call_handler=pipe_broke)
    healed = _FakeMcpSession([_FakeMcpTool(name="probe")])

    class _Reg(MCPRegistry):
        def _make_client(self, spec: MCPServerSpec) -> MCPClient:
            return MCPClient(spec, session=healed)

    reg = _Reg([MCPClient(MCPServerSpec.stdio("only", "noop"), session=broken)])
    r = await reg.call("probe", {"x": 1}, call_id="c1")
    assert r.ok, r.error
    assert broken.calls == [("probe", {"x": 1})]
    assert healed.calls == [("probe", {"x": 1})]


# ---------------------------------------------------------------------------
# Fix 3b — refresh() after a successful reconnect
# ---------------------------------------------------------------------------


async def test_reconnect_refreshes_tool_index() -> None:
    """The restarted server exposes a new tool; after the self-heal
    the registry's index must include it (no stale index)."""

    async def pipe_broke(name: str, args: dict[str, Any]) -> Any:
        raise ConnectionError("pipe closed")

    broken = _FakeMcpSession([_FakeMcpTool(name="probe")], call_handler=pipe_broke)
    healed = _FakeMcpSession(
        [_FakeMcpTool(name="probe"), _FakeMcpTool(name="brand_new")]
    )

    class _Reg(MCPRegistry):
        def _make_client(self, spec: MCPServerSpec) -> MCPClient:
            return MCPClient(spec, session=healed)

    reg = _Reg([MCPClient(MCPServerSpec.stdio("only", "noop"), session=broken)])
    r = await reg.call("probe", {}, call_id="c1")
    assert r.ok, r.error
    names = {d.name for d in await reg.list_tools()}
    assert "brand_new" in names
    # ...and the new tool is actually callable.
    r2 = await reg.call("brand_new", {}, call_id="c2")
    assert r2.ok


# ---------------------------------------------------------------------------
# Fix 3c — bare AND qualified names accepted at call time
# ---------------------------------------------------------------------------


async def test_unique_tool_callable_by_bare_and_qualified_name() -> None:
    session = _FakeMcpSession([_FakeMcpTool(name="probe")])
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=session)]
    )
    r1 = await reg.call("probe", {}, call_id="a")
    r2 = await reg.call("only.probe", {}, call_id="b")
    assert r1.ok and r2.ok
    # Both routed to the server under its unqualified name.
    assert session.calls == [("probe", {}), ("probe", {})]


async def test_dual_keys_do_not_duplicate_list_tools() -> None:
    session = _FakeMcpSession([_FakeMcpTool(name="probe")])
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=session)]
    )
    defs = await reg.list_tools()
    assert [d.name for d in defs] == ["probe"]


async def test_colliding_names_only_qualified() -> None:
    s1 = _FakeMcpSession([_FakeMcpTool(name="search")])
    s2 = _FakeMcpSession([_FakeMcpTool(name="search")])
    reg = MCPRegistry(
        [
            MCPClient(MCPServerSpec.stdio("alpha", "noop"), session=s1),
            MCPClient(MCPServerSpec.stdio("beta", "noop"), session=s2),
        ]
    )
    # The bare name is ambiguous — it must NOT silently pick a server.
    r = await reg.call("search", {}, call_id="c1")
    assert not r.ok
    assert r.error is not None and "unknown" in r.error.lower()
    r2 = await reg.call("alpha.search", {}, call_id="c2")
    assert r2.ok
    assert s1.calls == [("search", {})]
    assert s2.calls == []


# ---------------------------------------------------------------------------
# Fix 4 — image / binary content blocks aren't dropped
# ---------------------------------------------------------------------------


async def test_image_block_represented_in_output() -> None:
    image_block = SimpleNamespace(
        type="image", data="aGVsbG8=", mimeType="image/png"  # "hello", 5 bytes
    )

    async def handler(name: str, args: dict[str, Any]) -> _FakeCallResult:
        return _FakeCallResult(content=[image_block])

    session = _FakeMcpSession([_FakeMcpTool(name="shot")], call_handler=handler)
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=session)]
    )
    r = await reg.call("shot", {}, call_id="c1")
    assert r.ok
    assert r.output == "[image: image/png, 5 bytes]"


async def test_mixed_text_and_binary_blocks() -> None:
    image_block = SimpleNamespace(
        type="image", data="aGVsbG8=", mimeType="image/png"
    )
    blob_block = SimpleNamespace(
        type="resource",
        resource=SimpleNamespace(blob="AAAA", mimeType="application/octet-stream"),
    )

    async def handler(name: str, args: dict[str, Any]) -> _FakeCallResult:
        return _FakeCallResult(
            content=[_FakeContent(text="see attachment"), image_block, blob_block]
        )

    session = _FakeMcpSession([_FakeMcpTool(name="shot")], call_handler=handler)
    reg = MCPRegistry(
        [MCPClient(MCPServerSpec.stdio("only", "noop"), session=session)]
    )
    r = await reg.call("shot", {}, call_id="c1")
    assert r.ok
    assert r.output == (
        "see attachment\n"
        "[image: image/png, 5 bytes]\n"
        "[resource: application/octet-stream, 3 bytes]"
    )
