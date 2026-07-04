"""G14 — deep MCP surfaces: resources, prompts, listChanged, sampling.

Uses the fake-session patterns from ``tests/test_mcp.py`` (no real MCP
server needed). Only the sampling round-trip tests import ``mcp.types``
(skipped when the SDK isn't installed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from loomflow.core.errors import MCPError
from loomflow.mcp import MCPClient, MCPRegistry, MCPServerSpec

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fakes (mirroring tests/test_mcp.py, extended with resources/prompts)
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


def _text_resource(uri: str, text: str, mime: str = "text/plain") -> Any:
    return SimpleNamespace(uri=uri, mimeType=mime, text=text)


def _blob_resource(uri: str, blob: str, mime: str) -> Any:
    # No ``text`` attribute — mimics BlobResourceContents.
    return SimpleNamespace(uri=uri, mimeType=mime, blob=blob)


class _FakeMcpSession:
    """Enough of ``mcp.ClientSession`` for the deep-MCP surfaces."""

    def __init__(
        self,
        tools: list[_FakeMcpTool] | None = None,
        call_handler: Any | None = None,
        resources: list[Any] | None = None,
        resource_contents: dict[str, list[Any]] | None = None,
        read_handler: Any | None = None,
        prompts: list[Any] | None = None,
        prompt_results: dict[str, Any] | None = None,
    ) -> None:
        self._tools = tools or []
        self._handler = call_handler
        self._resources = resources or []
        self._resource_contents = resource_contents or {}
        self._read_handler = read_handler
        self._prompts = prompts or []
        self._prompt_results = prompt_results or {}
        self.initialized = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.list_tools_calls = 0
        self.list_resources_calls = 0
        self.reads: list[str] = []
        self.list_prompts_calls = 0
        self.prompt_gets: list[tuple[str, dict[str, str] | None]] = []

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

    async def list_resources(self) -> Any:
        self.list_resources_calls += 1
        return SimpleNamespace(resources=list(self._resources))

    async def read_resource(self, uri: str) -> Any:
        self.reads.append(str(uri))
        if self._read_handler is not None:
            return await self._read_handler(uri)
        return SimpleNamespace(contents=self._resource_contents.get(str(uri), []))

    async def list_prompts(self) -> Any:
        self.list_prompts_calls += 1
        return SimpleNamespace(prompts=list(self._prompts))

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> Any:
        self.prompt_gets.append((name, arguments))
        result = self._prompt_results.get(name)
        if result is None:
            raise RuntimeError(f"no such prompt: {name}")
        return result


def _client(name: str, session: _FakeMcpSession) -> MCPClient:
    return MCPClient(MCPServerSpec.stdio(name, "noop"), session=session)


_LIST_CHANGED = SimpleNamespace(
    root=SimpleNamespace(method="notifications/tools/list_changed")
)


# ---------------------------------------------------------------------------
# Resources — client wrappers + registry aggregation/routing
# ---------------------------------------------------------------------------


async def test_client_list_and_read_resources_pass_through() -> None:
    session = _FakeMcpSession(
        resources=[SimpleNamespace(uri="mem://a", name="a")],
        resource_contents={"mem://a": [_text_resource("mem://a", "alpha body")]},
    )
    client = _client("srv", session)
    resources = await client.list_resources()
    assert [str(r.uri) for r in resources] == ["mem://a"]
    result = await client.read_resource("mem://a")
    assert result.contents[0].text == "alpha body"
    assert session.reads == ["mem://a"]


async def test_registry_lists_resources_across_servers() -> None:
    s1 = _FakeMcpSession(
        resources=[
            SimpleNamespace(
                uri="mem://one", name="one", description="first", mimeType="text/plain"
            )
        ]
    )
    s2 = _FakeMcpSession(resources=[SimpleNamespace(uri="mem://two", name="two")])
    reg = MCPRegistry([_client("alpha", s1), _client("beta", s2)])

    listed = await reg.list_resources()
    by_uri = {r["uri"]: r for r in listed}
    # Unique URIs stay bare.
    assert set(by_uri) == {"mem://one", "mem://two"}
    assert by_uri["mem://one"]["server"] == "alpha"
    assert by_uri["mem://one"]["description"] == "first"
    assert by_uri["mem://one"]["mime_type"] == "text/plain"
    assert by_uri["mem://two"]["server"] == "beta"


async def test_duplicate_resource_uri_is_server_qualified() -> None:
    s1 = _FakeMcpSession(resources=[SimpleNamespace(uri="mem://cfg", name="c")])
    s2 = _FakeMcpSession(resources=[SimpleNamespace(uri="mem://cfg", name="c")])
    reg = MCPRegistry([_client("alpha", s1), _client("beta", s2)])
    uris = {r["uri"] for r in await reg.list_resources()}
    assert uris == {"alpha:mem://cfg", "beta:mem://cfg"}


async def test_read_resource_returns_text_verbatim() -> None:
    session = _FakeMcpSession(
        resources=[SimpleNamespace(uri="mem://doc", name="doc")],
        resource_contents={"mem://doc": [_text_resource("mem://doc", "hello world")]},
    )
    reg = MCPRegistry([_client("only", session)])
    assert await reg.read_resource("mem://doc") == "hello world"


async def test_read_resource_blob_becomes_placeholder_dict() -> None:
    session = _FakeMcpSession(
        resources=[SimpleNamespace(uri="mem://img", name="img")],
        resource_contents={
            # "aGVsbG8=" decodes to "hello" — 5 bytes.
            "mem://img": [_blob_resource("mem://img", "aGVsbG8=", "image/png")]
        },
    )
    reg = MCPRegistry([_client("only", session)])
    assert await reg.read_resource("mem://img") == {"mime": "image/png", "size": 5}


async def test_read_resource_mixed_contents_returns_list() -> None:
    session = _FakeMcpSession(
        resources=[SimpleNamespace(uri="mem://both", name="both")],
        resource_contents={
            "mem://both": [
                _text_resource("mem://both", "caption"),
                _blob_resource("mem://both", "AAAA", "application/octet-stream"),
            ]
        },
    )
    reg = MCPRegistry([_client("only", session)])
    assert await reg.read_resource("mem://both") == [
        "caption",
        {"mime": "application/octet-stream", "size": 3},
    ]


async def test_read_resource_qualified_uri_routes_to_named_server() -> None:
    body = [_text_resource("mem://cfg", "from beta")]
    s1 = _FakeMcpSession(resources=[SimpleNamespace(uri="mem://cfg", name="c")])
    s2 = _FakeMcpSession(
        resources=[SimpleNamespace(uri="mem://cfg", name="c")],
        resource_contents={"mem://cfg": body},
    )
    reg = MCPRegistry([_client("alpha", s1), _client("beta", s2)])
    assert await reg.read_resource("beta:mem://cfg") == "from beta"
    assert s1.reads == []
    assert s2.reads == ["mem://cfg"]  # bare URI forwarded to the session


async def test_read_resource_ambiguous_bare_uri_raises() -> None:
    s1 = _FakeMcpSession(resources=[SimpleNamespace(uri="mem://cfg", name="c")])
    s2 = _FakeMcpSession(resources=[SimpleNamespace(uri="mem://cfg", name="c")])
    reg = MCPRegistry([_client("alpha", s1), _client("beta", s2)])
    with pytest.raises(MCPError, match="multiple servers"):
        await reg.read_resource("mem://cfg")


async def test_read_unknown_resource_raises() -> None:
    session = _FakeMcpSession(resources=[SimpleNamespace(uri="mem://a", name="a")])
    reg = MCPRegistry([_client("only", session)])
    with pytest.raises(MCPError, match="unknown MCP resource"):
        await reg.read_resource("mem://ghost")


async def test_read_resource_explicit_server_bypasses_listing() -> None:
    """server= lets callers read URIs the server never listed
    (e.g. resource-template expansions)."""
    session = _FakeMcpSession(
        resource_contents={"mem://dyn/42": [_text_resource("mem://dyn/42", "row 42")]},
    )
    reg = MCPRegistry([_client("only", session)])
    assert await reg.read_resource("mem://dyn/42", server="only") == "row 42"
    with pytest.raises(MCPError, match="unknown MCP server"):
        await reg.read_resource("mem://dyn/42", server="ghost")


async def test_read_resource_retries_after_connection_error() -> None:
    """Same reconnect-and-retry gating as call(): a transport-style
    failure resets the client and retries once."""

    async def pipe_broke(uri: str) -> Any:
        raise ConnectionError("pipe closed")

    broken = _FakeMcpSession(
        resources=[SimpleNamespace(uri="mem://a", name="a")],
        read_handler=pipe_broke,
    )
    healed = _FakeMcpSession(
        resources=[SimpleNamespace(uri="mem://a", name="a")],
        resource_contents={"mem://a": [_text_resource("mem://a", "healed")]},
    )

    class _Reg(MCPRegistry):
        def _make_client(self, spec: MCPServerSpec) -> MCPClient:
            return MCPClient(spec, session=healed)

    reg = _Reg([_client("only", broken)])
    assert await reg.read_resource("mem://a") == "healed"
    assert broken.reads == ["mem://a"]
    assert healed.reads == ["mem://a"]


# ---------------------------------------------------------------------------
# Prompts — list/get with tool-style naming
# ---------------------------------------------------------------------------


def _prompt(name: str, description: str = "", arguments: Any = None) -> Any:
    return SimpleNamespace(name=name, description=description, arguments=arguments)


def _prompt_result(description: str, *texts: tuple[str, str]) -> Any:
    return SimpleNamespace(
        description=description,
        messages=[
            SimpleNamespace(role=role, content=SimpleNamespace(type="text", text=text))
            for role, text in texts
        ],
    )


async def test_list_prompts_bare_when_unique_qualified_on_collision() -> None:
    s1 = _FakeMcpSession(prompts=[_prompt("summarize"), _prompt("greet")])
    s2 = _FakeMcpSession(prompts=[_prompt("summarize")])
    reg = MCPRegistry([_client("alpha", s1), _client("beta", s2)])
    listed = await reg.list_prompts()
    names = {p["name"] for p in listed}
    assert names == {"alpha.summarize", "beta.summarize", "greet"}
    by_name = {p["name"]: p for p in listed}
    assert by_name["greet"]["server"] == "alpha"


async def test_get_prompt_routes_and_flattens_messages() -> None:
    session = _FakeMcpSession(
        prompts=[_prompt("greet", description="say hi")],
        prompt_results={
            "greet": _prompt_result("say hi", ("user", "Hello, Ada!"))
        },
    )
    reg = MCPRegistry([_client("only", session)])
    got = await reg.get_prompt("greet", {"name": "Ada"})
    assert got == {
        "description": "say hi",
        "messages": [{"role": "user", "content": "Hello, Ada!"}],
    }
    assert session.prompt_gets == [("greet", {"name": "Ada"})]


async def test_get_prompt_qualified_name_routes_on_collision() -> None:
    s1 = _FakeMcpSession(
        prompts=[_prompt("summarize")],
        prompt_results={"summarize": _prompt_result("a", ("user", "from alpha"))},
    )
    s2 = _FakeMcpSession(
        prompts=[_prompt("summarize")],
        prompt_results={"summarize": _prompt_result("b", ("user", "from beta"))},
    )
    reg = MCPRegistry([_client("alpha", s1), _client("beta", s2)])
    got = await reg.get_prompt("beta.summarize")
    assert got["messages"][0]["content"] == "from beta"
    # Bare (ambiguous) name must not silently pick a server.
    with pytest.raises(MCPError, match="unknown MCP prompt"):
        await reg.get_prompt("summarize")
    assert s1.prompt_gets == []
    assert s2.prompt_gets == [("summarize", None)]


async def test_get_unknown_prompt_raises() -> None:
    reg = MCPRegistry([_client("only", _FakeMcpSession(prompts=[_prompt("p")]))])
    with pytest.raises(MCPError, match="unknown MCP prompt"):
        await reg.get_prompt("ghost")


# ---------------------------------------------------------------------------
# listChanged notification → targeted re-pull
# ---------------------------------------------------------------------------


async def test_list_changed_notification_triggers_targeted_repull() -> None:
    """The client's message handler flags the server; the registry
    re-lists ONLY that server on its next operation."""
    s1 = _FakeMcpSession(tools=[_FakeMcpTool(name="old_tool")])
    s2 = _FakeMcpSession(tools=[_FakeMcpTool(name="stable_tool")])
    c1 = _client("alpha", s1)
    c2 = _client("beta", s2)
    reg = MCPRegistry([c1, c2])
    await reg.connect()
    assert s1.list_tools_calls == 1
    assert s2.list_tools_calls == 1

    # Server grows a tool and announces it.
    s1._tools.append(_FakeMcpTool(name="new_tool"))
    await c1._handle_incoming_message(_LIST_CHANGED)

    names = {d.name for d in await reg.list_tools()}
    assert "new_tool" in names
    assert s1.list_tools_calls == 2  # alpha re-listed
    assert s2.list_tools_calls == 1  # beta untouched
    await reg.aclose()


async def test_non_list_changed_messages_are_ignored() -> None:
    session = _FakeMcpSession(tools=[_FakeMcpTool(name="t")])
    client = _client("only", session)
    reg = MCPRegistry([client])
    await reg.connect()
    await client._handle_incoming_message(
        SimpleNamespace(root=SimpleNamespace(method="notifications/progress"))
    )
    await reg.list_tools()
    assert session.list_tools_calls == 1  # no spurious re-pull
    await reg.aclose()


async def test_mark_stale_is_safe_without_deadlock_from_foreign_thread() -> None:
    """The notification callback only flips a lock-guarded flag, so it
    is safe to invoke from any thread (the portal thread in prod)."""
    session = _FakeMcpSession(tools=[_FakeMcpTool(name="t")])
    client = _client("only", session)
    reg = MCPRegistry([client])
    await reg.connect()

    def _fire_from_thread() -> None:
        client._fire_tools_changed()

    await anyio.to_thread.run_sync(_fire_from_thread)
    await reg.list_tools()
    assert session.list_tools_calls == 2
    await reg.aclose()


async def test_watch_yields_event_after_list_changed_refresh() -> None:
    session = _FakeMcpSession(tools=[_FakeMcpTool(name="old_tool")])
    client = _client("only", session)
    reg = MCPRegistry([client])
    await reg.connect()

    # Subscription is registered eagerly at watch() call time, so the
    # event emitted by the refresh below is buffered for us.
    watcher = reg.watch()
    session._tools.append(_FakeMcpTool(name="new_tool"))
    await client._handle_incoming_message(_LIST_CHANGED)
    await reg.list_tools()  # drains the stale flag → refresh → event

    with anyio.fail_after(5):
        event = await watcher.__anext__()
    assert (event.kind, event.tool, event.server) == ("added", "new_tool", "only")
    await watcher.aclose()
    await reg.aclose()


async def test_watch_ends_when_registry_closes() -> None:
    reg = MCPRegistry([_client("only", _FakeMcpSession(tools=[]))])
    await reg.connect()
    watcher = reg.watch()
    await reg.aclose()
    with anyio.fail_after(5):
        with pytest.raises(StopAsyncIteration):
            await watcher.__anext__()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


async def test_sampling_handler_invoked_on_server_request() -> None:
    """The wrapped SDK callback forwards messages/preferences to the
    user handler and packages the reply as a CreateMessageResult."""
    types = pytest.importorskip("mcp.types")
    seen: dict[str, Any] = {}

    def handler(messages: Any, model_preferences: Any) -> str:
        seen["messages"] = messages
        seen["prefs"] = model_preferences
        return "the completion"

    client = MCPClient(
        MCPServerSpec.stdio("srv", "noop"),
        session=_FakeMcpSession(),
        sampling_handler=handler,
    )
    callback = client._make_sampling_callback()
    params = SimpleNamespace(messages=["m1"], modelPreferences={"hint": "small"})
    result = await callback(None, params)  # the SDK invokes it like this
    assert isinstance(result, types.CreateMessageResult)
    assert result.content.text == "the completion"
    assert result.role == "assistant"
    assert seen == {"messages": ["m1"], "prefs": {"hint": "small"}}


async def test_async_sampling_handler_supported() -> None:
    pytest.importorskip("mcp.types")

    async def handler(messages: Any, model_preferences: Any) -> str:
        return "async reply"

    client = MCPClient(
        MCPServerSpec.stdio("srv", "noop"),
        session=_FakeMcpSession(),
        sampling_handler=handler,
    )
    result = await client._make_sampling_callback()(
        None, SimpleNamespace(messages=[], modelPreferences=None)
    )
    assert result.content.text == "async reply"


async def test_sampling_handler_error_returned_as_error_data() -> None:
    types = pytest.importorskip("mcp.types")

    def handler(messages: Any, model_preferences: Any) -> str:
        raise RuntimeError("no model available")

    client = MCPClient(
        MCPServerSpec.stdio("srv", "noop"),
        session=_FakeMcpSession(),
        sampling_handler=handler,
    )
    result = await client._make_sampling_callback()(
        None, SimpleNamespace(messages=[], modelPreferences=None)
    )
    assert isinstance(result, types.ErrorData)
    assert "no model available" in result.message


async def test_sampling_handler_from_spec() -> None:
    def handler(messages: Any, model_preferences: Any) -> str:
        return "x"

    spec = MCPServerSpec.stdio("srv", "noop", sampling_handler=handler)
    client = MCPClient(spec, session=_FakeMcpSession())
    assert client._sampling_handler is handler


# ---------------------------------------------------------------------------
# SDK-capability gating (signature inspection)
# ---------------------------------------------------------------------------


class _ModernSessionCls:
    def __init__(
        self,
        read: Any,
        write: Any,
        sampling_callback: Any = None,
        message_handler: Any = None,
    ) -> None: ...


class _LegacySessionCls:
    def __init__(self, read: Any, write: Any) -> None: ...


async def test_session_kwargs_wires_supported_features() -> None:
    client = MCPClient(
        MCPServerSpec.stdio("srv", "noop"),
        sampling_handler=lambda messages, prefs: "y",
    )
    kwargs = client._session_kwargs(_ModernSessionCls)
    assert kwargs["message_handler"] == client._handle_incoming_message
    assert callable(kwargs["sampling_callback"])


async def test_session_kwargs_degrade_gracefully_on_old_sdk() -> None:
    """No sampling_callback / message_handler params → skipped
    silently instead of TypeError at ClientSession construction."""
    client = MCPClient(
        MCPServerSpec.stdio("srv", "noop"),
        sampling_handler=lambda messages, prefs: "y",
    )
    assert client._session_kwargs(_LegacySessionCls) == {}


async def test_session_kwargs_no_sampling_callback_without_handler() -> None:
    client = MCPClient(MCPServerSpec.stdio("srv", "noop"))
    kwargs = client._session_kwargs(_ModernSessionCls)
    assert "sampling_callback" not in kwargs
    assert "message_handler" in kwargs
