"""JeevesGateway tests — config, spec generation, ToolHost forwarding.

No real network calls: where forwarding is exercised, the gateway is
constructed with an injected :class:`MCPRegistry` whose clients use
fake sessions (the same fakes used in ``test_mcp.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from loomflow import Agent
from loomflow.core.errors import ConfigError
from loomflow.core.types import ToolCall
from loomflow.jeeves import (
    JEEVES_API_KEY_ENV,
    JEEVES_DEFAULT_BASE_URL,
    JEEVES_TOKEN_PREFIX,
    JeevesConfig,
    JeevesGateway,
    looks_like_jeeves_key,
)
from loomflow.mcp import MCPClient, MCPRegistry, MCPServerSpec
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Local fake session — same shape as ``test_mcp.py``'s
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
    def __init__(
        self,
        tools: list[_FakeMcpTool],
        call_handler: Any | None = None,
    ) -> None:
        self._tools = tools
        self._handler = call_handler
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self) -> None:
        pass

    async def list_tools(self) -> _FakeListResult:
        return _FakeListResult(tools=list(self._tools))

    async def call_tool(self, name: str, args: dict[str, Any]) -> _FakeCallResult:
        self.calls.append((name, dict(args)))
        if self._handler is not None:
            return await self._handler(name, args)
        return _FakeCallResult(content=[_FakeContent(text=f"called {name}")])


# ---------------------------------------------------------------------------
# Config and spec
# ---------------------------------------------------------------------------


def test_constants_match_documentation() -> None:
    assert JEEVES_API_KEY_ENV == "JEEVES_API_KEY"
    assert JEEVES_DEFAULT_BASE_URL == "https://jeeves.works/mcp"
    assert JEEVES_TOKEN_PREFIX == "jm_sk_"


def test_empty_api_key_rejected() -> None:
    with pytest.raises(ConfigError):
        JeevesGateway(JeevesConfig(api_key=""))


def test_from_env_reads_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(JEEVES_API_KEY_ENV, "jm_sk_test123")
    gateway = JeevesGateway.from_env()
    assert gateway.config.api_key == "jm_sk_test123"
    assert gateway.config.base_url == JEEVES_DEFAULT_BASE_URL
    assert gateway.server_name == "jeeves"


def test_from_env_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(JEEVES_API_KEY_ENV, "  jm_sk_padded  ")
    gateway = JeevesGateway.from_env()
    assert gateway.config.api_key == "jm_sk_padded"


def test_from_env_missing_key_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(JEEVES_API_KEY_ENV, raising=False)
    with pytest.raises(ConfigError, match=JEEVES_API_KEY_ENV):
        JeevesGateway.from_env()


def test_from_env_custom_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_JEEVES_KEY", "jm_sk_alt")
    gateway = JeevesGateway.from_env(env_var="MY_JEEVES_KEY")
    assert gateway.config.api_key == "jm_sk_alt"


def test_from_env_custom_base_url_and_server_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(JEEVES_API_KEY_ENV, "jm_sk_x")
    gateway = JeevesGateway.from_env(
        base_url="https://staging.jeeves.works/mcp",
        server_name="jeeves-staging",
    )
    assert gateway.config.base_url == "https://staging.jeeves.works/mcp"
    assert gateway.config.server_name == "jeeves-staging"


def test_as_mcp_server_produces_http_spec_with_url_path_auth() -> None:
    gateway = JeevesGateway(JeevesConfig(api_key="jm_sk_abc"))
    spec = gateway.as_mcp_server()
    assert isinstance(spec, MCPServerSpec)
    assert spec.transport == "http"
    assert spec.url == "https://jeeves.works/mcp/jm_sk_abc"
    assert spec.name == "jeeves"


def test_as_mcp_server_uses_custom_base_url() -> None:
    gateway = JeevesGateway(
        JeevesConfig(
            api_key="jm_sk_z",
            base_url="https://custom.example/v2",
            server_name="custom",
        )
    )
    spec = gateway.as_mcp_server()
    assert spec.url == "https://custom.example/v2/jm_sk_z"
    assert spec.name == "custom"


def test_as_registry_returns_mcp_registry_with_one_server() -> None:
    gateway = JeevesGateway(JeevesConfig(api_key="jm_sk_x"))
    registry = gateway.as_registry()
    assert isinstance(registry, MCPRegistry)
    assert registry.server_names == ["jeeves"]


# ---------------------------------------------------------------------------
# Token helper
# ---------------------------------------------------------------------------


def test_looks_like_jeeves_key() -> None:
    assert looks_like_jeeves_key("jm_sk_abc")
    assert not looks_like_jeeves_key("sk-anthropic-xxxx")
    assert not looks_like_jeeves_key("")


# ---------------------------------------------------------------------------
# ToolHost forwarding via injected fake registry
# ---------------------------------------------------------------------------


def _gateway_with_fake_session(
    tools: list[_FakeMcpTool],
    call_handler: Any | None = None,
) -> tuple[JeevesGateway, _FakeMcpSession]:
    session = _FakeMcpSession(tools, call_handler=call_handler)
    spec = MCPServerSpec.http("jeeves", "https://example/mcp/test")
    client = MCPClient(spec, session=session)
    registry = MCPRegistry([client])
    gateway = JeevesGateway(JeevesConfig(api_key="jm_sk_test"), registry=registry)
    return gateway, session


async def test_list_tools_forwards_to_underlying_registry() -> None:
    gateway, _ = _gateway_with_fake_session(
        [
            _FakeMcpTool(name="send_email", description="Send an email."),
            _FakeMcpTool(name="schedule_meeting"),
        ]
    )
    defs = await gateway.list_tools()
    names = {d.name for d in defs}
    assert names == {"send_email", "schedule_meeting"}


async def test_list_tools_query_filter_propagates() -> None:
    gateway, _ = _gateway_with_fake_session(
        [
            _FakeMcpTool(name="send_email", description="email out"),
            _FakeMcpTool(name="schedule_meeting", description="calendar"),
        ]
    )
    defs = await gateway.list_tools(query="email")
    assert {d.name for d in defs} == {"send_email"}


async def test_call_forwards_with_call_id() -> None:
    gateway, session = _gateway_with_fake_session(
        [_FakeMcpTool(name="ping")]
    )
    result = await gateway.call("ping", {"x": 1}, call_id="c1")
    assert result.ok
    assert result.call_id == "c1"
    assert session.calls == [("ping", {"x": 1})]


async def test_aclose_propagates_to_registry() -> None:
    gateway, _ = _gateway_with_fake_session([_FakeMcpTool(name="ping")])
    # Trigger lazy materialisation by calling list_tools first.
    await gateway.list_tools()
    await gateway.aclose()


# ---------------------------------------------------------------------------
# End-to-end: JeevesGateway in an Agent
# ---------------------------------------------------------------------------


async def test_agent_run_with_jeeves_gateway_dispatches_tool() -> None:
    async def email_handler(
        name: str, args: dict[str, Any]
    ) -> _FakeCallResult:
        recipient = args.get("to", "?")
        return _FakeCallResult(
            content=[_FakeContent(text=f"sent to {recipient}")]
        )

    gateway, session = _gateway_with_fake_session(
        [_FakeMcpTool(name="send_email", description="Send an email.")],
        call_handler=email_handler,
    )

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="send_email",
                        args={"to": "alice@example.com"},
                    )
                ]
            ),
            ScriptedTurn(text="Email queued."),
        ]
    )
    agent = Agent("you assist with mail", model=model, tools=gateway)

    result = await agent.run("send an email to alice")

    assert "Email queued" in result.output
    assert session.calls == [("send_email", {"to": "alice@example.com"})]
    assert not result.interrupted
