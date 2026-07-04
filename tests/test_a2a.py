"""Tests for ``loomflow.a2a`` — A2A v1.0 protocol support (G10).

No network sockets: the server is a plain ASGI-3 callable driven with
canned protocol messages (same harness shape as ``tests/test_serve.py``),
and the client is exercised against an in-test fake ``http=`` object
(the documented injection seam), so httpx is not required.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn, __version__
from loomflow.a2a import A2AClient, A2AError, AgentCard, serve_a2a
from loomflow.core.types import Message, ModelChunk, ToolCall, ToolDef, Usage
from loomflow.serve.app import ASGIApp

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# In-test ASGI harness (no httpx) — mirrors tests/test_serve.py
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self) -> None:
        self.status: int | None = None
        self.headers: dict[bytes, bytes] = {}
        self.body = b""
        self.completed = False

    def json(self) -> Any:
        return json.loads(self.body)


async def _request(
    app: ASGIApp,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    json_body: Any = None,
) -> _Response:
    """Drive ``app`` through one request/response cycle in-process."""
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
    scope: dict[str, Any] = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": [],
        "scheme": "http",
    }
    incoming: list[dict[str, Any]] = [
        {"type": "http.request", "body": body or b"", "more_body": False}
    ]
    never = anyio.Event()  # a receive after the body blocks (no disconnect)
    response = _Response()

    async def receive() -> dict[str, Any]:
        if incoming:
            return incoming.pop(0)
        await never.wait()
        return {"type": "http.disconnect"}  # pragma: no cover

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            response.status = int(message["status"])
            response.headers = {bytes(k).lower(): bytes(v) for k, v in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            response.body += bytes(message.get("body", b""))
            if not message.get("more_body", False):
                response.completed = True

    with anyio.fail_after(10):
        await app(scope, receive, send)
    return response


async def _rpc(
    app: ASGIApp, method: str, params: dict[str, Any], *, request_id: Any = 1
) -> _Response:
    """POST one JSON-RPC request to the A2A endpoint."""
    return await _request(
        app,
        "POST",
        "/",
        json_body={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
    )


def _user_message(text: str, *, context_id: str | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
        "messageId": "msg-test-1",
    }
    if context_id is not None:
        msg["contextId"] = context_id
    return msg


def _sse_results(body: bytes) -> list[dict[str, Any]]:
    """Parse SSE data-frames into their JSON-RPC ``result`` payloads."""
    results: list[dict[str, Any]] = []
    for block in body.decode("utf-8").strip().split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                envelope = json.loads(line[len("data: ") :])
                assert envelope["jsonrpc"] == "2.0"
                results.append(envelope["result"])
    return results


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


@pytest.fixture
def agent() -> Agent:
    return Agent("You are a test agent.", model="echo")


class _BoomModel:
    """Model whose every call raises — drives the failed-task path."""

    name = "boom"

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        output_schema: Any | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        prompt_caching: Any = None,
    ) -> tuple[str, list[ToolCall], Usage, str]:
        raise RuntimeError("kaboom")

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        output_schema: Any | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        prompt_caching: Any = None,
    ) -> AsyncIterator[ModelChunk]:
        raise RuntimeError("kaboom")
        yield  # pragma: no cover — makes this an async generator


class _RecordingAgent:
    """Wraps a real Agent, recording the kwargs each run() received."""

    def __init__(self, inner: Agent) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        prompt: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Any:
        self.calls.append({"prompt": prompt, "user_id": user_id, "session_id": session_id})
        return await self.inner.run(prompt, user_id=user_id, session_id=session_id)


class _FakeHttpResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeHttp:
    """Sync fake for the client's documented ``http=`` injection seam."""

    def __init__(
        self,
        *,
        post_response: _FakeHttpResponse | None = None,
        get_responses: dict[str, _FakeHttpResponse] | None = None,
    ) -> None:
        self.post_response = post_response
        self.get_responses = get_responses or {}
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []

    def post(self, url: str, json: dict[str, Any]) -> _FakeHttpResponse:
        self.posts.append((url, json))
        assert self.post_response is not None
        return self.post_response

    def get(self, url: str) -> _FakeHttpResponse:
        self.gets.append(url)
        return self.get_responses.get(url, _FakeHttpResponse(404, {}))


def _completed_task_payload(text: str, *, context_id: str = "ctx-remote") -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "id": "task-remote-1",
            "contextId": context_id,
            "kind": "task",
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "artifactId": "artifact-remote-1",
                    "name": "response",
                    "parts": [{"kind": "text", "text": text}],
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# Agent card
# ---------------------------------------------------------------------------


async def test_agent_card_well_known_and_legacy_paths(agent: Agent) -> None:
    app = serve_a2a(
        agent,
        name="test-bot",
        description="A bot under test.",
        url="https://bots.example/a2a",
        skills=[{"id": "echo", "name": "echo", "description": "Echoes text."}],
    )
    for path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
        resp = await _request(app, "GET", path)
        assert resp.status == 200
        card = resp.json()
        assert card["name"] == "test-bot"
        assert card["description"] == "A bot under test."
        assert card["url"] == "https://bots.example/a2a"
        assert card["version"] == __version__
        assert card["capabilities"] == {"streaming": True, "pushNotifications": False}
        assert card["skills"] == [{"id": "echo", "name": "echo", "description": "Echoes text."}]
        assert card["defaultInputModes"] == ["text"]
        assert card["defaultOutputModes"] == ["text"]
        # The payload round-trips through the pydantic model (conformance).
        AgentCard.model_validate(card)


async def test_agent_card_defaults(agent: Agent) -> None:
    app = serve_a2a(agent)
    resp = await _request(app, "GET", "/.well-known/agent-card.json")
    card = resp.json()
    assert card["name"] == "loomflow-agent"
    assert card["skills"][0]["id"] == "run"


async def test_card_path_wrong_method_405(agent: Agent) -> None:
    app = serve_a2a(agent)
    resp = await _request(app, "POST", "/.well-known/agent-card.json")
    assert resp.status == 405
    resp = await _request(app, "GET", "/")
    assert resp.status == 405
    resp = await _request(app, "GET", "/nope")
    assert resp.status == 404


# ---------------------------------------------------------------------------
# message/send
# ---------------------------------------------------------------------------


async def test_message_send_happy_path(agent: Agent) -> None:
    app = serve_a2a(agent)
    resp = await _rpc(app, "message/send", {"message": _user_message("hi")})
    assert resp.status == 200
    envelope = resp.json()
    assert envelope["jsonrpc"] == "2.0"
    assert envelope["id"] == 1
    task = envelope["result"]
    assert task["kind"] == "task"
    assert isinstance(task["id"], str) and task["id"]
    assert isinstance(task["contextId"], str) and task["contextId"]
    assert task["status"]["state"] == "completed"
    parts = task["artifacts"][0]["parts"]
    assert parts == [{"kind": "text", "text": "Echo: hi"}]
    # History carries the inbound user turn and the agent reply.
    roles = [m["role"] for m in task["history"]]
    assert roles == ["user", "agent"]


async def test_context_id_maps_to_same_session(agent: Agent) -> None:
    """The same contextId across sends must reach agent.run as the SAME
    loomflow session_id — verified with a ScriptedModel returning a
    different output per turn (so the two replies are distinguishable)."""
    scripted = Agent(
        "t",
        model=ScriptedModel([ScriptedTurn(text="first reply"), ScriptedTurn(text="second reply")]),
    )
    recording = _RecordingAgent(scripted)
    app = serve_a2a(recording)

    first = await _rpc(app, "message/send", {"message": _user_message("a", context_id="ctx-77")})
    second = await _rpc(app, "message/send", {"message": _user_message("b", context_id="ctx-77")})
    t1, t2 = first.json()["result"], second.json()["result"]
    assert t1["contextId"] == t2["contextId"] == "ctx-77"
    assert t1["artifacts"][0]["parts"][0]["text"] == "first reply"
    assert t2["artifacts"][0]["parts"][0]["text"] == "second reply"
    assert [c["session_id"] for c in recording.calls] == ["ctx-77", "ctx-77"]


async def test_message_send_generates_context_id_and_user_id(agent: Agent) -> None:
    recording = _RecordingAgent(agent)
    app = serve_a2a(recording)
    resp = await _rpc(
        app,
        "message/send",
        {"message": _user_message("hi"), "metadata": {"userId": "u-42"}},
    )
    task = resp.json()["result"]
    assert task["contextId"]  # generated when the message carried none
    assert recording.calls == [{"prompt": "hi", "user_id": "u-42", "session_id": task["contextId"]}]


async def test_message_send_agent_failure_yields_failed_task() -> None:
    app = serve_a2a(Agent("boom", model=_BoomModel()))
    resp = await _rpc(app, "message/send", {"message": _user_message("hi")})
    assert resp.status == 200
    task = resp.json()["result"]
    assert task["status"]["state"] == "failed"
    status_text = task["status"]["message"]["parts"][0]["text"]
    assert "kaboom" in status_text
    assert "Traceback" not in resp.body.decode("utf-8")


# ---------------------------------------------------------------------------
# tasks/get
# ---------------------------------------------------------------------------


async def test_tasks_get_returns_stored_task(agent: Agent) -> None:
    app = serve_a2a(agent)
    sent = await _rpc(app, "message/send", {"message": _user_message("hi")})
    task = sent.json()["result"]
    fetched = await _rpc(app, "tasks/get", {"id": task["id"]}, request_id=2)
    assert fetched.json()["result"] == task


async def test_tasks_get_unknown_id(agent: Agent) -> None:
    app = serve_a2a(agent)
    resp = await _rpc(app, "tasks/get", {"id": "task_nope"})
    error = resp.json()["error"]
    assert error["code"] == -32001
    resp = await _rpc(app, "tasks/get", {})
    assert resp.json()["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# JSON-RPC protocol errors
# ---------------------------------------------------------------------------


async def test_parse_error(agent: Agent) -> None:
    app = serve_a2a(agent)
    resp = await _request(app, "POST", "/", body=b"{not json")
    assert resp.status == 200  # JSON-RPC errors ride on HTTP 200
    envelope = resp.json()
    assert envelope["error"]["code"] == -32700
    assert envelope["id"] is None


async def test_invalid_request(agent: Agent) -> None:
    app = serve_a2a(agent)
    # Not an object.
    resp = await _request(app, "POST", "/", json_body=["nope"])
    assert resp.json()["error"]["code"] == -32600
    # Missing jsonrpc version marker.
    resp = await _request(app, "POST", "/", json_body={"method": "message/send"})
    assert resp.json()["error"]["code"] == -32600
    # Missing method.
    resp = await _request(app, "POST", "/", json_body={"jsonrpc": "2.0", "id": 1})
    assert resp.json()["error"]["code"] == -32600


async def test_method_not_found(agent: Agent) -> None:
    app = serve_a2a(agent)
    resp = await _rpc(app, "tasks/frobnicate", {})
    error = resp.json()["error"]
    assert error["code"] == -32601
    assert "tasks/frobnicate" in error["message"]


async def test_invalid_params(agent: Agent) -> None:
    app = serve_a2a(agent)
    # No message at all.
    resp = await _rpc(app, "message/send", {})
    assert resp.json()["error"]["code"] == -32602
    # Message with no text parts (v1 is text-only).
    resp = await _rpc(
        app,
        "message/send",
        {"message": {"role": "user", "parts": [{"kind": "data", "data": {"x": 1}}]}},
    )
    assert resp.json()["error"]["code"] == -32602
    # Structurally invalid message.
    resp = await _rpc(app, "message/send", {"message": {"role": "nobody", "parts": []}})
    assert resp.json()["error"]["code"] == -32602
    # params not an object.
    resp = await _request(
        app, "POST", "/", json_body={"jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": 7}
    )
    assert resp.json()["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# message/stream (coarse v1: working → artifact → completed)
# ---------------------------------------------------------------------------


async def test_message_stream_sse_working_then_completed(agent: Agent) -> None:
    app = serve_a2a(agent)
    resp = await _rpc(app, "message/stream", {"message": _user_message("hi", context_id="ctx-s")})
    assert resp.status == 200
    assert resp.headers[b"content-type"] == b"text/event-stream"
    assert resp.completed
    results = _sse_results(resp.body)
    kinds = [r["kind"] for r in results]
    assert kinds == ["status-update", "artifact-update", "status-update"]
    working, artifact, final = results
    assert working["status"]["state"] == "working"
    assert working["final"] is False
    assert artifact["artifact"]["parts"][0]["text"] == "Echo: hi"
    assert final["status"]["state"] == "completed"
    assert final["final"] is True
    assert {r["taskId"] for r in results} == {working["taskId"]}
    assert {r["contextId"] for r in results} == {"ctx-s"}


async def test_message_stream_failure_ends_failed() -> None:
    app = serve_a2a(Agent("boom", model=_BoomModel()))
    resp = await _rpc(app, "message/stream", {"message": _user_message("hi")})
    results = _sse_results(resp.body)
    assert results[0]["status"]["state"] == "working"
    assert results[-1]["status"]["state"] == "failed"
    assert results[-1]["final"] is True


async def test_message_stream_bad_params_plain_json_error(agent: Agent) -> None:
    app = serve_a2a(agent)
    resp = await _rpc(app, "message/stream", {})
    assert resp.headers[b"content-type"] == b"application/json"
    assert resp.json()["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# A2AClient (fake http seam — no httpx needed)
# ---------------------------------------------------------------------------


async def test_client_fetch_card() -> None:
    card_json = {
        "name": "remote-bot",
        "description": "d",
        "url": "https://r.example",
        "version": "1.2.3",
        "capabilities": {"streaming": True, "pushNotifications": False},
        "skills": [{"id": "s", "name": "s", "description": "d"}],
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
    }
    fake = _FakeHttp(
        get_responses={
            "https://r.example/.well-known/agent-card.json": _FakeHttpResponse(200, card_json)
        }
    )
    client = A2AClient("https://r.example/", http=fake)
    card = await client.fetch_card()
    assert isinstance(card, AgentCard)
    assert card.name == "remote-bot"
    assert card.capabilities.streaming is True


async def test_client_fetch_card_legacy_fallback() -> None:
    card_json = {"name": "old-bot", "description": "d"}
    fake = _FakeHttp(
        get_responses={
            "https://r.example/.well-known/agent.json": _FakeHttpResponse(200, card_json)
        }
    )
    client = A2AClient("https://r.example", http=fake)
    card = await client.fetch_card()
    assert card.name == "old-bot"
    assert fake.gets == [
        "https://r.example/.well-known/agent-card.json",
        "https://r.example/.well-known/agent.json",
    ]


async def test_client_send_extracts_artifact_text() -> None:
    fake = _FakeHttp(post_response=_FakeHttpResponse(200, _completed_task_payload("remote reply")))
    client = A2AClient("https://r.example", http=fake)
    reply = await client.send("do the thing", context_id="ctx-9")
    assert reply == "remote reply"
    url, payload = fake.posts[0]
    assert url == "https://r.example/"
    assert payload["jsonrpc"] == "2.0"
    assert payload["method"] == "message/send"
    message = payload["params"]["message"]
    assert message["role"] == "user"
    assert message["contextId"] == "ctx-9"
    assert message["parts"] == [{"kind": "text", "text": "do the thing"}]


async def test_client_send_raises_on_rpc_error() -> None:
    fake = _FakeHttp(
        post_response=_FakeHttpResponse(
            200,
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "unknown method"}},
        )
    )
    client = A2AClient("https://r.example", http=fake)
    with pytest.raises(A2AError) as excinfo:
        await client.send("hi")
    assert excinfo.value.code == -32601
    assert "unknown method" in str(excinfo.value)


async def test_client_send_raises_on_http_error_and_failed_task() -> None:
    client = A2AClient(
        "https://r.example", http=_FakeHttp(post_response=_FakeHttpResponse(503, {}))
    )
    with pytest.raises(A2AError, match="HTTP 503"):
        await client.send("hi")
    failed = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "id": "t1",
            "contextId": "c1",
            "kind": "task",
            "status": {
                "state": "failed",
                "message": {"role": "agent", "parts": [{"kind": "text", "text": "boom"}]},
            },
        },
    }
    client = A2AClient(
        "https://r.example", http=_FakeHttp(post_response=_FakeHttpResponse(200, failed))
    )
    with pytest.raises(A2AError, match="boom"):
        await client.send("hi")


# ---------------------------------------------------------------------------
# as_tool — a remote A2A agent as a delegate target in a local Agent
# ---------------------------------------------------------------------------


async def test_as_tool_end_to_end() -> None:
    fake = _FakeHttp(
        post_response=_FakeHttpResponse(200, _completed_task_payload("42 tickets today"))
    )
    remote = A2AClient("https://r.example", http=fake).as_tool(
        name="ticket_bot", description="Ask the remote ticket bot."
    )
    assert remote.name == "ticket_bot"
    local = Agent(
        "You coordinate.",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(id="c1", tool="ticket_bot", args={"prompt": "count tickets"})
                    ]
                ),
                ScriptedTurn(text="The remote bot says: 42 tickets today"),
            ]
        ),
        tools=[remote],
    )
    result = await local.run("how many tickets came in today?")
    assert result.output == "The remote bot says: 42 tickets today"
    # The delegate call really went over the (fake) wire as message/send.
    assert len(fake.posts) == 1
    sent_text = fake.posts[0][1]["params"]["message"]["parts"][0]["text"]
    assert sent_text == "count tickets"


# ---------------------------------------------------------------------------
# Loopback: A2AClient driving serve_a2a through an in-process ASGI bridge
# ---------------------------------------------------------------------------


class _AsgiHttp:
    """Adapter satisfying the client's http seam by calling the ASGI app
    in-process — proves client and server speak the same wire format."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def post(self, url: str, json: dict[str, Any]) -> _FakeHttpResponse:
        path = url.split("example", 1)[1] or "/"
        resp = await _request(self._app, "POST", path, json_body=json)
        return _FakeHttpResponse(resp.status or 500, resp.json())

    async def get(self, url: str) -> _FakeHttpResponse:
        path = url.split("example", 1)[1] or "/"
        resp = await _request(self._app, "GET", path)
        payload = resp.json() if resp.body else {}
        return _FakeHttpResponse(resp.status or 500, payload)


async def test_client_server_loopback(agent: Agent) -> None:
    app = serve_a2a(agent, name="loop-bot")
    client = A2AClient("https://loop.example", http=_AsgiHttp(app))
    card = await client.fetch_card()
    assert card.name == "loop-bot"
    reply = await client.send("round trip", context_id="ctx-loop")
    assert reply == "Echo: round trip"
