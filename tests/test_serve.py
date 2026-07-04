"""Tests for ``loomflow.serve`` — the pure-ASGI deployment surface (G12).

No network sockets: the app is a plain ASGI-3 callable, so every test
drives ``app(scope, receive, send)`` directly with canned protocol
messages via the tiny in-test harness below (httpx is not a test
dependency of this repo, so no ``httpx.ASGITransport``).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest

from loomflow import Agent, __version__
from loomflow.core.types import Event, Message, ModelChunk, ToolCall, ToolDef, Usage
from loomflow.serve import create_app
from loomflow.serve.app import MAX_BODY_BYTES, ASGIApp

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# In-test ASGI harness (no httpx)
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


def _parse_sse(body: bytes) -> list[tuple[str, str]]:
    """Parse an SSE stream into ``(event_name, data)`` pairs."""
    frames: list[tuple[str, str]] = []
    for block in body.decode("utf-8").strip().split("\n\n"):
        name, data = "", ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        frames.append((name, data))
    return frames


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


@pytest.fixture
def agent() -> Agent:
    return Agent("You are a test agent.", model="echo")


class _BoomModel:
    """Model whose every call raises — drives the 500 path."""

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


# ---------------------------------------------------------------------------
# /run
# ---------------------------------------------------------------------------


async def test_run_happy_path(agent: Agent) -> None:
    app = create_app(agent)
    resp = await _request(app, "POST", "/run", json_body={"prompt": "hi"})
    assert resp.status == 200
    assert resp.completed
    data = resp.json()
    assert data["output"] == "Echo: hi"
    assert isinstance(data["session_id"], str) and data["session_id"]
    usage = data["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["turns"] >= 1
    assert data["interrupted"] is False
    assert data["interruption_reason"] is None


async def test_run_missing_prompt(agent: Agent) -> None:
    app = create_app(agent)
    resp = await _request(app, "POST", "/run", json_body={"user_id": "u1"})
    assert resp.status == 400
    assert "prompt" in resp.json()["message"]


async def test_run_bad_json(agent: Agent) -> None:
    app = create_app(agent)
    resp = await _request(app, "POST", "/run", body=b"{not json")
    assert resp.status == 400
    assert resp.json()["error"] == "bad_request"
    # A JSON scalar is equally rejected — the body must be an object.
    resp = await _request(app, "POST", "/run", body=b'"just a string"')
    assert resp.status == 400


async def test_run_require_user_id(agent: Agent) -> None:
    app = create_app(agent, require_user_id=True)
    resp = await _request(app, "POST", "/run", json_body={"prompt": "hi"})
    assert resp.status == 422
    assert resp.json()["error"] == "missing_user_id"
    # With a user_id the same request goes through.
    resp = await _request(app, "POST", "/run", json_body={"prompt": "hi", "user_id": "u1"})
    assert resp.status == 200
    assert resp.json()["output"] == "Echo: hi"


async def test_run_oversized_body_413(agent: Agent) -> None:
    app = create_app(agent)
    resp = await _request(app, "POST", "/run", body=b"x" * (MAX_BODY_BYTES + 1))
    assert resp.status == 413
    assert resp.json()["error"] == "payload_too_large"


async def test_run_agent_error_500_shape() -> None:
    app = create_app(Agent("boom", model=_BoomModel()))
    resp = await _request(app, "POST", "/run", json_body={"prompt": "hi"})
    assert resp.status == 500
    data = resp.json()
    assert set(data) == {"error", "message"}
    assert isinstance(data["error"], str) and data["error"]
    assert "Traceback" not in resp.body.decode("utf-8")


# ---------------------------------------------------------------------------
# routing / health
# ---------------------------------------------------------------------------


async def test_health(agent: Agent) -> None:
    app = create_app(agent)
    resp = await _request(app, "GET", "/health")
    assert resp.status == 200
    assert resp.json() == {"status": "ok", "version": __version__}


async def test_unknown_route_404(agent: Agent) -> None:
    app = create_app(agent)
    resp = await _request(app, "GET", "/nope")
    assert resp.status == 404
    assert resp.json()["error"] == "not_found"


async def test_wrong_method_405(agent: Agent) -> None:
    app = create_app(agent)
    resp = await _request(app, "GET", "/run")
    assert resp.status == 405
    assert resp.headers[b"allow"] == b"POST"
    resp = await _request(app, "POST", "/health")
    assert resp.status == 405


# ---------------------------------------------------------------------------
# /stream
# ---------------------------------------------------------------------------


async def test_stream_sse_shape(agent: Agent) -> None:
    app = create_app(agent)
    resp = await _request(app, "POST", "/stream", json_body={"prompt": "hi"})
    assert resp.status == 200
    assert resp.headers[b"content-type"] == b"text/event-stream"
    assert resp.completed
    frames = _parse_sse(resp.body)
    kinds = [name for name, _ in frames]
    assert "started" in kinds
    assert "completed" in kinds
    assert kinds[-1] == "done"
    # Every agent-event frame carries the Event's JSON serialization.
    for name, data in frames[:-1]:
        payload = json.loads(data)
        assert payload["kind"] == name
        assert payload["session_id"]
    # The done frame carries the final RunResult payload.
    done = json.loads(frames[-1][1])
    assert done["output"] == "Echo: hi"


async def test_stream_validation_errors_are_json(agent: Agent) -> None:
    app = create_app(agent, require_user_id=True)
    resp = await _request(app, "POST", "/stream", json_body={"prompt": "hi"})
    assert resp.status == 422
    resp = await _request(app, "POST", "/stream", json_body={})
    assert resp.status == 400


async def test_stream_client_disconnect_cancels_run() -> None:
    """Breaking the SSE consumer must cancel the underlying run
    (the agent.stream break-to-cancel contract)."""

    class _EndlessAgent:
        def __init__(self) -> None:
            self.closed = anyio.Event()

        async def run(
            self,
            prompt: str,
            *,
            user_id: str | None = None,
            session_id: str | None = None,
            response_tone: str | None = None,
        ) -> Any:
            raise NotImplementedError

        async def stream(
            self,
            prompt: str,
            *,
            user_id: str | None = None,
            session_id: str | None = None,
            response_tone: str | None = None,
        ) -> AsyncIterator[Event]:
            try:
                while True:
                    yield Event.started("sess_endless", prompt)
                    await anyio.sleep(0.001)
            finally:
                self.closed.set()

    fake = _EndlessAgent()
    app = create_app(fake)
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/stream",
        "headers": [],
        "query_string": b"",
    }
    body_sent = False
    two_frames = anyio.Event()
    frames = 0

    async def receive() -> dict[str, Any]:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {
                "type": "http.request",
                "body": json.dumps({"prompt": "hi"}).encode("utf-8"),
                "more_body": False,
            }
        await two_frames.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal frames
        if message["type"] == "http.response.body" and message.get("body"):
            frames += 1
            if frames >= 2:
                two_frames.set()

    with anyio.fail_after(10):
        await app(scope, receive, send)
    assert fake.closed.is_set()


# ---------------------------------------------------------------------------
# /resume
# ---------------------------------------------------------------------------


async def test_resume_round_trip(agent: Agent) -> None:
    """run → resume against the same session_id round-trips: the resumed
    call continues the session (same id) and produces fresh output."""
    app = create_app(agent)
    first = await _request(app, "POST", "/run", json_body={"prompt": "hi"})
    assert first.status == 200
    session_id = first.json()["session_id"]

    resumed = await _request(
        app, "POST", "/resume", json_body={"session_id": session_id, "prompt": "again"}
    )
    assert resumed.status == 200
    data = resumed.json()
    assert data["session_id"] == session_id
    assert data["output"] == "Echo: again"
    assert set(data) == {
        "output",
        "session_id",
        "usage",
        "interrupted",
        "interruption_reason",
    }


async def test_resume_missing_session_id(agent: Agent) -> None:
    app = create_app(agent)
    resp = await _request(app, "POST", "/resume", json_body={"prompt": "hi"})
    assert resp.status == 400
    assert "session_id" in resp.json()["message"]


async def test_resume_feature_detects_from_checkpoint(agent: Agent) -> None:
    """``from_checkpoint`` in the body must not break against the thin
    resume signature (no such kwarg yet) — signature-gated, dropped."""
    app = create_app(agent)
    first = await _request(app, "POST", "/run", json_body={"prompt": "hi"})
    session_id = first.json()["session_id"]
    resp = await _request(
        app,
        "POST",
        "/resume",
        json_body={"session_id": session_id, "prompt": "next", "from_checkpoint": "latest"},
    )
    assert resp.status == 200
    assert resp.json()["output"] == "Echo: next"


async def test_resume_upgraded_signature_receives_from_checkpoint() -> None:
    """Against a checkpoint-style resume signature the kwarg IS passed."""
    calls: list[dict[str, Any]] = []
    real = Agent("t", model="echo")

    class _UpgradedAgent:
        async def run(
            self,
            prompt: str,
            *,
            user_id: str | None = None,
            session_id: str | None = None,
            response_tone: str | None = None,
        ) -> Any:
            return await real.run(prompt, user_id=user_id, session_id=session_id)

        def stream(
            self,
            prompt: str,
            *,
            user_id: str | None = None,
            session_id: str | None = None,
            response_tone: str | None = None,
        ) -> AsyncIterator[Event]:
            return real.stream(prompt, user_id=user_id, session_id=session_id)

        async def resume(
            self,
            prompt: str | None = None,
            *,
            session_id: str,
            from_checkpoint: str = "latest",
            user_id: str | None = None,
        ) -> Any:
            calls.append(
                {
                    "prompt": prompt,
                    "session_id": session_id,
                    "from_checkpoint": from_checkpoint,
                }
            )
            return await real.run(prompt or "resumed", session_id=session_id)

    app = create_app(_UpgradedAgent())
    resp = await _request(
        app,
        "POST",
        "/resume",
        json_body={"session_id": "sess_x", "from_checkpoint": "ckpt_7"},
    )
    assert resp.status == 200
    assert calls == [{"prompt": None, "session_id": "sess_x", "from_checkpoint": "ckpt_7"}]
    assert resp.json()["session_id"] == "sess_x"


async def test_resume_thin_signature_back_compat() -> None:
    """Against the legacy thin resume — ``resume(session_id, prompt, *,
    user_id=None)``, prompt required, no ``from_checkpoint`` — the kwarg
    is dropped and a missing prompt is substituted with ''."""
    calls: list[dict[str, Any]] = []
    real = Agent("t", model="echo")

    class _ThinAgent:
        async def run(
            self,
            prompt: str,
            *,
            user_id: str | None = None,
            session_id: str | None = None,
            response_tone: str | None = None,
        ) -> Any:
            return await real.run(prompt, user_id=user_id, session_id=session_id)

        def stream(
            self,
            prompt: str,
            *,
            user_id: str | None = None,
            session_id: str | None = None,
            response_tone: str | None = None,
        ) -> AsyncIterator[Event]:
            return real.stream(prompt, user_id=user_id, session_id=session_id)

        async def resume(self, session_id: str, prompt: str, *, user_id: str | None = None) -> Any:
            calls.append({"session_id": session_id, "prompt": prompt})
            return await real.run(prompt or "resumed", session_id=session_id)

    app = create_app(_ThinAgent())
    resp = await _request(
        app,
        "POST",
        "/resume",
        json_body={"session_id": "sess_thin", "from_checkpoint": "ckpt_ignored"},
    )
    assert resp.status == 200
    # from_checkpoint dropped (unsupported), prompt defaulted to "".
    assert calls == [{"session_id": "sess_thin", "prompt": ""}]


# ---------------------------------------------------------------------------
# lifespan
# ---------------------------------------------------------------------------


async def test_lifespan_protocol(agent: Agent) -> None:
    """uvicorn sends lifespan messages before/after serving; the app
    must acknowledge them."""
    app = create_app(agent)
    incoming: list[dict[str, Any]] = [
        {"type": "lifespan.startup"},
        {"type": "lifespan.shutdown"},
    ]
    acks: list[str] = []

    async def receive() -> dict[str, Any]:
        return incoming.pop(0)

    async def send(message: dict[str, Any]) -> None:
        acks.append(str(message["type"]))

    with anyio.fail_after(5):
        await app({"type": "lifespan"}, receive, send)
    assert acks == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
