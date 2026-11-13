"""Security regression tests — authentication on ``serve`` / ``a2a``,
A2A task ownership, and error bodies that don't leak exception text.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from typing import Any

import anyio
import pytest

from loomflow import Agent
from loomflow.a2a import serve_a2a
from loomflow.cli import main
from loomflow.core.types import ToolCall
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.security.permissions import (
    Mode,
    PerUserPermissions,
    StandardPermissions,
)
from loomflow.serve import (
    TRUSTED_CALLER,
    AuthError,
    AuthRequest,
    bearer_token_auth,
    create_app,
)
from loomflow.serve.auth import is_loopback_host
from loomflow.tools.registry import tool

pytestmark = pytest.mark.anyio

SECRET_DSN = "postgres://svc:hunter2@db.internal:5432/prod"


# ---------------------------------------------------------------------------
# In-test ASGI harness
# ---------------------------------------------------------------------------


async def _call(
    app: Any,
    method: str,
    path: str,
    body: Any = None,
    *,
    token: str | None = None,
    headers: dict[str, str] | None = None,
    server: tuple[str, int] | None = None,
) -> tuple[int, Any]:
    raw = json.dumps(body).encode() if body is not None else b""
    hdrs = dict(headers or {})
    if token is not None:
        hdrs["Authorization"] = f"Bearer {token}"
    scope: dict[str, Any] = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(k.encode(), v.encode()) for k, v in hdrs.items()],
        "query_string": b"",
        "scheme": "http",
    }
    if server is not None:
        scope["server"] = server
    incoming = [{"type": "http.request", "body": raw, "more_body": False}]
    never = anyio.Event()
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if incoming:
            return incoming.pop(0)
        await never.wait()
        return {"type": "http.disconnect"}  # pragma: no cover

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    with anyio.fail_after(10):
        await app(scope, receive, send)
    status = int(sent[0]["status"])
    text = b"".join(m.get("body", b"") for m in sent[1:]).decode()
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, text


class _Recorder:
    """Duck-typed agent that records who each run was for."""

    def __init__(self) -> None:
        self.user_ids: list[str | None] = []

    async def run(self, prompt: str, **kwargs: Any) -> Any:
        self.user_ids.append(kwargs.get("user_id"))
        return await Agent("x", model="echo").run(prompt)

    async def resume(self, *, session_id: str, prompt: str = "", **kwargs: Any) -> Any:
        self.user_ids.append(kwargs.get("user_id"))
        return await Agent("x", model="echo").run(prompt or "resume")

    def stream(self, prompt: str, **kwargs: Any) -> Any:
        self.user_ids.append(kwargs.get("user_id"))
        return Agent("x", model="echo").stream(prompt)


class _Boom:
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"connect failed {SECRET_DSN}")

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        async def gen() -> Any:
            raise RuntimeError(f"connect failed {SECRET_DSN}")
            yield  # pragma: no cover

        return gen()


TOKENS = {"tok-alice": "alice", "tok-bob": "bob"}


# ---------------------------------------------------------------------------
# 3. serve: auth hook
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/run", "/stream", "/resume"])
async def test_serve_rejects_unauthenticated_requests(path: str) -> None:
    rec = _Recorder()
    app = create_app(rec, auth=bearer_token_auth(TOKENS))
    body = {"prompt": "hi", "session_id": "s", "user_id": "alice"}
    for token in (None, "wrong", "tok-alic", ""):
        status, payload = await _call(app, "POST", path, body, token=token)
        assert status == 401, (path, token)
        assert payload["error"] == "unauthorized"
    assert rec.user_ids == []  # the agent never ran


async def test_serve_identity_is_the_user_id() -> None:
    rec = _Recorder()
    app = create_app(rec, auth=bearer_token_auth(TOKENS))
    status, _ = await _call(app, "POST", "/run", {"prompt": "hi"}, token="tok-alice")
    assert status == 200
    # A matching body user_id is fine too.
    status, _ = await _call(
        app, "POST", "/run", {"prompt": "hi", "user_id": "bob"}, token="tok-bob"
    )
    assert status == 200
    assert rec.user_ids == ["alice", "bob"]


@pytest.mark.parametrize("path", ["/run", "/stream", "/resume"])
async def test_serve_rejects_body_user_id_that_differs(path: str) -> None:
    rec = _Recorder()
    app = create_app(rec, auth=bearer_token_auth(TOKENS))
    status, payload = await _call(
        app,
        "POST",
        path,
        {"prompt": "hi", "session_id": "s", "user_id": "alice"},
        token="tok-bob",
    )
    assert status == 403
    assert payload["error"] == "forbidden"
    assert rec.user_ids == []


async def test_serve_privilege_escalation_by_asserting_user_id_is_closed() -> None:
    """The reviewed exploit: claim to be the BYPASS-mode admin."""
    ran: list[str] = []

    @tool(destructive=True)
    async def delete_everything(confirm: str) -> str:
        "wipe"
        ran.append(confirm)
        return "WIPED"

    def agent() -> Agent:
        model = ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[ToolCall(tool="delete_everything", args={"confirm": "y"})]
                ),
                ScriptedTurn(text="finished"),
            ]
        )
        perms = PerUserPermissions(
            policies={"admin_alice": StandardPermissions(mode=Mode.BYPASS)},
            default=StandardPermissions.strict(),
        )
        return Agent("x", model=model, tools=[delete_everything], permissions=perms)

    auth = bearer_token_auth({"tok-random": "random_user", "tok-admin": "admin_alice"})
    body = {"prompt": "wipe", "user_id": "admin_alice"}

    status, _ = await _call(create_app(agent(), auth=auth), "POST", "/run", body)
    assert status == 401 and not ran
    status, _ = await _call(
        create_app(agent(), auth=auth), "POST", "/run", body, token="tok-random"
    )
    assert status == 403 and not ran
    status, _ = await _call(
        create_app(agent(), auth=auth), "POST", "/run", {"prompt": "wipe"}, token="tok-random"
    )
    assert status == 200 and not ran  # strict policy applied to the real identity
    status, _ = await _call(
        create_app(agent(), auth=auth), "POST", "/run", body, token="tok-admin"
    )
    assert status == 200 and ran == ["y"]


async def test_serve_custom_hooks_sync_async_and_failing() -> None:
    rec = _Recorder()
    seen: list[AuthRequest] = []

    async def async_hook(request: AuthRequest) -> str | None:
        seen.append(request)
        return request.header("X-User")

    status, _ = await _call(
        create_app(rec, auth=async_hook), "POST", "/run", {"prompt": "hi"},
        headers={"x-user": "carol"},
    )
    assert status == 200 and rec.user_ids == ["carol"]
    assert seen[0].method == "POST" and seen[0].path == "/run"
    assert seen[0].headers["x-user"] == "carol"

    def sync_hook(request: AuthRequest) -> str | None:
        return "dave" if request.bearer_token == "t" else None

    status, _ = await _call(
        create_app(rec, auth=sync_hook), "POST", "/run", {"prompt": "hi"}, token="t"
    )
    assert status == 200 and rec.user_ids[-1] == "dave"

    def raises_auth_error(request: AuthRequest) -> str:
        raise AuthError("expired token for user=eve")

    def raises_other(request: AuthRequest) -> str:
        raise KeyError("boom")

    def returns_garbage(request: AuthRequest) -> Any:
        return 42

    for hook in (raises_auth_error, raises_other, returns_garbage, lambda r: ""):
        status, payload = await _call(
            create_app(rec, auth=hook), "POST", "/run", {"prompt": "hi"}
        )
        assert status == 401  # fails closed
        assert "eve" not in json.dumps(payload)
    assert len(rec.user_ids) == 2


async def test_serve_shared_token_is_a_trusted_caller() -> None:
    rec = _Recorder()
    app = create_app(rec, auth=bearer_token_auth("s3cret-token"), require_user_id=True)
    assert (await _call(app, "POST", "/run", {"prompt": "hi", "user_id": "u1"}))[0] == 401
    status, _ = await _call(
        app, "POST", "/run", {"prompt": "hi", "user_id": "u1"}, token="s3cret-token"
    )
    assert status == 200 and rec.user_ids == ["u1"]
    # require_user_id still applies to a trusted caller.
    status, _ = await _call(app, "POST", "/run", {"prompt": "hi"}, token="s3cret-token")
    assert status == 422

    pinned = create_app(rec, auth=bearer_token_auth("s3cret-token", user_id="svc"))
    status, _ = await _call(pinned, "POST", "/run", {"prompt": "hi"}, token="s3cret-token")
    assert status == 200 and rec.user_ids[-1] == "svc"
    status, _ = await _call(
        pinned, "POST", "/run", {"prompt": "hi", "user_id": "u1"}, token="s3cret-token"
    )
    assert status == 403


def test_bearer_token_auth_validation_and_sentinel() -> None:
    with pytest.raises(ValueError):
        bearer_token_auth("")
    with pytest.raises(ValueError):
        bearer_token_auth({})
    with pytest.raises(ValueError):
        bearer_token_auth({"tok": ""})
    with pytest.raises(ValueError):
        bearer_token_auth({"tok": "a"}, user_id="b")
    hook = bearer_token_auth("tok")
    ok = AuthRequest("POST", "/run", {"authorization": "bearer tok"})
    assert hook(ok) is TRUSTED_CALLER
    assert hook(AuthRequest("POST", "/run", {"authorization": "Basic tok"})) is None
    assert hook(AuthRequest("POST", "/run", {})) is None


async def test_serve_health_stays_open_and_no_auth_default_still_works() -> None:
    app = create_app(_Recorder(), auth=bearer_token_auth(TOKENS))
    assert (await _call(app, "GET", "/health"))[0] == 200
    # No auth configured: loopback dev flow is unchanged.
    rec = _Recorder()
    status, _ = await _call(
        create_app(rec), "POST", "/run", {"prompt": "hi", "user_id": "anyone"}
    )
    assert status == 200 and rec.user_ids == ["anyone"]


async def test_open_app_warns_once_on_non_loopback_bind(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app(_Recorder())
    with caplog.at_level(logging.WARNING, logger="loomflow.serve"):
        await _call(app, "GET", "/health", server=("127.0.0.1", 8000))
        assert not caplog.records
        await _call(app, "GET", "/health", server=("0.0.0.0", 8000))
        await _call(app, "GET", "/health", server=("0.0.0.0", 8000))
    warnings_ = [r for r in caplog.records if "WITHOUT authentication" in r.getMessage()]
    assert len(warnings_) == 1

    caplog.clear()
    guarded = create_app(_Recorder(), auth=bearer_token_auth("tok-12345"))
    with caplog.at_level(logging.WARNING, logger="loomflow.serve"):
        await _call(guarded, "GET", "/health", server=("0.0.0.0", 8000))
    assert not caplog.records


def test_is_loopback_host() -> None:
    for host in ("127.0.0.1", "localhost", "::1", "[::1]", "127.8.9.1", None):
        assert is_loopback_host(host)
    for host in ("0.0.0.0", "::", "", "10.0.0.5", "example.com"):
        assert not is_loopback_host(host)


# ---------------------------------------------------------------------------
# 3b. CLI: --auth-token / LOOM_SERVE_TOKEN / warning
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_serve(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    fake_uvicorn = types.ModuleType("uvicorn")

    def run(app: Any, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    fake_uvicorn.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr("loomflow.cli._load_target", lambda spec: _Recorder())
    monkeypatch.delenv("LOOM_SERVE_TOKEN", raising=False)
    return captured


def _run_sync(app: Any, body: Any, token: str | None) -> int:
    async def go() -> int:
        return (await _call(app, "POST", "/run", body, token=token))[0]

    return anyio.run(go)


def test_cli_serve_auth_token_flag_and_env(
    fake_serve: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["serve", "m:a", "--host", "0.0.0.0", "--auth-token", "cli-token"]) == 0
    assert "WITHOUT authentication" not in capsys.readouterr().err
    assert _run_sync(fake_serve["app"], {"prompt": "hi"}, None) == 401
    assert _run_sync(fake_serve["app"], {"prompt": "hi"}, "cli-token") == 200

    monkeypatch.setenv("LOOM_SERVE_TOKEN", "env-token")
    assert main(["serve", "m:a", "--host", "0.0.0.0"]) == 0
    assert "WITHOUT authentication" not in capsys.readouterr().err
    assert _run_sync(fake_serve["app"], {"prompt": "hi"}, "cli-token") == 401
    assert _run_sync(fake_serve["app"], {"prompt": "hi"}, "env-token") == 200


def test_cli_serve_warns_when_open_on_non_loopback(
    fake_serve: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["serve", "m:a", "--host", "0.0.0.0"]) == 0
    err = capsys.readouterr().err
    assert "WARNING: serving WITHOUT authentication on 0.0.0.0" in err
    # Loopback default: no warning, still open for local dev.
    assert main(["serve", "m:a"]) == 0
    assert "WITHOUT authentication" not in capsys.readouterr().err
    assert _run_sync(fake_serve["app"], {"prompt": "hi"}, None) == 200


# ---------------------------------------------------------------------------
# 9. error bodies
# ---------------------------------------------------------------------------


async def test_serve_500_is_generic_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="loomflow.serve"):
        status, payload = await _call(create_app(_Boom()), "POST", "/run", {"prompt": "x"})
    assert status == 500
    assert payload["error"] == "internal_error"
    assert "hunter2" not in json.dumps(payload)
    assert "RuntimeError" not in json.dumps(payload)
    error_id = payload["error_id"]
    assert error_id and error_id in payload["message"]
    # The detail is in the server log, under the same id.
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert error_id in logged and "hunter2" in logged


async def test_serve_stream_error_frame_is_generic() -> None:
    status, text = await _call(create_app(_Boom()), "POST", "/stream", {"prompt": "x"})
    assert status == 200
    assert "event: stream_error" in text
    assert "hunter2" not in text and "internal_error" in text


async def test_serve_debug_flag_restores_detail() -> None:
    status, payload = await _call(
        create_app(_Boom(), debug=True), "POST", "/run", {"prompt": "x"}
    )
    assert status == 500
    assert payload["error"] == "RuntimeError"
    assert "hunter2" in payload["message"]
    assert payload["error_id"]


# ---------------------------------------------------------------------------
# 3 + 4. A2A: auth and task ownership
# ---------------------------------------------------------------------------


def _msg(text: str, **extra: Any) -> dict[str, Any]:
    return {
        "role": "ROLE_USER",
        "messageId": "m1",
        "parts": [{"text": text}],
        **extra,
    }


async def _rpc(
    app: Any,
    method: str,
    params: dict[str, Any],
    *,
    token: str | None = None,
) -> tuple[int, Any]:
    return await _call(
        app,
        "POST",
        "/",
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        token=token,
    )


def _a2a_agent(text: str = "ALICE PRIVATE ANSWER: salary=250k") -> Agent:
    return Agent("x", model=ScriptedModel([ScriptedTurn(text=text)] * 8))


async def test_a2a_rejects_unauthenticated_rpc_but_serves_the_card() -> None:
    app = serve_a2a(_a2a_agent(), auth=bearer_token_auth(TOKENS))
    status, payload = await _rpc(app, "SendMessage", {"message": _msg("hi")})
    assert status == 401 and payload["error"] == "unauthorized"
    status, _ = await _rpc(app, "ListTasks", {}, token="nope")
    assert status == 401
    status, card = await _call(app, "GET", "/.well-known/agent-card.json")
    assert status == 200 and card["name"]


async def test_a2a_identity_is_user_id_and_mismatch_is_403() -> None:
    rec = _Recorder()
    app = serve_a2a(rec, auth=bearer_token_auth(TOKENS))
    status, payload = await _rpc(
        app, "SendMessage", {"message": _msg("hi")}, token="tok-alice"
    )
    assert status == 200 and "result" in payload
    assert rec.user_ids == ["alice"]
    for method in ("SendMessage", "SendStreamingMessage", "ListTasks", "GetTask"):
        status, payload = await _rpc(
            app,
            method,
            {"message": _msg("hi"), "id": "t", "metadata": {"userId": "alice"}},
            token="tok-bob",
        )
        assert status == 403, method
        assert payload["error"] == "forbidden"
    assert rec.user_ids == ["alice"]


async def test_a2a_tasks_are_scoped_to_the_authenticated_caller() -> None:
    app = serve_a2a(_a2a_agent(), auth=bearer_token_auth(TOKENS))
    _, sent = await _rpc(app, "SendMessage", {"message": _msg("secret q")}, token="tok-alice")
    task_id = sent["result"]["task"]["id"]

    _, listed = await _rpc(app, "ListTasks", {}, token="tok-bob")
    assert listed["result"]["tasks"] == [] and listed["result"]["totalSize"] == 0
    _, got = await _rpc(app, "GetTask", {"id": task_id}, token="tok-bob")
    assert got["error"]["code"] == -32001
    _, cancelled = await _rpc(app, "CancelTask", {"id": task_id}, token="tok-bob")
    assert cancelled["error"]["code"] == -32001
    # Same answer as a task that doesn't exist: no existence oracle.
    _, missing = await _rpc(app, "GetTask", {"id": "task_nope"}, token="tok-bob")
    assert missing["error"]["code"] == got["error"]["code"]

    _, mine = await _rpc(app, "ListTasks", {}, token="tok-alice")
    assert [t["id"] for t in mine["result"]["tasks"]] == [task_id]
    _, got = await _rpc(app, "GetTask", {"id": task_id}, token="tok-alice")
    assert "salary=250k" in json.dumps(got["result"])


async def test_a2a_without_auth_scopes_tasks_by_self_asserted_user() -> None:
    """The reviewed leak: an anonymous caller listing alice's tasks."""
    app = serve_a2a(_a2a_agent())
    _, sent = await _rpc(
        app,
        "SendMessage",
        {"message": _msg("alice confidential"), "metadata": {"userId": "alice"}},
    )
    task_id = sent["result"]["task"]["id"]
    _, listed = await _rpc(app, "ListTasks", {})
    assert "salary" not in json.dumps(listed)
    assert listed["result"]["tasks"] == []
    _, got = await _rpc(app, "GetTask", {"id": task_id})
    assert got["error"]["code"] == -32001
    _, got = await _rpc(app, "tasks/get", {"id": task_id, "metadata": {"userId": "bob"}})
    assert got["error"]["code"] == -32001
    _, mine = await _rpc(app, "ListTasks", {"metadata": {"userId": "alice"}})
    assert [t["id"] for t in mine["result"]["tasks"]] == [task_id]


async def test_a2a_client_task_id_cannot_overwrite_another_callers_task() -> None:
    app = serve_a2a(_a2a_agent("ORIGINAL"), auth=bearer_token_auth(TOKENS))
    _, sent = await _rpc(app, "SendMessage", {"message": _msg("q")}, token="tok-alice")
    task_id = sent["result"]["task"]["id"]

    for method in ("SendMessage", "SendStreamingMessage"):
        status, hijack = await _rpc(
            app, method, {"message": _msg("overwrite", taskId=task_id)}, token="tok-bob"
        )
        assert status == 200
        assert hijack["error"]["code"] == -32001, method

    _, got = await _rpc(app, "GetTask", {"id": task_id}, token="tok-alice")
    history = json.dumps(got["result"])
    assert "overwrite" not in history and "ORIGINAL" in history
    # The owner may still continue their own task id.
    _, again = await _rpc(
        app, "SendMessage", {"message": _msg("follow up", taskId=task_id)}, token="tok-alice"
    )
    assert again["result"]["task"]["id"] == task_id


async def test_a2a_failed_task_is_generic_unless_debug() -> None:
    _, sent = await _rpc(serve_a2a(_Boom()), "SendMessage", {"message": _msg("x")})
    blob = json.dumps(sent)
    assert sent["result"]["task"]["status"]["state"] == "TASK_STATE_FAILED"
    assert "hunter2" not in blob and "error_id=" in blob

    _, sent = await _rpc(
        serve_a2a(_Boom(), debug=True), "SendMessage", {"message": _msg("x")}
    )
    assert "hunter2" in json.dumps(sent)
