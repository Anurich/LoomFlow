"""A2A v1.0 on the wire, v0.3 kept working.

The card advertised protocolVersion 1.0 while the server spoke only the
v0.3 binding, so official v1.0 clients got "unknown method:
SendMessage". These tests drive the ASGI app over real HTTP framing
(httpx ASGITransport) and, when ``a2a-sdk`` is installed, with the
official client.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest

httpx = pytest.importorskip("httpx")

from loomflow import Agent  # noqa: E402
from loomflow.a2a import A2AClient, serve_a2a  # noqa: E402
from loomflow.core.types import RunResult  # noqa: E402

pytestmark = pytest.mark.anyio

BASE = "http://agent.test"


def _client(app: Any) -> Any:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE)


async def _rpc(
    http: Any, method: str, params: dict[str, Any], *, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = await http.post("/", json=body, headers=headers or {})
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    return data


def _v1_message(text: str, **extra: Any) -> dict[str, Any]:
    return {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": text}], **extra}


def _frames(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])["result"]
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _no_kind(obj: Any) -> bool:
    if isinstance(obj, dict):
        return "kind" not in obj and all(_no_kind(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_no_kind(v) for v in obj)
    return True


async def test_send_message_speaks_v1_shapes() -> None:
    async with _client(serve_a2a(Agent("x", model="echo"))) as http:
        reply = await _rpc(
            http, "SendMessage", {"message": _v1_message("hi")}, headers={"A2A-Version": "1.0"}
        )
    task = reply["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"] == [{"text": "Echo: hi"}]
    assert [m["role"] for m in task["history"]] == ["ROLE_USER", "ROLE_AGENT"]
    assert _no_kind(task)


async def test_v03_methods_keep_their_v03_shapes() -> None:
    message = {"messageId": "m1", "role": "user", "kind": "message",
               "parts": [{"kind": "text", "text": "hi"}]}
    async with _client(serve_a2a(Agent("x", model="echo"))) as http:
        reply = await _rpc(http, "message/send", {"message": message})
    task = reply["result"]
    assert task["kind"] == "task"
    assert task["status"]["state"] == "completed"
    assert task["artifacts"][0]["parts"] == [{"kind": "text", "text": "Echo: hi"}]


async def test_streaming_v1_begins_with_the_task() -> None:
    async with _client(serve_a2a(Agent("x", model="echo"))) as http:
        body = {"jsonrpc": "2.0", "id": 7, "method": "SendStreamingMessage",
                "params": {"message": _v1_message("hi")}}
        response = await http.post("/", json=body)
    frames = _frames(response.text)
    assert [next(iter(f)) for f in frames] == ["task", "artifactUpdate", "statusUpdate"]
    assert frames[0]["task"]["status"]["state"] == "TASK_STATE_WORKING"
    assert frames[1]["artifactUpdate"]["artifact"]["parts"] == [{"text": "Echo: hi"}]
    assert frames[2]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert _no_kind(frames) and "final" not in frames[2]["statusUpdate"]


async def test_versions_are_negotiated_by_header() -> None:
    async with _client(serve_a2a(Agent("x", model="echo"))) as http:
        bad = await _rpc(http, "SendMessage", {"message": _v1_message("hi")},
                         headers={"A2A-Version": "2.0"})
        assert bad["error"]["code"] == -32009
        # Under 1.0 semantics there is no message/send.
        wrong = await _rpc(http, "message/send", {"message": _v1_message("hi")},
                           headers={"A2A-Version": "1.0"})
        assert wrong["error"]["code"] == -32601
        # An empty header is 0.3 (spec 3.6.2).
        legacy = await _rpc(http, "message/send", {"message": _v1_message("hi")},
                            headers={"A2A-Version": "0.3"})
        assert legacy["result"]["status"]["state"] == "completed"


async def test_get_and_list_tasks() -> None:
    async with _client(serve_a2a(Agent("x", model="echo"))) as http:
        sent = await _rpc(http, "SendMessage", {"message": _v1_message("a", contextId="c1")})
        await _rpc(http, "SendMessage", {"message": _v1_message("b", contextId="c2")})
        task_id = sent["result"]["task"]["id"]
        got = await _rpc(http, "GetTask", {"id": task_id, "historyLength": 0})
        assert got["result"]["id"] == task_id and "history" not in got["result"]
        listed = await _rpc(http, "ListTasks", {"contextId": "c1"})
        assert [t["id"] for t in listed["result"]["tasks"]] == [task_id]
        page = await _rpc(http, "ListTasks", {"pageSize": 1})
        assert page["result"]["totalSize"] == 2 and page["result"]["nextPageToken"] == "1"
        done = await _rpc(http, "ListTasks", {"status": "TASK_STATE_COMPLETED"})
        assert done["result"]["totalSize"] == 2


class _SlowAgent:
    def __init__(self) -> None:
        self.started = anyio.Event()

    async def run(self, prompt: str, *, user_id: Any = None, session_id: Any = None) -> RunResult:
        self.started.set()
        await anyio.sleep(10)
        raise AssertionError("should have been cancelled")


async def test_cancel_task() -> None:
    slow = _SlowAgent()
    app = serve_a2a(slow)  # type: ignore[arg-type]
    async with _client(app) as http:
        frames: list[dict[str, Any]] = []

        async def stream() -> None:
            body = {"jsonrpc": "2.0", "id": 1, "method": "SendStreamingMessage",
                    "params": {"message": _v1_message("go", taskId="t-1")}}
            response = await http.post("/", json=body)
            frames.extend(_frames(response.text))

        async with anyio.create_task_group() as tg:
            tg.start_soon(stream)
            with anyio.fail_after(5):
                await slow.started.wait()
            canceled = await _rpc(http, "CancelTask", {"id": "t-1"})
            assert canceled["result"]["status"]["state"] == "TASK_STATE_CANCELED"
        assert frames[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_CANCELED"
        again = await _rpc(http, "CancelTask", {"id": "t-1"})
        assert again["error"]["code"] == -32002  # TaskNotCancelableError
        missing = await _rpc(http, "CancelTask", {"id": "nope"})
        assert missing["error"]["code"] == -32001


async def test_card_lists_both_interfaces_and_its_own_url() -> None:
    async with _client(serve_a2a(Agent("x", model="echo"))) as http:
        card = (await http.get("/.well-known/agent-card.json")).json()
    assert [(i["protocolVersion"], i["url"]) for i in card["supportedInterfaces"]] == [
        ("1.0", BASE + "/"),
        ("0.3", BASE + "/"),
    ]
    assert card["url"] == BASE + "/"
    assert all(skill["tags"] for skill in card["skills"])


async def test_client_speaks_v1_and_falls_back_to_v03() -> None:
    async with _client(serve_a2a(Agent("x", model="echo"))) as http:
        client = A2AClient(BASE, http=http)
        assert await client.send("hi") == "Echo: hi"
        assert client.protocol_version == "1.0"

    calls: list[str] = []

    async def v03_only(scope: Any, receive: Any, send: Any) -> None:
        body = json.loads((await receive())["body"])
        calls.append(body["method"])
        if body["method"] != "message/send":
            result: dict[str, Any] = {"error": {"code": -32601, "message": "Method not found"}}
        else:
            result = {"result": {"kind": "task", "id": "t", "contextId": "c",
                                 "status": {"state": "completed"},
                                 "artifacts": [{"artifactId": "a",
                                                "parts": [{"kind": "text", "text": "old"}]}]}}
        payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], **result}).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": payload})

    async with _client(v03_only) as http:
        client = A2AClient(BASE, http=http)
        assert await client.send("hi") == "old"
        assert client.protocol_version == "0.3"
        assert calls == ["SendMessage", "message/send"]
        pinned = A2AClient(BASE, http=http, protocol_version="1.0")
        with pytest.raises(Exception, match="Method not found"):
            await pinned.send("hi")


async def test_official_a2a_sdk_client_talks_to_loomflow() -> None:
    pytest.importorskip("a2a.client")
    from a2a.client import A2ACardResolver, ClientConfig, create_client
    from a2a.types import a2a_pb2 as pb

    async with _client(serve_a2a(Agent("x", model="echo"))) as http:
        card = await A2ACardResolver(http, BASE).get_agent_card()
        for streaming in (False, True):
            client = await create_client(card, ClientConfig(streaming=streaming, httpx_client=http))
            request = pb.SendMessageRequest(
                message=pb.Message(message_id="m1", role=pb.ROLE_USER, parts=[pb.Part(text="hi")])
            )
            events = [ev async for ev in client.send_message(request)]
            texts = [
                part.text
                for ev in events
                for artifact in (
                    list(ev.task.artifacts) if ev.HasField("task") else
                    [ev.artifact_update.artifact] if ev.HasField("artifact_update") else []
                )
                for part in artifact.parts
            ]
            assert "Echo: hi" in texts
