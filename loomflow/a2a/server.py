"""Expose a loomflow Agent as an A2A v1.0 endpoint (G10).

``serve_a2a(agent)`` returns a plain ASGI-3 callable — same
framework-free posture as :mod:`loomflow.serve` (whose low-level
request/response helpers this module reuses). It runs under any ASGI
server and mounts inside FastAPI/Starlette apps::

    from loomflow.a2a import serve_a2a
    app = serve_a2a(agent, name="research-bot", url="https://bots.example/a2a")

Routes:

* ``GET /.well-known/agent-card.json`` — the discovery card
  (``/.well-known/agent.json`` is also accepted for pre-1.0 clients).
* ``POST /`` — the JSON-RPC 2.0 endpoint. Methods:

  * ``message/send`` — extract the text parts of ``params.message``,
    run the agent (the message's ``contextId`` — generated when absent
    — becomes the loomflow ``session_id``; ``params.metadata.userId``
    becomes ``user_id``), and return a completed :class:`Task` whose
    output is a single text-part artifact. Agent exceptions come back
    as a task in the ``failed`` state (with the error text in
    ``status.message``), not as a JSON-RPC error — execution failure
    is task-level, protocol failure is RPC-level.
  * ``tasks/get`` — return the last known task by id from an
    in-memory, bounded task table (v1: process-local, no durable
    store; restarting the server forgets tasks).
  * ``message/stream`` — SSE stream of JSON-RPC responses. v1 is
    **coarse**: one ``status-update (working)`` frame, then the run
    executes to completion, then an ``artifact-update`` frame and a
    final ``status-update (completed | failed)`` frame. Per-token
    streaming via ``agent.stream`` is future work.

  Unknown methods → ``-32601``; malformed envelope → ``-32600``;
  unparseable body → ``-32700``; bad params → ``-32602``. JSON-RPC
  errors are returned with HTTP 200 per convention.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import ValidationError

from .. import __version__
from ..core.ids import new_id
from ..core.types import RunResult
from ..serve.app import (
    ASGIApp,
    Receive,
    Scope,
    Send,
    _ClientDisconnect,
    _read_body,
    _send_error,
    _send_json,
    _unwrap_exception,
)
from .types import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    TASK_NOT_FOUND,
    AgentCard,
    AgentSkill,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatus,
    TaskStatusUpdateEvent,
    message_text,
    text_artifact,
    text_message,
)

__all__ = ["MAX_TASKS", "serve_a2a"]

MAX_TASKS = 1024
"""In-memory task-table cap; oldest tasks are evicted past this."""

_CARD_PATHS = (
    "/.well-known/agent-card.json",  # A2A v1.0 well-known path
    "/.well-known/agent.json",  # legacy pre-1.0 path, still probed by SDKs
)


class A2AServableAgent(Protocol):
    """The structural surface ``serve_a2a`` needs from an agent.

    Only ``run`` — v1 does not drive ``agent.stream`` (see the coarse
    ``message/stream`` note in the module docstring). The concrete
    :class:`loomflow.Agent` satisfies it.
    """

    async def run(
        self,
        prompt: str,
        *,
        user_id: str | None = ...,
        session_id: str | None = ...,
    ) -> RunResult: ...


def _rpc_result(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": dict(result)}


def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _coerce_skills(
    skills: Sequence[AgentSkill | Mapping[str, Any]] | None,
    default_description: str,
) -> list[AgentSkill]:
    if not skills:
        return [AgentSkill(id="run", name="run", description=default_description)]
    return [s if isinstance(s, AgentSkill) else AgentSkill.model_validate(s) for s in skills]


def serve_a2a(
    agent: A2AServableAgent,
    *,
    name: str | None = None,
    description: str | None = None,
    url: str = "",
    skills: Sequence[AgentSkill | Mapping[str, Any]] | None = None,
) -> ASGIApp:
    """Wrap ``agent`` in a framework-free A2A v1.0 ASGI application.

    ``name`` / ``description`` populate the agent card (defaults:
    ``"loomflow-agent"`` / a generic one-liner). ``url`` is the card's
    advertised endpoint — set it to the externally reachable URL when
    deploying. ``skills`` accepts :class:`AgentSkill` instances or
    plain ``{id, name, description}`` mappings; when omitted a single
    catch-all ``run`` skill is advertised.
    """
    card_description = description or "A loomflow agent exposed over the A2A protocol."
    card = AgentCard(
        name=name or "loomflow-agent",
        description=card_description,
        url=url,
        version=__version__,
        skills=_coerce_skills(skills, card_description),
    )
    # Bounded, insertion-ordered task table (dict preserves order).
    tasks: dict[str, Task] = {}

    def _store(task: Task) -> None:
        tasks[task.id] = task
        while len(tasks) > MAX_TASKS:
            tasks.pop(next(iter(tasks)))

    # -- param parsing shared by message/send and message/stream ----------

    def _parse_message_params(
        params: Mapping[str, Any],
    ) -> tuple[Message, str, str, str | None] | str:
        """Return ``(message, text, context_id, user_id)`` or an error string."""
        raw = params.get("message")
        if not isinstance(raw, Mapping):
            return "params.message must be a Message object"
        try:
            message = Message.model_validate(dict(raw))
        except ValidationError as exc:
            return f"invalid message: {exc.error_count()} validation error(s)"
        text = message_text(message)
        if not text:
            return "message contains no text parts (v1 supports text parts only)"
        context_id = message.contextId or new_id("ctx")
        metadata = params.get("metadata")
        user_id: str | None = None
        if isinstance(metadata, Mapping):
            candidate = metadata.get("userId")
            if isinstance(candidate, str) and candidate:
                user_id = candidate
        return message, text, context_id, user_id

    async def _execute(message: Message, text: str, context_id: str, user_id: str | None) -> Task:
        """Run the agent and materialize the outcome as a Task."""
        task_id = message.taskId or new_id("task")
        try:
            result = await agent.run(text, user_id=user_id, session_id=context_id)
        except Exception as exc:  # noqa: BLE001 — wire boundary: failure → failed task
            cause = _unwrap_exception(exc)
            return Task(
                id=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state="failed",
                    message=text_message(
                        "agent", f"{type(cause).__name__}: {cause}", contextId=context_id
                    ),
                ),
                history=[message],
            )
        return Task(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(state="completed"),
            artifacts=[text_artifact(result.output, name="response")],
            history=[message, text_message("agent", result.output, contextId=context_id)],
        )

    # -- JSON-RPC method handlers ------------------------------------------

    async def _message_send(request_id: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        parsed = _parse_message_params(params)
        if isinstance(parsed, str):
            return _rpc_error(request_id, INVALID_PARAMS, parsed)
        message, text, context_id, user_id = parsed
        task = await _execute(message, text, context_id, user_id)
        _store(task)
        return _rpc_result(request_id, task.model_dump(exclude_none=True))

    async def _tasks_get(request_id: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        task_id = params.get("id")
        if not isinstance(task_id, str) or not task_id:
            return _rpc_error(request_id, INVALID_PARAMS, "missing or empty 'id'")
        task = tasks.get(task_id)
        if task is None:
            return _rpc_error(request_id, TASK_NOT_FOUND, f"task not found: {task_id}")
        return _rpc_result(request_id, task.model_dump(exclude_none=True))

    async def _message_stream(
        request_id: Any, params: Mapping[str, Any], send: Send
    ) -> dict[str, Any] | None:
        """Coarse v1 streaming (see module docstring).

        Returns an error envelope for the normal JSON path when params
        are bad; returns ``None`` once the SSE response has started.
        """
        parsed = _parse_message_params(params)
        if isinstance(parsed, str):
            return _rpc_error(request_id, INVALID_PARAMS, parsed)
        message, text, context_id, user_id = parsed
        task_id = message.taskId or new_id("task")

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                    (b"cache-control", b"no-cache"),
                    (b"x-accel-buffering", b"no"),
                ],
            }
        )

        async def _frame(result: Mapping[str, Any]) -> None:
            payload = json.dumps(_rpc_result(request_id, result))
            await send(
                {
                    "type": "http.response.body",
                    "body": f"data: {payload}\n\n".encode(),
                    "more_body": True,
                }
            )

        working = TaskStatusUpdateEvent(
            taskId=task_id, contextId=context_id, status=TaskStatus(state="working")
        )
        await _frame(working.model_dump(exclude_none=True))
        task = await _execute(message, text, context_id, user_id)
        task.id = task_id  # keep the announced id even if _execute re-derived it
        _store(task)
        for artifact in task.artifacts:
            update = TaskArtifactUpdateEvent(
                taskId=task_id, contextId=context_id, artifact=artifact
            )
            await _frame(update.model_dump(exclude_none=True))
        final = TaskStatusUpdateEvent(
            taskId=task_id, contextId=context_id, status=task.status, final=True
        )
        await _frame(final.model_dump(exclude_none=True))
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        return None

    # -- HTTP plumbing -------------------------------------------------------

    async def _handle_rpc(receive: Receive, send: Send) -> None:
        raw = await _read_body(receive)
        if raw is None:
            await _send_json(send, 200, _rpc_error(None, INVALID_REQUEST, "request body too large"))
            return
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            await _send_json(send, 200, _rpc_error(None, PARSE_ERROR, "body is not valid JSON"))
            return
        if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
            await _send_json(
                send,
                200,
                _rpc_error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 request object"),
            )
            return
        request_id = data.get("id")
        method = data.get("method")
        if not isinstance(method, str) or not method:
            await _send_json(send, 200, _rpc_error(request_id, INVALID_REQUEST, "missing 'method'"))
            return
        params = data.get("params", {})
        if not isinstance(params, dict):
            await _send_json(
                send, 200, _rpc_error(request_id, INVALID_PARAMS, "params must be an object")
            )
            return

        if method == "message/send":
            await _send_json(send, 200, await _message_send(request_id, params))
        elif method == "tasks/get":
            await _send_json(send, 200, await _tasks_get(request_id, params))
        elif method == "message/stream":
            envelope = await _message_stream(request_id, params, send)
            if envelope is not None:
                await _send_json(send, 200, envelope)
        else:
            await _send_json(
                send, 200, _rpc_error(request_id, METHOD_NOT_FOUND, f"unknown method: {method}")
            )

    async def _handle_lifespan(receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await _handle_lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        path = str(scope.get("path", ""))
        method = str(scope.get("method", "GET")).upper()
        try:
            if path in _CARD_PATHS:
                if method != "GET":
                    await _send_error(send, 405, "method_not_allowed", f"use GET for {path}")
                    return
                await _send_json(send, 200, card.model_dump(exclude_none=True))
                return
            if path == "/":
                if method != "POST":
                    await _send_error(send, 405, "method_not_allowed", "use POST for /")
                    return
                await _handle_rpc(receive, send)
                return
            await _send_error(send, 404, "not_found", f"no route for {path}")
        except _ClientDisconnect:
            return

    return app
