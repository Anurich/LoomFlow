"""Expose a loomflow Agent as an A2A endpoint, v1.0 and v0.3 (G10).

``serve_a2a(agent)`` returns a plain ASGI-3 callable — same
framework-free posture as :mod:`loomflow.serve` (whose low-level
request/response helpers this module reuses). It runs under any ASGI
server and mounts inside FastAPI/Starlette apps::

    from loomflow.a2a import serve_a2a
    app = serve_a2a(agent, name="research-bot", url="https://bots.example/a2a")

Routes:

* ``GET /.well-known/agent-card.json`` — the discovery card
  (``/.well-known/agent.json`` is also accepted for pre-1.0 clients).
  It lists a v1.0 and a v0.3 JSON-RPC interface in
  ``supportedInterfaces`` and keeps the v0.3 ``url`` /
  ``protocolVersion`` fields for older clients.
* ``POST /`` — the JSON-RPC 2.0 endpoint, both bindings:

  ============================  ==================  =====================
  operation                     v1.0 method         v0.3 method
  ============================  ==================  =====================
  send a message                ``SendMessage``     ``message/send``
  send, streaming (SSE)         ``SendStreaming``   ``message/stream``
                                ``Message``
  get a task                    ``GetTask``         ``tasks/get``
  cancel a task                 ``CancelTask``      ``tasks/cancel``
  list tasks                    ``ListTasks``       —
  ============================  ==================  =====================

  The version is the request's ``A2A-Version`` header (or query
  parameter) when present (an unsupported one is ``-32009``), else
  the method name decides. v1.0 requests get v1.0 shapes (ProtoJSON
  enums, ``{"task": ...}`` results); v0.3 requests get v0.3 ones.

  Sending runs the agent: the message's ``contextId`` (generated when
  absent) becomes the loomflow ``session_id``, ``params.metadata
  .userId`` becomes ``user_id``, and the reply comes back as a
  completed task with one text artifact. Agent exceptions come back as
  a ``failed`` task, not a JSON-RPC error — execution failure is
  task-level, protocol failure is RPC-level. Streaming is coarse: the
  task (working), then the artifact, then the final status. A task
  that is still running can be cancelled; a finished one is
  ``-32002``. Tasks live in a bounded in-memory table (restarting the
  server forgets them). Subscribing, push notifications and the
  extended card answer with their spec error codes.

  Unknown methods → ``-32601``; malformed envelope → ``-32600``;
  unparseable body → ``-32700``; bad params → ``-32602``. JSON-RPC
  errors are returned with HTTP 200 per convention.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol
from urllib.parse import parse_qs

import anyio
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
    EXTENDED_CARD_NOT_CONFIGURED,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PUSH_NOTIFICATION_NOT_SUPPORTED,
    TASK_NOT_CANCELABLE,
    TASK_NOT_FOUND,
    TERMINAL_STATES,
    UNSUPPORTED_OPERATION,
    VERSION_NOT_SUPPORTED,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatus,
    TaskStatusUpdateEvent,
    from_v1,
    message_text,
    text_artifact,
    text_message,
    to_v1,
)

__all__ = ["MAX_TASKS", "serve_a2a"]

MAX_TASKS = 1024
"""In-memory task-table cap; oldest tasks are evicted past this."""

_CARD_PATHS = (
    "/.well-known/agent-card.json",  # A2A v1.0 well-known path
    "/.well-known/agent.json",  # legacy pre-1.0 path, still probed by SDKs
)

V1 = "1.0"
V03 = "0.3"

# Operation names per protocol version.
_METHODS: dict[str, dict[str, str]] = {
    V1: {
        "SendMessage": "send",
        "SendStreamingMessage": "stream",
        "GetTask": "get",
        "CancelTask": "cancel",
        "ListTasks": "list",
        "SubscribeToTask": "subscribe",
        "GetExtendedAgentCard": "extended_card",
        "CreateTaskPushNotificationConfig": "push",
        "GetTaskPushNotificationConfig": "push",
        "ListTaskPushNotificationConfigs": "push",
        "DeleteTaskPushNotificationConfig": "push",
    },
    V03: {
        "message/send": "send",
        "message/stream": "stream",
        "tasks/get": "get",
        "tasks/cancel": "cancel",
        "tasks/resubscribe": "subscribe",
        "agent/getAuthenticatedExtendedCard": "extended_card",
        "tasks/pushNotificationConfig/set": "push",
        "tasks/pushNotificationConfig/get": "push",
        "tasks/pushNotificationConfig/list": "push",
        "tasks/pushNotificationConfig/delete": "push",
    },
}


class A2AServableAgent(Protocol):
    """The structural surface ``serve_a2a`` needs from an agent.

    Only ``run`` — streaming is coarse (see the module docstring). The
    concrete :class:`loomflow.Agent` satisfies it.
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
        return [
            AgentSkill(id="run", name="run", description=default_description, tags=["general"])
        ]
    return [s if isinstance(s, AgentSkill) else AgentSkill.model_validate(s) for s in skills]


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return bytes(value).decode("latin-1").strip()
    return None


def _requested_version(scope: Scope) -> str | None:
    """``Major.Minor`` from the ``A2A-Version`` header or query parameter."""
    raw = _header(scope, b"a2a-version")
    if raw is None:
        query = parse_qs(bytes(scope.get("query_string") or b"").decode("latin-1"))
        values = query.get("A2A-Version") or query.get("a2a-version")
        raw = values[0] if values else None
    if not raw:
        return None
    return ".".join(raw.split(".")[:2])


def _base_url(scope: Scope) -> str:
    host = _header(scope, b"host") or "localhost"
    scheme = str(scope.get("scheme") or "http")
    root = str(scope.get("root_path") or "")
    return f"{scheme}://{host}{root}/"


def _history(task: dict[str, Any], length: Any) -> dict[str, Any]:
    """Apply ``historyLength`` (spec 3.2.4): unset = all, 0 = none."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        return task
    if length == 0:
        return {k: v for k, v in task.items() if k != "history"}
    if "history" in task:
        return {**task, "history": task["history"][-length:]}
    return task


def serve_a2a(
    agent: A2AServableAgent,
    *,
    name: str | None = None,
    description: str | None = None,
    url: str = "",
    skills: Sequence[AgentSkill | Mapping[str, Any]] | None = None,
) -> ASGIApp:
    """Wrap ``agent`` in a framework-free A2A ASGI application.

    ``name`` / ``description`` populate the agent card (defaults:
    ``"loomflow-agent"`` / a generic one-liner). ``url`` is the card's
    advertised endpoint; when empty, the URL the card was requested
    at is used. ``skills`` accepts :class:`AgentSkill` instances or
    plain ``{id, name, description, tags}`` mappings; when omitted a
    single catch-all ``run`` skill is advertised.
    """
    card_description = description or "A loomflow agent exposed over the A2A protocol."
    card_skills = _coerce_skills(skills, card_description)

    def _card(scope: Scope) -> dict[str, Any]:
        endpoint = url or _base_url(scope)
        card = AgentCard(
            name=name or "loomflow-agent",
            description=card_description,
            supportedInterfaces=[
                AgentInterface(url=endpoint, protocolVersion=V1),
                AgentInterface(url=endpoint, protocolVersion=V03),
            ],
            url=endpoint,
            version=__version__,
            protocolVersion=V03,
            preferredTransport="JSONRPC",
            skills=card_skills,
        )
        return card.model_dump(exclude_none=True)

    # Bounded, insertion-ordered task table (dict preserves order).
    tasks: dict[str, Task] = {}
    # Tasks still running, with the scope that cancels them.
    running: dict[str, anyio.CancelScope] = {}

    def _store(task: Task) -> None:
        tasks[task.id] = task
        while len(tasks) > MAX_TASKS:
            tasks.pop(next(iter(tasks)))

    def _render(task: Task, version: str) -> dict[str, Any]:
        dumped = task.model_dump(exclude_none=True)
        return to_v1(dumped) if version == V1 else dumped

    # -- param parsing shared by the send operations -----------------------

    def _parse_message_params(
        params: Mapping[str, Any],
    ) -> tuple[Message, str, str, str | None] | str:
        """Return ``(message, text, context_id, user_id)`` or an error string."""
        raw = params.get("message")
        if not isinstance(raw, Mapping):
            return "params.message must be a Message object"
        try:
            # Either wire shape is accepted whatever the version.
            message = Message.model_validate(from_v1(dict(raw), kind="message"))
        except ValidationError as exc:
            return f"invalid message: {exc.error_count()} validation error(s)"
        text = message_text(message)
        if not text:
            return "message contains no text parts (only text parts are supported)"
        context_id = message.contextId or new_id("ctx")
        metadata = params.get("metadata")
        user_id: str | None = None
        if isinstance(metadata, Mapping):
            candidate = metadata.get("userId")
            if isinstance(candidate, str) and candidate:
                user_id = candidate
        return message, text, context_id, user_id

    async def _execute(
        message: Message, text: str, context_id: str, user_id: str | None, task_id: str
    ) -> Task:
        """Run the agent (cancellably) and materialize the outcome as a Task."""
        scope = anyio.CancelScope()
        running[task_id] = scope
        result: RunResult | None = None
        try:
            with scope:
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
        finally:
            running.pop(task_id, None)
        if result is None:  # cancelled through CancelTask
            return Task(
                id=task_id,
                contextId=context_id,
                status=TaskStatus(state="canceled"),
                history=[message],
            )
        return Task(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(state="completed"),
            artifacts=[text_artifact(result.output, name="response")],
            history=[message, text_message("agent", result.output, contextId=context_id)],
        )

    # -- operations ----------------------------------------------------------

    async def _send(request_id: Any, params: Mapping[str, Any], version: str) -> dict[str, Any]:
        parsed = _parse_message_params(params)
        if isinstance(parsed, str):
            return _rpc_error(request_id, INVALID_PARAMS, parsed)
        message, text, context_id, user_id = parsed
        task_id = message.taskId or new_id("task")
        _store(
            Task(id=task_id, contextId=context_id, status=TaskStatus(state="working"),
                 history=[message])
        )
        task = await _execute(message, text, context_id, user_id, task_id)
        _store(task)
        rendered = _render(task, version)
        return _rpc_result(request_id, {"task": rendered} if version == V1 else rendered)

    async def _get(request_id: Any, params: Mapping[str, Any], version: str) -> dict[str, Any]:
        task_id = params.get("id")
        if not isinstance(task_id, str) or not task_id:
            return _rpc_error(request_id, INVALID_PARAMS, "missing or empty 'id'")
        task = tasks.get(task_id)
        if task is None:
            return _rpc_error(request_id, TASK_NOT_FOUND, f"task not found: {task_id}")
        return _rpc_result(request_id, _history(_render(task, version), params.get("historyLength")))

    async def _cancel(request_id: Any, params: Mapping[str, Any], version: str) -> dict[str, Any]:
        task_id = params.get("id")
        if not isinstance(task_id, str) or not task_id:
            return _rpc_error(request_id, INVALID_PARAMS, "missing or empty 'id'")
        task = tasks.get(task_id)
        if task is None:
            return _rpc_error(request_id, TASK_NOT_FOUND, f"task not found: {task_id}")
        scope = running.get(task_id)
        if task.status.state in TERMINAL_STATES or scope is None:
            return _rpc_error(
                request_id,
                TASK_NOT_CANCELABLE,
                f"task {task_id} is {task.status.state} and can't be canceled",
            )
        scope.cancel()
        canceled = task.model_copy(update={"status": TaskStatus(state="canceled")})
        _store(canceled)
        return _rpc_result(request_id, _render(canceled, version))

    async def _list(request_id: Any, params: Mapping[str, Any], version: str) -> dict[str, Any]:
        context_id = params.get("contextId")
        state = params.get("status")
        state = from_v1({"state": state})["state"] if isinstance(state, str) else None
        size = params.get("pageSize")
        page_size = size if isinstance(size, int) and 1 <= size <= 100 else 50
        token = params.get("pageToken")
        offset = int(token) if isinstance(token, str) and token.isdigit() else 0
        matching = [
            t
            for t in reversed(tasks.values())  # newest first
            if (not context_id or t.contextId == context_id)
            and (state is None or t.status.state == state)
        ]
        page = matching[offset : offset + page_size]
        rendered = [_history(_render(t, version), params.get("historyLength")) for t in page]
        if params.get("includeArtifacts") is False:
            rendered = [{k: v for k, v in t.items() if k != "artifacts"} for t in rendered]
        more = offset + page_size < len(matching)
        return _rpc_result(
            request_id,
            {
                "tasks": rendered,
                "nextPageToken": str(offset + page_size) if more else "",
                "pageSize": page_size,
                "totalSize": len(matching),
            },
        )

    async def _stream(
        request_id: Any, params: Mapping[str, Any], version: str, send: Send
    ) -> dict[str, Any] | None:
        """Coarse streaming (see module docstring).

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

        working = Task(
            id=task_id, contextId=context_id, status=TaskStatus(state="working"),
            history=[message],
        )
        _store(working)
        if version == V1:
            # A v1.0 task stream begins with the Task itself.
            await _frame({"task": _render(working, V1)})
        else:
            update = TaskStatusUpdateEvent(
                taskId=task_id, contextId=context_id, status=TaskStatus(state="working")
            )
            await _frame(update.model_dump(exclude_none=True))
        task = await _execute(message, text, context_id, user_id, task_id)
        _store(task)
        for artifact in task.artifacts:
            event = TaskArtifactUpdateEvent(taskId=task_id, contextId=context_id, artifact=artifact)
            dumped = event.model_dump(exclude_none=True)
            await _frame({"artifactUpdate": to_v1(dumped)} if version == V1 else dumped)
        final = TaskStatusUpdateEvent(
            taskId=task_id, contextId=context_id, status=task.status, final=True
        )
        dumped = final.model_dump(exclude_none=True)
        await _frame({"statusUpdate": to_v1(dumped)} if version == V1 else dumped)
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        return None

    def _unsupported(code: int, what: str) -> Callable[..., Awaitable[dict[str, Any]]]:
        async def handler(request_id: Any, params: Mapping[str, Any], version: str) -> dict[str, Any]:
            return _rpc_error(request_id, code, what)

        return handler

    handlers: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
        "send": _send,
        "get": _get,
        "cancel": _cancel,
        "list": _list,
        "subscribe": _unsupported(UNSUPPORTED_OPERATION, "task subscription is not supported"),
        "push": _unsupported(
            PUSH_NOTIFICATION_NOT_SUPPORTED, "push notifications are not supported"
        ),
        "extended_card": _unsupported(
            EXTENDED_CARD_NOT_CONFIGURED, "no authenticated extended agent card"
        ),
    }

    # -- HTTP plumbing -------------------------------------------------------

    async def _handle_rpc(scope: Scope, receive: Receive, send: Send) -> None:
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
        if params is None:
            params = {}
        if not isinstance(params, dict):
            await _send_json(
                send, 200, _rpc_error(request_id, INVALID_PARAMS, "params must be an object")
            )
            return

        version = _requested_version(scope)
        if version is not None and version not in _METHODS:
            await _send_json(
                send,
                200,
                _rpc_error(
                    request_id,
                    VERSION_NOT_SUPPORTED,
                    f"A2A-Version {version} is not supported (supported: {V1}, {V03})",
                ),
            )
            return
        if version is None:
            # No header: the method name says which binding it is.
            version = V1 if method in _METHODS[V1] else V03
        operation = _METHODS[version].get(method)
        if operation is None:
            await _send_json(
                send,
                200,
                _rpc_error(
                    request_id, METHOD_NOT_FOUND, f"unknown method for A2A {version}: {method}"
                ),
            )
            return
        if operation == "stream":
            envelope = await _stream(request_id, params, version, send)
            if envelope is not None:
                await _send_json(send, 200, envelope)
            return
        await _send_json(send, 200, await handlers[operation](request_id, params, version))

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
                await _send_json(send, 200, _card(scope))
                return
            if path == "/":
                if method != "POST":
                    await _send_error(send, 405, "method_not_allowed", "use POST for /")
                    return
                await _handle_rpc(scope, receive, send)
                return
            await _send_error(send, 404, "not_found", f"no route for {path}")
        except _ClientDisconnect:
            return

    return app
