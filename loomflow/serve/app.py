"""Pure-ASGI deployment surface for a loomflow Agent (G12).

``create_app(agent)`` returns a plain ASGI-3 callable — no FastAPI or
Starlette dependency, keeping the core dependency-light ethos. Because
it speaks the raw ``scope / receive / send`` protocol, it runs under
any ASGI server (uvicorn, hypercorn, daphne) and can be **mounted
inside** a FastAPI/Starlette app when you want to compose it with an
existing service::

    from fastapi import FastAPI
    from loomflow.serve import create_app

    api = FastAPI()
    api.mount("/agent", create_app(agent))   # any ASGI app mounts

Routes:

* ``POST /run``    — JSON ``{prompt, user_id?, session_id?, tone?}`` →
  the :class:`~loomflow.RunResult` as JSON.
* ``POST /stream`` — same body → ``text/event-stream`` of agent
  :class:`~loomflow.Event`\\ s, terminated by an ``event: done`` frame
  carrying the final result payload. A client disconnect cancels the
  underlying run (the stream generator is closed, which triggers the
  agent's break-to-cancel contract).
* ``POST /resume`` — ``{session_id, prompt?, from_checkpoint?}`` →
  same shape as ``/run``, via ``agent.resume(...)``.
* ``GET /health``  — liveness + version.

The ``output_schema`` request field is accepted but **ignored in v1**
(structured output over the wire needs a schema registry; planned).

Concurrency model: the app holds ONE shared Agent — multi-tenant by
design — and each request simply awaits ``agent.run(...)`` with the
``user_id`` from the request body as the tenancy partition. No worker
pool, no long-lived state: serverless-friendly.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, MutableMapping
from contextlib import suppress
from typing import Any, Protocol, cast

import anyio

from .. import __version__
from ..core.types import Event, EventKind, RunResult

__all__ = ["ASGIApp", "MAX_BODY_BYTES", "ServableAgent", "create_app"]

# --- ASGI-3 protocol aliases (h11-level dicts; no framework types) ---
ASGIMessage = MutableMapping[str, Any]
Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[ASGIMessage]]
Send = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_Handler = Callable[[Receive, Send], Awaitable[None]]

MAX_BODY_BYTES = 1024 * 1024
"""Request-body cap. Larger bodies get a 413 before any JSON parsing."""


class ServableAgent(Protocol):
    """The structural surface the ASGI app needs from an agent.

    The concrete :class:`loomflow.Agent` satisfies it; any duck-typed
    stand-in with these methods (plus a ``resume`` coroutine method,
    accessed reflectively — see :func:`_call_resume`) also works.
    """

    async def run(
        self,
        prompt: str,
        *,
        user_id: str | None = ...,
        session_id: str | None = ...,
        response_tone: str | None = ...,
    ) -> RunResult: ...

    def stream(
        self,
        prompt: str,
        *,
        user_id: str | None = ...,
        session_id: str | None = ...,
        response_tone: str | None = ...,
    ) -> AsyncIterator[Event]: ...


class _ClientDisconnect(Exception):
    """Raised internally when the client goes away mid-request."""


# ---------------------------------------------------------------------------
# Small protocol helpers
# ---------------------------------------------------------------------------


async def _read_body(receive: Receive) -> bytes | None:
    """Drain the request body. ``None`` means the 1MB cap was exceeded."""
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise _ClientDisconnect
        body = bytes(message.get("body", b""))
        total += len(body)
        if total > MAX_BODY_BYTES:
            return None
        chunks.append(body)
        if not message.get("more_body", False):
            return b"".join(chunks)


async def _send_json(
    send: Send,
    status: int,
    payload: Mapping[str, Any],
    *,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps(dict(payload)).encode("utf-8")
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _send_error(send: Send, status: int, error: str, message: str) -> None:
    await _send_json(send, status, {"error": error, "message": message})


def _opt_str(data: Mapping[str, Any], key: str) -> str | None:
    """Optional string field: non-string / empty values read as absent."""
    value = data.get(key)
    return value if isinstance(value, str) and value else None


def _result_payload(result: RunResult) -> dict[str, Any]:
    """Wire shape for a :class:`RunResult` (``model_dump``-style)."""
    return {
        "output": result.output,
        "session_id": result.session_id,
        "usage": {
            "input_tokens": result.tokens_in,
            "cached_input_tokens": result.cached_tokens_in,
            "cache_write_tokens": result.cache_write_tokens,
            "output_tokens": result.tokens_out,
            "total_tokens": result.total_tokens,
            "cost_usd": result.cost_usd,
            "turns": result.turns,
        },
        "interrupted": result.interrupted,
        "interruption_reason": result.interruption_reason,
    }


def _unwrap_exception(exc: BaseException) -> BaseException:
    """Peel single-member ExceptionGroups (anyio task-group wrapping)."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


async def _parse_json_object(receive: Receive, send: Send) -> dict[str, Any] | None:
    """Read + parse the body as a JSON object, or respond 413/400 and
    return ``None``."""
    raw = await _read_body(receive)
    if raw is None:
        await _send_error(
            send,
            413,
            "payload_too_large",
            f"request body exceeds {MAX_BODY_BYTES} bytes",
        )
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        await _send_error(send, 400, "bad_request", "request body is not valid JSON")
        return None
    if not isinstance(data, dict):
        await _send_error(send, 400, "bad_request", "request body must be a JSON object")
        return None
    return data


async def _call_resume(
    agent: ServableAgent,
    *,
    session_id: str,
    prompt: str | None,
    user_id: str | None,
    from_checkpoint: str | None,
) -> RunResult:
    """Invoke ``agent.resume`` defensively across both known signatures.

    Feature detection (via ``inspect.signature``) instead of version
    pinning, because ``resume`` is being upgraded in-flight:

    * **thin resume** (current): ``resume(session_id, prompt, *,
      user_id=None, ...)`` — ``prompt`` is required, no
      ``from_checkpoint``.
    * **checkpoint resume** (upgraded): ``resume(prompt=None, *,
      session_id, from_checkpoint="latest", ...)``.

    Both are callable with keyword args ``session_id=`` / ``prompt=``,
    so we always pass keywords and gate the optional kwargs:

    * ``from_checkpoint`` is passed only when the signature declares it
      (or takes ``**kwargs``); silently dropped against the thin resume
      — the thin path has no checkpoints to select from.
    * ``prompt`` is passed when the caller supplied one; when absent
      and the signature *requires* a prompt (thin resume), an empty
      string is substituted so the round-trip still works.
    * ``user_id`` is passed only when supplied AND accepted.
    """
    resume = cast(Any, agent).resume
    params: Mapping[str, inspect.Parameter]
    try:
        params = inspect.signature(resume).parameters
    except (TypeError, ValueError):  # builtins / exotic callables
        params = {}
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def _supports(name: str) -> bool:
        return accepts_var_kw or name in params

    kwargs: dict[str, Any] = {"session_id": session_id}
    if prompt is not None:
        kwargs["prompt"] = prompt
    else:
        prompt_param = params.get("prompt")
        if prompt_param is not None and prompt_param.default is inspect.Parameter.empty:
            kwargs["prompt"] = ""  # thin resume: prompt is required
    if user_id is not None and _supports("user_id"):
        kwargs["user_id"] = user_id
    if from_checkpoint is not None and _supports("from_checkpoint"):
        kwargs["from_checkpoint"] = from_checkpoint
    result = await resume(**kwargs)
    return cast(RunResult, result)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(agent: ServableAgent, *, require_user_id: bool = False) -> ASGIApp:
    """Wrap ``agent`` in a framework-free ASGI application.

    ``require_user_id=True`` makes the app reject any ``/run`` /
    ``/stream`` / ``/resume`` request whose body lacks a ``user_id``
    with a 422 — the multi-tenant deployment posture where anonymous
    runs must not share the default partition.
    """

    async def _parse_run_request(
        receive: Receive, send: Send
    ) -> tuple[str, str | None, str | None, str | None] | None:
        """Shared /run + /stream body handling.

        Returns ``(prompt, user_id, session_id, tone)`` or ``None``
        when an error response has already been sent.
        """
        data = await _parse_json_object(receive, send)
        if data is None:
            return None
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            await _send_error(send, 400, "bad_request", "missing or empty 'prompt'")
            return None
        user_id = _opt_str(data, "user_id")
        if require_user_id and user_id is None:
            await _send_error(
                send, 422, "missing_user_id", "'user_id' is required by this deployment"
            )
            return None
        return prompt, user_id, _opt_str(data, "session_id"), _opt_str(data, "tone")

    async def _handle_run(receive: Receive, send: Send) -> None:
        parsed = await _parse_run_request(receive, send)
        if parsed is None:
            return
        prompt, user_id, session_id, tone = parsed
        try:
            result = await agent.run(
                prompt, user_id=user_id, session_id=session_id, response_tone=tone
            )
        except Exception as exc:  # noqa: BLE001 — wire boundary: no tracebacks
            cause = _unwrap_exception(exc)
            await _send_error(send, 500, type(cause).__name__, str(cause))
            return
        await _send_json(send, 200, _result_payload(result))

    async def _handle_stream(receive: Receive, send: Send) -> None:
        parsed = await _parse_run_request(receive, send)
        if parsed is None:
            return
        prompt, user_id, session_id, tone = parsed
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

        final_result: Any = None
        disconnected = False

        async def _frame(name: str, data: str) -> None:
            await send(
                {
                    "type": "http.response.body",
                    "body": f"event: {name}\ndata: {data}\n\n".encode(),
                    "more_body": True,
                }
            )

        try:
            async with anyio.create_task_group() as tg:

                async def _watch_disconnect() -> None:
                    nonlocal disconnected
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            disconnected = True
                            # Cancelling the group aborts the async-for
                            # below; closing the generator cancels the
                            # producer (break-to-cancel contract).
                            tg.cancel_scope.cancel()
                            return

                tg.start_soon(_watch_disconnect)
                events = agent.stream(
                    prompt,
                    user_id=user_id,
                    session_id=session_id,
                    response_tone=tone,
                )
                try:
                    async for event in events:
                        if event.kind is EventKind.COMPLETED:
                            final_result = event.payload.get("result")
                        await _frame(event.kind.value, event.model_dump_json())
                finally:
                    # Close the generator explicitly (break-to-cancel:
                    # this cancels the producing run), shielded so a
                    # disconnect-triggered cancellation can't skip it.
                    aclose = getattr(events, "aclose", None)
                    if aclose is not None:
                        with anyio.CancelScope(shield=True):
                            await aclose()
                    tg.cancel_scope.cancel()  # release the watcher
        except Exception as exc:  # noqa: BLE001 — headers sent; emit an error frame
            cause = _unwrap_exception(exc)
            with suppress(Exception):
                await _frame(
                    "error",
                    json.dumps({"error": type(cause).__name__, "message": str(cause)}),
                )
        if disconnected:
            return  # nobody is listening; the run was cancelled above
        with suppress(Exception):
            await _frame("done", json.dumps(final_result) if final_result is not None else "{}")
            await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _handle_resume(receive: Receive, send: Send) -> None:
        data = await _parse_json_object(receive, send)
        if data is None:
            return
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            await _send_error(send, 400, "bad_request", "missing or empty 'session_id'")
            return
        user_id = _opt_str(data, "user_id")
        if require_user_id and user_id is None:
            await _send_error(
                send, 422, "missing_user_id", "'user_id' is required by this deployment"
            )
            return
        try:
            result = await _call_resume(
                agent,
                session_id=session_id,
                prompt=_opt_str(data, "prompt"),
                user_id=user_id,
                from_checkpoint=_opt_str(data, "from_checkpoint"),
            )
        except Exception as exc:  # noqa: BLE001 — wire boundary: no tracebacks
            cause = _unwrap_exception(exc)
            await _send_error(send, 500, type(cause).__name__, str(cause))
            return
        await _send_json(send, 200, _result_payload(result))

    async def _handle_health(receive: Receive, send: Send) -> None:
        await _send_json(send, 200, {"status": "ok", "version": __version__})

    routes: dict[str, dict[str, _Handler]] = {
        "/run": {"POST": _handle_run},
        "/stream": {"POST": _handle_stream},
        "/resume": {"POST": _handle_resume},
        "/health": {"GET": _handle_health},
    }

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
            return  # websockets etc. are out of scope for v1
        path = str(scope.get("path", ""))
        method = str(scope.get("method", "GET")).upper()
        by_method = routes.get(path)
        if by_method is None:
            await _send_error(send, 404, "not_found", f"no route for {path}")
            return
        handler = by_method.get(method)
        if handler is None:
            allow = ", ".join(sorted(by_method))
            await _send_json(
                send,
                405,
                {
                    "error": "method_not_allowed",
                    "message": f"{method} not allowed for {path}",
                },
                extra_headers=[(b"allow", allow.encode("ascii"))],
            )
            return
        try:
            await handler(receive, send)
        except _ClientDisconnect:
            return  # client went away before we could respond

    return app
