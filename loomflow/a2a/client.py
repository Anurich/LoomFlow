"""Client for remote A2A agents, usable as a loomflow tool (G10).

::

    from loomflow.a2a import A2AClient

    remote = A2AClient("https://bots.example/a2a")
    card = await remote.fetch_card()
    reply = await remote.send("summarize today's tickets")

    # ...or hand the remote agent to a local Agent as a delegate:
    agent = Agent("You coordinate.", tools=[remote.as_tool(name="ticket_bot")])

The client speaks A2A v1.0 (``SendMessage``, ``A2A-Version: 1.0``)
and, by default, falls back to v0.3 (``message/send``) when the remote
doesn't know the v1.0 method or version — so it works with v1.0
servers, v0.3 servers and earlier loomflow releases. Pass
``protocol_version="1.0"`` or ``"0.3"`` to pin one (spec 3.6.3: a
client that needs v1.0 features shouldn't silently fall back).

httpx is imported lazily on first use — the client works without it as
long as an ``http=`` object is injected (the pattern the tests use).
"""

from __future__ import annotations

import inspect
from typing import Any, Literal

from ..core.ids import new_id
from ..tools.registry import Tool
from .types import (
    METHOD_NOT_FOUND,
    VERSION_NOT_SUPPORTED,
    A2AError,
    AgentCard,
    Message,
    Task,
    from_v1,
    message_text,
    text_message,
    to_v1,
)

__all__ = ["A2AClient"]

_CARD_PATH = "/.well-known/agent-card.json"
_LEGACY_CARD_PATH = "/.well-known/agent.json"


def _accepts_headers(method: Any) -> bool:
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return True
    return "headers" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, else return it as-is.

    Lets the injected ``http=`` object be either a real async client
    (``httpx.AsyncClient``) or a plain-sync test fake.
    """
    if inspect.isawaitable(value):
        return await value
    return value


class A2AClient:
    """Talk to a remote A2A agent over JSON-RPC.

    ``base_url`` is the agent's endpoint root (the JSON-RPC POST goes
    to ``base_url + "/"``; the card is fetched from the well-known
    path under it).

    ``http=`` is the injection seam: any object with ``post(url,
    json=...)`` and ``get(url)`` whose (optionally awaitable) return
    value exposes ``.status_code`` and ``.json()`` works — an
    ``httpx.AsyncClient``, a wrapper adding auth headers, or an
    in-test fake. When omitted, an ``httpx.AsyncClient`` is created
    lazily on first use (``pip install httpx`` or
    ``pip install 'loomflow[a2a]'``); call :meth:`aclose` to release
    it, or use the client as an async context manager.
    """

    def __init__(
        self,
        base_url: str,
        *,
        http: Any | None = None,
        protocol_version: Literal["auto", "1.0", "0.3"] = "auto",
    ) -> None:
        if protocol_version not in ("auto", "1.0", "0.3"):
            raise ValueError("protocol_version must be 'auto', '1.0' or '0.3'")
        self._base_url = base_url.rstrip("/")
        self._http = http
        self._owns_http = False
        self._pinned = protocol_version != "auto"
        self._version = "1.0" if protocol_version == "auto" else protocol_version

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> A2AClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the lazily created httpx client (no-op for injected http)."""
        if self._owns_http and self._http is not None:
            await _maybe_await(self._http.aclose())
            self._http = None
            self._owns_http = False

    def _ensure_http(self) -> Any:
        if self._http is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover — env-dependent
                raise ImportError(
                    "A2AClient needs httpx for network calls — "
                    "pip install httpx (or: pip install 'loomflow[a2a]')"
                ) from exc
            self._http = httpx.AsyncClient()
            self._owns_http = True
        return self._http

    # -- protocol calls ------------------------------------------------------

    async def fetch_card(self) -> AgentCard:
        """GET the agent's discovery card.

        Tries the v1.0 well-known path first, falling back to the
        legacy ``agent.json`` path on 404.
        """
        http = self._ensure_http()
        response = await _maybe_await(http.get(self._base_url + _CARD_PATH))
        if response.status_code == 404:
            response = await _maybe_await(http.get(self._base_url + _LEGACY_CARD_PATH))
        if response.status_code != 200:
            raise A2AError(f"agent card fetch failed: HTTP {response.status_code}")
        return AgentCard.model_validate(await _maybe_await(response.json()))

    @property
    def protocol_version(self) -> str:
        """The A2A version this client is speaking (after any fallback)."""
        return self._version

    async def _call(self, method: str, params: dict[str, Any]) -> Any:
        """POST one JSON-RPC request; return ``result`` or raise A2AError."""
        http = self._ensure_http()
        payload = {"jsonrpc": "2.0", "id": new_id("rpc"), "method": method, "params": params}
        post = http.post
        if _accepts_headers(post):
            request = post(
                self._base_url + "/", json=payload, headers={"A2A-Version": self._version}
            )
        else:  # a minimal injected http object: post(url, json=...)
            request = post(self._base_url + "/", json=payload)
        response = await _maybe_await(request)
        if response.status_code != 200:
            raise A2AError(f"{method} failed: HTTP {response.status_code}")
        data = await _maybe_await(response.json())
        if not isinstance(data, dict):
            raise A2AError(f"{method}: response is not a JSON object")
        error = data.get("error")
        if isinstance(error, dict):
            raise A2AError(
                str(error.get("message", "unknown error")),
                code=error.get("code") if isinstance(error.get("code"), int) else None,
            )
        return data.get("result")

    async def send(self, text: str, *, context_id: str | None = None) -> str:
        """Send a message (``SendMessage`` / ``message/send``) and
        return the agent's text reply.

        Pass the same ``context_id`` across calls to hold one remote
        conversation (the server maps it to its session). Raises
        :class:`A2AError` on JSON-RPC errors, malformed results, and
        tasks that come back ``failed``.
        """
        message = text_message("user", text, contextId=context_id).model_dump(exclude_none=True)
        try:
            result = await self._send(message)
        except A2AError as exc:
            if self._pinned or self._version != "1.0" or exc.code not in (
                METHOD_NOT_FOUND,
                VERSION_NOT_SUPPORTED,
            ):
                raise
            self._version = "0.3"  # a v0.3-only remote: speak its dialect
            result = await self._send(message)
        if not isinstance(result, dict):
            raise A2AError("send: result is not an object")
        # The spec allows either a Task or a bare Message as the result
        # (v1.0 wraps it as {"task": ...} / {"message": ...}).
        if "message" in result and "task" not in result and "status" not in result:
            return message_text(Message.model_validate(from_v1(result["message"], kind="message")))
        if result.get("kind") == "message":
            return message_text(Message.model_validate(result))
        raw_task = result.get("task", result)
        task = Task.model_validate(from_v1(raw_task, kind="task"))
        if task.status.state == "failed":
            detail = message_text(task.status.message) if task.status.message else ""
            raise A2AError(f"remote task failed{': ' + detail if detail else ''}")
        texts = [
            part.text
            for artifact in task.artifacts
            for part in artifact.parts
            if part.kind == "text" and isinstance(part.text, str)
        ]
        return "\n".join(texts)

    async def _send(self, message: dict[str, Any]) -> Any:
        if self._version == "1.0":
            return await self._call("SendMessage", {"message": to_v1(message)})
        return await self._call("message/send", {"message": message})

    # -- loomflow integration ------------------------------------------------

    def as_tool(self, name: str | None = None, description: str | None = None) -> Tool:
        """Wrap :meth:`send` as a loomflow :class:`Tool`.

        The returned tool takes ``prompt`` (required) and an optional
        ``context_id`` (pass the same value across calls to keep one
        remote conversation), making the remote A2A agent a delegate
        target in any loomflow agent's ``tools=[...]``.
        """

        async def _delegate(prompt: str, context_id: str = "") -> str:
            return await self.send(prompt, context_id=context_id or None)

        return Tool(
            name=name or "a2a_delegate",
            description=description
            or f"Delegate a task to the remote A2A agent at {self._base_url} "
            "and return its text reply.",
            fn=_delegate,
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "context_id": {"type": "string"},
                },
                "required": ["prompt"],
            },
        )
