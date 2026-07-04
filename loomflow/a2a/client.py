"""Client for remote A2A agents, usable as a loomflow tool (G10).

::

    from loomflow.a2a import A2AClient

    remote = A2AClient("https://bots.example/a2a")
    card = await remote.fetch_card()
    reply = await remote.send("summarize today's tickets")

    # ...or hand the remote agent to a local Agent as a delegate:
    agent = Agent("You coordinate.", tools=[remote.as_tool(name="ticket_bot")])

httpx is imported lazily on first use — the client works without it as
long as an ``http=`` object is injected (the pattern the tests use).
"""

from __future__ import annotations

import inspect
from typing import Any

from ..core.ids import new_id
from ..tools.registry import Tool
from .types import A2AError, AgentCard, Message, Task, message_text, text_message

__all__ = ["A2AClient"]

_CARD_PATH = "/.well-known/agent-card.json"
_LEGACY_CARD_PATH = "/.well-known/agent.json"


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

    def __init__(self, base_url: str, *, http: Any | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http
        self._owns_http = False

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

    async def _call(self, method: str, params: dict[str, Any]) -> Any:
        """POST one JSON-RPC request; return ``result`` or raise A2AError."""
        http = self._ensure_http()
        payload = {"jsonrpc": "2.0", "id": new_id("rpc"), "method": method, "params": params}
        response = await _maybe_await(http.post(self._base_url + "/", json=payload))
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
        """``message/send`` and return the agent's text reply.

        Pass the same ``context_id`` across calls to hold one remote
        conversation (the server maps it to its session). Raises
        :class:`A2AError` on JSON-RPC errors, malformed results, and
        tasks that come back ``failed``.
        """
        message = text_message("user", text, contextId=context_id)
        result = await self._call(
            "message/send", {"message": message.model_dump(exclude_none=True)}
        )
        if not isinstance(result, dict):
            raise A2AError("message/send: result is not an object")
        # The spec allows either a Task or a bare Message as the result.
        if result.get("kind") == "message":
            return message_text(Message.model_validate(result))
        task = Task.model_validate(result)
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
