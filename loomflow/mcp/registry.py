"""``ToolHost`` backed by N MCP servers.

The registry connects all clients in parallel through an
``anyio.create_task_group``, builds a name index, and routes calls.

Tool name collisions across servers are auto-disambiguated:

* If a tool name is unique across all servers, agents see the bare name
  (``get_weather``).
* If two servers expose the same name, both are registered as
  ``server.tool`` (``city_api.get_weather``, ``noaa.get_weather``).

Either form is accepted at call time: the qualified ``server.tool``
key is always indexed, and the bare name is indexed too whenever it
is unambiguous.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import anyio

from ..core.errors import MCPError
from ..core.types import ToolDef, ToolEvent, ToolResult
from .client import MCPClient
from .spec import MCPServerSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _IndexEntry:
    """One routable tool: which server owns it, the *unqualified*
    name that server knows it by, and the outward-facing ToolDef."""

    server: str
    mcp_name: str
    tool_def: ToolDef


class MCPRegistry:
    """Aggregates many :class:`MCPClient` instances into a single ``ToolHost``."""

    def __init__(
        self,
        items: list[MCPServerSpec | MCPClient] | None = None,
    ) -> None:
        clients: dict[str, MCPClient] = {}
        for item in items or []:
            if isinstance(item, MCPClient):
                clients[item.name] = item
            elif isinstance(item, MCPServerSpec):
                clients[item.name] = MCPClient(item)
            else:
                raise TypeError(
                    f"MCPRegistry items must be MCPServerSpec or MCPClient, "
                    f"got {type(item).__name__}"
                )
        self._clients = clients
        self._tool_index: dict[str, _IndexEntry] = {}
        # Cached per-server tool lists (raw SDK descriptors) from the
        # last successful pull; lets ``_refresh_server`` rebuild the
        # full index after re-listing only ONE server.
        self._per_server_tools: dict[str, list[Any]] = {}
        # Servers whose last ``list_tools`` pull failed — skipped from
        # the index but retried on the next ``refresh()``.
        self._unavailable: set[str] = set()
        self._connected = False

    # ---- introspection --------------------------------------------------

    @property
    def server_names(self) -> list[str]:
        return list(self._clients.keys())

    @property
    def unavailable(self) -> set[str]:
        """Names of servers whose last tool pull failed.

        These servers contribute no tools to the index; a subsequent
        :meth:`refresh` (or a successful targeted re-pull) clears
        them. Returned as a copy — mutate-proof for callers.
        """
        return set(self._unavailable)

    # ---- lifecycle ------------------------------------------------------

    async def connect(self) -> None:
        """Connect every client in parallel and rebuild the index.

        Per-client connect failures are isolated (logged, not
        raised): the failing server surfaces in :attr:`unavailable`
        after :meth:`refresh` because its ``list_tools`` pull re-
        attempts the connect and fails there. One dead server must
        not make every other server unusable.
        """
        if self._connected:
            return

        async def _connect_one(client: MCPClient) -> None:
            try:
                await client.connect()
            except Exception as exc:  # noqa: BLE001 — isolate per-server failures
                logger.warning(
                    "MCP server %r failed to connect: %s", client.name, exc
                )

        async with anyio.create_task_group() as tg:
            for client in self._clients.values():
                tg.start_soon(_connect_one, client)
        await self.refresh()
        self._connected = True

    async def aclose(self) -> None:
        """Close every client, isolating per-client failures.

        Each close is wrapped so one raising client can't cancel its
        siblings mid-teardown (which would leak portal threads /
        subprocesses). Every client is always closed; collected
        errors are re-raised afterwards as a single
        :class:`ExceptionGroup`.
        """
        errors: list[Exception] = []

        async def _close_one(client: MCPClient) -> None:
            try:
                await client.aclose()
            except Exception as exc:  # noqa: BLE001 — close the rest regardless
                errors.append(exc)

        async with anyio.create_task_group() as tg:
            for client in self._clients.values():
                tg.start_soon(_close_one, client)
        self._tool_index = {}
        self._per_server_tools = {}
        self._unavailable = set()
        self._connected = False
        if errors:
            raise ExceptionGroup("errors while closing MCP clients", errors)

    async def __aenter__(self) -> MCPRegistry:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # ---- index management -----------------------------------------------

    async def refresh(self) -> None:
        """Re-pull tool lists from every client and rebuild the index.

        Per-server failures are isolated: a server whose
        ``list_tools`` raises is skipped (warning logged, name
        recorded in :attr:`unavailable`) so one flaky server doesn't
        take down every other server's tools. Failed servers get
        retried on the next ``refresh()``.
        """
        per_server: dict[str, list[Any]] = {}
        failed: set[str] = set()
        async with anyio.create_task_group() as tg:

            async def _pull(server_name: str, client: MCPClient) -> None:
                try:
                    per_server[server_name] = await client.list_tools()
                except Exception as exc:  # noqa: BLE001 — isolate flaky servers
                    failed.add(server_name)
                    logger.warning(
                        "MCP server %r failed to list tools; "
                        "skipping its tools: %s",
                        server_name,
                        exc,
                    )

            for server_name, client in self._clients.items():
                tg.start_soon(_pull, server_name, client)

        self._per_server_tools = per_server
        self._unavailable = failed
        self._tool_index = _build_index(per_server)

    async def _refresh_server(self, server_name: str) -> None:
        """Re-pull ONE server's tools and rebuild the index from the
        cached per-server lists.

        Used after :meth:`_reset_client` so a single reconnect
        doesn't re-list every healthy server. Raises whatever the
        pull raises — the caller treats the refresh as best-effort.
        """
        client = self._clients.get(server_name)
        if client is None:
            return
        self._per_server_tools[server_name] = await client.list_tools()
        self._unavailable.discard(server_name)
        self._tool_index = _build_index(self._per_server_tools)

    # ---- ToolHost protocol ----------------------------------------------

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        await self.connect()
        # A tool can be indexed under two keys (bare + qualified);
        # dedupe by entry identity so each tool is listed once.
        defs: list[ToolDef] = []
        seen: set[int] = set()
        for entry in self._tool_index.values():
            if id(entry) not in seen:
                seen.add(id(entry))
                defs.append(entry.tool_def)
        if query:
            q = query.lower()
            defs = [
                d
                for d in defs
                if q in d.name.lower() or q in d.description.lower()
            ]
        return defs

    async def call(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> ToolResult:
        await self.connect()
        entry = self._tool_index.get(tool)
        if entry is None:
            return ToolResult.error_(
                call_id=call_id, message=f"unknown MCP tool: {tool}"
            )
        server_name = entry.server
        client = self._clients[server_name]

        try:
            sdk_result = await client.call_tool(entry.mcp_name, dict(args))
        except Exception as first_exc:  # noqa: BLE001 — see retry path
            # The session may have died (network drop, server restart,
            # broken pipe). Reset it and retry ONCE so the agent loop
            # self-heals without bubbling a transport hiccup up as a
            # permanent tool failure. We ONLY do this for errors that
            # look like connection/transport failures: a tool-execution
            # error may mean the server already performed a side effect,
            # and silently re-running it could repeat that side effect.
            # We do NOT retry beyond once — repeated failures get
            # surfaced to the caller, who can decide whether the
            # underlying tool call is idempotent.
            if not _is_connection_error(first_exc):
                return ToolResult.error_(call_id=call_id, message=str(first_exc))
            reset_ok = await self._reset_client(server_name)
            if not reset_ok:
                return ToolResult.error_(call_id=call_id, message=str(first_exc))
            # The reconnected server may expose a different tool set
            # (it might have restarted with new capabilities); re-pull
            # THIS server only — the healthy siblings' cached tool
            # lists are still good, so a full refresh would be N-1
            # wasted round-trips. Best-effort: a refresh failure must
            # not mask the retry below.
            try:
                await self._refresh_server(server_name)
            except Exception:  # noqa: BLE001 — keep the (stale) index
                pass
            # _reset_client swaps in a fresh client; re-fetch so we
            # don't keep talking to the broken one.
            client = self._clients[server_name]
            try:
                sdk_result = await client.call_tool(entry.mcp_name, dict(args))
            except Exception as second_exc:  # noqa: BLE001
                return ToolResult.error_(
                    call_id=call_id, message=str(second_exc)
                )

        is_error = bool(getattr(sdk_result, "isError", False))
        output = _extract_output(sdk_result)
        if is_error:
            return ToolResult.error_(
                call_id=call_id,
                message=output if isinstance(output, str) else str(output),
            )
        return ToolResult.success(call_id=call_id, output=output)

    async def _reset_client(self, server_name: str) -> bool:
        """Tear down + reopen one client's session. Returns ``True``
        if reconnection succeeded, ``False`` otherwise.

        Errors during close are swallowed — the existing session is
        already considered broken, so we just want a fresh one. A
        ``False`` return means the reset itself failed; callers
        should not retry against this client until they try again
        from a healthy state.
        """
        client = self._clients.get(server_name)
        if client is None:
            return False
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001 — the session is broken; closing may fail
            pass
        # Replace with a fresh client bound to the same spec so the
        # AsyncExitStack inside the closed one stays out of our way.
        fresh = self._make_client(client.spec)
        self._clients[server_name] = fresh
        try:
            await fresh.connect()
        except Exception:  # noqa: BLE001 — surface as ToolResult.error_
            return False
        return True

    def _make_client(self, spec: MCPServerSpec) -> MCPClient:
        """Construct a fresh :class:`MCPClient` for ``spec``.

        Subclass / monkeypatch hook used by :meth:`_reset_client` so
        tests can inject a pre-baked fake client after a reconnect
        without touching the production constructor path.
        """
        return MCPClient(spec)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        """``listChanged`` notifications. Not yet implemented; yields nothing."""
        empty: tuple[ToolEvent, ...] = ()
        for ev in empty:
            yield ev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_connection_error(exc: BaseException) -> bool:
    """Heuristic: does ``exc`` look like a connect/transport failure
    (safe to reconnect-and-retry) rather than a tool-execution error
    (retrying could repeat a server-side side effect)?
    """
    if isinstance(
        exc,
        (
            ConnectionError,  # incl. BrokenPipeError / ConnectionResetError
            EOFError,
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
            anyio.EndOfStream,
            MCPError,  # raised only from the client's connect phase
        ),
    ):
        return True
    # httpx connect-phase errors (streamable-http transport), detected
    # structurally so we don't have to import httpx. Only errors raised
    # BEFORE a request could have been delivered qualify — a mid-call
    # read/write failure may mean the server already ran the tool.
    for klass in type(exc).__mro__:
        if klass.__module__.split(".", 1)[0] == "httpx" and klass.__name__ in (
            "ConnectError",
            "ConnectTimeout",
        ):
            return True
    return False


def _annotated_destructive(t: Any) -> bool:
    """Map MCP tool annotations onto :attr:`ToolDef.destructive`.

    ``destructiveHint`` is authoritative when present; a
    ``readOnlyHint`` of ``False`` is a weaker fallback signal used
    only when ``destructiveHint`` is absent.
    """
    ann = getattr(t, "annotations", None)
    if ann is None:
        return False

    def _get(key: str) -> Any:
        if isinstance(ann, Mapping):
            return ann.get(key)
        return getattr(ann, key, None)

    destructive_hint = _get("destructiveHint")
    if destructive_hint is not None:
        return bool(destructive_hint)
    return _get("readOnlyHint") is False


def _build_index(
    per_server: dict[str, list[Any]],
) -> dict[str, _IndexEntry]:
    """Build the name index with auto-disambiguation.

    Every tool is indexed under its qualified ``server.name`` key.
    A name that's unique across all servers is ALSO indexed under
    the bare key (and displayed with the bare name); a name that
    appears in multiple servers is displayed qualified and gets no
    bare key (it would be ambiguous).
    """
    counts: dict[str, int] = {}
    for tools in per_server.values():
        for t in tools:
            tname = getattr(t, "name", None)
            if tname:
                counts[tname] = counts.get(tname, 0) + 1

    index: dict[str, _IndexEntry] = {}
    for server_name, tools in per_server.items():
        for t in tools:
            tname = getattr(t, "name", None)
            if not tname:
                continue
            unique = counts.get(tname, 0) == 1
            display = tname if unique else f"{server_name}.{tname}"
            tool_def = ToolDef(
                name=display,
                description=getattr(t, "description", "") or "",
                input_schema=getattr(t, "inputSchema", None) or {"type": "object"},
                server=server_name,
                destructive=_annotated_destructive(t),
            )
            entry = _IndexEntry(
                server=server_name, mcp_name=tname, tool_def=tool_def
            )
            index[f"{server_name}.{tname}"] = entry
            if unique:
                index[tname] = entry
    return index


def _describe_binary_block(block: Any) -> str | None:
    """Minimal textual stand-in for a non-text content block (image /
    audio / embedded binary resource) so the model learns the tool
    returned *something* instead of the block being dropped.
    """
    resource = getattr(block, "resource", None)
    data = getattr(block, "data", None)
    if data is None and resource is not None:
        data = getattr(resource, "blob", None)
    if data is None:
        return None
    mime = getattr(block, "mimeType", None)
    if mime is None and resource is not None:
        mime = getattr(resource, "mimeType", None)
    block_type = getattr(block, "type", None) or "binary"
    if isinstance(data, (bytes, bytearray)):
        size = len(data)
    elif isinstance(data, str):
        # base64 payload — report the (approximate) decoded size.
        size = (len(data.rstrip("=")) * 3) // 4
    else:
        return None
    return f"[{block_type}: {mime or 'unknown media type'}, {size} bytes]"


def _extract_output(sdk_result: Any) -> Any:
    """Pull a usable Python value out of an MCP ``CallToolResult``.

    Preference order:

    1. ``structuredContent`` if present (newer MCP versions).
    2. Concatenated text from text-typed content blocks, with
       non-text blocks (images, audio, binary resources) represented
       by a minimal ``[image: image/png, N bytes]`` placeholder.
    3. The raw ``content`` list if nothing representable was found.
    """
    structured = getattr(sdk_result, "structuredContent", None)
    if structured is not None:
        return structured

    blocks = getattr(sdk_result, "content", None) or []
    pieces: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            pieces.append(text)
            continue
        placeholder = _describe_binary_block(block)
        if placeholder is not None:
            pieces.append(placeholder)

    if pieces:
        return pieces[0] if len(pieces) == 1 else "\n".join(pieces)
    return blocks
