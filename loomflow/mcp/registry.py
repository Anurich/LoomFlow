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

Prompts follow the same bare-when-unique / qualified-always naming
(``server.prompt``); resources are qualified with ``server:uri`` when
the same URI is exposed by more than one server.

listChanged design
------------------
Each client's session lives in its own portal thread; the SDK invokes
the message handler *inside that thread's event loop*. To avoid ever
blocking or deadlocking the portal, the notification path is split in
two: the client-side handler synchronously flips a lock-guarded
"stale" flag on the registry (:meth:`MCPRegistry._mark_stale` — safe
from any thread, never awaits), and the registry lazily re-pulls the
flagged server(s) on its next operation (:meth:`_drain_stale`), in
the caller's event loop. Index rebuilds diff old vs. new and emit
:class:`ToolEvent`\\ s to :meth:`watch` subscribers.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import anyio
from anyio.streams.memory import MemoryObjectSendStream

from ..core.errors import MCPError
from ..core.types import ToolDef, ToolEvent, ToolResult
from .client import MCPClient
from .spec import MCPServerSpec

logger = logging.getLogger(__name__)

#: Buffered ToolEvents per ``watch()`` subscriber; slow consumers
#: lose events beyond this rather than blocking index rebuilds.
_WATCH_BUFFER = 64


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
        # Servers flagged stale by a ``listChanged`` notification.
        # Written from client portal threads (via ``_mark_stale``),
        # drained in the registry's own loop — hence a *threading*
        # lock, not an anyio one (it's held for nanoseconds and never
        # across an await).
        self._stale_lock = threading.Lock()
        self._stale: set[str] = set()
        # Live ``watch()`` subscribers.
        self._watchers: list[MemoryObjectSendStream[ToolEvent]] = []
        # Resource / prompt routing state (built lazily on first list).
        self._resource_owners: dict[str, list[str]] = {}
        self._resources_pulled = False
        self._prompt_index: dict[str, tuple[str, str]] = {}
        self._prompts_pulled = False
        for client in clients.values():
            client.on_tools_changed = self._mark_stale

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
        with self._stale_lock:
            self._stale = set()
        self._resource_owners = {}
        self._resources_pulled = False
        self._prompt_index = {}
        self._prompts_pulled = False
        # End every watch() iterator.
        watchers, self._watchers = self._watchers, []
        for send in watchers:
            send.close()
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
        self._set_index(_build_index(per_server))

    async def _refresh_server(self, server_name: str) -> None:
        """Re-pull ONE server's tools and rebuild the index from the
        cached per-server lists.

        Used after :meth:`_reset_client` and after a ``listChanged``
        notification, so a single change doesn't re-list every healthy
        server. Raises whatever the pull raises — the caller treats
        the refresh as best-effort.
        """
        client = self._clients.get(server_name)
        if client is None:
            return
        self._per_server_tools[server_name] = await client.list_tools()
        self._unavailable.discard(server_name)
        self._set_index(_build_index(self._per_server_tools))

    def _set_index(self, new_index: dict[str, _IndexEntry]) -> None:
        """Swap in a rebuilt index, emitting diff events to watchers."""
        old_index = self._tool_index
        self._tool_index = new_index
        if not self._watchers:
            return
        old_defs = _defs_by_display_name(old_index)
        new_defs = _defs_by_display_name(new_index)
        events: list[ToolEvent] = []
        for name, entry in new_defs.items():
            previous = old_defs.get(name)
            if previous is None:
                events.append(
                    ToolEvent(kind="added", tool=name, server=entry.server)
                )
            elif previous.tool_def != entry.tool_def:
                events.append(
                    ToolEvent(kind="updated", tool=name, server=entry.server)
                )
        for name, entry in old_defs.items():
            if name not in new_defs:
                events.append(
                    ToolEvent(kind="removed", tool=name, server=entry.server)
                )
        for send in list(self._watchers):
            for event in events:
                try:
                    send.send_nowait(event)
                except (
                    anyio.WouldBlock,
                    anyio.ClosedResourceError,
                    anyio.BrokenResourceError,
                ):
                    # Slow or gone subscriber — never block a refresh.
                    break

    # ---- listChanged plumbing ---------------------------------------------

    def _mark_stale(self, server_name: str) -> None:
        """Flag ``server_name`` for a lazy targeted re-pull.

        Called from :attr:`MCPClient.on_tools_changed`, i.e. possibly
        from a client's portal thread — must stay synchronous,
        non-blocking, and thread-safe (see module docstring).
        """
        with self._stale_lock:
            self._stale.add(server_name)

    async def _drain_stale(self) -> None:
        """Re-pull any servers flagged by ``listChanged`` notifications.

        Runs in the registry caller's event loop (never the portal
        thread). Best-effort: a failed re-pull keeps the previous
        (possibly stale) tool list and logs — the server will heal via
        the normal reconnect-and-retry path on its next call.
        """
        with self._stale_lock:
            if not self._stale:
                return
            stale, self._stale = self._stale, set()
        for server_name in stale:
            try:
                await self._refresh_server(server_name)
            except Exception as exc:  # noqa: BLE001 — keep the previous index
                logger.warning(
                    "MCP server %r: re-pull after listChanged failed; "
                    "keeping previous tool list: %s",
                    server_name,
                    exc,
                )

    # ---- ToolHost protocol ----------------------------------------------

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        await self.connect()
        await self._drain_stale()
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
        await self._drain_stale()
        entry = self._tool_index.get(tool)
        if entry is None:
            return ToolResult.error_(
                call_id=call_id, message=f"unknown MCP tool: {tool}"
            )
        mcp_name = entry.mcp_name

        async def _do_call(client: MCPClient) -> Any:
            return await client.call_tool(mcp_name, dict(args))

        try:
            sdk_result = await self._call_on_server(entry.server, _do_call)
        except Exception as exc:  # noqa: BLE001 — surface as ToolResult.error_
            return ToolResult.error_(call_id=call_id, message=str(exc))

        is_error = bool(getattr(sdk_result, "isError", False))
        output = _extract_output(sdk_result)
        if is_error:
            return ToolResult.error_(
                call_id=call_id,
                message=output if isinstance(output, str) else str(output),
            )
        return ToolResult.success(call_id=call_id, output=output)

    async def _call_on_server(
        self,
        server_name: str,
        op: Callable[[MCPClient], Awaitable[Any]],
    ) -> Any:
        """Run ``op`` against one server with reconnect-and-retry.

        The session may have died (network drop, server restart,
        broken pipe). Reset it and retry ONCE so the agent loop
        self-heals without bubbling a transport hiccup up as a
        permanent failure. We ONLY do this for errors that look like
        connection/transport failures: an execution error may mean the
        server already performed a side effect, and silently re-running
        it could repeat that side effect. We do NOT retry beyond once —
        repeated failures get surfaced to the caller, who can decide
        whether the underlying operation is idempotent.

        Raises the FIRST exception when it isn't retryable or the
        reconnect itself fails (that's what explains the failure to the
        caller); raises the second exception if the retry also fails.
        """
        client = self._clients[server_name]
        try:
            return await op(client)
        except Exception as first_exc:
            if not _is_connection_error(first_exc):
                raise
            if not await self._reset_client(server_name):
                raise first_exc from None
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
            return await op(self._clients[server_name])

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
        fresh.on_tools_changed = self._mark_stale
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

    def watch(self) -> AsyncIterator[ToolEvent]:
        """Yield :class:`ToolEvent` diffs as the tool index changes.

        Fires whenever a refresh (full, targeted, or listChanged-
        driven) rebuilds the index with additions/removals/updates.
        The subscriber is registered eagerly (at call time, not first
        iteration) so no event between subscribe and iterate is lost.
        Iterates until the registry is closed (or the subscriber
        breaks out); each subscriber has a bounded buffer — events
        beyond it are dropped rather than blocking refreshes.
        """
        send, receive = anyio.create_memory_object_stream[ToolEvent](_WATCH_BUFFER)
        self._watchers.append(send)

        async def _iterate() -> AsyncIterator[ToolEvent]:
            try:
                async with receive:
                    async for event in receive:
                        yield event
            finally:
                if send in self._watchers:
                    self._watchers.remove(send)
                send.close()

        return _iterate()

    # ---- resources --------------------------------------------------------

    async def list_resources(self) -> list[dict[str, Any]]:
        """Aggregate resource listings across every server.

        Returns one dict per resource:
        ``{"uri", "server", "name", "description", "mime_type"}``.
        ``uri`` is the server's own URI when unique across servers,
        or ``server:uri``-qualified when two servers expose the same
        URI. Per-server failures are isolated (logged + skipped),
        mirroring :meth:`refresh`.
        """
        await self.connect()
        await self._drain_stale()
        per_server = await self._pull_all("list_resources")
        counts: dict[str, int] = {}
        for resources in per_server.values():
            for resource in resources:
                uri = _uri_of(resource)
                if uri:
                    counts[uri] = counts.get(uri, 0) + 1
        owners: dict[str, list[str]] = {}
        out: list[dict[str, Any]] = []
        for server_name, resources in per_server.items():
            for resource in resources:
                uri = _uri_of(resource)
                if not uri:
                    continue
                owners.setdefault(uri, []).append(server_name)
                display = uri if counts[uri] == 1 else f"{server_name}:{uri}"
                out.append(
                    {
                        "uri": display,
                        "server": server_name,
                        "name": getattr(resource, "name", "") or "",
                        "description": getattr(resource, "description", "") or "",
                        "mime_type": getattr(resource, "mimeType", None),
                    }
                )
        self._resource_owners = owners
        self._resources_pulled = True
        return out

    async def read_resource(
        self, uri: str, *, server: str | None = None
    ) -> Any:
        """Read one resource, routing to the owning server.

        Accepts a bare URI (routed via the last listing when exactly
        one server owns it), a ``server:uri``-qualified URI, or an
        explicit ``server=`` override (which also allows reading URIs
        the server never listed, e.g. resource-template expansions).

        Text contents are returned verbatim (single block → ``str``);
        blob contents become a ``{"mime": ..., "size": ...}``
        placeholder, consistent with how binary tool-result blocks are
        represented. Multiple content blocks come back as a list.
        """
        await self.connect()
        await self._drain_stale()
        server_name, bare_uri = await self._route_resource(uri, server)

        async def _do_read(client: MCPClient) -> Any:
            return await client.read_resource(bare_uri)

        result = await self._call_on_server(server_name, _do_read)
        return _extract_resource_contents(result)

    async def _route_resource(
        self, uri: str, server: str | None
    ) -> tuple[str, str]:
        """Resolve ``uri`` (+ optional explicit ``server``) to
        ``(server_name, bare_uri)`` or raise :class:`MCPError`."""
        if server is not None:
            if server not in self._clients:
                raise MCPError(f"unknown MCP server: {server}")
            prefix = f"{server}:"
            bare = uri[len(prefix):] if uri.startswith(prefix) else uri
            return server, bare
        head, sep, rest = uri.partition(":")
        if sep and head in self._clients:
            # ``server:uri`` qualification. A server named after a URI
            # scheme (e.g. "file") would shadow bare URIs of that
            # scheme — the explicit ``server=`` argument disambiguates.
            return head, rest
        if not self._resources_pulled:
            await self.list_resources()
        owners = self._resource_owners.get(uri, [])
        if len(owners) == 1:
            return owners[0], uri
        if len(owners) > 1:
            raise MCPError(
                f"resource {uri!r} is exposed by multiple servers "
                f"({', '.join(sorted(owners))}); qualify as 'server:uri' "
                f"or pass server="
            )
        raise MCPError(f"unknown MCP resource: {uri}")

    # ---- prompts ----------------------------------------------------------

    async def list_prompts(self) -> list[dict[str, Any]]:
        """Aggregate prompt listings across every server.

        Returns one dict per prompt:
        ``{"name", "server", "description", "arguments"}``. Names use
        the same bare-when-unique / ``server.name``-qualified scheme
        as tools; both forms are accepted by :meth:`get_prompt`.
        Per-server failures are isolated (logged + skipped).
        """
        await self.connect()
        await self._drain_stale()
        per_server = await self._pull_all("list_prompts")
        counts: dict[str, int] = {}
        for prompts in per_server.values():
            for prompt in prompts:
                pname = getattr(prompt, "name", None)
                if pname:
                    counts[pname] = counts.get(pname, 0) + 1
        index: dict[str, tuple[str, str]] = {}
        out: list[dict[str, Any]] = []
        for server_name, prompts in per_server.items():
            for prompt in prompts:
                pname = getattr(prompt, "name", None)
                if not pname:
                    continue
                unique = counts.get(pname, 0) == 1
                display = pname if unique else f"{server_name}.{pname}"
                index[f"{server_name}.{pname}"] = (server_name, pname)
                if unique:
                    index[pname] = (server_name, pname)
                out.append(
                    {
                        "name": display,
                        "server": server_name,
                        "description": getattr(prompt, "description", "") or "",
                        "arguments": getattr(prompt, "arguments", None),
                    }
                )
        self._prompt_index = index
        self._prompts_pulled = True
        return out

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Fetch one prompt (bare or ``server.name``-qualified).

        Returns ``{"description": str | None, "messages": [{"role",
        "content"}, ...]}`` with text content verbatim and binary
        blocks as placeholders.
        """
        await self.connect()
        await self._drain_stale()
        if not self._prompts_pulled:
            await self.list_prompts()
        entry = self._prompt_index.get(name)
        if entry is None:
            raise MCPError(f"unknown MCP prompt: {name}")
        server_name, bare_name = entry

        async def _do_get(client: MCPClient) -> Any:
            return await client.get_prompt(bare_name, arguments)

        result = await self._call_on_server(server_name, _do_get)
        return _extract_prompt(result)

    async def _pull_all(self, method: str) -> dict[str, list[Any]]:
        """Call ``method`` (``list_resources`` / ``list_prompts``) on
        every client in parallel, isolating per-server failures the
        same way :meth:`refresh` does for tools."""
        per_server: dict[str, list[Any]] = {}
        async with anyio.create_task_group() as tg:

            async def _pull(server_name: str, client: MCPClient) -> None:
                try:
                    per_server[server_name] = await getattr(client, method)()
                except Exception as exc:  # noqa: BLE001 — isolate flaky servers
                    logger.warning(
                        "MCP server %r failed %s; skipping it: %s",
                        server_name,
                        method,
                        exc,
                    )

            for server_name, client in self._clients.items():
                tg.start_soon(_pull, server_name, client)
        return per_server


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


def _defs_by_display_name(
    index: dict[str, _IndexEntry],
) -> dict[str, _IndexEntry]:
    """Dedupe an index (bare + qualified keys may alias one entry)
    down to one entry per outward-facing tool name."""
    return {entry.tool_def.name: entry for entry in index.values()}


def _uri_of(resource: Any) -> str:
    """Stringify a resource descriptor's ``uri`` (SDK gives AnyUrl)."""
    uri = getattr(resource, "uri", None)
    return str(uri) if uri else ""


def _base64_size(data: str) -> int:
    """Approximate decoded byte size of a base64 payload."""
    return (len(data.rstrip("=")) * 3) // 4


def _blob_placeholder(content: Any) -> dict[str, Any] | None:
    """``{"mime", "size"}`` stand-in for a blob resource content block,
    consistent with how binary tool-result blocks are represented."""
    blob = getattr(content, "blob", None)
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray)):
        size = len(blob)
    elif isinstance(blob, str):
        size = _base64_size(blob)
    else:
        return None
    return {"mime": getattr(content, "mimeType", None), "size": size}


def _extract_resource_contents(sdk_result: Any) -> Any:
    """Pull usable values out of an MCP ``ReadResourceResult``.

    Text contents come back verbatim; blob contents as
    ``{"mime", "size"}`` placeholders. A single content block is
    unwrapped; multiple blocks return a list. An empty result returns
    the raw ``contents`` list (usually ``[]``).
    """
    contents = getattr(sdk_result, "contents", None) or []
    pieces: list[Any] = []
    for content in contents:
        text = getattr(content, "text", None)
        if isinstance(text, str):
            pieces.append(text)
            continue
        placeholder = _blob_placeholder(content)
        if placeholder is not None:
            pieces.append(placeholder)
    if not pieces:
        return contents
    return pieces[0] if len(pieces) == 1 else pieces


def _extract_prompt(sdk_result: Any) -> dict[str, Any]:
    """Flatten an MCP ``GetPromptResult`` into a plain dict.

    ``{"description": ..., "messages": [{"role", "content"}, ...]}``
    with text content verbatim and non-text blocks represented by the
    same minimal placeholder used for tool results.
    """
    messages: list[dict[str, Any]] = []
    for message in getattr(sdk_result, "messages", None) or []:
        block = getattr(message, "content", None)
        text = getattr(block, "text", None)
        content: Any
        if isinstance(text, str):
            content = text
        else:
            content = _describe_binary_block(block) or block
        messages.append(
            {"role": getattr(message, "role", None), "content": content}
        )
    return {
        "description": getattr(sdk_result, "description", None),
        "messages": messages,
    }


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
