"""``ToolHost`` backed by N MCP servers.

The registry connects all clients in parallel through an
``anyio.create_task_group``, builds a name index, and routes calls.

Tool name collisions across servers are auto-disambiguated:

* If a tool name is unique across all servers, agents see the bare name
  (``get_weather``).
* If two servers expose the same name, both are registered as
  ``server.tool`` (``city_api.get_weather``, ``noaa.get_weather``).

Either form is accepted at call time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import anyio

from ..core.types import ToolDef, ToolEvent, ToolResult
from .client import MCPClient
from .spec import MCPServerSpec


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
        self._tool_index: dict[str, tuple[str, ToolDef]] = {}
        self._connected = False

    # ---- introspection --------------------------------------------------

    @property
    def server_names(self) -> list[str]:
        return list(self._clients.keys())

    # ---- lifecycle ------------------------------------------------------

    async def connect(self) -> None:
        """Connect every client in parallel and rebuild the index."""
        if self._connected:
            return
        async with anyio.create_task_group() as tg:
            for client in self._clients.values():
                tg.start_soon(client.connect)
        await self.refresh()
        self._connected = True

    async def aclose(self) -> None:
        async with anyio.create_task_group() as tg:
            for client in self._clients.values():
                tg.start_soon(client.aclose)
        self._tool_index = {}
        self._connected = False

    async def __aenter__(self) -> MCPRegistry:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # ---- index management -----------------------------------------------

    async def refresh(self) -> None:
        """Re-pull tool lists from every client and rebuild the index."""
        per_server: dict[str, list[Any]] = {}
        async with anyio.create_task_group() as tg:

            async def _pull(server_name: str, client: MCPClient) -> None:
                per_server[server_name] = await client.list_tools()

            for server_name, client in self._clients.items():
                tg.start_soon(_pull, server_name, client)

        self._tool_index = _build_index(per_server)

    # ---- ToolHost protocol ----------------------------------------------

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        await self.connect()
        defs = [d for _, d in self._tool_index.values()]
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
        server_name, tool_def = entry
        client = self._clients[server_name]

        # The client expects the *unqualified* tool name (the name MCP
        # itself knows). When we qualified it as ``server.tool`` for
        # disambiguation, strip the prefix back off here.
        bare_name = tool_def.name.split(".", 1)[-1] if "." in tool_def.name else tool_def.name

        try:
            sdk_result = await client.call_tool(bare_name, dict(args))
        except Exception as exc:  # noqa: BLE001 — surface SDK errors as ToolResult
            return ToolResult.error_(call_id=call_id, message=str(exc))

        is_error = bool(getattr(sdk_result, "isError", False))
        output = _extract_output(sdk_result)
        if is_error:
            return ToolResult.error_(
                call_id=call_id,
                message=output if isinstance(output, str) else str(output),
            )
        return ToolResult.success(call_id=call_id, output=output)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        """``listChanged`` notifications. Not yet implemented; yields nothing."""
        empty: tuple[ToolEvent, ...] = ()
        for ev in empty:
            yield ev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_index(
    per_server: dict[str, list[Any]],
) -> dict[str, tuple[str, ToolDef]]:
    """Build the name index with auto-disambiguation.

    A name that's unique across all servers gets the bare key. A name
    that appears in multiple servers gets both ``server.name`` keys.
    """
    counts: dict[str, int] = {}
    for tools in per_server.values():
        for t in tools:
            tname = getattr(t, "name", None)
            if tname:
                counts[tname] = counts.get(tname, 0) + 1

    index: dict[str, tuple[str, ToolDef]] = {}
    for server_name, tools in per_server.items():
        for t in tools:
            tname = getattr(t, "name", None)
            if not tname:
                continue
            unique = counts.get(tname, 0) == 1
            key = tname if unique else f"{server_name}.{tname}"
            tool_def = ToolDef(
                name=key,
                description=getattr(t, "description", "") or "",
                input_schema=getattr(t, "inputSchema", None) or {"type": "object"},
                server=server_name,
            )
            index[key] = (server_name, tool_def)
    return index


def _extract_output(sdk_result: Any) -> Any:
    """Pull a usable Python value out of an MCP ``CallToolResult``.

    Preference order:

    1. ``structuredContent`` if present (newer MCP versions).
    2. Concatenated text from text-typed content blocks.
    3. The raw ``content`` list if no text blocks were found.
    """
    structured = getattr(sdk_result, "structuredContent", None)
    if structured is not None:
        return structured

    blocks = getattr(sdk_result, "content", None) or []
    text_pieces: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            text_pieces.append(text)

    if text_pieces:
        return text_pieces[0] if len(text_pieces) == 1 else "\n".join(text_pieces)
    return blocks
