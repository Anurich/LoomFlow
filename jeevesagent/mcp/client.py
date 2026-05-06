"""Per-server MCP client wrapping ``mcp.ClientSession`` lifetime.

The ``mcp`` SDK is imported lazily inside :meth:`MCPClient.connect`.
Tests can bypass the real connection entirely by passing a
``session=`` kwarg whose object exposes the methods we use:
``initialize()``, ``list_tools()``, ``call_tool(name, args)``.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from ..core.errors import MCPError
from .spec import MCPServerSpec


class MCPClient:
    """One client per MCP server. Holds the live ``ClientSession``."""

    def __init__(
        self,
        spec: MCPServerSpec,
        *,
        session: Any | None = None,
    ) -> None:
        self._spec = spec
        self._session: Any | None = session
        self._stack: AsyncExitStack | None = None

    # ---- properties -----------------------------------------------------

    @property
    def spec(self) -> MCPServerSpec:
        return self._spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    # ---- lifecycle ------------------------------------------------------

    async def connect(self) -> None:
        """Open the transport and initialise the session.

        No-op if already connected (or a fake session was injected at
        construction time).
        """
        if self._session is not None:
            return

        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            read, write = await self._open_transport(stack)
            try:
                from mcp.client.session import (  # type: ignore[import-not-found, import-untyped]
                    ClientSession,
                )
            except ImportError as exc:  # pragma: no cover — depends on user env
                raise MCPError(
                    "MCP SDK not installed. "
                    "Install with: pip install 'jeevesagent[mcp]'"
                ) from exc
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
            self._stack = stack
        except BaseException:
            await stack.aclose()
            raise

    async def aclose(self) -> None:
        """Tear down the session and underlying transport."""
        if self._stack is not None:
            try:
                await self._stack.aclose()
            finally:
                self._stack = None
                self._session = None
        else:
            self._session = None

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # ---- protocol surface -----------------------------------------------

    async def list_tools(self) -> list[Any]:
        """Return whatever the SDK gave us — a list of tool descriptors.

        Each descriptor has ``name``, ``description``, ``inputSchema``.
        We don't translate to :class:`ToolDef` here — the registry does
        that, since it also assigns names with disambiguation.
        """
        await self.connect()
        if self._session is None:
            raise MCPError(f"MCP client {self._spec.name!r}: session not initialised")
        result = await self._session.list_tools()
        return list(getattr(result, "tools", result) or [])

    async def call_tool(
        self, name: str, args: dict[str, Any]
    ) -> Any:
        """Invoke ``name`` with ``args``. Returns the SDK's CallToolResult."""
        await self.connect()
        if self._session is None:
            raise MCPError(f"MCP client {self._spec.name!r}: session not initialised")
        return await self._session.call_tool(name, args)

    # ---- transport plumbing ---------------------------------------------

    async def _open_transport(
        self, stack: AsyncExitStack
    ) -> tuple[Any, Any]:
        """Open the right transport for the spec; return ``(read, write)``."""
        if self._spec.transport == "stdio":
            try:
                from mcp.client.stdio import (  # type: ignore[import-not-found, import-untyped]
                    StdioServerParameters,
                    stdio_client,
                )
            except ImportError as exc:  # pragma: no cover
                raise MCPError(
                    "MCP SDK not installed. "
                    "Install with: pip install 'jeevesagent[mcp]'"
                ) from exc
            if not self._spec.command:
                raise MCPError(
                    f"stdio MCP spec {self._spec.name!r} has no command set"
                )
            params = StdioServerParameters(
                command=self._spec.command,
                args=list(self._spec.args),
                env=dict(self._spec.env) if self._spec.env else None,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            return read, write

        if self._spec.transport == "http":
            try:
                from mcp.client.streamable_http import (  # type: ignore[import-not-found, import-untyped]
                    streamablehttp_client,
                )
            except ImportError as exc:  # pragma: no cover
                raise MCPError(
                    "MCP SDK not installed. "
                    "Install with: pip install 'jeevesagent[mcp]'"
                ) from exc
            if not self._spec.url:
                raise MCPError(
                    f"http MCP spec {self._spec.name!r} has no url set"
                )
            ctx = streamablehttp_client(
                self._spec.url,
                headers=dict(self._spec.headers) if self._spec.headers else None,
            )
            triplet = await stack.enter_async_context(ctx)
            # streamablehttp_client returns (read, write, get_session_id)
            read, write = triplet[0], triplet[1]
            return read, write

        raise MCPError(f"unsupported transport: {self._spec.transport!r}")
