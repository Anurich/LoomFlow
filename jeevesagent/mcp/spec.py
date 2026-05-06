"""Declarative MCP server descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MCPServerSpec:
    """How to find and talk to a single MCP server.

    Construct via the class methods :meth:`stdio` or :meth:`http` rather
    than the bare constructor — they enforce the right combination of
    fields per transport.
    """

    name: str
    transport: Literal["stdio", "http"]

    # stdio transport
    command: str | None = None
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()

    # http transport
    url: str | None = None
    headers: tuple[tuple[str, str], ...] = ()

    # Optional notes (free-form, used in error messages)
    description: str = field(default="")

    @classmethod
    def stdio(
        cls,
        name: str,
        command: str,
        args: list[str] | tuple[str, ...] | None = None,
        env: dict[str, str] | None = None,
        *,
        description: str = "",
    ) -> MCPServerSpec:
        """Spawn ``command`` as a subprocess and speak JSON-RPC over its stdio."""
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=tuple(args or ()),
            env=tuple((env or {}).items()),
            description=description,
        )

    @classmethod
    def http(
        cls,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        *,
        description: str = "",
    ) -> MCPServerSpec:
        """Connect to ``url`` via Streamable HTTP transport."""
        return cls(
            name=name,
            transport="http",
            url=url,
            headers=tuple((headers or {}).items()),
            description=description,
        )
