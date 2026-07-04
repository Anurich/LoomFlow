"""Declarative MCP server descriptions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

#: User-supplied handler for MCP *sampling* requests (a server asking
#: the client for an LLM completion). Called as
#: ``handler(messages, model_preferences)`` where ``messages`` is the
#: SDK's list of ``SamplingMessage`` objects and ``model_preferences``
#: the (possibly ``None``) ``ModelPreferences``. May be sync or async;
#: must return the completion text (``str``). loomflow does NOT
#: auto-wire a Model here — sampling is opt-in and user-controlled.
SamplingHandler = Callable[[Any, Any], Any]


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

    # Optional handler for server-initiated sampling requests
    # (``sampling/createMessage``). Excluded from equality/hash so two
    # otherwise-identical specs still compare equal.
    sampling_handler: SamplingHandler | None = field(default=None, compare=False)

    @classmethod
    def stdio(
        cls,
        name: str,
        command: str,
        args: list[str] | tuple[str, ...] | None = None,
        env: dict[str, str] | None = None,
        *,
        description: str = "",
        sampling_handler: SamplingHandler | None = None,
    ) -> MCPServerSpec:
        """Spawn ``command`` as a subprocess and speak JSON-RPC over its stdio."""
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=tuple(args or ()),
            env=tuple((env or {}).items()),
            description=description,
            sampling_handler=sampling_handler,
        )

    @classmethod
    def http(
        cls,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        *,
        description: str = "",
        sampling_handler: SamplingHandler | None = None,
    ) -> MCPServerSpec:
        """Connect to ``url`` via Streamable HTTP transport."""
        return cls(
            name=name,
            transport="http",
            url=url,
            headers=tuple((headers or {}).items()),
            description=description,
            sampling_handler=sampling_handler,
        )
