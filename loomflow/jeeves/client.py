"""``JeevesGateway`` — convenience wrapper around the Jeeves MCP gateway.

The class is itself a :class:`~loomflow.core.protocols.ToolHost`; it
lazy-builds a one-server :class:`~loomflow.mcp.MCPRegistry` on first
use and forwards every protocol method to it. That means three usage
patterns work out of the box:

* **One-liner** — drop straight into ``Agent``::

      Agent("...", tools=JeevesGateway.from_env())

* **Compose with other MCP servers**::

      MCPRegistry([
          JeevesGateway.from_env().as_mcp_server(),
          MCPServerSpec.stdio("git", "uvx", ["mcp-server-git"]),
      ])

* **Build the registry directly** for explicit lifecycle management::

      gateway = JeevesGateway.from_env()
      registry = gateway.as_registry()
      async with registry:
          ...
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from ..core.errors import ConfigError
from ..core.types import ToolDef, ToolEvent, ToolResult
from ..mcp.registry import MCPRegistry
from ..mcp.spec import MCPServerSpec

JEEVES_DEFAULT_BASE_URL = "https://jeeves.works/mcp"
JEEVES_API_KEY_ENV = "JEEVES_API_KEY"
JEEVES_TOKEN_PREFIX = "jm_sk_"
JEEVES_DEFAULT_SERVER_NAME = "jeeves"


@dataclass(frozen=True)
class JeevesConfig:
    """Connection details for the Jeeves Gateway."""

    api_key: str
    base_url: str = JEEVES_DEFAULT_BASE_URL
    server_name: str = JEEVES_DEFAULT_SERVER_NAME


class JeevesGateway:
    """ToolHost-shaped wrapper around the Jeeves Gateway."""

    def __init__(
        self,
        config: JeevesConfig,
        *,
        registry: MCPRegistry | None = None,
    ) -> None:
        if not config.api_key:
            raise ConfigError("JeevesGateway requires a non-empty api_key")
        self._cfg = config
        # ``registry`` is an injection seam: production users leave it
        # ``None`` and we lazy-create a real one on first use; tests pass
        # a pre-built MCPRegistry with fake clients to bypass network.
        self._registry: MCPRegistry | None = registry

    # ---- factory ---------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        env_var: str = JEEVES_API_KEY_ENV,
        base_url: str | None = None,
        server_name: str = JEEVES_DEFAULT_SERVER_NAME,
    ) -> JeevesGateway:
        """Build a gateway from the ``JEEVES_API_KEY`` environment variable."""
        key = os.environ.get(env_var, "").strip()
        if not key:
            raise ConfigError(
                f"{env_var} env var is not set. Either export it with a "
                "Jeeves API key (jm_sk_...) or pass JeevesConfig(api_key=...) "
                "to JeevesGateway directly."
            )
        return cls(
            JeevesConfig(
                api_key=key,
                base_url=base_url or JEEVES_DEFAULT_BASE_URL,
                server_name=server_name,
            )
        )

    # ---- introspection ---------------------------------------------------

    @property
    def config(self) -> JeevesConfig:
        return self._cfg

    @property
    def server_name(self) -> str:
        return self._cfg.server_name

    # ---- spec / registry construction ------------------------------------

    def as_mcp_server(self) -> MCPServerSpec:
        """Return the :class:`MCPServerSpec` describing this gateway."""
        return MCPServerSpec.http(
            name=self._cfg.server_name,
            url=f"{self._cfg.base_url}/{self._cfg.api_key}",
            description="Jeeves Gateway",
        )

    def as_registry(self) -> MCPRegistry:
        """Return a one-server :class:`MCPRegistry` rooted at this gateway."""
        return MCPRegistry([self.as_mcp_server()])

    # ---- ToolHost protocol -- forward to underlying registry -------------

    def _ensure_registry(self) -> MCPRegistry:
        if self._registry is None:
            self._registry = self.as_registry()
        return self._registry

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        return await self._ensure_registry().list_tools(query=query)

    async def call(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> ToolResult:
        return await self._ensure_registry().call(tool, args, call_id=call_id)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        async for event in self._ensure_registry().watch():
            yield event

    # ---- lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        if self._registry is not None:
            try:
                await self._registry.aclose()
            finally:
                self._registry = None

    async def __aenter__(self) -> JeevesGateway:
        # Eagerly connect when used as an async context manager.
        await self._ensure_registry().connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


def looks_like_jeeves_key(value: str) -> bool:
    """Return ``True`` if ``value`` matches the Jeeves API-key shape.

    The check is intentionally permissive — it only verifies the
    well-known ``jm_sk_`` prefix so callers can warn on obviously-wrong
    inputs without blocking unconventional formats the server may
    accept.
    """
    return bool(value) and value.startswith(JEEVES_TOKEN_PREFIX)
