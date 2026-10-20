"""Pass-through sandbox baseline.

Wraps a :class:`ToolHost` without adding any restrictions. Exists so
the wrapping pattern is documented and tested, and so users can
construct an explicit "no isolation" layer in code review without
ambiguity.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from ...core.protocols import ToolHost
from ...core.types import ToolDef, ToolEvent, ToolResult


def register_inside(wrapper: Any, tool: Any) -> None:
    """Register ``tool`` on the host a sandbox wraps, so every call to
    it passes through the sandbox (used for tools added after
    construction, such as a skill's bundled tools). A wrapped host
    with no ``register`` gets an extension layer to hold it."""
    inner = wrapper._inner
    if not hasattr(inner, "register"):
        from ...architecture.tool_host_wrappers import ExtendedToolHost

        inner = ExtendedToolHost(inner, [])
        wrapper._inner = inner
    inner.register(tool)


class NoSandbox:
    """Pass-through wrapper around a :class:`ToolHost`."""

    def __init__(self, inner: ToolHost) -> None:
        self._inner = inner

    @property
    def inner(self) -> ToolHost:
        return self._inner

    def register(self, tool: Any) -> None:
        """Add ``tool`` under this sandbox (see :func:`register_inside`)."""
        register_inside(self, tool)

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        return await self._inner.list_tools(query=query)

    async def call(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> ToolResult:
        return await self._inner.call(tool, args, call_id=call_id)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        async for event in self._inner.watch():
            yield event
