"""Tool-host wrappers used by multi-agent architectures.

A :class:`~loomflow.core.protocols.ToolHost` is a black box from
the agent's point of view. Architectures that need to inject extra
tools per run (Supervisor's ``delegate``, Swarm's ``handoff``, the
``Agent.run(extra_tools=...)`` per-run kwarg) build an
:class:`ExtendedToolHost` that combines a base host with a fixed
list of extra :class:`Tool` instances.

Why a wrapper rather than mutating the base host?

* User-provided agents stay untouched — running an agent inside a
  supervisor doesn't permanently add a ``delegate`` tool to that
  agent's host.
* Additive only — extras coexist with the base's tools; conflicts
  resolve in favour of the extras (the architecture's tool wins
  over a same-named user tool, which is what the architecture wants).
* Same shape as the :class:`ToolHost` protocol — drop-in.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

from ..core.types import ToolDef, ToolEvent, ToolResult

if TYPE_CHECKING:
    from ..core.protocols import ToolHost
    from ..tools.registry import Tool


class ExtendedToolHost:
    """Combine a base :class:`ToolHost` with N extra :class:`Tool`\\ s.

    ``list_tools`` returns the base's defs plus the extras' defs.
    ``call`` dispatches to the matching extra by name; falls through
    to the base for everything else. Extras win on name conflict.
    """

    def __init__(
        self, base: ToolHost, extras: list[Tool]
    ) -> None:
        self._base = base
        self._extras = list(extras)
        self._extras_by_name: dict[str, Tool] = {
            t.name: t for t in extras
        }

    def register(self, item: Tool) -> Tool:
        """Mutably append a Tool to the extras pool.

        Mirrors :meth:`InProcessToolHost.register` so callers
        (notably the skills system, which lazy-registers Tools when
        ``load_skill`` fires) can add to either host kind without
        special-casing."""
        self._extras.append(item)
        self._extras_by_name[item.name] = item
        return item

    async def list_tools(
        self, *, query: str | None = None
    ) -> list[ToolDef]:
        defs = list(await self._base.list_tools(query=query))
        for t in self._extras:
            d = t.to_def()
            if query is None or (
                query.lower() in d.name.lower()
                or query.lower() in d.description.lower()
            ):
                defs.append(d)
        return defs

    async def call(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> ToolResult:
        extra = self._extras_by_name.get(tool)
        if extra is not None:
            try:
                output = await extra.execute(args)
            except Exception as exc:  # noqa: BLE001
                return ToolResult.error_(
                    call_id=call_id, message=str(exc)
                )
            return ToolResult.success(call_id=call_id, output=output)
        return await self._base.call(tool, args, call_id=call_id)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        async for ev in self._base.watch():
            yield ev
