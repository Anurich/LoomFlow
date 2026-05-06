"""Permission decisions for tool calls.

Three modes mirror the Claude Agent SDK so users don't relearn:

- ``DEFAULT`` — allow non-destructive tools, ask on destructive
- ``ACCEPT_EDITS`` — auto-approve filesystem writes; otherwise like default
- ``BYPASS`` — allow everything (CI / sandbox use only)

Allow- and deny-lists win over modes; deny-list wins over allow-list.
The decision flow:

    1. Tool in deny-list → deny
    2. Allow-list set and tool not in it → deny
    3. Mode == BYPASS → allow
    4. Mode == ACCEPT_EDITS and call is a non-destructive edit → allow
    5. Tool is destructive → ask
    6. Otherwise → allow
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from ..core.types import PermissionDecision, ToolCall


class Mode(StrEnum):
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS = "bypassPermissions"


class AllowAll:
    """Trivial permission policy: every call is allowed.

    The default for :class:`Agent` when no permissions are configured.
    """

    async def check(
        self, call: ToolCall, *, context: Mapping[str, Any]
    ) -> PermissionDecision:
        return PermissionDecision.allow_()


class StandardPermissions:
    """Mode + allow/deny-list permission policy."""

    def __init__(
        self,
        *,
        mode: Mode = Mode.DEFAULT,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
    ) -> None:
        self._mode = mode
        self._allowed = set(allowed_tools) if allowed_tools is not None else None
        self._denied = set(denied_tools or [])

    async def check(
        self, call: ToolCall, *, context: Mapping[str, Any]
    ) -> PermissionDecision:
        if call.tool in self._denied:
            return PermissionDecision.deny_(f"{call.tool}: denied by policy")
        if self._allowed is not None and call.tool not in self._allowed:
            return PermissionDecision.deny_(f"{call.tool}: not in allow-list")
        if self._mode == Mode.BYPASS:
            return PermissionDecision.allow_()
        if call.is_destructive() and self._mode != Mode.ACCEPT_EDITS:
            return PermissionDecision.ask_(
                f"{call.tool}: destructive call requires approval"
            )
        return PermissionDecision.allow_()

    @classmethod
    def strict(cls) -> StandardPermissions:
        """Convenience: default-mode permissions with no overrides."""
        return cls(mode=Mode.DEFAULT)
