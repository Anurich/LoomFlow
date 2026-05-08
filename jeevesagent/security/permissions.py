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
        self,
        call: ToolCall,
        *,
        context: Mapping[str, Any],
        user_id: str | None = None,
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
        self,
        call: ToolCall,
        *,
        context: Mapping[str, Any],
        user_id: str | None = None,
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


class PerUserPermissions:
    """Map ``user_id`` to a different permission policy.

    The common multi-tenant shape: admins get one policy, paid
    users get another, free users get a third. Construct with a
    mapping of ``user_id -> Permissions`` plus a ``default``
    fallback for unmapped users (including ``None``)::

        from jeevesagent import (
            PerUserPermissions, StandardPermissions, Mode,
        )

        permissions = PerUserPermissions(
            policies={
                "admin_alice": StandardPermissions(mode=Mode.BYPASS),
                "paid_user_42": StandardPermissions(
                    mode=Mode.ACCEPT_EDITS,
                ),
            },
            default=StandardPermissions(
                mode=Mode.DEFAULT,
                denied_tools=["bash", "delete_user"],
            ),
        )
        agent = Agent("...", permissions=permissions)

    Each ``check`` call routes to the policy keyed by ``user_id``
    (the live :class:`~jeevesagent.RunContext`'s value, threaded
    through by the agent loop). When no policy matches, the
    ``default`` decides — most apps want a strict default and add
    permissive policies for trusted users.
    """

    def __init__(
        self,
        *,
        policies: Mapping[str | None, Any],
        default: Any,
    ) -> None:
        # ``Any`` for the policy values because the exact shape is
        # the :class:`~jeevesagent.Permissions` protocol — narrowing
        # to a specific class would lock out custom impls.
        self._policies = dict(policies)
        self._default = default

    async def check(
        self,
        call: ToolCall,
        *,
        context: Mapping[str, Any],
        user_id: str | None = None,
    ) -> PermissionDecision:
        policy = self._policies.get(user_id, self._default)
        # Forward both the call + context + user_id to the underlying
        # policy. Older Permissions impls without the user_id kwarg
        # fall back via the ``except TypeError`` so the framework
        # never breaks on a legacy custom policy embedded inside a
        # PerUserPermissions mapping.
        try:
            return await policy.check(  # type: ignore[no-any-return]
                call, context=context, user_id=user_id
            )
        except TypeError:
            return await policy.check(  # type: ignore[no-any-return]
                call, context=context
            )
