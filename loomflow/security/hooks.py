"""User-registered lifecycle callbacks.

Hooks run in a timeout-shielded scope so a buggy callback can't hang
the loop. Pre-tool hooks can deny a call (first deny wins), and a
pre-tool hook that raises or times out denies it too: a policy check
that never answered hasn't approved anything (set
``timeout_denies=False`` to let timed-out calls through instead). Post-tool hooks are best-effort and can
never affect the result.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import anyio

from ..core.types import Event, PermissionDecision, ToolCall, ToolResult

PreToolHook = Callable[[ToolCall], Awaitable[PermissionDecision | None]]
PostToolHook = Callable[[ToolCall, ToolResult], Awaitable[None]]
EventHook = Callable[[Event], Awaitable[None]]


@dataclass
class HookRegistry:
    """Implements :class:`~loomflow.core.protocols.HookHost`."""

    pre_tool_hooks: list[PreToolHook] = field(default_factory=list)
    post_tool_hooks: list[PostToolHook] = field(default_factory=list)
    event_hooks: list[EventHook] = field(default_factory=list)

    hook_timeout_s: float = 5.0
    timeout_denies: bool = True

    # ---- registration ----------------------------------------------------

    def register_pre_tool(self, hook: PreToolHook) -> PreToolHook:
        self.pre_tool_hooks.append(hook)
        return hook

    def register_post_tool(self, hook: PostToolHook) -> PostToolHook:
        self.post_tool_hooks.append(hook)
        return hook

    def register_event(self, hook: EventHook) -> EventHook:
        self.event_hooks.append(hook)
        return hook

    # ---- HookHost protocol ----------------------------------------------

    async def pre_tool(
        self, call: ToolCall, *, user_id: str | None = None
    ) -> PermissionDecision:
        """Run all pre-tool hooks. First deny wins; otherwise allow.

        The ``user_id`` kwarg is forwarded for protocol parity (M9);
        the bundled :class:`HookRegistry` doesn't itself dispatch
        per-user, but custom :class:`HookHost` implementations can
        route on it. Individual hook callables continue to receive
        only ``(call,)`` to keep the existing decorator API stable;
        hooks that need the user_id can call
        :func:`get_run_context` themselves.
        """
        for hook in self.pre_tool_hooks:
            decision: PermissionDecision | None = None
            name = getattr(hook, "__name__", type(hook).__name__)
            with anyio.move_on_after(self.hook_timeout_s) as scope:
                try:
                    decision = await hook(call)
                except Exception as exc:  # noqa: BLE001 — a broken check denies
                    return PermissionDecision.deny_(
                        f"{call.tool}: pre-tool hook {name!r} raised "
                        f"{type(exc).__name__}: {exc}"
                    )
            if scope.cancelled_caught and self.timeout_denies:
                return PermissionDecision.deny_(
                    f"{call.tool}: pre-tool hook {name!r} timed out "
                    f"after {self.hook_timeout_s}s"
                )
            if decision is not None and decision.deny:
                return decision
        return PermissionDecision.allow_()

    async def post_tool(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        user_id: str | None = None,
    ) -> None:
        """Best-effort post-tool callbacks. Failures and timeouts are
        absorbed so they cannot affect the result the loop returns.
        ``user_id`` follows the same forwarded-but-not-required
        pattern as :meth:`pre_tool`."""
        for hook in self.post_tool_hooks:
            with anyio.move_on_after(self.hook_timeout_s):
                try:
                    await hook(call, result)
                except Exception:  # noqa: BLE001 — hooks must never break the loop
                    continue

    async def on_event(self, event: Event) -> None:
        for hook in self.event_hooks:
            with anyio.move_on_after(self.hook_timeout_s):
                try:
                    await hook(event)
                except Exception:  # noqa: BLE001
                    continue
