"""``send_message`` tool — continue a conversation with a persistent worker.

The companion to ``delegate``. Where ``delegate(target, instructions)``
spawns or re-engages a worker by ROLE (and historically reset the
worker's context each call), ``send_message(to=<worker_id>, content)``
addresses a SPECIFIC worker by its persistent ID and reuses the
worker's stable session_id — so the worker remembers everything it
was told before.

This is the Claude-Code ``SendMessage`` pattern (see
``tools/SendMessageTool/`` in their codebase) ported to loomflow's
conventions: a plain tool factory (matches ``make_plan_tools``,
``make_workspace_tools``), no new Protocol, no SDK imports, lazy
imports for any framework deps that would create circulars.

The registry the tool reads from is
:attr:`Agent._worker_registry` — populated eagerly by
``Team.supervisor(persistent_subagents=True)``. The tool's closure
holds a reference to the same dict; mutations to the dict (new
worker registrations after construction) are visible immediately.

Multi-tenant safety: the tool reads ``user_id`` from the live
:class:`RunContext` via :func:`get_run_context` and rejects cross-
user invocations with a clear tool-result error. Worker handles
pin their ``user_id`` on first touch (by either ``delegate`` or
``send_message``) and never change it.

Concurrency: the per-handle :class:`anyio.Lock` serialises
concurrent calls to the SAME worker (a coordinator running
``send_message`` + ``delegate`` in parallel for the same role
would otherwise corrupt the worker's session). Calls to DIFFERENT
workers fan out in parallel — the lock is per-handle, not global.
"""

from __future__ import annotations

from typing import Any

import anyio

from ..core.context import get_run_context, inherit_ambient_memory
from ..core.protocols import Memory
from ..core.types import Event
from .registry import Tool


def make_send_message_tool(
    registry: dict[str, Any],
    *,
    session: object,
    memory: Memory,
    tool_name: str = "send_message",
    event_sink: object | None = None,
) -> Tool:
    """Build the ``send_message`` tool the coordinator calls.

    Args:
        registry: the coordinator Agent's ``_worker_registry`` —
            a ``dict[str, _WorkerHandle]`` populated by
            ``Team.supervisor``. Tool closes over the reference;
            adding workers to the dict after construction is
            visible to the tool immediately.
        session: the coordinator's :class:`AgentSession`. Passed
            into :class:`SubagentInvocation` so the worker's
            token / cost usage rolls up into the coordinator's
            ``RunResult.cost_usd`` / tokens.
        tool_name: name the model sees. Default ``"send_message"``
            matches Claude Code's convention. Override only if
            you have a name collision.
        event_sink: optional memory-object send stream the tool
            emits architecture events into (``subagent.message_sent``
            / ``subagent.message_completed``). Pass the supervisor's
            event channel when wiring; ``None`` skips telemetry.

    Returns:
        A :class:`Tool` registered into the coordinator's
        :class:`ExtendedToolHost`.

    The tool's docstring (visible to the model) explicitly
    distinguishes ``send_message`` from ``delegate`` — the model
    needs to learn "use send_message to continue, use delegate
    to start fresh."
    """

    async def _send_message(to: str, content: str) -> str:
        # 1. Lookup. Unknown IDs return an error string (NOT a
        #    raise) — matches ``delegate`` which returns
        #    ``f"Error: unknown worker {target!r}"``. Models
        #    handle string errors gracefully; raised exceptions
        #    crash the whole turn.
        handle = registry.get(to)
        if handle is None:
            known = sorted(registry.keys())
            return (
                f"Error: unknown worker id {to!r}. "
                f"Known workers: {known}"
            )

        # 2. Multi-tenant safety check. ``get_run_context()``
        #    reads the contextvar Agent._loop installed at the
        #    top of the current run. ``user_id=None`` is the
        #    single-tenant case (REPL with no explicit user_id);
        #    we don't reject those — but a server running multi-
        #    user must surface a clear refusal when user_ids
        #    disagree.
        run_ctx = get_run_context()
        caller_user = run_ctx.user_id
        if (
            handle.user_id is not None
            and caller_user is not None
            and handle.user_id != caller_user
        ):
            return (
                f"Error: worker {to!r} belongs to user_id "
                f"{handle.user_id!r} but the current run is "
                f"user_id {caller_user!r}. Cross-tenant "
                "send_message is rejected for multi-tenant safety."
            )

        # 3. Telemetry — fire-and-forget, never block the call.
        if event_sink is not None:
            try:
                ev = Event.architecture_event(
                    getattr(session, "session_id", "")
                    or getattr(session, "id", ""),
                    "subagent.message_sent",
                    payload={
                        "worker_id": to,
                        "role": handle.role,
                    },
                )
                await event_sink.send(ev)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 — telemetry is best-effort
                pass

        # 4. Per-handle lock — serialise concurrent invocations
        #    targeting the SAME worker. Different workers stay
        #    parallel. anyio.Lock cooperates with cancellation
        #    via the surrounding task group.
        async with handle.lock:
            handle.touch(user_id=caller_user)

            # Lazy import: helpers.SubagentInvocation pulls in the
            # whole architecture package; doing the import at
            # module-load time would risk a cycle with
            # ``loomflow.tools.__init__`` → loomflow.__init__ →
            # loomflow.agent.api (which imports loomflow.tools).
            from ..architecture.helpers import SubagentInvocation

            try:
                invocation = SubagentInvocation(
                    handle.agent,
                    content,
                    session_id=handle.session_id,
                    rollup_into=session,  # type: ignore[arg-type]
                )
                # Drain the events into the sink so token-level
                # streaming from the worker reaches the parent's
                # stream. Without this, send_message would buffer
                # everything until the worker finishes.
                #
                # Memory propagation: install the coordinator's
                # memory as ambient so a worker constructed without
                # explicit ``memory=`` inherits it. anyio's
                # contextvar inheritance carries it into the
                # SubagentInvocation's internal task-group spawn
                # of ``agent.run``.
                with inherit_ambient_memory(memory):
                    async for ev in invocation.events():
                        if event_sink is not None:
                            try:
                                await event_sink.send(ev)  # type: ignore[attr-defined]
                            except Exception:  # noqa: BLE001
                                pass
                    output = str(invocation.result.get("output", ""))
            except anyio.get_cancelled_exc_class():
                # Cancellation must propagate — the parent's task
                # group is shutting down. Re-raise immediately;
                # do NOT swallow into the broad except below.
                raise
            except Exception as exc:  # noqa: BLE001 — return as tool-error
                # Worker raised mid-conversation. Returning a
                # string lets the coordinator decide what to do
                # (retry / apologize / pick a different worker)
                # without killing the whole coordinator turn.
                return (
                    f"Error: worker {to!r} raised "
                    f"{type(exc).__name__}: {exc}"
                )

        # 5. Final telemetry.
        if event_sink is not None:
            try:
                ev = Event.architecture_event(
                    getattr(session, "session_id", "")
                    or getattr(session, "id", ""),
                    "subagent.message_completed",
                    payload={
                        "worker_id": to,
                        "role": handle.role,
                        "output_length": len(output),
                    },
                )
                await event_sink.send(ev)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

        return output

    # Static method-descriptions go in the tool body so the model
    # learns the delegate-vs-send_message contract.
    description = (
        "Continue a conversation with an EXISTING subagent by its "
        "persistent worker_id. The subagent remembers your prior "
        "delegations and resumes its full context. Use this "
        "instead of `delegate` when you want a worker to BUILD ON "
        "its earlier work (iterate on code it wrote, follow up "
        "on research, refine a draft). Pass `to=<worker_id>` "
        "from a prior `delegate` result (the bracketed prefix at "
        "the top of the response) and `content=<message>`."
    )

    return Tool(
        name=tool_name,
        description=description,
        fn=_send_message,
        input_schema={
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": (
                        "Worker ID from a prior `delegate` "
                        "response (the bracketed prefix)."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "The follow-up message to send to the "
                        "worker. The worker will reply in the "
                        "same conversation thread."
                    ),
                },
            },
            "required": ["to", "content"],
        },
    )
