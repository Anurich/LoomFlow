"""ReAct: the canonical observe-think-act loop.

Each turn:

1. Check budget; emit warning / exceeded.
2. Call the model with current messages + available tools.
3. Stream tokens, accumulate text + tool calls + usage.
4. If no tool calls, the model is done; break.
5. Otherwise, dispatch all tool calls in parallel through hooks
   → permissions → tool host. Append results to messages.
6. Loop.

This is the v0.1.x default behaviour, lifted verbatim out of
``Agent._loop`` and behind the :class:`Architecture` protocol.
Behaviour is identical; the refactor only changes the shape.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import anyio
from anyio.streams.memory import MemoryObjectSendStream

from ..core.types import (
    Episode,
    Event,
    Message,
    PermissionDecision,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from ..security.audit import AuditLog
from .base import AgentSession, Dependencies

if TYPE_CHECKING:
    from ..agent.api import Agent


class ReAct:
    """Observe-think-act in a tight loop.

    The default architecture for every :class:`Agent`. Other
    architectures wrap or replace this strategy; see ``Subagent.md``.

    ``max_turns`` overrides ``Dependencies.max_turns`` for this
    architecture only — useful when wrapping ReAct inside another
    architecture that sets its own per-leaf cap (Reflexion,
    Plan-and-Execute, etc.). ``None`` means "use whatever the Agent
    was configured with".
    """

    name = "react"

    def __init__(self, *, max_turns: int | None = None) -> None:
        self._max_turns_override = max_turns

    def declared_workers(self) -> dict[str, Agent]:
        return {}

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # 1. Seed context (system prompt + memory recall + user prompt).
        session.messages.extend(
            await _build_seed_messages(deps, session.instructions, prompt)
        )

        max_turns = (
            self._max_turns_override
            if self._max_turns_override is not None
            else deps.max_turns
        )

        # 2. The ReAct loop.
        while True:
            if session.turns >= max_turns:
                session.interrupted = True
                session.interruption_reason = "max_turns_exceeded"
                break

            status = await deps.budget.allows_step()
            if status.blocked:
                session.interrupted = True
                session.interruption_reason = f"budget:{status.reason}"
                await deps.telemetry.emit_metric(
                    "jeeves.budget.exceeded",
                    1,
                    session_id=session.id,
                    reason=status.reason,
                )
                yield Event.budget_exceeded(session.id, status)
                break
            if status.warn:
                yield Event.budget_warning(session.id, status)

            session.turns += 1

            async with deps.telemetry.trace(
                "jeeves.turn",
                turn=session.turns,
                session_id=session.id,
            ):
                # 2a. Model call — stream chunks, yield model_chunk events.
                tool_defs = await deps.tools.list_tools()
                text_parts: list[str] = []
                tool_calls: list[ToolCall] = []
                usage = Usage()

                async with deps.telemetry.trace(
                    "jeeves.model.stream",
                    model=deps.model.name,
                    turn=session.turns,
                    session_id=session.id,
                    tool_count=len(tool_defs),
                ):
                    chunks = deps.runtime.stream_step(
                        f"model_call_{session.turns}",
                        deps.model.stream,
                        session.messages,
                        tools=tool_defs or None,
                    )
                    async for chunk in chunks:
                        yield Event.model_chunk(session.id, chunk)
                        if chunk.kind == "text" and chunk.text is not None:
                            text_parts.append(chunk.text)
                        elif (
                            chunk.kind == "tool_call"
                            and chunk.tool_call is not None
                        ):
                            tool_calls.append(chunk.tool_call)
                        elif chunk.kind == "finish" and chunk.usage is not None:
                            usage = chunk.usage

                text = "".join(text_parts)

                # 2b. Update budget + telemetry + cumulative usage.
                await deps.budget.consume(
                    tokens_in=usage.input_tokens,
                    tokens_out=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                )
                await deps.telemetry.emit_metric(
                    "jeeves.tokens.input",
                    usage.input_tokens,
                    session_id=session.id,
                    model=deps.model.name,
                )
                await deps.telemetry.emit_metric(
                    "jeeves.tokens.output",
                    usage.output_tokens,
                    session_id=session.id,
                    model=deps.model.name,
                )
                if usage.cost_usd:
                    await deps.telemetry.emit_metric(
                        "jeeves.cost.usd",
                        usage.cost_usd,
                        session_id=session.id,
                        model=deps.model.name,
                    )
                session.cumulative_usage = _add_usage(
                    session.cumulative_usage, usage
                )
                session.output = text

                session.messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=text,
                        tool_calls=tuple(tool_calls),
                    )
                )

                # 2c. No tool calls = model is done.
                if not tool_calls:
                    break

                # 2d. Dispatch tool calls in parallel; yield events as they
                # happen via an internal memory channel.
                results: list[ToolResult | None] = [None] * len(tool_calls)
                send, receive = anyio.create_memory_object_stream[Event](
                    max_buffer_size=len(tool_calls) * 4
                )

                async with anyio.create_task_group() as outer_tg:
                    outer_tg.start_soon(
                        _dispatch_all_tools, deps, session, tool_calls, results, send
                    )
                    async with receive:
                        async for ev in receive:
                            yield ev

                # 2e. Append tool results to messages in the order calls
                # were emitted (preserves model's expected ordering).
                for r, c in zip(results, tool_calls, strict=True):
                    final = (
                        r
                        if r is not None
                        else ToolResult.error_(c.id, "no_result")
                    )
                    session.messages.append(
                        Message(
                            role=Role.TOOL,
                            content=_format_tool_message(final),
                            tool_call_id=final.call_id,
                        )
                    )


# ---------------------------------------------------------------------------
# Helpers (module-level so multiple architectures can reuse)
# ---------------------------------------------------------------------------


async def _build_seed_messages(
    deps: Dependencies, instructions: str, prompt: str
) -> list[Message]:
    """Construct the initial message list: system + memory recall + user."""
    messages: list[Message] = [
        Message(role=Role.SYSTEM, content=instructions),
    ]

    blocks = await deps.memory.working()
    if blocks:
        block_text = "\n\n".join(b.format() for b in blocks)
        messages.append(Message(role=Role.SYSTEM, content=block_text))

    facts: list[Any] = []
    try:
        facts = await deps.memory.recall_facts(prompt, limit=5)
    except AttributeError:
        # 0.1.x backends without recall_facts: silent fallback.
        facts = []

    episodes = await deps.memory.recall(prompt, kind="episodic", limit=3)

    recall_parts: list[str] = []
    if facts:
        recall_parts.append(
            "Known facts:\n" + "\n".join(f"- {f.format()}" for f in facts)
        )
    if episodes:
        recall_parts.append(
            "Relevant past episodes:\n"
            + "\n".join(f"- {e.format()}" for e in episodes)
        )
    if recall_parts:
        messages.append(
            Message(role=Role.SYSTEM, content="\n\n".join(recall_parts))
        )

    messages.append(Message(role=Role.USER, content=prompt))
    return messages


async def _dispatch_all_tools(
    deps: Dependencies,
    session: AgentSession,
    tool_calls: list[ToolCall],
    results: list[ToolResult | None],
    send: MemoryObjectSendStream[Event],
) -> None:
    """Spawn one ``_run_one_tool`` task per call and close the parent
    sender once all workers complete. Worker clones of ``send`` keep
    the receive channel alive until the last worker exits."""
    async with send:
        async with anyio.create_task_group() as inner_tg:
            for i, call in enumerate(tool_calls):
                inner_tg.start_soon(
                    _run_one_tool,
                    deps,
                    session,
                    call,
                    i,
                    results,
                    send.clone(),
                )


async def _run_one_tool(
    deps: Dependencies,
    session: AgentSession,
    call: ToolCall,
    slot: int,
    results: list[ToolResult | None],
    send: MemoryObjectSendStream[Event],
) -> None:
    """Per-call worker: emit tool_call, run the tool through hooks +
    permissions + runtime.step, write result into ``results[slot]``,
    emit tool_result. Used by the parallel dispatch loop."""
    async with send:
        await send.send(Event.tool_call(session.id, call))
        await _audit(
            deps.audit_log,
            session.id,
            "model",
            "tool_call",
            {
                "tool": call.tool,
                "call_id": call.id,
                "args": dict(call.args),
                "destructive": call.destructive,
                "turn": session.turns,
            },
        )
        result = await _run_single_tool(deps, call, turn=session.turns, slot=slot)
        results[slot] = result
        await _audit(
            deps.audit_log,
            session.id,
            "system",
            "tool_result",
            {
                "tool": call.tool,
                "call_id": result.call_id,
                "ok": result.ok,
                "denied": result.denied,
                "error": result.error,
                "reason": result.reason,
                "turn": session.turns,
            },
        )
        await send.send(Event.tool_result(session.id, result))


async def _run_single_tool(
    deps: Dependencies, call: ToolCall, *, turn: int, slot: int
) -> ToolResult:
    """Hooks → permission → journaled tool host call → post-hook."""
    started = anyio.current_time()
    result: ToolResult

    async with deps.telemetry.trace(
        "jeeves.tool",
        tool=call.tool,
        call_id=call.id,
        turn=turn,
    ):
        hook_decision: PermissionDecision = await deps.hooks.pre_tool(call)
        if hook_decision.deny:
            result = ToolResult.denied_(
                call.id, hook_decision.reason or "denied by hook"
            )
        else:
            perm = await deps.permissions.check(call, context={})
            if perm.deny:
                result = ToolResult.denied_(
                    call.id, perm.reason or "denied by policy"
                )
            elif perm.ask and not hook_decision.allow:
                result = ToolResult.denied_(
                    call.id,
                    perm.reason or "approval required; no approver",
                )
            else:
                try:
                    result = await deps.runtime.step(
                        f"tool_call_{turn}_{slot}",
                        deps.tools.call,
                        call.tool,
                        call.args,
                        call_id=call.id,
                        idempotency_key=call.idempotency_key(),
                    )
                except Exception as exc:  # noqa: BLE001
                    result = ToolResult.error_(call.id, str(exc))

                await deps.hooks.post_tool(call, result)

    elapsed_ms = (anyio.current_time() - started) * 1000
    await deps.telemetry.emit_metric(
        "jeeves.tool.duration_ms",
        elapsed_ms,
        tool=call.tool,
        ok=result.ok,
        denied=result.denied,
    )
    return result


async def _audit(
    audit_log: AuditLog | None,
    session_id: str,
    actor: str,
    action: str,
    payload: dict[str, Any],
) -> None:
    if audit_log is None:
        return
    await audit_log.append(
        session_id=session_id,
        actor=actor,
        action=action,
        payload=payload,
    )


def _format_tool_message(result: ToolResult) -> str:
    if result.ok:
        return str(result.output)
    if result.denied:
        return f"DENIED: {result.reason or 'no reason given'}"
    return f"ERROR: {result.error or 'unknown'}"


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cost_usd=a.cost_usd + b.cost_usd,
    )


# Public re-exports for callers that want to compose with ReAct
# helpers (e.g. Reflexion will reuse _build_seed_messages with a
# lessons-prefix injected).
__all__ = ["ReAct", "Episode"]
