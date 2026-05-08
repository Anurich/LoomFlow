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

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import anyio

from ..core._deprecation import warn_legacy_protocol
from ..core.types import (
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
from .helpers import add_usage

# Module-level singleton no-op async context manager. ``contextlib.nullcontext``
# implements both the sync and async protocols (since Python 3.10), so we can
# reuse a single instance everywhere a hot path wants to *maybe* enter a
# telemetry span via ``async with (NULL_CTX if fast else tel.trace(...)):``.
_NULL_CTX: contextlib.AbstractAsyncContextManager[None] = contextlib.nullcontext()

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

            # Budget gate. ``NoBudget.allows_step()`` returns OK every
            # time — when ``fast_budget`` is True we skip the call
            # entirely.
            if not deps.fast_budget:
                # Pass user_id so per-user budget caps fire
                # correctly. Older budget impls without the kwarg
                # fall back via the ``except TypeError``.
                try:
                    status = await deps.budget.allows_step(
                        user_id=deps.context.user_id
                    )
                except TypeError:
                    status = await deps.budget.allows_step()
                if status.blocked:
                    session.interrupted = True
                    session.interruption_reason = f"budget:{status.reason}"
                    if not deps.fast_telemetry:
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

            turn_trace: contextlib.AbstractAsyncContextManager[Any] = (
                _NULL_CTX
                if deps.fast_telemetry
                else deps.telemetry.trace(
                    "jeeves.turn",
                    turn=session.turns,
                    session_id=session.id,
                )
            )

            async with turn_trace:
                # 2a. Model call.
                #
                # Non-streaming hot path: when nobody's reading from
                # ``agent.stream()``, prefer the model adapter's
                # single-shot ``complete()`` method. Skips per-token
                # async-generator yields + per-chunk Event
                # constructions and uses a non-streaming HTTP call
                # on the wire (no SSE overhead). About 100-200 ms
                # per turn faster on token-heavy responses.
                #
                # Streaming path: yield each ModelChunk as a
                # ``model_chunk`` Event so a ``stream()`` consumer
                # sees tokens as they arrive.
                tool_defs = await deps.tools.list_tools()
                tool_calls: list[ToolCall] = []
                usage = Usage()

                if not deps.streaming and hasattr(deps.model, "complete"):
                    model_trace: contextlib.AbstractAsyncContextManager[Any] = (
                        _NULL_CTX
                        if deps.fast_telemetry
                        else deps.telemetry.trace(
                            "jeeves.model.complete",
                            model=deps.model.name,
                            turn=session.turns,
                            session_id=session.id,
                            tool_count=len(tool_defs),
                        )
                    )
                    async with model_trace:
                        if deps.fast_runtime:
                            # Inline path: skip ``runtime.step`` wrapping
                            # (no idempotency_key derivation, no journal
                            # write — InProcRuntime's step is a literal
                            # ``await fn(*args)``).
                            text, tool_calls, usage, _finish_reason = (
                                await deps.model.complete(
                                    session.messages,
                                    tools=tool_defs or None,
                                )
                            )
                        else:
                            text, tool_calls, usage, _finish_reason = (
                                await deps.runtime.step(  # type: ignore[func-returns-value]
                                    f"model_call_{session.turns}",
                                    deps.model.complete,
                                    session.messages,
                                    tools=tool_defs or None,
                                )
                            )
                else:
                    text_parts: list[str] = []
                    stream_trace: contextlib.AbstractAsyncContextManager[Any] = (
                        _NULL_CTX
                        if deps.fast_telemetry
                        else deps.telemetry.trace(
                            "jeeves.model.stream",
                            model=deps.model.name,
                            turn=session.turns,
                            session_id=session.id,
                            tool_count=len(tool_defs),
                        )
                    )
                    async with stream_trace:
                        if deps.fast_runtime:
                            chunks = deps.model.stream(
                                session.messages,
                                tools=tool_defs or None,
                            )
                        else:
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
                if not deps.fast_budget:
                    try:
                        await deps.budget.consume(
                            tokens_in=usage.input_tokens,
                            tokens_out=usage.output_tokens,
                            cost_usd=usage.cost_usd,
                            user_id=deps.context.user_id,
                        )
                    except TypeError:
                        # Legacy budget impls without the user_id
                        # kwarg — keep working.
                        await deps.budget.consume(
                            tokens_in=usage.input_tokens,
                            tokens_out=usage.output_tokens,
                            cost_usd=usage.cost_usd,
                        )
                if not deps.fast_telemetry:
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
                session.cumulative_usage = add_usage(
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

                # 2d. Dispatch tool calls in parallel.
                #
                # Two paths picked from ``deps.streaming``:
                #
                # - **Buffered (default for ``agent.run``)**: events
                #   for each tool are appended to a per-call list
                #   while the task runs. After all tasks complete
                #   we yield the lists in call order. One task
                #   group, no memory channel, no per-call clones —
                #   ~25-35% faster end-to-end on tool-heavy turns
                #   in the JeevesAgent vs LangChain bench.
                #
                # - **Streaming (``agent.stream``)**: events flow
                #   through a memory-object channel as tasks emit
                #   them. Slower but preserves arrival-order
                #   semantics so a consumer that breaks out of the
                #   stream cancels long-running tools promptly.
                results: list[ToolResult | None] = [None] * len(tool_calls)

                if deps.streaming:
                    async for ev in _dispatch_streaming(
                        deps, session, tool_calls, results
                    ):
                        yield ev
                else:
                    events_per_call: list[list[Event]] = [
                        [] for _ in tool_calls
                    ]
                    async with anyio.create_task_group() as tg:
                        for i, call in enumerate(tool_calls):
                            tg.start_soon(
                                _run_one_tool,
                                deps,
                                session,
                                call,
                                i,
                                results,
                                events_per_call[i],
                            )
                    for event_list in events_per_call:
                        for ev in event_list:
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
    """Construct the initial message list: system + memory recall +
    rehydrated session history + current user prompt.

    Two layers of memory feed the model:

    * **Cross-session recall** (``recall_facts`` + ``recall_episodes``)
      — surfaces relevant context from OTHER conversations the same
      user has had. Returned as a single SYSTEM block above the
      message log, partitioned by ``deps.context.user_id``.

    * **Within-session continuity** (``session_messages``) —
      rehydrates this conversation's prior user/assistant turns as
      real :class:`Message` history so the model sees the chat
      thread, not just a recall summary. Driven by
      ``deps.context.session_id``: reusing the same id continues
      the conversation.

    Recall episodes that belong to *this same session* are filtered
    out before the SYSTEM block is built — those would just
    duplicate content the rehydrated message history already
    carries.
    """
    user_id = deps.context.user_id
    session_id = deps.context.session_id
    messages: list[Message] = [
        Message(role=Role.SYSTEM, content=instructions),
    ]

    # Working blocks are user-partitioned (M9) — pass the run's
    # user_id so alice's pinned context never bleeds into bob's
    # seed prompt. Fall back gracefully for legacy custom Memory
    # implementations whose ``working()`` predates the kwarg.
    try:
        blocks = await deps.memory.working(user_id=user_id)
    except TypeError:
        blocks = await deps.memory.working()
    if blocks:
        block_text = "\n\n".join(b.format() for b in blocks)
        messages.append(Message(role=Role.SYSTEM, content=block_text))

    facts: list[Any] = []
    try:
        facts = await deps.memory.recall_facts(
            prompt, limit=5, user_id=user_id
        )
    except (AttributeError, TypeError):
        # 0.1.x backends without recall_facts, or older signatures
        # missing ``user_id``: silent fallback.
        facts = []

    try:
        episodes = await deps.memory.recall(
            prompt, kind="episodic", limit=3, user_id=user_id
        )
    except TypeError:
        # Backends that haven't picked up the user_id kwarg yet.
        episodes = await deps.memory.recall(
            prompt, kind="episodic", limit=3
        )

    # Cross-session recall only — drop any episode that belongs to
    # this same conversation (its content will be rehydrated as a
    # real chat turn below).
    if session_id is not None:
        episodes = [e for e in episodes if e.session_id != session_id]

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

    # Rehydrate this conversation's prior turns so the model sees
    # real chat history rather than relying only on semantic
    # recall. Backends without persisted message logs return [] —
    # in which case we fall through to the current-prompt-only path.
    if session_id is not None:
        try:
            history = await deps.memory.session_messages(
                session_id, user_id=user_id, limit=20
            )
        except (AttributeError, TypeError):
            history = []
        messages.extend(history)

    messages.append(Message(role=Role.USER, content=prompt))
    return messages


async def _dispatch_streaming(
    deps: Dependencies,
    session: AgentSession,
    tool_calls: list[ToolCall],
    results: list[ToolResult | None],
) -> AsyncIterator[Event]:
    """Streaming path for tool dispatch — events flow as they happen.

    Each worker emits its tool_call event BEFORE running the tool
    (via the channel) and its tool_result event AFTER, so a stream
    consumer can break the iterator on a tool_call event and the
    surrounding task group cancels the in-flight worker promptly.
    """
    from anyio.streams.memory import MemoryObjectSendStream

    send, receive = anyio.create_memory_object_stream[Event](
        max_buffer_size=len(tool_calls) * 4
    )

    async def _stream_one(
        slot: int,
        call: ToolCall,
        sender: MemoryObjectSendStream[Event],
    ) -> None:
        async with sender:
            await sender.send(Event.tool_call(session.id, call))
            if not deps.fast_audit:
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
                    user_id=deps.context.user_id,
                )
            result = await _run_single_tool(
                deps, call, turn=session.turns, slot=slot
            )
            results[slot] = result
            if not deps.fast_audit:
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
                    user_id=deps.context.user_id,
                )
            await sender.send(Event.tool_result(session.id, result))

    async with anyio.create_task_group() as tg:
        async with send:
            for i, call in enumerate(tool_calls):
                tg.start_soon(_stream_one, i, call, send.clone())
        async with receive:
            async for ev in receive:
                yield ev


async def _run_one_tool(
    deps: Dependencies,
    session: AgentSession,
    call: ToolCall,
    slot: int,
    results: list[ToolResult | None],
    event_buffer: list[Event],
) -> None:
    """Per-call worker: append tool_call event, run the tool through
    hooks + permissions + runtime.step, write result into
    ``results[slot]``, append tool_result event. Buffered into
    ``event_buffer`` so the caller can yield events in deterministic
    call order after the parallel dispatch completes."""
    event_buffer.append(Event.tool_call(session.id, call))
    if not deps.fast_audit:
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
            user_id=deps.context.user_id,
        )
    result = await _run_single_tool(deps, call, turn=session.turns, slot=slot)
    results[slot] = result
    if not deps.fast_audit:
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
            user_id=deps.context.user_id,
        )
    event_buffer.append(Event.tool_result(session.id, result))


async def _run_single_tool(
    deps: Dependencies, call: ToolCall, *, turn: int, slot: int
) -> ToolResult:
    """Hooks → permission → journaled tool host call → post-hook.

    Layers (hooks / permissions / runtime / telemetry) are skipped
    when their corresponding ``fast_*`` flag on ``deps`` is set —
    e.g. ``AllowAll`` permissions short-circuit the ``check`` call,
    an empty ``HookRegistry`` short-circuits ``pre_tool`` /
    ``post_tool`` dispatch, and ``InProcRuntime`` lets us inline
    the tool host call (skipping idempotency-key derivation).
    """
    started = anyio.current_time()
    result: ToolResult

    tool_trace: contextlib.AbstractAsyncContextManager[Any] = (
        _NULL_CTX
        if deps.fast_telemetry
        else deps.telemetry.trace(
            "jeeves.tool",
            tool=call.tool,
            call_id=call.id,
            turn=turn,
        )
    )

    async with tool_trace:
        run_user_id = deps.context.user_id
        if deps.fast_hooks:
            hook_decision: PermissionDecision = PermissionDecision.allow_()
        else:
            try:
                hook_decision = await deps.hooks.pre_tool(
                    call, user_id=run_user_id
                )
            except TypeError:
                # Legacy HookHost without the kwarg. Warn once
                # per-process so callers know to add it.
                warn_legacy_protocol("HookHost", "pre_tool")
                hook_decision = await deps.hooks.pre_tool(call)

        if hook_decision.deny:
            result = ToolResult.denied_(
                call.id, hook_decision.reason or "denied by hook"
            )
        else:
            if deps.fast_permissions:
                # AllowAll always allows — skip the dataclass round-trip.
                perm: PermissionDecision = PermissionDecision.allow_()
            else:
                try:
                    perm = await deps.permissions.check(
                        call, context={}, user_id=run_user_id
                    )
                except TypeError:
                    # Legacy Permissions without the kwarg. Warn
                    # once per-process so callers know to add it.
                    warn_legacy_protocol("Permissions", "check")
                    perm = await deps.permissions.check(call, context={})
            # Decide whether to execute, then either set a deny
            # result here or fall into the execute-and-post-hook
            # block below. ``execute_call`` stays True only when
            # every gate (deny / ask) approved.
            execute_call = True
            # When ``deps.fast_hooks`` is True we never actually
            # ran a hook — the ``allow_()`` we pre-set above is a
            # default, not an explicit approval. Distinguish so an
            # ``ask`` from permissions doesn't get silently bypassed
            # by the absence of a hook layer.
            hook_explicitly_allowed = (
                (not deps.fast_hooks) and hook_decision.allow
            )
            if perm.deny:
                result = ToolResult.denied_(
                    call.id, perm.reason or "denied by policy"
                )
                execute_call = False
            elif perm.ask and not hook_explicitly_allowed:
                # Permissions returned ``ask`` — route the decision
                # through the configured approval handler. When no
                # handler is wired, fall back to a deny so the agent
                # never silently bypasses the approval gate.
                approved = await _resolve_ask_decision(
                    call,
                    deps.approval_handler,
                    run_user_id,
                )
                if not approved:
                    result = ToolResult.denied_(
                        call.id,
                        perm.reason or (
                            "approval required; no approver"
                            if deps.approval_handler is None
                            else "approval declined"
                        ),
                    )
                    execute_call = False
            if execute_call:
                try:
                    if deps.fast_runtime:
                        result = await deps.tools.call(
                            call.tool,
                            call.args,
                            call_id=call.id,
                        )
                    else:
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

                if not deps.fast_hooks:
                    try:
                        await deps.hooks.post_tool(
                            call, result, user_id=run_user_id
                        )
                    except TypeError:
                        warn_legacy_protocol("HookHost", "post_tool")
                        await deps.hooks.post_tool(call, result)

    if not deps.fast_telemetry:
        elapsed_ms = (anyio.current_time() - started) * 1000
        await deps.telemetry.emit_metric(
            "jeeves.tool.duration_ms",
            elapsed_ms,
            tool=call.tool,
            ok=result.ok,
            denied=result.denied,
        )
    return result


async def _resolve_ask_decision(
    call: ToolCall,
    handler: Any,  # ApprovalHandler | None — Any keeps the import flat
    user_id: str | None,
) -> bool:
    """Translate a ``Decision.ask_`` permissions outcome into an
    allow/deny by invoking the registered approval handler.

    Returns ``True`` when the handler approves the call, ``False``
    when it declines, when no handler is wired, or when the handler
    raises. A raising handler is treated as a deny (and logged) so
    a buggy approval flow never silently green-lights a tool the
    policy explicitly wanted gated.
    """
    if handler is None:
        return False
    try:
        return bool(await handler(call, user_id))
    except Exception as exc:  # noqa: BLE001 — defensive: handlers
        # may raise from UI plumbing / network failures. We must
        # not turn a buggy approval flow into a permissive one.
        logging.getLogger("jeevesagent.architecture.react").warning(
            "approval_handler raised for tool=%s; treating as deny: %s",
            call.tool,
            exc,
        )
        return False


async def _audit(
    audit_log: AuditLog | None,
    session_id: str,
    actor: str,
    action: str,
    payload: dict[str, Any],
    *,
    user_id: str | None = None,
) -> None:
    if audit_log is None:
        return
    try:
        await audit_log.append(
            session_id=session_id,
            actor=actor,
            action=action,
            payload=payload,
            user_id=user_id,
        )
    except TypeError:
        # Legacy AuditLog impls without the user_id kwarg.
        warn_legacy_protocol("AuditLog", "append")
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


__all__ = ["ReAct"]
