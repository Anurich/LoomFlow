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
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import anyio

from ..core.types import (
    Event,
    Image,
    Message,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from .base import AgentSession, Dependencies
from .helpers import (
    _NULL_CTX,
    budget_gate,
    consume_usage,
    run_gated_tool,
)

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
        # 1. Seed context (system prompt + memory recall + user prompt)
        # — but ONLY on first entry. On re-invocation (the stop-hook
        # Ralph loop calls ``run()`` again on the same session) the
        # seed context is already in ``session.messages``; re-seeding
        # would duplicate the system prompt + memory recall +
        # rehydrated history on every iteration. Per the
        # re-invocation contract (base.py), the new ``prompt`` is
        # appended as a fresh USER turn instead.
        if not session.messages:
            session.messages.extend(
                await _build_seed_messages(deps, session.instructions, prompt)
            )
        else:
            session.messages.append(
                Message(role=Role.USER, content=prompt)
            )

        # 1a. Snip — bounded-window trim of the rehydrated message
        # list before the first model call. Pure list slicing; no
        # API call. ``deps.fast_snip`` short-circuits when
        # ``snip_window=0`` (the default) so the call site costs
        # zero allocation. See :mod:`loomflow.agent.snip` for the
        # slicing rules — snips at user-message boundaries so
        # tool_call / tool_result pairs stay intact.
        if not deps.fast_snip:
            from ..agent.snip import snip_messages
            snipped, dropped = snip_messages(
                session.messages, deps.snip_window
            )
            if dropped > 0:
                session.messages = snipped
                yield Event.architecture_event(
                    session.id,
                    "messages_snipped",
                    dropped=dropped,
                    kept=len(snipped),
                    window_turns=deps.snip_window,
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

            # Budget gate — shared helper (user_id forwarding +
            # legacy fallback + fast_budget skip live in ONE place).
            blocked, gate_events = await budget_gate(deps, session)
            for gate_event in gate_events:
                yield gate_event
            if blocked:
                break

            session.turns += 1

            turn_trace: contextlib.AbstractAsyncContextManager[Any] = (
                _NULL_CTX
                if deps.fast_telemetry
                else deps.telemetry.trace(
                    "loom.turn",
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
                # One ``{name: destructive}`` map per turn, built
                # from the defs we just fetched — passed down into
                # the tool dispatch so the destructive-flag backstop
                # never needs a second ``list_tools()`` round-trip
                # per call.
                destructive_map = {
                    d.name: d.destructive for d in tool_defs
                }
                tool_calls: list[ToolCall] = []
                usage = Usage()

                if not deps.streaming and hasattr(deps.model, "complete"):
                    model_trace: contextlib.AbstractAsyncContextManager[Any] = (
                        _NULL_CTX
                        if deps.fast_telemetry
                        else deps.telemetry.trace(
                            "loom.model.complete",
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
                                    output_schema=deps.output_schema,
                                    effort=deps.effort,
                                    strict_effort=deps.strict_effort,
                                    prompt_caching=deps.prompt_caching,
                                )
                            )
                        else:
                            text, tool_calls, usage, _finish_reason = (
                                await deps.runtime.step(  # type: ignore[func-returns-value]
                                    f"model_call_{session.turns}",
                                    deps.model.complete,
                                    session.messages,
                                    tools=tool_defs or None,
                                    output_schema=deps.output_schema,
                                    effort=deps.effort,
                                    strict_effort=deps.strict_effort,
                                    prompt_caching=deps.prompt_caching,
                                )
                            )
                else:
                    text_parts: list[str] = []
                    stream_trace: contextlib.AbstractAsyncContextManager[Any] = (
                        _NULL_CTX
                        if deps.fast_telemetry
                        else deps.telemetry.trace(
                            "loom.model.stream",
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
                                output_schema=deps.output_schema,
                                effort=deps.effort,
                                strict_effort=deps.strict_effort,
                                prompt_caching=deps.prompt_caching,
                            )
                        else:
                            chunks = deps.runtime.stream_step(
                                f"model_call_{session.turns}",
                                deps.model.stream,
                                session.messages,
                                tools=tool_defs or None,
                                output_schema=deps.output_schema,
                                effort=deps.effort,
                                strict_effort=deps.strict_effort,
                                prompt_caching=deps.prompt_caching,
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
                # ``count_turn=False`` — ReAct already incremented
                # its turn counter at the top of the loop.
                await consume_usage(
                    deps, session, usage, count_turn=False
                )
                if not deps.fast_telemetry:
                    await deps.telemetry.emit_metric(
                        "loom.tokens.input",
                        usage.input_tokens,
                        session_id=session.id,
                        model=deps.model.name,
                    )
                    await deps.telemetry.emit_metric(
                        "loom.tokens.output",
                        usage.output_tokens,
                        session_id=session.id,
                        model=deps.model.name,
                    )
                    if usage.cost_usd:
                        await deps.telemetry.emit_metric(
                            "loom.cost.usd",
                            usage.cost_usd,
                            session_id=session.id,
                            model=deps.model.name,
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
                #   in the Loom vs LangChain bench.
                #
                # - **Streaming (``agent.stream``)**: events flow
                #   through a memory-object channel as tasks emit
                #   them. Slower but preserves arrival-order
                #   semantics so a consumer that breaks out of the
                #   stream cancels long-running tools promptly.
                results: list[ToolResult | None] = [None] * len(tool_calls)

                if deps.streaming:
                    async for ev in _dispatch_streaming(
                        deps, session, tool_calls, results,
                        destructive_map,
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
                                destructive_map,
                            )
                    for event_list in events_per_call:
                        for ev in event_list:
                            yield ev

                # 2e. Append tool results to messages in the order calls
                # were emitted (preserves model's expected ordering).
                # When a tool-result summariser is wired (Agent
                # constructed with ``tool_result_summarizer=``) and
                # the rendered result is larger than the threshold,
                # we replace the verbatim text with a model-generated
                # summary before it enters conversation history — see
                # :mod:`loomflow.tools.result_summarizer` for the
                # fall-back semantics (failures + empty summaries
                # both ship the original verbatim).
                for r, c in zip(results, tool_calls, strict=True):
                    final = (
                        r
                        if r is not None
                        else ToolResult.error_(c.id, "no_result")
                    )
                    msg_content = _format_tool_message(final)
                    if (
                        not deps.fast_tool_summary
                        and deps.tool_result_summarizer is not None
                        and len(msg_content)
                        > deps.tool_result_summary_threshold
                    ):
                        # Lazy import: ``tools.result_summarizer``
                        # would create a tools→architecture cycle
                        # at module-load time.
                        from ..tools.result_summarizer import (
                            summarize_tool_result,
                        )
                        original_len = len(msg_content)
                        msg_content = await summarize_tool_result(
                            msg_content,
                            tool_name=c.tool,
                            summarizer=deps.tool_result_summarizer,
                            threshold=(
                                deps.tool_result_summary_threshold
                            ),
                        )
                        if len(msg_content) != original_len:
                            yield Event.architecture_event(
                                session.id,
                                "tool_result_summarized",
                                tool=c.tool,
                                call_id=final.call_id,
                                original_chars=original_len,
                                summary_chars=len(msg_content),
                            )
                    # Unconditional hard cap — the floor beneath the
                    # optional summariser above. A pathological
                    # multi-megabyte tool output must never blow up
                    # the next model call's input tokens.
                    max_chars = deps.tool_result_max_chars
                    if len(msg_content) > max_chars:
                        dropped_chars = len(msg_content) - max_chars
                        msg_content = (
                            msg_content[:max_chars]
                            + f"\n…[truncated {dropped_chars} chars]"
                        )
                        yield Event.architecture_event(
                            session.id,
                            "tool_result_truncated",
                            tool=c.tool,
                            call_id=final.call_id,
                            kept_chars=max_chars,
                            truncated_chars=dropped_chars,
                        )
                    session.messages.append(
                        Message(
                            role=Role.TOOL,
                            content=msg_content,
                            tool_call_id=final.call_id,
                        )
                    )
                    # Record this tool call for plan_write's
                    # strong-mode verification. Skip plan_write
                    # itself — letting the model verify a step by
                    # the same plan call that marked it done
                    # defeats the whole point. Only SUCCESSFUL
                    # calls count: a denied or errored tool call
                    # is not evidence that work happened, so it
                    # must not be usable as ``verified_by`` for a
                    # DONE transition. ``record_tool_call`` is a
                    # no-op when living_plan isn't enabled.
                    if c.tool != "plan_write" and final.ok:
                        from ..tools.plan import record_tool_call
                        record_tool_call(str(final.call_id))


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

    # Vision: attach any images passed for this run (carried on the run
    # context's metadata under ``_loom_images`` as a list of Image — or
    # of dicts with {data, media_type}). Adapters fold them into the
    # provider's multimodal format. Empty/absent → a plain text message.
    images: tuple[Image, ...] = ()
    try:
        from ..core import get_run_context

        ctx = get_run_context()
        raw = (ctx.metadata or {}).get("_loom_images") if ctx else None
        if raw:
            coerced: list[Image] = []
            for it in raw:
                if isinstance(it, Image):
                    coerced.append(it)
                elif isinstance(it, dict) and it.get("data"):
                    coerced.append(Image(
                        data=str(it["data"]),
                        media_type=str(it.get("media_type", "image/png")),
                    ))
            images = tuple(coerced)
    except Exception:  # noqa: BLE001 — vision is best-effort, never break a run
        images = ()

    messages.append(Message(role=Role.USER, content=prompt, images=images))
    return messages


async def _dispatch_streaming(
    deps: Dependencies,
    session: AgentSession,
    tool_calls: list[ToolCall],
    results: list[ToolResult | None],
    destructive_map: dict[str, bool] | None = None,
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
            result = await run_gated_tool(
                deps,
                session,
                call,
                slot=slot,
                destructive_map=destructive_map,
            )
            results[slot] = result
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
    destructive_map: dict[str, bool] | None = None,
) -> None:
    """Per-call worker: append tool_call event, run the tool through
    the shared gated executor (audit → hooks → permissions →
    approval → timeout-bounded host call → audit), write result into
    ``results[slot]``, append tool_result event. Buffered into
    ``event_buffer`` so the caller can yield events in deterministic
    call order after the parallel dispatch completes."""
    event_buffer.append(Event.tool_call(session.id, call))
    result = await run_gated_tool(
        deps,
        session,
        call,
        slot=slot,
        destructive_map=destructive_map,
    )
    results[slot] = result
    event_buffer.append(Event.tool_result(session.id, result))


def _format_tool_message(result: ToolResult) -> str:
    if result.ok:
        return str(result.output)
    if result.denied:
        return f"DENIED: {result.reason or 'no reason given'}"
    return f"ERROR: {result.error or 'unknown'}"


__all__ = ["ReAct"]
