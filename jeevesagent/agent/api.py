"""The public ``Agent`` class.

Conventions:

* Pass a string of instructions for a working agent backed by sensible
  defaults: :class:`EchoModel`, :class:`InMemoryMemory`,
  :class:`InProcRuntime`, :class:`NoBudget`, :class:`AllowAll`,
  :class:`HookRegistry`, an empty :class:`InProcessToolHost`.
* Pass ``tools=[fn_or_Tool, ...]`` to register Python callables; the
  agent wraps them in an in-process :class:`ToolHost`.
* Override any subsystem by passing a concrete implementation of the
  matching protocol from :mod:`jeevesagent.core.protocols`.

Two execution surfaces share a single internal loop:

* :meth:`Agent.run` runs to completion and returns a :class:`RunResult`.
* :meth:`Agent.stream` returns an ``AsyncIterator[Event]`` of milestones
  as they happen — STARTED, MODEL_CHUNK, TOOL_CALL, TOOL_RESULT,
  BUDGET_WARNING/EXCEEDED, ERROR, COMPLETED.

Internally, :meth:`_loop` accepts an ``emit`` callback and threads it
through every milestone. ``run()`` passes a no-op emit; ``stream()``
pipes events through an :func:`anyio.create_memory_object_stream` so a
slow consumer applies backpressure to the loop instead of buffering
unboundedly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import anyio

from ..core.ids import new_id
from ..core.protocols import (
    Budget,
    HookHost,
    Memory,
    Model,
    Permissions,
    Runtime,
    Telemetry,
    ToolHost,
)
from ..core.types import (
    Episode,
    Event,
    Message,
    ModelChunk,
    PermissionDecision,
    Role,
    RunResult,
    ToolCall,
    ToolResult,
    Usage,
)
from ..governance.budget import NoBudget
from ..memory.inmemory import InMemoryMemory
from ..model.echo import EchoModel
from ..observability.tracing import NoTelemetry
from ..runtime.inproc import InProcRuntime
from ..security.audit import AuditLog
from ..security.hooks import HookRegistry, PostToolHook, PreToolHook
from ..security.permissions import AllowAll
from ..tools.registry import InProcessToolHost, Tool

DEFAULT_MAX_TURNS = 50
DEFAULT_STREAM_BUFFER = 128

Emit = Callable[[Event], Awaitable[None]]


class Agent:
    """A fully-async, MCP-native, model-agnostic agent harness."""

    def __init__(
        self,
        instructions: str,
        *,
        model: Model | str | None = None,
        memory: Memory | None = None,
        runtime: Runtime | None = None,
        budget: Budget | None = None,
        permissions: Permissions | None = None,
        hooks: HookRegistry | None = None,
        tools: list[Tool | Callable[..., object]] | ToolHost | None = None,
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        auto_consolidate: bool = False,
    ) -> None:
        self._instructions = instructions
        self._model: Model = _resolve_model(model)
        self._memory: Memory = memory if memory is not None else InMemoryMemory()
        self._runtime: Runtime = runtime if runtime is not None else InProcRuntime()
        self._budget: Budget = budget if budget is not None else NoBudget()
        self._permissions: Permissions = (
            permissions if permissions is not None else AllowAll()
        )
        self._hooks = hooks if hooks is not None else HookRegistry()
        self._tool_host: ToolHost = _coerce_tool_host(tools)
        self._telemetry: Telemetry = (
            telemetry if telemetry is not None else NoTelemetry()
        )
        self._audit_log: AuditLog | None = audit_log
        self._max_turns = max_turns
        self._auto_consolidate = auto_consolidate

    # ---- hook decorators (user-facing sugar) ----------------------------

    def before_tool(self, fn: PreToolHook) -> PreToolHook:
        """Register a pre-tool hook. First denial wins; allow otherwise."""
        return self._hooks.register_pre_tool(fn)

    def after_tool(self, fn: PostToolHook) -> PostToolHook:
        """Register a best-effort post-tool callback."""
        return self._hooks.register_post_tool(fn)

    @property
    def hooks(self) -> HookHost:
        return self._hooks

    async def consolidate(self) -> None:
        """Manually trigger memory consolidation.

        Convenience for ``await agent._memory.consolidate()``. Useful
        when ``auto_consolidate=False`` (the default) and you want to
        batch consolidation at a controlled cadence — e.g. once a day,
        or before shutdown.
        """
        await self._memory.consolidate()

    async def _audit(
        self,
        *,
        session_id: str,
        actor: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        if self._audit_log is None:
            return
        await self._audit_log.append(
            session_id=session_id,
            actor=actor,
            action=action,
            payload=payload,
        )

    # ---- public API ------------------------------------------------------

    async def run(
        self, prompt: str, *, session_id: str | None = None
    ) -> RunResult:
        """Run the agent to completion and return its :class:`RunResult`.

        Pass ``session_id`` to resume a journaled run — when paired with
        a durable runtime (e.g. :class:`SqliteRuntime`), already-completed
        steps replay from the journal instead of re-executing. Without a
        durable runtime, ``session_id`` just labels the run.
        """
        return await self._loop(prompt, emit=_noop_emit, session_id=session_id)

    async def resume(
        self, session_id: str, prompt: str
    ) -> RunResult:
        """Resume a previously-interrupted run from its journal.

        Equivalent to ``agent.run(prompt, session_id=session_id)``.
        Exists as a separate method so the intent is explicit at the
        call site and to match the surface advertised by the engineering
        plan.
        """
        return await self.run(prompt, session_id=session_id)

    async def stream(
        self, prompt: str, *, session_id: str | None = None
    ) -> AsyncIterator[Event]:
        """Stream :class:`Event`\\ s as the loop produces them.

        The loop runs as a background task; events are pushed through a
        bounded memory stream so a slow consumer applies backpressure.
        Breaking out of the iteration cancels the producer cleanly.
        ``session_id`` works the same as :meth:`run`'s — pass an
        existing one to resume against a durable runtime's journal.
        """
        send, receive = anyio.create_memory_object_stream[Event](
            max_buffer_size=DEFAULT_STREAM_BUFFER
        )

        async def _produce() -> None:
            try:
                await self._loop(prompt, emit=send.send, session_id=session_id)
            except Exception as exc:  # noqa: BLE001 — surface as ERROR + re-raise
                with anyio.CancelScope(shield=True):
                    await send.send(Event.error("", exc))
                raise
            finally:
                send.close()

        async with anyio.create_task_group() as tg:
            tg.start_soon(_produce)
            try:
                async with receive:
                    async for event in receive:
                        yield event
            finally:
                tg.cancel_scope.cancel()

    # ---- the loop --------------------------------------------------------

    async def _loop(
        self,
        prompt: str,
        *,
        emit: Emit,
        session_id: str | None = None,
    ) -> RunResult:
        started_at = datetime.now(UTC)
        # Caller-supplied session_id enables journal replay when paired
        # with a durable runtime; auto-generated otherwise.
        if session_id is None:
            session_id = new_id("sess")
        loop_started = anyio.current_time()

        # Open a runtime session so journal-backed runtimes can record
        # every step against this run; in-process runtimes treat it as
        # a no-op.
        async with (
            self._runtime.session(session_id),
            self._telemetry.trace(
                "jeeves.run",
                session_id=session_id,
                max_turns=self._max_turns,
                model=self._model.name,
            ),
        ):
            await self._audit(
                session_id=session_id,
                actor="user",
                action="run_started",
                payload={
                    "prompt": prompt[:500],
                    "model": self._model.name,
                    "max_turns": self._max_turns,
                },
            )
            await emit(Event.started(session_id, prompt))

            messages = await self._seed_context(prompt)

            turns = 0
            output_text = ""
            cumulative = Usage()
            interrupted = False
            reason: str | None = None

            while True:
                if turns >= self._max_turns:
                    interrupted = True
                    reason = "max_turns_exceeded"
                    break

                status = await self._budget.allows_step()
                if status.blocked:
                    interrupted = True
                    reason = f"budget:{status.reason}"
                    await self._telemetry.emit_metric(
                        "jeeves.budget.exceeded",
                        1,
                        session_id=session_id,
                        reason=status.reason,
                    )
                    await emit(Event.budget_exceeded(session_id, status))
                    break
                if status.warn:
                    await emit(Event.budget_warning(session_id, status))

                turns += 1
                async with self._telemetry.trace(
                    "jeeves.turn", turn=turns, session_id=session_id
                ):
                    text, tool_calls, usage = await self._take_one_turn(
                        messages, turns, session_id=session_id, emit=emit
                    )

                    await self._budget.consume(
                        tokens_in=usage.input_tokens,
                        tokens_out=usage.output_tokens,
                        cost_usd=usage.cost_usd,
                    )
                    await self._telemetry.emit_metric(
                        "jeeves.tokens.input",
                        usage.input_tokens,
                        session_id=session_id,
                        model=self._model.name,
                    )
                    await self._telemetry.emit_metric(
                        "jeeves.tokens.output",
                        usage.output_tokens,
                        session_id=session_id,
                        model=self._model.name,
                    )
                    if usage.cost_usd:
                        await self._telemetry.emit_metric(
                            "jeeves.cost.usd",
                            usage.cost_usd,
                            session_id=session_id,
                            model=self._model.name,
                        )
                    cumulative = _add_usage(cumulative, usage)
                    output_text = text

                    messages.append(
                        Message(
                            role=Role.ASSISTANT,
                            content=text,
                            tool_calls=tuple(tool_calls),
                        )
                    )

                    if not tool_calls:
                        break

                    results = await self._dispatch_tools(
                        tool_calls,
                        turns,
                        session_id=session_id,
                        emit=emit,
                    )
                    for r in results:
                        messages.append(
                            Message(
                                role=Role.TOOL,
                                content=_format_tool_message(r),
                                tool_call_id=r.call_id,
                            )
                        )

            await self._runtime.step(
                f"persist_episode_{turns}",
                self._memory.remember,
                Episode(
                    session_id=session_id,
                    input=prompt,
                    output=output_text,
                ),
            )

            result = RunResult(
                session_id=session_id,
                output=output_text,
                turns=turns,
                tokens_in=cumulative.input_tokens,
                tokens_out=cumulative.output_tokens,
                cost_usd=cumulative.cost_usd,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                interrupted=interrupted,
                interruption_reason=reason,
            )

            elapsed_ms = (anyio.current_time() - loop_started) * 1000
            await self._telemetry.emit_metric(
                "jeeves.session.duration_ms",
                elapsed_ms,
                session_id=session_id,
                interrupted=interrupted,
                turns=turns,
            )

            # Auto-consolidate runs after the response is finalized but
            # before the COMPLETED event so observers see it as part of
            # the same run. Failures are surfaced as ERROR events but
            # never break the run — consolidation is best-effort.
            if self._auto_consolidate:
                try:
                    await self._memory.consolidate()
                except Exception as exc:  # noqa: BLE001
                    await emit(Event.error(session_id, exc))

            await self._audit(
                session_id=session_id,
                actor="system",
                action="run_completed",
                payload={
                    "turns": turns,
                    "interrupted": interrupted,
                    "interruption_reason": reason,
                    "tokens_in": cumulative.input_tokens,
                    "tokens_out": cumulative.output_tokens,
                    "cost_usd": cumulative.cost_usd,
                    "elapsed_ms": elapsed_ms,
                },
            )
            await emit(Event.completed(session_id, result.model_dump(mode="json")))
            return result

    # ---- internals -------------------------------------------------------

    async def _seed_context(self, prompt: str) -> list[Message]:
        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=self._instructions),
        ]

        blocks = await self._memory.working()
        if blocks:
            block_text = "\n\n".join(b.format() for b in blocks)
            messages.append(Message(role=Role.SYSTEM, content=block_text))

        # Pull semantic facts when the memory backend exposes a
        # FactStore via ``.facts``. Backends without one are skipped.
        fact_store = getattr(self._memory, "facts", None)
        facts: list[Any] = []
        if fact_store is not None:
            try:
                facts = await fact_store.recall_text(prompt, limit=5)
            except AttributeError:
                # Some attribute wearing the name "facts" doesn't satisfy
                # the FactStore protocol; ignore silently.
                facts = []

        episodes = await self._memory.recall(prompt, kind="episodic", limit=3)

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

    async def _take_one_turn(
        self,
        messages: list[Message],
        turn: int,
        *,
        session_id: str,
        emit: Emit,
    ) -> tuple[str, list[ToolCall], Usage]:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = Usage()

        tool_defs = await self._tool_host.list_tools()

        async with self._telemetry.trace(
            "jeeves.model.stream",
            model=self._model.name,
            turn=turn,
            session_id=session_id,
            tool_count=len(tool_defs),
        ):
            chunks = self._runtime.stream_step(
                f"model_call_{turn}",
                self._model.stream,
                messages,
                tools=tool_defs or None,
            )
            chunk: ModelChunk
            async for chunk in chunks:
                await emit(Event.model_chunk(session_id, chunk))
                if chunk.kind == "text" and chunk.text is not None:
                    text_parts.append(chunk.text)
                elif chunk.kind == "tool_call" and chunk.tool_call is not None:
                    tool_calls.append(chunk.tool_call)
                elif chunk.kind == "finish" and chunk.usage is not None:
                    usage = chunk.usage

        return "".join(text_parts), tool_calls, usage

    async def _dispatch_tools(
        self,
        calls: list[ToolCall],
        turn: int,
        *,
        session_id: str,
        emit: Emit,
    ) -> list[ToolResult]:
        """Run all tool calls in parallel through a structured task group.

        Each result slot is pre-allocated so we can write into it from a
        spawned task without locks — order is preserved by index.
        """
        results: list[ToolResult | None] = [None] * len(calls)

        async def _run_one(i: int, call: ToolCall) -> None:
            await emit(Event.tool_call(session_id, call))
            await self._audit(
                session_id=session_id,
                actor="model",
                action="tool_call",
                payload={
                    "tool": call.tool,
                    "call_id": call.id,
                    "args": dict(call.args),
                    "destructive": call.destructive,
                    "turn": turn,
                },
            )
            r = await self._run_single_tool(call, turn=turn, slot=i)
            results[i] = r
            await self._audit(
                session_id=session_id,
                actor="system",
                action="tool_result",
                payload={
                    "tool": call.tool,
                    "call_id": r.call_id,
                    "ok": r.ok,
                    "denied": r.denied,
                    "error": r.error,
                    "reason": r.reason,
                    "turn": turn,
                },
            )
            await emit(Event.tool_result(session_id, r))

        async with anyio.create_task_group() as tg:
            for i, call in enumerate(calls):
                tg.start_soon(_run_one, i, call)

        return [
            r if r is not None else ToolResult.error_(c.id, "no_result")
            for r, c in zip(results, calls, strict=True)
        ]

    async def _run_single_tool(
        self, call: ToolCall, *, turn: int, slot: int
    ) -> ToolResult:
        started = anyio.current_time()
        result: ToolResult

        async with self._telemetry.trace(
            "jeeves.tool",
            tool=call.tool,
            call_id=call.id,
            turn=turn,
        ):
            # 1. User pre-tool hooks first. A hook denial short-circuits.
            hook_decision: PermissionDecision = await self._hooks.pre_tool(call)
            if hook_decision.deny:
                result = ToolResult.denied_(
                    call.id, hook_decision.reason or "denied by hook"
                )
            else:
                # 2. System permission policy. ``ask`` becomes deny in this
                #    slice (no interactive UI); a hook can override by
                #    returning allow.
                perm = await self._permissions.check(call, context={})
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
                    # 3. Execute through a journaled runtime step so the
                    #    result is cached for replay. The host is
                    #    responsible for surfacing errors as
                    #    ToolResult.error_; we still wrap defensively.
                    try:
                        result = await self._runtime.step(
                            f"tool_call_{turn}_{slot}",
                            self._tool_host.call,
                            call.tool,
                            call.args,
                            call_id=call.id,
                            idempotency_key=call.idempotency_key(),
                        )
                    except Exception as exc:  # noqa: BLE001
                        result = ToolResult.error_(call.id, str(exc))

                    # 4. Best-effort post-tool hooks (timeout-shielded).
                    await self._hooks.post_tool(call, result)

        elapsed_ms = (anyio.current_time() - started) * 1000
        await self._telemetry.emit_metric(
            "jeeves.tool.duration_ms",
            elapsed_ms,
            tool=call.tool,
            ok=result.ok,
            denied=result.denied,
        )
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_emit(_event: Event) -> None:
    return None


def _resolve_model(spec: Model | str | None) -> Model:
    """Resolve a string spec or instance to a concrete :class:`Model`.

    Strings dispatch by prefix:

    * ``claude-*`` -> :class:`AnthropicModel`
    * ``gpt-*`` / ``o1-*`` / ``o3-*`` -> :class:`OpenAIModel`
    * ``echo`` -> :class:`EchoModel`

    Anything else passes through if it duck-types as ``Model``; otherwise
    a ``ValueError`` is raised. ``None`` returns the default ``EchoModel``.
    """
    if spec is None:
        return EchoModel()
    if not isinstance(spec, str):
        return spec
    if spec.startswith("claude-"):
        from ..model.anthropic import AnthropicModel
        return AnthropicModel(spec)
    if spec.startswith(("gpt-", "o1-", "o3-")):
        from ..model.openai import OpenAIModel
        return OpenAIModel(spec)
    if spec == "echo":
        return EchoModel()
    raise ValueError(
        f"unknown model spec: {spec!r}. Pass a Model instance directly "
        "or use a recognised prefix (claude-, gpt-, o1-, o3-, echo)."
    )


def _coerce_tool_host(
    tools: list[Tool | Callable[..., object]] | ToolHost | None,
) -> ToolHost:
    if tools is None:
        return InProcessToolHost([])
    # Duck-type: anything with ``list_tools`` and ``call`` is a host.
    if hasattr(tools, "list_tools") and hasattr(tools, "call"):
        return tools  # type: ignore[return-value]
    if isinstance(tools, list):
        return InProcessToolHost(tools)
    raise TypeError(f"unsupported tools= argument: {type(tools).__name__}")


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
