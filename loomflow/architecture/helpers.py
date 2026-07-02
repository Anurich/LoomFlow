"""Cross-architecture helpers.

Small utilities multiple architectures need. Putting them here keeps
each architecture's module focused on its strategy and avoids
circular re-implementation:

* :func:`text_only_model_call` — run a single model call with no
  tools, collecting the response text and usage. Used by Self-Refine
  (critic / refiner), Reflexion (evaluator / reflector),
  Plan-and-Execute (planner / replanner), Router (classifier), and
  any other architecture that needs a one-shot structured LLM call.
* :func:`add_usage` — sum two :class:`Usage` records.
* :func:`budget_gate` — the canonical pre-step budget check
  (``allows_step`` with ``user_id`` forwarding + legacy fallback,
  blocked/warn event construction, session interruption marking).
  ONE implementation so per-user budget caps can't silently drift
  out of sync between architectures.
* :func:`consume_usage` — the canonical post-model-call accounting
  (``budget.consume`` with ``user_id`` + legacy fallback, cumulative
  usage rollup, optional turn increment).
* :func:`parse_fenced_json` / :func:`strip_markdown_fences` —
  tolerant JSON parsing of model output that may be wrapped in
  markdown code fences. Used by every architecture with a
  structured-output planner / coordinator / critic step.
* :func:`run_single_tool` — the gated tool executor shared by ReAct
  and ReWOO: hooks.pre_tool → permissions.check → approval handler
  → (timeout-bounded) tool host call → hooks.post_tool. Any
  architecture that executes model-planned tool calls MUST route
  through this so destructive tools hit the same gates everywhere.
* :func:`run_gated_tool` — :func:`run_single_tool` wrapped with the
  tool_call / tool_result audit-log writes.
* :func:`parse_score` — extract a 0-1 confidence number from
  free-form evaluator output. Used by Reflexion and Tree of Thoughts;
  any architecture with an evaluator step.
* :class:`SubagentInvocation` — run a sub-:class:`Agent` and stream
  its events through to the parent's generator while capturing the
  final :class:`RunResult` separately. Used by Swarm, Supervisor,
  Router, ActorCritic, Debate, and Blackboard so the inner agent's
  ``MODEL_CHUNK`` / ``TOOL_CALL`` / ``TOOL_RESULT`` events surface
  in the outermost ``agent.stream(...)`` consumer.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

import anyio

from ..core._deprecation import warn_legacy_protocol
from ..core.context import RunContext, get_run_context
from ..core.types import (
    Event,
    Message,
    PermissionDecision,
    ToolCall,
    ToolResult,
    Usage,
)
from ..security.audit import AuditLog, wants_full_transcripts
from .base import Dependencies

if TYPE_CHECKING:
    from ..agent.api import Agent
    from ..core.protocols import Model
    from ..tools.registry import Tool
    from .base import AgentSession

# Module-level singleton no-op async context manager.
# ``contextlib.nullcontext`` implements both the sync and async
# protocols (since Python 3.10), so we can reuse a single instance
# everywhere a hot path wants to *maybe* enter a telemetry span via
# ``async with (_NULL_CTX if fast else tel.trace(...)):``.
_NULL_CTX: contextlib.AbstractAsyncContextManager[None] = contextlib.nullcontext()


async def text_only_model_call(
    deps: Dependencies,
    step_name: str,
    messages: list[Message],
    model: Model | None = None,
    *,
    output_schema: Any | None = None,
) -> tuple[str, Usage]:
    """Run a single text-only model call (no tools exposed).

    Returns ``(text, usage)``. Used for one-shot structured prompts
    (critique, evaluation, classification, planning, synthesis).

    Mirrors ReAct's model-call paths:

    * **Non-streaming fast path** — when nobody is reading from
      ``agent.stream()`` and the adapter exposes ``complete()``,
      prefer the single-shot call (no SSE overhead, no per-chunk
      allocations). Journaled via ``runtime.step`` unless
      ``deps.fast_runtime`` lets us inline.
    * **Streaming path** — journaled ``runtime.stream_step`` so
      replays are deterministic.

    ``deps.prompt_caching`` is always forwarded so adapters can apply
    ``cache_control`` markers / cache keys on these calls too.

    ``model`` overrides which model handles the call; it defaults to
    ``deps.model`` so existing callers are unchanged. The
    :class:`~loomflow.GoalStopHook` passes ``deps.goal_checker`` here so
    a cheap checker model can judge the stop condition.

    ``output_schema`` lets final-answer-producing call sites (ReWOO
    solver, Plan-and-Execute synthesizer, SelfRefine refiner) forward
    ``deps.output_schema`` so native structured-output adapters can
    constrain the response. Defaults to ``None`` — intermediate calls
    (critics, planners, classifiers) must NOT be schema-constrained.
    """
    mdl = model if model is not None else deps.model

    if not deps.streaming and hasattr(mdl, "complete"):
        if deps.fast_runtime:
            text, _tool_calls, usage, _finish_reason = await mdl.complete(
                messages,
                tools=None,
                output_schema=output_schema,
                effort=deps.effort,
                strict_effort=deps.strict_effort,
                prompt_caching=deps.prompt_caching,
            )
        else:
            text, _tool_calls, usage, _finish_reason = (
                await deps.runtime.step(  # type: ignore[func-returns-value]
                    step_name,
                    mdl.complete,
                    messages,
                    tools=None,
                    output_schema=output_schema,
                    effort=deps.effort,
                    strict_effort=deps.strict_effort,
                    prompt_caching=deps.prompt_caching,
                )
            )
        return text, usage

    text_parts: list[str] = []
    usage = Usage()

    chunks = deps.runtime.stream_step(
        step_name,
        mdl.stream,
        messages,
        tools=None,
        output_schema=output_schema,
        effort=deps.effort,
        strict_effort=deps.strict_effort,
        prompt_caching=deps.prompt_caching,
    )
    async for chunk in chunks:
        if chunk.kind == "text" and chunk.text is not None:
            text_parts.append(chunk.text)
        elif chunk.kind == "finish" and chunk.usage is not None:
            usage = chunk.usage

    return "".join(text_parts), usage


# ---------------------------------------------------------------------------
# Budget helpers — ONE implementation of the allows_step / consume
# dance so per-user caps can't silently drift between architectures.
# ---------------------------------------------------------------------------


async def budget_gate(
    deps: Dependencies, session: AgentSession
) -> tuple[bool, list[Event]]:
    """Canonical pre-step budget check.

    Returns ``(blocked, events)``. Callers yield the returned events
    and stop iterating when ``blocked`` is True (``session.interrupted``
    / ``interruption_reason`` are already set by then).

    Always forwards ``deps.context.user_id`` to ``allows_step`` so
    per-user budget caps fire in EVERY architecture, with the
    ``except TypeError`` fallback for legacy budget impls that
    predate the kwarg. Skipped entirely on the ``fast_budget`` path
    (``NoBudget`` returns OK unconditionally).
    """
    if deps.fast_budget:
        return False, []
    try:
        status = await deps.budget.allows_step(
            user_id=deps.context.user_id
        )
    except TypeError:
        # Legacy budget impls without the user_id kwarg.
        status = await deps.budget.allows_step()
    if status.blocked:
        session.interrupted = True
        session.interruption_reason = f"budget:{status.reason}"
        if not deps.fast_telemetry:
            await deps.telemetry.emit_metric(
                "loom.budget.exceeded",
                1,
                session_id=session.id,
                reason=status.reason,
            )
        return True, [Event.budget_exceeded(session.id, status)]
    if status.warn:
        return False, [Event.budget_warning(session.id, status)]
    return False, []


async def consume_usage(
    deps: Dependencies,
    session: AgentSession,
    usage: Usage,
    *,
    count_turn: bool = True,
) -> None:
    """Canonical post-model-call accounting.

    ``budget.consume`` with ``user_id`` forwarding (+ legacy
    fallback), cumulative-usage rollup on the session, and — for
    architectures that count each helper model call as a turn —
    the turn increment. ReAct increments its turn counter at the
    top of its loop, so it passes ``count_turn=False``.
    """
    if not deps.fast_budget:
        try:
            await deps.budget.consume(
                tokens_in=usage.input_tokens,
                tokens_out=usage.output_tokens,
                cost_usd=usage.cost_usd,
                user_id=deps.context.user_id,
            )
        except TypeError:
            # Legacy budget impls without the user_id kwarg.
            await deps.budget.consume(
                tokens_in=usage.input_tokens,
                tokens_out=usage.output_tokens,
                cost_usd=usage.cost_usd,
            )
    session.cumulative_usage = add_usage(session.cumulative_usage, usage)
    if count_turn:
        session.turns += 1


def usage_from_result_dict(result: Mapping[str, Any]) -> Usage:
    """Convert a ``RunResult``-shaped dict (``SubagentInvocation
    .result``) into a :class:`Usage`.

    Mind the field-name swap: RunResult dicts use ``tokens_in`` /
    ``tokens_out`` / ``cached_tokens_in`` while ``Usage`` uses
    ``input_tokens`` / ``output_tokens`` / ``cached_input_tokens``
    — the rest line up.
    """
    return Usage(
        input_tokens=int(result.get("tokens_in", 0) or 0),
        cached_input_tokens=int(
            result.get("cached_tokens_in", 0) or 0
        ),
        cache_write_tokens=int(
            result.get("cache_write_tokens", 0) or 0
        ),
        output_tokens=int(result.get("tokens_out", 0) or 0),
        cost_usd=float(result.get("cost_usd", 0) or 0),
    )


async def consume_worker_usage(
    deps: Dependencies,
    worker: Any,
    usage: Usage,
) -> None:
    """Charge a completed sub-agent's spend against the PARENT budget.

    Budget-only counterpart of :func:`consume_usage`. Multi-agent
    architectures roll a worker's usage into
    ``session.cumulative_usage`` via ``SubagentInvocation(
    rollup_into=session)`` (or an inline ``add_usage``), so calling
    :func:`consume_usage` here would double-count the session
    totals — this helper touches ONLY ``deps.budget``.

    Skipped when:

    * ``deps.fast_budget`` — parent budget is ``NoBudget``.
    * the worker :class:`Agent` shares the parent's budget INSTANCE
      — the worker's own run already consumed against it, and
      charging again here would double-bill.
    * the usage is all-zero (nothing to charge).
    """
    if deps.fast_budget:
        return
    if getattr(worker, "budget", None) is deps.budget:
        return
    if not (
        usage.input_tokens
        or usage.cached_input_tokens
        or usage.output_tokens
        or usage.cost_usd
    ):
        return
    try:
        await deps.budget.consume(
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cost_usd=usage.cost_usd,
            user_id=deps.context.user_id,
        )
    except TypeError:
        # Legacy budget impls without the user_id kwarg.
        await deps.budget.consume(
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )


# ---------------------------------------------------------------------------
# Fenced-JSON parsing — shared by every architecture that asks the
# model for structured JSON and needs to tolerate markdown fences.
# ---------------------------------------------------------------------------


def strip_markdown_fences(text: str) -> str:
    """Strip a wrapping markdown code fence (```/```json) if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_fenced_json(text: str) -> Any | None:
    """``json.loads`` tolerant of markdown code fences.

    Returns the parsed object, or ``None`` on parse failure —
    callers decide what a failed parse means (empty plan, no-op
    coordinator decision, zero-score critique, ...).
    """
    try:
        return json.loads(strip_markdown_fences(text))
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Gated tool execution — shared by ReAct and ReWOO (and any future
# architecture that executes model-planned tool calls). Hooks →
# permissions → approval handler → journaled + timeout-bounded tool
# host call → post-hook, with audit-log writes in the *_gated_*
# wrapper. ONE implementation so a destructive tool hits the same
# gates and audit trail regardless of which architecture planned it.
# ---------------------------------------------------------------------------


def tool_result_payload(
    *,
    call: ToolCall,
    result: ToolResult,
    turn: int,
    audit_log: AuditLog | None,
) -> dict[str, Any]:
    """Build the ``tool_result`` audit payload, adding the result
    body when a :class:`FullTranscriptAuditLog` is wired."""
    payload: dict[str, Any] = {
        "tool": call.tool,
        "call_id": result.call_id,
        "ok": result.ok,
        "denied": result.denied,
        "error": result.error,
        "reason": result.reason,
        "turn": turn,
    }
    if wants_full_transcripts(audit_log):
        payload["output"] = result.output
        payload["duration_ms"] = result.duration_ms
    return payload


async def audit(
    audit_log: AuditLog | None,
    session_id: str,
    actor: str,
    action: str,
    payload: dict[str, Any],
    *,
    user_id: str | None = None,
) -> None:
    """Append an audit entry, tolerating legacy AuditLog impls."""
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
        logging.getLogger("loomflow.architecture.helpers").warning(
            "approval_handler raised for tool=%s; treating as deny: %s",
            call.tool,
            exc,
        )
        return False


async def run_single_tool(
    deps: Dependencies,
    call: ToolCall,
    *,
    turn: int,
    slot: int,
    step_name: str | None = None,
    destructive_map: Mapping[str, bool] | None = None,
) -> ToolResult:
    """Hooks → permission → journaled tool host call → post-hook.

    Layers (hooks / permissions / runtime / telemetry) are skipped
    when their corresponding ``fast_*`` flag on ``deps`` is set —
    e.g. ``AllowAll`` permissions short-circuit the ``check`` call,
    an empty ``HookRegistry`` short-circuits ``pre_tool`` /
    ``post_tool`` dispatch, and ``InProcRuntime`` lets us inline
    the tool host call (skipping idempotency-key derivation).

    ``step_name`` names the ``runtime.step`` for journaled replay;
    defaults to the ReAct-shaped ``tool_call_{turn}_{slot}``.
    Architectures with their own deterministic naming (ReWOO's
    ``rewoo_step_{id}``) pass it explicitly.

    ``destructive_map`` (``{tool_name: destructive}``) lets callers
    who already fetched ``list_tools()`` this turn stamp the
    destructive flag without a second round-trip to the tool host;
    when ``None`` we fall back to fetching the defs here.

    ``deps.tool_timeout_s`` bounds the actual tool execution with
    ``anyio.fail_after`` — a stuck tool becomes a
    ``ToolResult.error_`` the model can react to instead of hanging
    the whole run.
    """
    started = anyio.current_time()
    result: ToolResult

    tool_trace: contextlib.AbstractAsyncContextManager[Any] = (
        _NULL_CTX
        if deps.fast_telemetry
        else deps.telemetry.trace(
            "loom.tool",
            tool=call.tool,
            call_id=call.id,
            turn=turn,
        )
    )

    async with tool_trace:
        run_user_id = deps.context.user_id
        # Stamp ``call.destructive`` from the tool host BEFORE any
        # permission check. Background: ``ToolCall.destructive``
        # defaults to False, and model adapters (openai, anthropic)
        # construct ToolCall from the model's tool_use response
        # without consulting the original Tool's ``destructive``
        # flag — so a call to a ``destructive=True`` tool would
        # arrive at permissions.check with ``destructive=False`` and
        # auto-approve, bypassing the approval handler entirely.
        # Tool.to_def() now propagates the flag (registry.py), so
        # any well-behaved adapter could stamp it themselves; this
        # block is the defensive backstop that fixes the bug for
        # adapters (current OpenAI/Anthropic ones included) that
        # don't. Prefer the caller-supplied ``destructive_map``
        # (built once per turn from an already-fetched list_tools)
        # over a fresh round-trip to the tool host.
        if not call.destructive:
            if destructive_map is not None:
                if destructive_map.get(call.tool, False):
                    call = call.model_copy(update={"destructive": True})
            else:
                try:
                    defs = await deps.tools.list_tools()
                    for d in defs:
                        if d.name == call.tool and d.destructive:
                            call = call.model_copy(
                                update={"destructive": True}
                            )
                            break
                except Exception as exc:  # noqa: BLE001 — host may be flaky; never crash here
                    # FAIL CLOSED. If we can't fetch the tool defs we
                    # don't know whether this tool is destructive —
                    # letting the call proceed as destructive=False
                    # would silently skip the ask/approval gate for a
                    # possibly-destructive tool. Stamping True routes
                    # it through the approval handler; the worst case
                    # of the wrong polarity is an unnecessary
                    # approval prompt, not an ungated destructive
                    # call.
                    logging.getLogger(
                        "loomflow.architecture.helpers"
                    ).warning(
                        "list_tools() failed while stamping the "
                        "destructive flag for tool=%s; failing "
                        "closed (treating call as destructive): %s",
                        call.tool,
                        exc,
                    )
                    call = call.model_copy(
                        update={"destructive": True}
                    )
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
                resolved_step = (
                    step_name
                    if step_name is not None
                    else f"tool_call_{turn}_{slot}"
                )
                timeout_s = deps.tool_timeout_s
                timeout_cm: contextlib.AbstractContextManager[Any] = (
                    anyio.fail_after(timeout_s)
                    if timeout_s is not None
                    else contextlib.nullcontext()
                )
                try:
                    with timeout_cm:
                        if deps.fast_runtime:
                            result = await deps.tools.call(
                                call.tool,
                                call.args,
                                call_id=call.id,
                            )
                        else:
                            result = await deps.runtime.step(
                                resolved_step,
                                deps.tools.call,
                                call.tool,
                                call.args,
                                call_id=call.id,
                                idempotency_key=call.idempotency_key(),
                            )
                except TimeoutError:
                    result = ToolResult.error_(
                        call.id,
                        f"tool timed out after {timeout_s}s",
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
            "loom.tool.duration_ms",
            elapsed_ms,
            tool=call.tool,
            ok=result.ok,
            denied=result.denied,
        )
    return result


async def run_gated_tool(
    deps: Dependencies,
    session: AgentSession,
    call: ToolCall,
    *,
    slot: int,
    step_name: str | None = None,
    destructive_map: Mapping[str, bool] | None = None,
) -> ToolResult:
    """:func:`run_single_tool` bracketed by audit-log writes.

    Emits the ``tool_call`` audit entry before execution and the
    ``tool_result`` entry after, exactly as ReAct's dispatch workers
    do — so architectures that execute planned tool calls outside a
    ReAct turn (ReWOO) leave the same audit trail.
    """
    if not deps.fast_audit:
        await audit(
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
    result = await run_single_tool(
        deps,
        call,
        turn=session.turns,
        slot=slot,
        step_name=step_name,
        destructive_map=destructive_map,
    )
    if not deps.fast_audit:
        await audit(
            deps.audit_log,
            session.id,
            "system",
            "tool_result",
            tool_result_payload(
                call=call,
                result=result,
                turn=session.turns,
                audit_log=deps.audit_log,
            ),
            user_id=deps.context.user_id,
        )
    return result


def add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        cached_input_tokens=a.cached_input_tokens + b.cached_input_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cost_usd=a.cost_usd + b.cost_usd,
    )


_SCORE_LINE_RE = re.compile(
    r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
)
# Prose fallback: DECIMAL forms only (``0.85``, ``1.0``). Bare
# integers are deliberately excluded — "0 errors found" or "1 issue
# remains" must not parse as a score of 0.0 / 1.0.
_FALLBACK_DECIMAL_RE = re.compile(r"\b(0?\.\d+|1\.0+)\b")
# Whole-line fallback: a line that is NOTHING but a plausible 0-1
# number ("0", "1", "0.7") is unambiguous even for bare integers.
_WHOLE_LINE_NUMBER_RE = re.compile(r"[01](?:\.\d+)?")


class SubagentInvocation:
    """Run a sub-Agent and stream its events to the parent.

    Use this from any multi-agent architecture instead of calling
    ``await worker.run(prompt, ...)`` directly. The plain ``run()``
    drops events on the floor; this helper forwards them to the
    parent generator so token-level streaming works end-to-end.

    Usage from inside an architecture's ``run()`` async generator::

        invocation = SubagentInvocation(
            worker, prompt, session_id="...", extra_tools=[...]
        )
        async for event in invocation.events():
            yield event
        result = invocation.result   # dict version of RunResult

    Filtering policy:

    * **Suppressed** in the parent stream: ``STARTED`` and
      ``COMPLETED`` from the sub-agent — those are internal framing
      events; the parent owns its own STARTED/COMPLETED. The
      sub-agent's ``RunResult`` (carried in its ``COMPLETED`` event
      payload) is captured into ``self.result`` for the architecture
      to read.
    * **Forwarded** as-is: ``MODEL_CHUNK`` (token-level streaming),
      ``TOOL_CALL`` / ``TOOL_RESULT`` (with full args / output),
      ``BUDGET_WARNING`` / ``BUDGET_EXCEEDED``, ``ERROR``,
      ``ARCHITECTURE_EVENT`` (so a nested architecture's progress
      events bubble up too).
    """

    def __init__(
        self,
        agent: Agent,
        prompt: str,
        *,
        session_id: str | None = None,
        context: RunContext | None = None,
        extra_tools: list[Tool] | None = None,
        buffer_size: int = 128,
        rollup_into: AgentSession | None = None,
    ) -> None:
        self._agent = agent
        self._prompt = prompt
        self._session_id = session_id
        # Sub-agents inherit the parent's :class:`RunContext` by
        # default — read the live context off the contextvar that
        # ``Agent._loop`` installed when the parent run started.
        # That propagates ``user_id`` and ``metadata`` down a
        # multi-agent tree without each architecture having to
        # plumb them by hand. ``session_id`` (if supplied) overrides
        # the parent's so each spawn gets its own conversation
        # thread; if not, the framework auto-generates a fresh one.
        # When called outside an active parent run,
        # ``get_run_context`` returns the empty default — sub-agents
        # then run anonymously, same as direct ``Agent.run`` with
        # no kwargs.
        base_ctx = context if context is not None else get_run_context()
        # Subagent parent-attribution (0.10.18): record the parent's
        # session_id + run_id under reserved metadata keys so the
        # child agent (and any downstream telemetry / audit /
        # custom tools) can attribute its work back to the spawning
        # parent. Reserved keys are namespaced with ``_loomflow_``
        # so they can't collide with user metadata. When the
        # caller supplied an explicit ``context``, we still augment
        # — the explicit context wins on user_id/session_id, but
        # parent attribution is additive metadata that helps
        # observability without overriding intent.
        parent_session = base_ctx.session_id
        parent_run = base_ctx.run_id
        if parent_session or parent_run:
            merged_metadata: dict[str, Any] = dict(base_ctx.metadata)
            if parent_session:
                merged_metadata.setdefault(
                    "_loomflow_parent_session_id", parent_session
                )
            if parent_run:
                merged_metadata.setdefault(
                    "_loomflow_parent_run_id", parent_run
                )
            base_ctx = base_ctx.with_overrides(
                metadata=merged_metadata
            )
        self._context = base_ctx
        self._extra_tools = extra_tools
        self._buffer_size = buffer_size
        # When provided, the sub-agent's ``RunResult`` usage (input /
        # cached / cache-write / output tokens + cost) is rolled into
        # ``rollup_into.cumulative_usage`` the moment the sub-agent's
        # ``completed`` event fires. Without this, every architecture
        # using SubagentInvocation silently under-counts: the
        # parent's RunResult.cost_usd would reflect only the parent's
        # own model calls, not the worker's. Architectures pass
        # ``rollup_into=session`` from their ``run()`` to get correct
        # accounting "for free." Optional so callers outside the
        # architecture protocol can still use the helper.
        self._rollup_into = rollup_into
        self.result: dict[str, Any] = {}

    async def events(self) -> AsyncIterator[Event]:
        """Yield the sub-agent's events (filtered) as they happen.

        After the iterator drains, ``self.result`` contains the
        sub-agent's :class:`RunResult` as a dict (with ``output``,
        ``turns``, ``tokens_in``, ``tokens_out``, ``cost_usd``,
        ``interrupted``, ``interruption_reason``).
        """
        send, receive = anyio.create_memory_object_stream[Event](
            max_buffer_size=self._buffer_size
        )

        async def _capture(ev: Event) -> None:
            if ev.kind.value == "completed":
                # Capture the result dict; don't forward — parent
                # emits its own COMPLETED at the end of its own loop.
                result = ev.payload.get("result", {}) or {}
                self.result.update(result)
                # Roll worker usage into the parent session so the
                # parent's RunResult.cost_usd / tokens reflect work
                # the sub-agent did (field-name swap handled by
                # ``usage_from_result_dict``).
                if self._rollup_into is not None:
                    sub_usage = usage_from_result_dict(result)
                    self._rollup_into.cumulative_usage = add_usage(
                        self._rollup_into.cumulative_usage, sub_usage
                    )
            elif ev.kind.value == "started":
                # Suppress sub-agent's STARTED; parent owns framing.
                return
            else:
                await send.send(ev)

        async def _run_worker() -> None:
            try:
                await self._agent.run(
                    self._prompt,
                    session_id=self._session_id,
                    context=self._context,
                    extra_tools=self._extra_tools,
                    emit=_capture,
                )
            finally:
                await send.aclose()

        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_worker)
            async with receive:
                async for ev in receive:
                    yield ev


def parse_score(text: str) -> float:
    """Extract a 0-1 score from free-form evaluator output.

    Parsing order:

    1. The ``score: X`` (or ``score=X``) pattern anywhere in the
       text — the documented evaluator format.
    2. A line that consists of NOTHING but a 0-1 number (``"0.7"``,
       ``"1"``) — unambiguous even for bare integers.
    3. A decimal-form number in prose (``"scored 0.6 overall"``).
       Bare integers in prose are deliberately NOT matched — a
       critique like ``"0 errors found"`` or ``"1 issue remains"``
       is not a score and previously parsed as 0.0 / 1.0.

    Clamps to ``[0.0, 1.0]``. Returns 0.0 on parse failure (treated
    as a failed evaluation — let the caller decide what that means).

    Used by :class:`~loomflow.architecture.Reflexion` (attempt
    score) and :class:`~loomflow.architecture.TreeOfThoughts`
    (per-thought evaluation).
    """
    match = _SCORE_LINE_RE.search(text)
    if match is not None:
        try:
            value = float(match.group(1))
        except ValueError:
            return 0.0
        return max(0.0, min(1.0, value))
    for line in text.strip().splitlines():
        if _WHOLE_LINE_NUMBER_RE.fullmatch(line.strip()):
            return max(0.0, min(1.0, float(line.strip())))
    fallback = _FALLBACK_DECIMAL_RE.search(text)
    if fallback is None:
        return 0.0
    try:
        value = float(fallback.group(1))
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, value))
