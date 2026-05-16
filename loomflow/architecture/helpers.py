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

import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import anyio

from ..core.context import RunContext, get_run_context
from ..core.types import Event, Message, Usage
from .base import Dependencies

if TYPE_CHECKING:
    from ..agent.api import Agent
    from ..tools.registry import Tool
    from .base import AgentSession


async def text_only_model_call(
    deps: Dependencies,
    step_name: str,
    messages: list[Message],
) -> tuple[str, Usage]:
    """Run a single text-only model call through ``runtime.step``.

    Returns ``(text, usage)``. The call is journaled so replays are
    deterministic, but no tools are exposed — used for one-shot
    structured prompts (critique, evaluation, classification,
    planning).
    """
    text_parts: list[str] = []
    usage = Usage()

    chunks = deps.runtime.stream_step(
        step_name,
        deps.model.stream,
        messages,
        tools=None,
        effort=deps.effort,
        strict_effort=deps.strict_effort,
    )
    async for chunk in chunks:
        if chunk.kind == "text" and chunk.text is not None:
            text_parts.append(chunk.text)
        elif chunk.kind == "finish" and chunk.usage is not None:
            usage = chunk.usage

    return "".join(text_parts), usage


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
_FALLBACK_NUMBER_RE = re.compile(r"\b(0\.\d+|1\.0+|0|1)\b")


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
                # the sub-agent did. Mind the field-name swap:
                # RunResult dict uses ``tokens_in`` / ``tokens_out``
                # / ``cached_tokens_in`` while ``Usage`` uses
                # ``input_tokens`` / ``output_tokens`` /
                # ``cached_input_tokens`` — the rest line up.
                if self._rollup_into is not None:
                    sub_usage = Usage(
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

    Prefers the ``score: X`` (or ``score=X``) pattern; falls back to
    any plausible number in the text. Clamps to ``[0.0, 1.0]``.
    Returns 0.0 on parse failure (treated as a failed evaluation —
    let the caller decide what that means).

    Used by :class:`~loomflow.architecture.Reflexion` (attempt
    score) and :class:`~loomflow.architecture.TreeOfThoughts`
    (per-thought evaluation).
    """
    match = _SCORE_LINE_RE.search(text)
    if match is None:
        match = _FALLBACK_NUMBER_RE.search(text)
    if match is None:
        return 0.0
    try:
        value = float(match.group(1))
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, value))
