"""Supervisor: workers + a ``delegate`` tool injected into the loop.

Anthropic Multi-Agent Research System (2026 internal report) +
Anthropic Agent Teams (Feb 2026). The 2026 production consensus:
**hierarchical Supervisor is the multi-agent pattern that earns its
cost in production.** Anthropic reports +90.2% on their MA research
benchmark vs single-agent baseline.

Pattern
-------

The supervisor itself runs an architecture (default
:class:`ReAct`). Its tool host is augmented with one extra tool:
``delegate(worker, instructions)``. When the supervising model
calls ``delegate``, the named worker :class:`Agent` runs to
completion with the supervisor's instructions and returns its final
answer as the tool result.

Because :class:`ReAct`'s tool dispatch is already parallel
(:func:`anyio.create_task_group` over all tool calls in a turn),
**the supervisor gets parallel delegation for free** — emit two
``delegate`` calls in one turn and both workers run concurrently.

Replay correctness
------------------
Each ``delegate`` call is wrapped by ``runtime.step`` at the
parent's tool dispatch layer (see ReAct), so the worker's full
:class:`RunResult.output` is journaled in the parent's session.
Replays return the cached worker output without re-running the
worker. The worker is itself an :class:`Agent` and uses a
collision-free session id (parent + worker name + a fresh ULID)
when it does run.

Composition
-----------
* Workers can be any architecture themselves (DeepAgent worker for
  research, ActorCritic worker for code, plain Agent for simple
  specialists).
* Workers can be supervisors (nested teams).
* Wrap Supervisor in Reflexion for cross-session learning of which
  worker handles which intent best.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import TYPE_CHECKING

import anyio
from anyio.streams.memory import MemoryObjectSendStream

from ..core.ids import new_id
from ..core.types import Event
from ..tools.registry import Tool
from .base import AgentSession, Architecture, Dependencies
from .helpers import SubagentInvocation
from .react import ReAct
from .tool_host_wrappers import ExtendedToolHost

if TYPE_CHECKING:
    from ..agent.api import Agent


DEFAULT_SUPERVISOR_TEMPLATE = """\
You are a supervisor coordinating specialist worker agents.

For each task you receive:
1. Decide which workers are needed.
2. Call `delegate(worker, instructions)` to invoke a specialist.
3. Each worker runs independently and returns its final answer.
4. Synthesize worker outputs into a unified response.

You can delegate multiple workers in a single turn — they will run
in parallel. Be specific in the ``instructions`` you pass; workers
do NOT see the user's original message, only what you write.

Available workers:
{worker_descriptions}
"""


class Supervisor:
    """Coordinator + workers, glued by a ``delegate`` tool.

    The supervisor's base architecture (default :class:`ReAct`) sees
    a fresh ``delegate(worker, instructions)`` tool that routes calls
    to the named worker :class:`Agent`. Worker outputs come back as
    tool results just like any other tool call.

    Constructor
    -----------
    * ``workers``: dict mapping role-names to fully-built
      :class:`Agent` instances. Names must be valid identifiers
      (the model emits them as the ``worker`` argument).
    * ``base``: the architecture the supervisor itself runs.
      Default :class:`ReAct`. Wrap inside :class:`Reflexion` to
      learn delegation patterns across runs.
    * ``instructions_template``: format string with
      ``{worker_descriptions}``. Default teaches the supervisor
      to delegate effectively. The agent's own ``instructions``
      are *prepended* (so domain context survives).
    * ``delegate_tool_name``: defaults to ``"delegate"``. Customize
      to avoid clashes with user-defined tools that happen to have
      the same name.
    """

    name = "supervisor"

    def __init__(
        self,
        *,
        workers: dict[str, Agent],
        base: Architecture | None = None,
        instructions_template: str | None = None,
        delegate_tool_name: str = "delegate",
    ) -> None:
        if not workers:
            raise ValueError("Supervisor requires at least one worker")
        self._workers: dict[str, Agent] = dict(workers)
        self._base: Architecture = base if base is not None else ReAct()
        self._template = (
            instructions_template or DEFAULT_SUPERVISOR_TEMPLATE
        )
        self._delegate_name = delegate_tool_name

    def declared_workers(self) -> dict[str, Agent]:
        return dict(self._workers)

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # 1. Set up a shared event channel so worker events emitted
        #    from inside the delegate tool can stream through to the
        #    supervisor's outer generator alongside the base
        #    architecture's events. Without this, MODEL_CHUNK and
        #    TOOL_CALL events from the workers would be dropped on
        #    the floor.
        send_chan, recv_chan = anyio.create_memory_object_stream[Event](
            max_buffer_size=128
        )

        # 2. Build the delegate tool that routes to workers AND
        #    forwards worker events into the shared channel.
        delegate_tool = _make_delegate_tool(
            self._workers,
            session.id,
            tool_name=self._delegate_name,
            event_sink=send_chan,
        )

        # 3. Wrap the parent's ToolHost so the model sees `delegate`
        #    alongside whatever tools the parent already had.
        wrapped_host = ExtendedToolHost(deps.tools, [delegate_tool])

        # 3. Compose instructions: user's domain prompt + supervisor
        #    template (with worker descriptions). Worker descriptions
        #    use each worker Agent's own ``instructions`` so the
        #    supervisor knows what each one does.
        worker_lines = []
        for wname, wagent in self._workers.items():
            desc = (wagent.instructions or "").strip()
            # Trim long instructions so the supervisor prompt stays
            # focused. 200 chars is enough for "you are a Python
            # coder with access to fs and bash" style summaries.
            if len(desc) > 200:
                desc = desc[:197] + "..."
            worker_lines.append(
                f"  - {wname}: {desc or '(no description)'}"
            )
        worker_descriptions = "\n".join(worker_lines)

        supervisor_section = self._template.format(
            worker_descriptions=worker_descriptions
        )
        composed_instructions = (
            f"{session.instructions}\n\n---\n\n{supervisor_section}"
            if session.instructions
            else supervisor_section
        )

        original_instructions = session.instructions
        session.instructions = composed_instructions

        sup_deps = replace(deps, tools=wrapped_host)

        yield Event.architecture_event(
            session.id,
            "supervisor.workers_ready",
            workers=list(self._workers.keys()),
        )

        # 4. Run the base architecture in a background task; both
        #    its events AND any worker events emitted from the
        #    delegate tool flow through the shared channel. We yield
        #    from the channel concurrently.
        async def _run_base() -> None:
            try:
                async for event in self._base.run(
                    session, sup_deps, prompt
                ):
                    await send_chan.send(event)
            finally:
                await send_chan.aclose()

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(_run_base)
                async with recv_chan:
                    async for ev in recv_chan:
                        yield ev
        finally:
            # Restore the original instructions on session even
            # though the session is single-use; harmless and keeps
            # the abstraction clean for tests that re-use sessions.
            session.instructions = original_instructions

        yield Event.architecture_event(
            session.id,
            "supervisor.completed",
            workers=list(self._workers.keys()),
        )


# ---------------------------------------------------------------------------
# delegate tool + tool-host wrapper
# ---------------------------------------------------------------------------


def _make_delegate_tool(
    workers: dict[str, Agent],
    parent_session_id: str,
    *,
    tool_name: str,
    event_sink: MemoryObjectSendStream[Event] | None = None,
) -> Tool:
    """Build a :class:`Tool` whose ``execute`` routes to the named
    worker :class:`Agent` and returns its final output.

    Each invocation generates a fresh ULID-suffixed session_id for
    the worker; replay correctness for the parent comes from the
    parent's own runtime journal caching the tool result, so the
    worker's session_id only matters during the first execution.

    When ``event_sink`` is provided, the worker's streaming events
    (model chunks, nested tool calls, tool results, architecture
    progress) are forwarded into the channel so the supervisor's
    outer ``stream(...)`` consumer sees token-by-token output. When
    ``None``, the worker's events are silently dropped (legacy
    behaviour, kept as a fallback).
    """

    async def _delegate(worker: str, instructions: str) -> str:
        agent = workers.get(worker)
        if agent is None:
            known = ", ".join(sorted(workers))
            return (
                f"Error: unknown worker {worker!r}. Known: {known}"
            )
        suffix = new_id("del")
        worker_session_id = (
            f"{parent_session_id}__delegate_{worker}_{suffix}"
        )

        if event_sink is None:
            # No event sink → fall back to plain run() (events lost).
            result = await agent.run(
                instructions, session_id=worker_session_id
            )
            return result.output

        # Stream worker events into the supervisor's shared channel
        # using a clone of the send half (clone keeps the channel
        # alive past this tool call's lifetime).
        invocation = SubagentInvocation(
            agent, instructions, session_id=worker_session_id
        )
        async with event_sink.clone() as sink:
            async for ev in invocation.events():
                await sink.send(ev)
        return str(invocation.result.get("output", ""))

    return Tool(
        name=tool_name,
        description=(
            "Delegate a subtask to a named specialist worker. "
            "The worker runs independently and returns its final "
            "answer as a string."
        ),
        fn=_delegate,
        input_schema={
            "type": "object",
            "properties": {
                "worker": {
                    "type": "string",
                    "description": (
                        "Name of the worker to delegate to. Must be "
                        "one of the configured workers."
                    ),
                },
                "instructions": {
                    "type": "string",
                    "description": (
                        "Task description for the worker. Be "
                        "specific — the worker does not see the "
                        "user's original message."
                    ),
                },
            },
            "required": ["worker", "instructions"],
        },
    )


