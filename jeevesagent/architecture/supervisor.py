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

from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..core.ids import new_id
from ..core.types import Event, ToolDef, ToolEvent, ToolResult
from ..tools.registry import Tool
from .base import AgentSession, Architecture, Dependencies
from .react import ReAct

if TYPE_CHECKING:
    from ..agent.api import Agent
    from ..core.protocols import ToolHost


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
        # 1. Build the delegate tool that routes to workers.
        delegate_tool = _make_delegate_tool(
            self._workers,
            session.id,
            tool_name=self._delegate_name,
        )

        # 2. Wrap the parent's ToolHost so the model sees `delegate`
        #    alongside whatever tools the parent already had.
        wrapped_host = _SupervisorToolHost(deps.tools, delegate_tool)

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

        # 4. Run the base architecture. ReAct sees the delegate tool
        #    and decides when to call it; multiple delegate calls in
        #    one turn are dispatched in parallel by ReAct's existing
        #    parallel tool dispatch.
        try:
            async for event in self._base.run(session, sup_deps, prompt):
                yield event
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
) -> Tool:
    """Build a :class:`Tool` whose ``execute`` routes to the named
    worker :class:`Agent` and returns its final output.

    Each invocation generates a fresh ULID-suffixed session_id for
    the worker; replay correctness for the parent comes from the
    parent's own runtime journal caching the tool result, so the
    worker's session_id only matters during the first execution.
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
        result = await agent.run(
            instructions, session_id=worker_session_id
        )
        return result.output

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


class _SupervisorToolHost:
    """Wraps a base :class:`ToolHost` and adds one ``delegate`` tool.

    All other tool calls are forwarded to the wrapped base host, so
    workers + parent-defined tools coexist transparently.
    """

    def __init__(self, base: ToolHost, delegate_tool: Tool) -> None:
        self._base = base
        self._delegate = delegate_tool

    async def list_tools(
        self, *, query: str | None = None
    ) -> list[ToolDef]:
        defs = list(await self._base.list_tools(query=query))
        delegate_def = self._delegate.to_def()
        if query is None or (
            query.lower() in delegate_def.name.lower()
            or query.lower() in delegate_def.description.lower()
        ):
            defs.append(delegate_def)
        return defs

    async def call(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> ToolResult:
        if tool == self._delegate.name:
            try:
                output = await self._delegate.execute(args)
            except Exception as exc:  # noqa: BLE001
                return ToolResult.error_(
                    call_id=call_id, message=str(exc)
                )
            return ToolResult.success(call_id=call_id, output=output)
        return await self._base.call(tool, args, call_id=call_id)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        async for ev in self._base.watch():
            yield ev
