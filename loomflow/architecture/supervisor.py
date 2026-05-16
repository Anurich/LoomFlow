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
from typing import TYPE_CHECKING, Any

import anyio
from anyio.streams.memory import MemoryObjectSendStream

from ..core.context import get_run_context
from ..core.ids import new_id
from ..core.types import Event, Usage
from ..tools.registry import Tool
from .base import AgentSession, Architecture, Dependencies
from .helpers import SubagentInvocation, add_usage
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
4. Either synthesize worker outputs into a unified response, OR
   call `forward_message(worker)` if a single worker's output IS
   already the final answer the user wants. Forwarding skips a
   paraphrase round-trip: the worker's output is returned verbatim
   as YOUR final response. End your turn immediately after a
   forward_message call.

You can delegate multiple workers in a single turn — they will run
in parallel. Be specific in the ``instructions`` you pass; workers
do NOT see the user's original message, only what you write.

5. To CONTINUE a conversation with a worker (build on its earlier
   work rather than starting fresh), call
   ``send_message(to=<worker_id>, content=...)``. The ``worker_id``
   is printed at the top of every ``delegate`` response in square
   brackets like ``[worker_id: worker_coder_01J...]``. The worker
   remembers its full prior context — use ``send_message`` for
   iterations / follow-ups and ``delegate`` to start a fresh
   conversation on a different topic.

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
    * ``forward_tool_name``: defaults to ``"forward_message"``. The
      supervisor calls this with a worker name to return that
      worker's last output VERBATIM as the supervisor's final
      response. Skips a synthesis round-trip — the
      `langchain.com/blog/benchmarking-multi-agent-architectures`_
      benchmark showed +50% quality on tasks where the supervisor
      would otherwise paraphrase a worker's output.
    """

    name = "supervisor"

    def __init__(
        self,
        *,
        workers: dict[str, Agent],
        base: Architecture | None = None,
        instructions_template: str | None = None,
        delegate_tool_name: str = "delegate",
        forward_tool_name: str = "forward_message",
        worker_registry: dict[str, Any] | None = None,
        role_to_worker_id: dict[str, str] | None = None,
    ) -> None:
        if not workers:
            raise ValueError("Supervisor requires at least one worker")
        self._workers: dict[str, Agent] = dict(workers)
        self._base: Architecture = base if base is not None else ReAct()
        self._template = (
            instructions_template or DEFAULT_SUPERVISOR_TEMPLATE
        )
        self._delegate_name = delegate_tool_name
        self._forward_name = forward_tool_name
        # ``worker_registry`` + ``role_to_worker_id`` are the
        # persistent-subagent wiring. None on both = legacy
        # stateless-per-delegate behavior (preserved for tests +
        # callers that explicitly opt out via
        # ``Team.supervisor(persistent_subagents=False)``).
        self._worker_registry: dict[str, Any] | None = worker_registry
        self._role_to_worker_id: dict[str, str] | None = (
            role_to_worker_id
        )

    def declared_workers(self) -> dict[str, Agent]:
        return dict(self._workers)

    def add_worker(self, name: str, agent: Agent) -> None:
        """Register a worker between runs.

        Safe to call between :meth:`Agent.run` invocations on the
        agent that owns this supervisor; the new worker becomes
        available for ``delegate(name, ...)`` on the next run.
        Calling mid-run is undefined — the supervisor's prompt is
        composed at run start.
        """
        if not name or not name.isidentifier():
            raise ValueError(
                f"worker name {name!r} must be a valid identifier"
            )
        self._workers[name] = agent

    def remove_worker(self, name: str) -> Agent | None:
        """Unregister a worker by name. Returns the removed Agent
        if it was registered, ``None`` otherwise. Same lifecycle
        rules as :meth:`add_worker`."""
        return self._workers.pop(name, None)

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

        # 2. Build the delegate + forward_message tools.
        #    - delegate routes to workers and forwards events.
        #    - forward_message lets the supervisor return a
        #      worker's last output verbatim, bypassing a
        #      paraphrase round-trip.
        # Both tools share the ``last_outputs`` dict (delegate
        # writes, forward_message reads). ``forward_request``
        # captures the worker text the supervisor wants forwarded;
        # we override session.output with it after the base
        # architecture completes.
        last_outputs: dict[str, str] = {}
        forward_request: dict[str, str] = {}

        delegate_tool = _make_delegate_tool(
            self._workers,
            session,
            tool_name=self._delegate_name,
            event_sink=send_chan,
            last_outputs=last_outputs,
            worker_registry=self._worker_registry,
            role_to_worker_id=self._role_to_worker_id,
        )
        forward_tool = _make_forward_message_tool(
            last_outputs=last_outputs,
            forward_request=forward_request,
            tool_name=self._forward_name,
            worker_names=list(self._workers.keys()),
        )

        # 3. Wrap the parent's ToolHost so the model sees `delegate`
        #    + `forward_message` (+ `send_message` when persistent
        #    subagents are enabled) alongside whatever tools the
        #    parent already had.
        extra_tools = [delegate_tool, forward_tool]
        if self._worker_registry is not None:
            # Lazy import to avoid loomflow.tools → loomflow.agent
            # circular at module-load. The tool's closure holds a
            # ref to the same registry dict that ``delegate``
            # writes to; mutations are visible immediately.
            from ..tools.send_message import make_send_message_tool
            send_msg_tool = make_send_message_tool(
                self._worker_registry,
                session=session,
                event_sink=send_chan,
            )
            extra_tools.append(send_msg_tool)
        wrapped_host = ExtendedToolHost(deps.tools, extra_tools)

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

        # If the supervisor model called forward_message at any
        # point, override the final output with the captured worker
        # text. The model's own final assistant message (typically
        # "[done]" or similar after the forward call) is discarded.
        if "output" in forward_request:
            session.output = forward_request["output"]
            yield Event.architecture_event(
                session.id,
                "supervisor.forwarded",
                worker=forward_request.get("worker", ""),
            )

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
    session: AgentSession,
    *,
    tool_name: str,
    event_sink: MemoryObjectSendStream[Event] | None = None,
    last_outputs: dict[str, str] | None = None,
    worker_registry: dict[str, Any] | None = None,
    role_to_worker_id: dict[str, str] | None = None,
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

    Both code paths roll the worker's token / cost usage into the
    parent ``session.cumulative_usage`` so the parent's
    ``RunResult.cost_usd`` reflects the worker's spend — without
    this every consumer of ``Team.supervisor`` silently under-counts.
    """

    async def _delegate(worker: str, instructions: str) -> str:
        agent = workers.get(worker)
        if agent is None:
            known = ", ".join(sorted(workers))
            return (
                f"Error: unknown worker {worker!r}. Known: {known}"
            )

        # Persistent-subagents path: the worker has a stable
        # session_id in the registry. Reusing it across delegate +
        # send_message calls is the load-bearing bit — loomflow's
        # Memory rehydrates prior episodes for the same
        # (user_id, session_id), so the worker REMEMBERS its
        # earlier delegations.
        #
        # Stateless legacy path (worker_registry is None): generate
        # a fresh session_id per call. Preserves the pre-0.10.10
        # behavior for ``Team.supervisor(persistent_subagents=False)``.
        worker_id_for_return: str | None = None
        if (
            worker_registry is not None
            and role_to_worker_id is not None
            and worker in role_to_worker_id
        ):
            handle = worker_registry[role_to_worker_id[worker]]
            worker_id_for_return = handle.worker_id
            # Pin user_id on first touch + reject cross-user.
            caller_user = get_run_context().user_id
            if (
                handle.user_id is not None
                and caller_user is not None
                and handle.user_id != caller_user
            ):
                return (
                    f"Error: worker {worker!r} "
                    f"({handle.worker_id}) belongs to user_id "
                    f"{handle.user_id!r} but the current run is "
                    f"user_id {caller_user!r}. Cross-tenant "
                    "delegation is rejected."
                )
            # Lock + touch happen INSIDE the handle so concurrent
            # delegate-to-same-worker calls serialise.
            async with handle.lock:
                handle.touch(user_id=caller_user)
                worker_session_id = handle.session_id
                if event_sink is None:
                    result = await agent.run(
                        instructions,
                        session_id=worker_session_id,
                        context=get_run_context(),
                    )
                    session.cumulative_usage = add_usage(
                        session.cumulative_usage,
                        Usage(
                            input_tokens=result.tokens_in,
                            cached_input_tokens=result.cached_tokens_in,
                            cache_write_tokens=result.cache_write_tokens,
                            output_tokens=result.tokens_out,
                            cost_usd=result.cost_usd,
                        ),
                    )
                    output = result.output
                else:
                    invocation = SubagentInvocation(
                        agent,
                        instructions,
                        session_id=worker_session_id,
                        rollup_into=session,
                    )
                    async with event_sink.clone() as sink:
                        async for ev in invocation.events():
                            await sink.send(ev)
                    output = str(invocation.result.get("output", ""))
            if last_outputs is not None:
                last_outputs[worker] = output
            # Prefix return with the worker_id so the model
            # learns the ID and can use it later via
            # ``send_message(to=<worker_id>, ...)``.
            return f"[worker_id: {worker_id_for_return}]\n{output}"

        # Legacy stateless path.
        suffix = new_id("del")
        worker_session_id = (
            f"{session.id}__delegate_{worker}_{suffix}"
        )

        if event_sink is None:
            # No event sink → fall back to plain run() (events lost).
            # Inherit the parent's RunContext (user_id + metadata) so
            # the worker runs in the same namespace partition as the
            # supervisor; ``session_id`` is the worker-specific one
            # we just derived, overriding the parent's session.
            result = await agent.run(
                instructions,
                session_id=worker_session_id,
                context=get_run_context(),
            )
            # Roll worker usage into the parent's session so cost
            # accounting is correct (mirrors what
            # SubagentInvocation does in the streaming branch).
            session.cumulative_usage = add_usage(
                session.cumulative_usage,
                Usage(
                    input_tokens=result.tokens_in,
                    cached_input_tokens=result.cached_tokens_in,
                    cache_write_tokens=result.cache_write_tokens,
                    output_tokens=result.tokens_out,
                    cost_usd=result.cost_usd,
                ),
            )
            output = result.output
            if last_outputs is not None:
                last_outputs[worker] = output
            return output

        # Stream worker events into the supervisor's shared channel
        # using a clone of the send half (clone keeps the channel
        # alive past this tool call's lifetime).
        invocation = SubagentInvocation(
            agent,
            instructions,
            session_id=worker_session_id,
            rollup_into=session,
        )
        async with event_sink.clone() as sink:
            async for ev in invocation.events():
                await sink.send(ev)
        output = str(invocation.result.get("output", ""))
        if last_outputs is not None:
            last_outputs[worker] = output
        return output

    worker_names = list(workers.keys())
    worker_descriptions = "\n".join(
        f"  - {name}: {(a.instructions or '').strip()[:120]}"
        for name, a in workers.items()
    )

    return Tool(
        name=tool_name,
        description=(
            "Delegate a subtask to a named specialist worker. "
            "The worker runs independently and returns its final "
            "answer as a string.\n\n"
            f"Available workers:\n{worker_descriptions}"
        ),
        fn=_delegate,
        input_schema={
            "type": "object",
            "properties": {
                "worker": {
                    "type": "string",
                    # Strict-schema providers (Anthropic, OpenAI
                    # strict mode) reject calls outside this list,
                    # so hallucinated worker names never reach our
                    # tool implementation.
                    "enum": worker_names,
                    "description": (
                        "Name of the worker to delegate to. "
                        "Must be one of: "
                        f"{', '.join(worker_names)}."
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


def _make_forward_message_tool(
    *,
    last_outputs: dict[str, str],
    forward_request: dict[str, str],
    tool_name: str,
    worker_names: list[str],
) -> Tool:
    """Build the ``forward_message`` tool.

    Reads from ``last_outputs`` (populated by the delegate tool)
    and writes the chosen worker's output into ``forward_request``.
    The supervisor's run loop checks ``forward_request`` after the
    base architecture finishes; if set, the agent's final output
    is overridden with the captured text — no synthesis round-trip.
    """

    async def _forward(worker: str) -> str:
        output = last_outputs.get(worker)
        if output is None:
            known = ", ".join(sorted(last_outputs)) or "(none yet)"
            return (
                f"Error: no captured output for worker {worker!r}. "
                f"You must call delegate({worker}, ...) first. "
                f"Workers with captured output: {known}"
            )
        forward_request["output"] = output
        forward_request["worker"] = worker
        return (
            f"[forward_message recorded — {worker}'s last output "
            "will be returned verbatim as the final response. End "
            "your turn now without writing any additional text.]"
        )

    return Tool(
        name=tool_name,
        description=(
            "Return a worker's last delegated output VERBATIM as "
            "the supervisor's final response. Use this when one "
            "worker already produced exactly what the user asked "
            "for and synthesis would just paraphrase it. End your "
            "turn immediately after calling — no additional text."
        ),
        fn=_forward,
        input_schema={
            "type": "object",
            "properties": {
                "worker": {
                    "type": "string",
                    # Constrains to the actual worker pool. The
                    # tool body still checks ``last_outputs`` so a
                    # forward before any delegate returns a clear
                    # error.
                    "enum": worker_names,
                    "description": (
                        "Name of the worker whose last output "
                        "should be forwarded. Must be one of: "
                        f"{', '.join(worker_names)}."
                    ),
                },
            },
            "required": ["worker"],
        },
    )

