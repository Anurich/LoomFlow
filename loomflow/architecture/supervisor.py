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

from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import anyio
from anyio.streams.memory import MemoryObjectSendStream

from ..core.context import get_run_context, inherit_ambient_memory
from ..core.ids import new_id
from ..core.protocols import Memory, Model
from ..core.types import Event, Usage
from ..tools.registry import Tool
from .base import AgentSession, Architecture, Dependencies
from .helpers import (
    SubagentInvocation,
    add_usage,
    consume_worker_usage,
    usage_from_result_dict,
)
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

SPAWN_SECTION_TEMPLATE = """\


You may also CREATE new specialist workers mid-run with
`spawn_worker(role, instructions)`:
- ``role`` must be a valid Python identifier not already in the
  roster; ``instructions`` become the new worker's system prompt.
- At most {max_spawned} workers can be spawned per run.
- A spawned worker is immediately delegable via `delegate(role, ...)`
  and exists only for the rest of THIS run.
Prefer the existing roster; spawn only when no current worker fits.
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
    * ``allow_spawn``: when ``True``, a ``spawn_worker(role,
      instructions)`` tool is injected alongside ``delegate`` so the
      coordinator model can create NEW specialist workers mid-run.
      Spawned workers are **ephemeral** — they live in a per-run
      overlay merged with the fixed roster for delegate lookup and
      die when the run ends (never persisted into the agent-level
      registry, so no cross-run / cross-user leakage). Default
      ``False`` (no behavior change).
    * ``max_spawned``: hard cap on spawns **per run**. Exceeding it
      returns an error string to the model (not an exception).
    * ``spawn_template``: template :class:`Agent` a spawned worker
      clones its model + tools from. When ``None``, spawned workers
      use the coordinator's own model and get no tools. v1 keeps the
      tool surface minimal on purpose: the model may NOT choose the
      worker's tools or model (future work — see G15 roadmap spec).
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
        allow_spawn: bool = False,
        max_spawned: int = 5,
        spawn_template: Agent | None = None,
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
        if max_spawned < 0:
            raise ValueError(
                f"max_spawned must be >= 0, got {max_spawned}"
            )
        self._allow_spawn = allow_spawn
        self._max_spawned = max_spawned
        self._spawn_template: Agent | None = spawn_template

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

        # Spawn support (``allow_spawn=True``): the delegate tool
        # closes over a PER-RUN copy of the roster plus a per-run
        # overlay of spawned worker handles. Spawned workers are
        # never written into ``self._workers`` or (durably) into the
        # agent-level worker registry — they die with the run.
        run_workers: dict[str, Agent] = (
            dict(self._workers) if self._allow_spawn else self._workers
        )
        spawn_overlay: dict[str, Any] | None = (
            {} if self._allow_spawn else None
        )
        spawned_ids: list[str] = []

        def _build_worker_tools() -> list[Tool]:
            """(Re)build delegate + forward_message from the CURRENT
            roster. Called once at run start and again after every
            spawn so the delegate enum/description reflect spawned
            workers (ReAct re-fetches ``list_tools()`` each turn, so
            re-registering on the ExtendedToolHost is enough)."""
            return [
                _make_delegate_tool(
                    run_workers,
                    session,
                    tool_name=self._delegate_name,
                    memory=deps.memory,
                    event_sink=send_chan,
                    last_outputs=last_outputs,
                    worker_registry=self._worker_registry,
                    role_to_worker_id=self._role_to_worker_id,
                    deps=deps,
                    spawn_overlay=spawn_overlay,
                ),
                _make_forward_message_tool(
                    last_outputs=last_outputs,
                    forward_request=forward_request,
                    tool_name=self._forward_name,
                    worker_names=list(run_workers.keys()),
                ),
            ]

        # 3. Wrap the parent's ToolHost so the model sees `delegate`
        #    + `forward_message` (+ `send_message` when persistent
        #    subagents are enabled, + `spawn_worker` when
        #    ``allow_spawn=True``) alongside whatever tools the
        #    parent already had.
        extra_tools = _build_worker_tools()
        if self._worker_registry is not None:
            # Lazy import to avoid loomflow.tools → loomflow.agent
            # circular at module-load. The tool's closure holds a
            # ref to the same registry dict that ``delegate``
            # writes to; mutations are visible immediately.
            from ..tools.send_message import make_send_message_tool
            send_msg_tool = make_send_message_tool(
                self._worker_registry,
                session=session,
                memory=deps.memory,
                event_sink=send_chan,
            )
            extra_tools.append(send_msg_tool)
        wrapped_host = ExtendedToolHost(deps.tools, extra_tools)

        if self._allow_spawn:
            assert spawn_overlay is not None  # narrowed for mypy

            def _rebuild_worker_tools() -> None:
                # ``register`` replaces by name (extras are keyed by
                # tool name), so the delegate/forward defs the model
                # sees next turn carry the updated roster enum.
                for t in _build_worker_tools():
                    wrapped_host.register(t)

            spawn_tool = _make_spawn_worker_tool(
                run_workers,
                spawn_overlay,
                session=session,
                template=self._spawn_template,
                fallback_model=deps.model,
                max_spawned=self._max_spawned,
                shared_registry=self._worker_registry,
                spawned_ids=spawned_ids,
                rebuild_worker_tools=_rebuild_worker_tools,
                event_sink=send_chan,
                delegate_name=self._delegate_name,
            )
            wrapped_host.register(spawn_tool)

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
        if self._allow_spawn:
            supervisor_section += SPAWN_SECTION_TEMPLATE.format(
                max_spawned=self._max_spawned
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
            # Ephemeral-spawn cleanup: spawned handles were mirrored
            # into the SHARED agent-level registry (the same dict the
            # ``send_message`` tool closes over — preserving the
            # coordinator._worker_registry identity invariant) so
            # ``send_message(to=<spawned id>)`` works mid-run. Pop
            # them here so nothing spawned survives the run (no
            # cross-run / cross-user leakage). IDs are ULIDs, so
            # concurrent runs only ever pop their own entries.
            if self._worker_registry is not None:
                for spawned_wid in spawned_ids:
                    self._worker_registry.pop(spawned_wid, None)

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
            # ``run_workers`` so spawned roles show up in telemetry.
            workers=list(run_workers.keys()),
        )


# ---------------------------------------------------------------------------
# delegate tool + tool-host wrapper
# ---------------------------------------------------------------------------


def _mark_if_interrupted(
    output: str, *, interrupted: bool, reason: str | None
) -> str:
    """Prefix a worker's output with an interruption marker.

    When a delegated worker's run was cut short (budget exhausted,
    ``max_turns`` hit, ...), its partial output would otherwise be
    indistinguishable from a completed answer — the coordinator
    model would happily synthesize on top of truncated work. The
    marker makes the truncation explicit so the model can retry,
    re-delegate, or caveat its answer.
    """
    if not interrupted:
        return output
    return f"[interrupted: {reason or 'unknown'}]\n{output}"


def _make_delegate_tool(
    workers: dict[str, Agent],
    session: AgentSession,
    *,
    tool_name: str,
    memory: Memory,
    event_sink: MemoryObjectSendStream[Event] | None = None,
    last_outputs: dict[str, str] | None = None,
    worker_registry: dict[str, Any] | None = None,
    role_to_worker_id: dict[str, str] | None = None,
    deps: Dependencies | None = None,
    spawn_overlay: dict[str, Any] | None = None,
) -> Tool:
    """Build a :class:`Tool` whose ``execute`` routes to the named
    worker :class:`Agent` and returns its final output.

    ``spawn_overlay`` (``allow_spawn=True`` only) maps a spawned
    role → its per-run :class:`_WorkerHandle`. Spawned handles win
    the lookup over the fixed roster and go through the exact same
    ``acquire_worker_session`` discipline (user pinning + lock), so
    tenant isolation holds for spawned workers too.

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
    When ``deps`` is supplied, the worker's spend is ALSO charged
    against the parent's budget via
    :func:`~loomflow.architecture.helpers.consume_worker_usage`
    (budget-only — cumulative_usage is never double-counted, and
    workers sharing the parent's budget instance are skipped).
    """

    async def _charge_parent_budget(
        worker_agent: Agent, usage: Usage
    ) -> None:
        if deps is not None:
            await consume_worker_usage(deps, worker_agent, usage)

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
        handle: Any = None
        if spawn_overlay is not None and worker in spawn_overlay:
            # Spawned-this-run worker: per-run overlay handle. Same
            # session/lock/pinning discipline as persistent workers,
            # but the handle dies with the run.
            handle = spawn_overlay[worker]
        elif (
            worker_registry is not None
            and role_to_worker_id is not None
            and worker in role_to_worker_id
        ):
            handle = worker_registry[role_to_worker_id[worker]]
        if handle is not None:
            worker_id_for_return = handle.worker_id
            caller_user = get_run_context().user_id
            # Cross-user check + lock + touch live in the shared
            # :func:`acquire_worker_session` helper (used by every
            # architecture) — the lock is held for the whole worker
            # run so concurrent delegate-to-same-worker calls
            # serialise. Lazy import: module scope would pull
            # loomflow.agent at architecture import time.
            from ..agent.worker_registry import (
                CrossUserWorkerError,
                acquire_worker_session,
            )
            try:
                async with acquire_worker_session(
                    handle, caller_user
                ):
                    worker_session_id = handle.session_id
                    # Memory propagation: install the coordinator's
                    # memory as ambient so a worker constructed
                    # without an explicit ``memory=`` inherits it.
                    # Matches the propagation Workflow.stream does.
                    # Anyio task-group spawns inherit the
                    # contextvar, so SubagentInvocation (which fires
                    # agent.run inside a task group) also sees the
                    # ambient.
                    with inherit_ambient_memory(memory):
                        if event_sink is None:
                            result = await agent.run(
                                instructions,
                                session_id=worker_session_id,
                                context=get_run_context(),
                            )
                            worker_usage = Usage(
                                input_tokens=result.tokens_in,
                                cached_input_tokens=result.cached_tokens_in,
                                cache_write_tokens=result.cache_write_tokens,
                                output_tokens=result.tokens_out,
                                cost_usd=result.cost_usd,
                            )
                            session.cumulative_usage = add_usage(
                                session.cumulative_usage,
                                worker_usage,
                            )
                            await _charge_parent_budget(
                                agent, worker_usage
                            )
                            output = _mark_if_interrupted(
                                result.output,
                                interrupted=result.interrupted,
                                reason=result.interruption_reason,
                            )
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
                            await _charge_parent_budget(
                                agent,
                                usage_from_result_dict(
                                    invocation.result
                                ),
                            )
                            output = _mark_if_interrupted(
                                str(invocation.result.get("output", "")),
                                interrupted=bool(
                                    invocation.result.get("interrupted", False)
                                ),
                                reason=invocation.result.get(
                                    "interruption_reason"
                                ),
                            )
            except CrossUserWorkerError as exc:
                return f"Error: {exc}"
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
            # Memory propagation: same rationale as the persistent
            # path above — worker without explicit memory= inherits
            # the coordinator's.
            with inherit_ambient_memory(memory):
                result = await agent.run(
                    instructions,
                    session_id=worker_session_id,
                    context=get_run_context(),
                )
            # Roll worker usage into the parent's session so cost
            # accounting is correct (mirrors what
            # SubagentInvocation does in the streaming branch),
            # and charge it against the parent budget.
            worker_usage = Usage(
                input_tokens=result.tokens_in,
                cached_input_tokens=result.cached_tokens_in,
                cache_write_tokens=result.cache_write_tokens,
                output_tokens=result.tokens_out,
                cost_usd=result.cost_usd,
            )
            session.cumulative_usage = add_usage(
                session.cumulative_usage, worker_usage
            )
            await _charge_parent_budget(agent, worker_usage)
            output = _mark_if_interrupted(
                result.output,
                interrupted=result.interrupted,
                reason=result.interruption_reason,
            )
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
        with inherit_ambient_memory(memory):
            async with event_sink.clone() as sink:
                async for ev in invocation.events():
                    await sink.send(ev)
        await _charge_parent_budget(
            agent, usage_from_result_dict(invocation.result)
        )
        output = _mark_if_interrupted(
            str(invocation.result.get("output", "")),
            interrupted=bool(
                invocation.result.get("interrupted", False)
            ),
            reason=invocation.result.get("interruption_reason"),
        )
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



# ---------------------------------------------------------------------------
# spawn_worker tool (G15 — dynamic model-driven agent spawning)
# ---------------------------------------------------------------------------


def _make_spawn_worker_tool(
    run_workers: dict[str, Agent],
    spawn_overlay: dict[str, Any],
    *,
    session: AgentSession,
    template: Agent | None,
    fallback_model: Model,
    max_spawned: int,
    shared_registry: dict[str, Any] | None,
    spawned_ids: list[str],
    rebuild_worker_tools: Callable[[], None],
    event_sink: MemoryObjectSendStream[Event] | None = None,
    tool_name: str = "spawn_worker",
    delegate_name: str = "delegate",
) -> Tool:
    """Build the ``spawn_worker(role, instructions)`` tool.

    Design notes (v1 — deliberately minimal surface):

    * ``role`` must be a Python identifier (reuses the same
      ``_VALID_ROLE`` rule as :func:`new_worker_id` /
      :meth:`Supervisor.add_worker`) and must not collide with an
      existing roster entry.
    * ``instructions`` become the spawned worker's system prompt.
      The model may NOT pick the worker's tools or model — those are
      inherited from ``template`` (or the coordinator's model with
      no tools when no template is set). Letting the model choose
      tools/model is future work (see G15 in the roadmap).
    * Every spawn appends the new role to ``run_workers`` (the
      per-run roster copy the delegate tool closes over) and a
      :class:`_WorkerHandle` to ``spawn_overlay`` — then calls
      ``rebuild_worker_tools()`` so the delegate/forward enums the
      model sees on its NEXT turn include the new role. All state is
      per-run: nothing spawned survives ``Supervisor.run``.
    * The handle is also mirrored into ``shared_registry`` (the
      agent-level dict the ``send_message`` tool closes over) so
      ``send_message(to=<spawned worker_id>)`` works mid-run;
      ``Supervisor.run``'s ``finally`` pops those entries, keeping
      spawned workers ephemeral.
    * ``user_id`` is pinned at spawn time to the spawning run's user
      (cross-tenant delegation to a spawned worker is rejected by
      the shared ``acquire_worker_session`` discipline, exactly like
      persistent workers).
    * ``max_spawned`` is enforced per run; exceeding it returns an
      error string to the model, never an exception.
    """

    async def _spawn(role: str, instructions: str) -> str:
        # Lazy import — module scope would pull loomflow.agent at
        # architecture import time (same pattern as ``_delegate``).
        from datetime import UTC, datetime

        from ..agent.api import Agent as _Agent
        from ..agent.worker_registry import (
            _VALID_ROLE,
            _WorkerHandle,
            new_worker_id,
        )

        def _roster() -> str:
            return ", ".join(sorted(run_workers)) or "(none)"

        if len(spawned_ids) >= max_spawned:
            return (
                f"Error: spawn limit reached — {max_spawned} "
                f"worker(s) may be spawned per run and you have "
                f"already spawned {len(spawned_ids)}. Delegate to "
                f"an existing worker instead. Roster: {_roster()}"
            )
        if not _VALID_ROLE.match(role):
            return (
                f"Error: invalid role {role!r} — the role must be "
                "a Python identifier (letters, digits, underscores; "
                "not starting with a digit)."
            )
        if role in run_workers:
            return (
                f"Error: a worker named {role!r} already exists. "
                f"Pick a new role name or delegate to it directly. "
                f"Roster: {_roster()}"
            )
        if not instructions.strip():
            return (
                "Error: instructions must be a non-empty system "
                "prompt for the new worker."
            )

        # v1 inheritance: model + tools come from the template; with
        # no template the worker shares the coordinator's model and
        # gets no tools. Memory is NOT passed — the delegate path
        # installs the coordinator's memory as ambient
        # (``inherit_ambient_memory``), so a worker built without an
        # explicit ``memory=`` inherits it, same as fixed workers.
        if template is not None:
            worker_agent = _Agent(
                instructions,
                model=template.model,
                tools=template.tool_host,
            )
        else:
            worker_agent = _Agent(instructions, model=fallback_model)

        worker_id = new_worker_id(role)
        handle = _WorkerHandle(
            worker_id=worker_id,
            role=role,
            agent=worker_agent,
            # Run-scoped session id — the handle dies with the run,
            # so unlike ``persistent_*`` sessions this one is never
            # revisited by a later run.
            session_id=f"{session.id}__spawned_{worker_id}",
            # Pin to the spawning run's user immediately: a spawned
            # worker belongs to the caller from birth (first-touch
            # pinning would leave a None window). ``None`` for
            # anonymous runs keeps first-touch semantics.
            user_id=get_run_context().user_id,
            created_at=datetime.now(UTC),
        )

        run_workers[role] = worker_agent
        spawn_overlay[role] = handle
        if shared_registry is not None:
            shared_registry[worker_id] = handle
        spawned_ids.append(worker_id)
        # Refresh delegate/forward defs so the next model turn's
        # tool list carries the new role in enum + description.
        rebuild_worker_tools()

        if event_sink is not None:
            async with event_sink.clone() as sink:
                await sink.send(
                    Event.architecture_event(
                        session.id,
                        "supervisor.worker_spawned",
                        role=role,
                        worker_id=worker_id,
                        spawned_count=len(spawned_ids),
                    )
                )

        return (
            f"Spawned worker {role!r} [worker_id: {worker_id}] "
            f"({len(spawned_ids)}/{max_spawned} spawned this run). "
            f"It is now available via "
            f"{delegate_name}(worker={role!r}, instructions=...). "
            f"Current roster: {_roster()}"
        )

    return Tool(
        name=tool_name,
        description=(
            "Create a NEW specialist worker mid-run and add it to "
            "the delegation roster. `role` must be a valid Python "
            "identifier not already in the roster; `instructions` "
            "become the new worker's system prompt. The worker "
            "inherits its model and tools from the team's spawn "
            "template (you cannot choose them) and exists only for "
            f"the rest of this run. At most {max_spawned} worker(s) "
            "can be spawned per run. After spawning, use "
            f"`{delegate_name}(worker=<role>, ...)` to give it work."
        ),
        fn=_spawn,
        input_schema={
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": (
                        "Short identifier for the new worker (e.g. "
                        "'fact_checker'). Must be a valid Python "
                        "identifier and not already in the roster."
                    ),
                },
                "instructions": {
                    "type": "string",
                    "description": (
                        "System prompt for the new worker — define "
                        "its specialty, style, and constraints. Be "
                        "specific; this is ALL the worker knows "
                        "about its job."
                    ),
                },
            },
            "required": ["role", "instructions"],
        },
    )
