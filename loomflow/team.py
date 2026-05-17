"""Team — sibling-style builders for multi-agent architectures.

Coming from LangGraph, CrewAI, AutoGen, or the OpenAI Agents SDK,
you'd expect to construct a multi-agent team like this::

    team = create_supervisor([researcher, writer], model="gpt-4")
    result = await team.invoke(...)

In Loom the same shape is :class:`Team`::

    team = Team.supervisor(
        workers={"researcher": researcher, "writer": writer},
        instructions="manage the pipeline",
        model="gpt-4.1-mini",
    )
    result = await team.run("write me a blog post")

Under the hood :class:`Team` returns a regular :class:`Agent` —
exactly what ``Agent(architecture=Supervisor(...))`` would produce.
The two shapes are interchangeable; :class:`Team` is the
"familiar-from-other-frameworks" facade for the common case, while
the nested ``Agent(architecture=...)`` form remains the path for
**recursive composition** (wrapping a Supervisor in Reflexion, etc.).

Provided builders
-----------------

* :meth:`Team.supervisor` — coordinator + workers
* :meth:`Team.swarm` — peer agents handing off control
* :meth:`Team.router` — classify-and-dispatch
* :meth:`Team.debate` — N debaters + optional judge
* :meth:`Team.actor_critic` — actor + critic pair
* :meth:`Team.blackboard` — shared workspace + coordinator

Each method exposes **every** :class:`Agent` constructor kwarg
explicitly (rather than forwarding through ``**kwargs``) so that
IDEs and type-checkers surface the full parameter list with proper
types, defaults, and docstrings on hover.

Standalone-run helper
---------------------

:func:`run_architecture` builds a minimal :class:`Agent` shell
around any :class:`Architecture` instance and runs it. Useful for
**testing orchestrators in isolation** without wiring a full Agent
yourself::

    sup = Supervisor(workers={"a": agent_a})
    result = await run_architecture(sup, "do the thing", model="gpt-4.1-mini")
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from .agent.api import DEFAULT_MAX_TURNS, Agent
from .architecture import (
    ActorCritic,
    Architecture,
    BlackboardArchitecture,
    MultiAgentDebate,
    Router,
    RouterRoute,
    Supervisor,
    Swarm,
)
from .architecture.swarm import Handoff
from .core.types import RunResult

if TYPE_CHECKING:
    from .core.protocols import (
        Budget,
        Memory,
        Model,
        Permissions,
        Runtime,
        Telemetry,
        ToolHost,
    )
    from .security.audit import AuditLog
    from .security.hooks import HookRegistry
    from .tools.registry import Tool


# Type alias for the same ``tools=`` argument :class:`Agent` accepts —
# spelled out once so each builder can reference it cleanly.
_ToolsArg = (
    "list[Tool | Callable[..., object]] | ToolHost | Tool | "
    "Callable[..., object] | None"
)


def _attach_workspace_to_workers(
    workers: dict[str, Agent],
) -> None:
    """Stamp each worker with its team-name + teammates so when the
    parent coordinator's ambient workspace is picked up at run-time,
    the worker writes notes attributed to its team role (not the
    generic ``"agent"`` default).

    Mutates each worker's ``_workspace_name`` and
    ``_workspace_teammates`` post-construction. This is intentional:
    "team membership" is something the Team builder assigns, not
    something the agent decided about itself. Workers keep their
    own ``_workspace = None`` so the workflow-style ambient
    inheritance kicks in — they don't pin their own.
    """
    names = list(workers.keys())
    for name, worker in workers.items():
        worker._workspace_name = name  # noqa: SLF001 — intentional
        worker._workspace_teammates = names  # noqa: SLF001 — intentional


class Team:
    """Namespace for multi-agent team builders.

    Every classmethod returns a fully-built :class:`Agent` whose
    architecture is the corresponding multi-agent strategy. The
    returned Agent has the standard ``run`` / ``stream`` / etc.
    interface — call sites don't change between single-agent and
    team agents.
    """

    # -----------------------------------------------------------------
    # Supervisor
    # -----------------------------------------------------------------

    @staticmethod
    def supervisor(
        workers: dict[str, Agent],
        *,
        instructions: str = "",
        # --- forwarded Agent kwargs (explicit so IDEs autocomplete) ---
        model: Model | str | None = None,
        memory: Memory | None = None,
        runtime: Runtime | None = None,
        budget: Budget | None = None,
        permissions: Permissions | None = None,
        hooks: HookRegistry | None = None,
        tools: (
            list[Tool | Callable[..., object]]
            | ToolHost
            | Tool
            | Callable[..., object]
            | None
        ) = None,
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        auto_consolidate: bool = False,
        skills: list[Any] | None = None,
        workspace: Any | str | None = None,
        living_plan: Any = None,
        stop_hooks: list[Any] | None = None,
        max_stop_hook_iterations: int = 15,
        prompt_caching: bool | Mapping[str, Any] | None = None,
        tool_result_summarizer: Model | str | None = None,
        tool_result_summary_threshold: int = 500,
        persist_tool_transcripts: bool = False,
        tool_transcript_max_bytes: int = 50_000,
        snip_window: int = 0,
        auto_compact_at_tokens: int | None = None,
        auto_compact_summariser: Model | str | None = None,
        auto_compact_keep_recent_turns: int = 4,
        retry_policy: Any | None = None,
        auto_extract: bool | None = None,
        approval_handler: Any | None = None,
        secrets: Any | None = None,
        response_tone: str | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        # --- supervisor-specific options ---
        instructions_template: str | None = None,
        delegate_tool_name: str = "delegate",
        forward_tool_name: str = "forward_message",
        persistent_subagents: bool = True,
    ) -> Agent:
        """Build a coordinator Agent that delegates to ``workers``.

        The coordinator can call ``delegate(worker, instructions)``
        to dispatch a subtask, or ``forward_message(worker)`` to
        return a worker's output verbatim. Multiple delegations in
        one turn run in parallel.

        ``workspace`` wires a shared notebook onto the coordinator
        and every worker. The worker's dict key becomes its author
        identity in the notebook (so notes show up as
        ``[researcher]`` rather than the generic ``[agent]``).
        Workers pick up the workspace via ambient inheritance from
        the coordinator's run, so you don't have to rebuild them.

        ``prompt_caching=True`` (or the dict form ``{"enabled": True,
        "ttl": "1h"}``) enables provider-native caching on the
        coordinator. Workers cache independently — set
        ``prompt_caching=`` on each worker Agent at construction.
        """
        coord_ws: Any = workspace
        if workspace is not None:
            _attach_workspace_to_workers(workers)
            # Resolve the spec, then re-wrap with the coordinator's
            # identity. ``ws.member(...)`` collapses the three old
            # kwargs into one.
            from .workspace.resolver import resolve_workspace
            resolved = resolve_workspace(workspace)
            if resolved is not None and hasattr(resolved, "member"):
                coord_ws = resolved.member(
                    "coordinator", teammates=list(workers.keys())
                )
        # Build the registry FIRST so the Supervisor architecture
        # can take a ref. When ``persistent_subagents=False``, both
        # dicts stay empty and Supervisor falls back to legacy
        # stateless-per-delegate behavior (no send_message tool,
        # no per-handle locks, fresh ULID session_id per delegate).
        worker_registry: dict[str, Any] = {}
        role_to_worker_id: dict[str, str] = {}
        if persistent_subagents:
            from .agent.worker_registry import build_worker_registry
            worker_registry, role_to_worker_id = build_worker_registry(
                workers
            )

        coordinator = Agent(
            instructions=instructions,
            model=model,
            memory=memory,
            runtime=runtime,
            budget=budget,
            permissions=permissions,
            hooks=hooks,
            tools=tools,
            telemetry=telemetry,
            audit_log=audit_log,
            max_turns=max_turns,
            auto_consolidate=auto_consolidate,
            skills=skills,
            workspace=coord_ws,
            living_plan=living_plan,
            stop_hooks=stop_hooks,
            max_stop_hook_iterations=max_stop_hook_iterations,
            prompt_caching=prompt_caching,
            tool_result_summarizer=tool_result_summarizer,
            tool_result_summary_threshold=(
                tool_result_summary_threshold
            ),
            persist_tool_transcripts=persist_tool_transcripts,
            tool_transcript_max_bytes=tool_transcript_max_bytes,
            snip_window=snip_window,
            auto_compact_at_tokens=auto_compact_at_tokens,
            auto_compact_summariser=auto_compact_summariser,
            auto_compact_keep_recent_turns=(
                auto_compact_keep_recent_turns
            ),
            retry_policy=retry_policy,
            auto_extract=auto_extract,
            approval_handler=approval_handler,
            secrets=secrets,
            response_tone=response_tone,
            effort=effort,
            strict_effort=strict_effort,
            architecture=Supervisor(
                workers=workers,
                instructions_template=instructions_template,
                delegate_tool_name=delegate_tool_name,
                forward_tool_name=forward_tool_name,
                worker_registry=(
                    worker_registry if persistent_subagents else None
                ),
                role_to_worker_id=(
                    role_to_worker_id if persistent_subagents else None
                ),
            ),
        )
        # Stamp the registry onto the coordinator Agent so the
        # send_message tool's closure (built inside Supervisor.run)
        # and external introspection (tests, observability) point
        # at the SAME dict.
        if persistent_subagents:
            coordinator._worker_registry.update(worker_registry)
        return coordinator

    # -----------------------------------------------------------------
    # Swarm
    # -----------------------------------------------------------------

    @staticmethod
    def swarm(
        agents: dict[str, Agent | Handoff],
        entry_agent: str,
        *,
        instructions: str = "",
        model: Model | str | None = None,
        memory: Memory | None = None,
        runtime: Runtime | None = None,
        budget: Budget | None = None,
        permissions: Permissions | None = None,
        hooks: HookRegistry | None = None,
        tools: (
            list[Tool | Callable[..., object]]
            | ToolHost
            | Tool
            | Callable[..., object]
            | None
        ) = None,
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        auto_consolidate: bool = False,
        skills: list[Any] | None = None,
        workspace: Any | str | None = None,
        living_plan: Any = None,
        stop_hooks: list[Any] | None = None,
        max_stop_hook_iterations: int = 15,
        prompt_caching: bool | Mapping[str, Any] | None = None,
        tool_result_summarizer: Model | str | None = None,
        tool_result_summary_threshold: int = 500,
        persist_tool_transcripts: bool = False,
        tool_transcript_max_bytes: int = 50_000,
        snip_window: int = 0,
        auto_compact_at_tokens: int | None = None,
        auto_compact_summariser: Model | str | None = None,
        auto_compact_keep_recent_turns: int = 4,
        retry_policy: Any | None = None,
        auto_extract: bool | None = None,
        approval_handler: Any | None = None,
        secrets: Any | None = None,
        response_tone: str | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        # --- swarm-specific options ---
        max_handoffs: int = 8,
        detect_cycles: bool = True,
        pass_full_history: bool = True,
        handoff_tool_name: str = "handoff",
        persistent_subagents: bool = True,
    ) -> Agent:
        """Build a peer-swarm of agents that hand off control via a
        ``handoff`` tool (or per-target ``transfer_to_<name>`` tools
        when peers are wrapped in :class:`Handoff` with an
        ``input_type``).

        ``entry_agent`` is the peer that receives the first message.

        ``workspace`` wires a shared notebook across every peer.
        Each peer's dict key becomes its author identity in the
        notebook so handoffs leave a clear trail of who wrote what.

        ``persistent_subagents=True`` (default) registers each peer
        with a stable ``worker_<role>_<ULID>`` ID and reuses one
        session per peer across every handoff — peer agents
        accumulate memory across handoffs and across multiple
        ``Agent.run()`` invocations. Set to ``False`` to restore
        legacy per-handoff stateless behaviour.

        ``prompt_caching=`` enables provider-native caching on the
        coordinator (see :meth:`Team.supervisor` for accepted shapes).
        Peers cache independently via their own ``Agent`` ctor.
        """
        entry_ws: Any = workspace
        # Unwrap Handoff configs to plain Agent map up front — needed
        # for both workspace wiring and the worker registry.
        raw_agents: dict[str, Agent] = {
            k: (v.agent if isinstance(v, Handoff) else v)
            for k, v in agents.items()
        }
        if workspace is not None:
            # Unwrap Handoff configs so we can mutate the underlying
            # Agent's workspace identity.
            _attach_workspace_to_workers(raw_agents)
            from .workspace.resolver import resolve_workspace
            resolved = resolve_workspace(workspace)
            if resolved is not None and hasattr(resolved, "member"):
                entry_ws = resolved.member(
                    entry_agent, teammates=list(agents.keys())
                )
        worker_registry = None
        role_to_worker_id = None
        if persistent_subagents:
            from .agent.worker_registry import build_worker_registry
            worker_registry, role_to_worker_id = build_worker_registry(
                raw_agents
            )
        coordinator = Agent(
            instructions=instructions,
            model=model,
            memory=memory,
            runtime=runtime,
            budget=budget,
            permissions=permissions,
            hooks=hooks,
            tools=tools,
            telemetry=telemetry,
            audit_log=audit_log,
            max_turns=max_turns,
            auto_consolidate=auto_consolidate,
            skills=skills,
            workspace=entry_ws,
            living_plan=living_plan,
            stop_hooks=stop_hooks,
            max_stop_hook_iterations=max_stop_hook_iterations,
            prompt_caching=prompt_caching,
            tool_result_summarizer=tool_result_summarizer,
            tool_result_summary_threshold=(
                tool_result_summary_threshold
            ),
            persist_tool_transcripts=persist_tool_transcripts,
            tool_transcript_max_bytes=tool_transcript_max_bytes,
            snip_window=snip_window,
            auto_compact_at_tokens=auto_compact_at_tokens,
            auto_compact_summariser=auto_compact_summariser,
            auto_compact_keep_recent_turns=(
                auto_compact_keep_recent_turns
            ),
            retry_policy=retry_policy,
            auto_extract=auto_extract,
            approval_handler=approval_handler,
            secrets=secrets,
            response_tone=response_tone,
            effort=effort,
            strict_effort=strict_effort,
            architecture=Swarm(
                agents=agents,
                entry_agent=entry_agent,
                max_handoffs=max_handoffs,
                detect_cycles=detect_cycles,
                pass_full_history=pass_full_history,
                handoff_tool_name=handoff_tool_name,
                worker_registry=worker_registry,
                role_to_worker_id=role_to_worker_id,
            ),
        )
        if persistent_subagents and worker_registry is not None:
            coordinator._worker_registry = worker_registry
        return coordinator

    # -----------------------------------------------------------------
    # Router
    # -----------------------------------------------------------------

    @staticmethod
    def router(
        routes: list[RouterRoute],
        *,
        instructions: str = "",
        model: Model | str | None = None,
        memory: Memory | None = None,
        runtime: Runtime | None = None,
        budget: Budget | None = None,
        permissions: Permissions | None = None,
        hooks: HookRegistry | None = None,
        tools: (
            list[Tool | Callable[..., object]]
            | ToolHost
            | Tool
            | Callable[..., object]
            | None
        ) = None,
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        auto_consolidate: bool = False,
        skills: list[Any] | None = None,
        workspace: Any | str | None = None,
        living_plan: Any = None,
        stop_hooks: list[Any] | None = None,
        max_stop_hook_iterations: int = 15,
        prompt_caching: bool | Mapping[str, Any] | None = None,
        tool_result_summarizer: Model | str | None = None,
        tool_result_summary_threshold: int = 500,
        persist_tool_transcripts: bool = False,
        tool_transcript_max_bytes: int = 50_000,
        snip_window: int = 0,
        auto_compact_at_tokens: int | None = None,
        auto_compact_summariser: Model | str | None = None,
        auto_compact_keep_recent_turns: int = 4,
        retry_policy: Any | None = None,
        auto_extract: bool | None = None,
        approval_handler: Any | None = None,
        secrets: Any | None = None,
        response_tone: str | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        # --- router-specific options ---
        fallback_route: str | None = None,
        require_confidence_above: float = 0.0,
        classifier_prompt: str | None = None,
        persistent_subagents: bool = True,
    ) -> Agent:
        """Build a router that classifies once and dispatches to
        ONE specialist :class:`Agent`. Cheaper than Supervisor for
        tasks with clear specialist boundaries (one classifier call
        + one specialist run, no synthesis pass).

        ``persistent_subagents=True`` (default) gives each route's
        specialist a stable persistent session, so successive routes
        to the same specialist reuse memory (the typical case in a
        long REPL: route to ``billing`` once, route to ``billing``
        again — the second call sees the first conversation).

        ``prompt_caching=`` enables provider-native caching on the
        classifier-coordinator (see :meth:`Team.supervisor` for
        accepted shapes). Specialists cache via their own ``Agent``.
        """
        worker_registry = None
        role_to_worker_id = None
        if persistent_subagents:
            from .agent.worker_registry import build_worker_registry
            route_agents = {r.name: r.agent for r in routes}
            worker_registry, role_to_worker_id = build_worker_registry(
                route_agents
            )
        coordinator = Agent(
            instructions=instructions,
            model=model,
            memory=memory,
            runtime=runtime,
            budget=budget,
            permissions=permissions,
            hooks=hooks,
            tools=tools,
            telemetry=telemetry,
            audit_log=audit_log,
            max_turns=max_turns,
            auto_consolidate=auto_consolidate,
            skills=skills,
            workspace=workspace,
            living_plan=living_plan,
            stop_hooks=stop_hooks,
            max_stop_hook_iterations=max_stop_hook_iterations,
            prompt_caching=prompt_caching,
            tool_result_summarizer=tool_result_summarizer,
            tool_result_summary_threshold=(
                tool_result_summary_threshold
            ),
            persist_tool_transcripts=persist_tool_transcripts,
            tool_transcript_max_bytes=tool_transcript_max_bytes,
            snip_window=snip_window,
            auto_compact_at_tokens=auto_compact_at_tokens,
            auto_compact_summariser=auto_compact_summariser,
            auto_compact_keep_recent_turns=(
                auto_compact_keep_recent_turns
            ),
            retry_policy=retry_policy,
            auto_extract=auto_extract,
            approval_handler=approval_handler,
            secrets=secrets,
            response_tone=response_tone,
            effort=effort,
            strict_effort=strict_effort,
            architecture=Router(
                routes=routes,
                fallback_route=fallback_route,
                require_confidence_above=require_confidence_above,
                classifier_prompt=classifier_prompt,
                worker_registry=worker_registry,
                role_to_worker_id=role_to_worker_id,
            ),
        )
        if persistent_subagents and worker_registry is not None:
            coordinator._worker_registry = worker_registry
        return coordinator

    # -----------------------------------------------------------------
    # Debate
    # -----------------------------------------------------------------

    @staticmethod
    def debate(
        debaters: list[Agent],
        *,
        judge: Agent | None = None,
        instructions: str = "",
        model: Model | str | None = None,
        memory: Memory | None = None,
        runtime: Runtime | None = None,
        budget: Budget | None = None,
        permissions: Permissions | None = None,
        hooks: HookRegistry | None = None,
        tools: (
            list[Tool | Callable[..., object]]
            | ToolHost
            | Tool
            | Callable[..., object]
            | None
        ) = None,
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        auto_consolidate: bool = False,
        skills: list[Any] | None = None,
        workspace: Any | str | None = None,
        living_plan: Any = None,
        stop_hooks: list[Any] | None = None,
        max_stop_hook_iterations: int = 15,
        prompt_caching: bool | Mapping[str, Any] | None = None,
        tool_result_summarizer: Model | str | None = None,
        tool_result_summary_threshold: int = 500,
        persist_tool_transcripts: bool = False,
        tool_transcript_max_bytes: int = 50_000,
        snip_window: int = 0,
        auto_compact_at_tokens: int | None = None,
        auto_compact_summariser: Model | str | None = None,
        auto_compact_keep_recent_turns: int = 4,
        retry_policy: Any | None = None,
        auto_extract: bool | None = None,
        approval_handler: Any | None = None,
        secrets: Any | None = None,
        response_tone: str | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        # --- debate-specific options ---
        rounds: int = 2,
        convergence_check: bool = True,
        convergence_similarity: float = 0.85,
        debater_instructions: str | None = None,
        judge_instructions: str | None = None,
        persistent_subagents: bool = True,
    ) -> Agent:
        """Build a multi-agent debate where ``debaters`` argue for
        ``rounds`` (with optional convergence early-exit). If
        ``judge`` is provided, the judge synthesizes a final
        answer; otherwise majority vote wins.

        ``persistent_subagents=True`` (default) registers each
        ``debater_<i>`` (and ``judge`` when provided) with a stable
        session so they remember prior debates across multiple
        ``Agent.run()`` invocations on the same coordinator. Set to
        ``False`` to restore the legacy per-round stateless behaviour.

        ``prompt_caching=`` enables provider-native caching on the
        coordinator (see :meth:`Team.supervisor` for accepted shapes).
        Debaters cache via their own ``Agent`` ctor.
        """
        worker_registry = None
        role_to_worker_id = None
        if persistent_subagents:
            from .agent.worker_registry import build_worker_registry
            debate_agents: dict[str, Agent] = {
                f"debater_{i}": d for i, d in enumerate(debaters)
            }
            if judge is not None:
                debate_agents["judge"] = judge
            worker_registry, role_to_worker_id = build_worker_registry(
                debate_agents
            )
        coordinator = Agent(
            instructions=instructions,
            model=model,
            memory=memory,
            runtime=runtime,
            budget=budget,
            permissions=permissions,
            hooks=hooks,
            tools=tools,
            telemetry=telemetry,
            audit_log=audit_log,
            max_turns=max_turns,
            auto_consolidate=auto_consolidate,
            skills=skills,
            workspace=workspace,
            living_plan=living_plan,
            stop_hooks=stop_hooks,
            max_stop_hook_iterations=max_stop_hook_iterations,
            prompt_caching=prompt_caching,
            tool_result_summarizer=tool_result_summarizer,
            tool_result_summary_threshold=(
                tool_result_summary_threshold
            ),
            persist_tool_transcripts=persist_tool_transcripts,
            tool_transcript_max_bytes=tool_transcript_max_bytes,
            snip_window=snip_window,
            auto_compact_at_tokens=auto_compact_at_tokens,
            auto_compact_summariser=auto_compact_summariser,
            auto_compact_keep_recent_turns=(
                auto_compact_keep_recent_turns
            ),
            retry_policy=retry_policy,
            auto_extract=auto_extract,
            approval_handler=approval_handler,
            secrets=secrets,
            response_tone=response_tone,
            effort=effort,
            strict_effort=strict_effort,
            architecture=MultiAgentDebate(
                debaters=debaters,
                judge=judge,
                rounds=rounds,
                convergence_check=convergence_check,
                convergence_similarity=convergence_similarity,
                debater_instructions=debater_instructions,
                judge_instructions=judge_instructions,
                worker_registry=worker_registry,
                role_to_worker_id=role_to_worker_id,
            ),
        )
        if persistent_subagents and worker_registry is not None:
            coordinator._worker_registry = worker_registry
        return coordinator

    # -----------------------------------------------------------------
    # Actor-Critic
    # -----------------------------------------------------------------

    @staticmethod
    def actor_critic(
        actor: Agent,
        critic: Agent,
        *,
        instructions: str = "",
        model: Model | str | None = None,
        memory: Memory | None = None,
        runtime: Runtime | None = None,
        budget: Budget | None = None,
        permissions: Permissions | None = None,
        hooks: HookRegistry | None = None,
        tools: (
            list[Tool | Callable[..., object]]
            | ToolHost
            | Tool
            | Callable[..., object]
            | None
        ) = None,
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        auto_consolidate: bool = False,
        skills: list[Any] | None = None,
        workspace: Any | str | None = None,
        living_plan: Any = None,
        stop_hooks: list[Any] | None = None,
        max_stop_hook_iterations: int = 15,
        prompt_caching: bool | Mapping[str, Any] | None = None,
        tool_result_summarizer: Model | str | None = None,
        tool_result_summary_threshold: int = 500,
        persist_tool_transcripts: bool = False,
        tool_transcript_max_bytes: int = 50_000,
        snip_window: int = 0,
        auto_compact_at_tokens: int | None = None,
        auto_compact_summariser: Model | str | None = None,
        auto_compact_keep_recent_turns: int = 4,
        retry_policy: Any | None = None,
        auto_extract: bool | None = None,
        approval_handler: Any | None = None,
        secrets: Any | None = None,
        response_tone: str | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        # --- actor-critic-specific options ---
        max_rounds: int = 3,
        approval_threshold: float = 0.9,
        critique_template: str | None = None,
        refine_template: str | None = None,
        persistent_subagents: bool = True,
    ) -> Agent:
        """Build an actor-critic pair where the critic reviews the
        actor's output (with structured JSON scoring + rubric) and
        the actor refines below ``approval_threshold``.

        ``persistent_subagents=True`` (default) registers the
        ``actor`` and ``critic`` agents with stable sessions so
        their memory carries across rounds AND across multiple
        ``Agent.run()`` invocations — the critic remembers what it
        already flagged; the actor remembers what it already refined.

        ``prompt_caching=`` enables provider-native caching on the
        coordinator (see :meth:`Team.supervisor` for accepted shapes).
        Actor/critic cache via their own ``Agent`` ctor.
        """
        worker_registry = None
        role_to_worker_id = None
        if persistent_subagents:
            from .agent.worker_registry import build_worker_registry
            worker_registry, role_to_worker_id = build_worker_registry(
                {"actor": actor, "critic": critic}
            )
        coordinator = Agent(
            instructions=instructions,
            model=model,
            memory=memory,
            runtime=runtime,
            budget=budget,
            permissions=permissions,
            hooks=hooks,
            tools=tools,
            telemetry=telemetry,
            audit_log=audit_log,
            max_turns=max_turns,
            auto_consolidate=auto_consolidate,
            skills=skills,
            workspace=workspace,
            living_plan=living_plan,
            stop_hooks=stop_hooks,
            max_stop_hook_iterations=max_stop_hook_iterations,
            prompt_caching=prompt_caching,
            tool_result_summarizer=tool_result_summarizer,
            tool_result_summary_threshold=(
                tool_result_summary_threshold
            ),
            persist_tool_transcripts=persist_tool_transcripts,
            tool_transcript_max_bytes=tool_transcript_max_bytes,
            snip_window=snip_window,
            auto_compact_at_tokens=auto_compact_at_tokens,
            auto_compact_summariser=auto_compact_summariser,
            auto_compact_keep_recent_turns=(
                auto_compact_keep_recent_turns
            ),
            retry_policy=retry_policy,
            auto_extract=auto_extract,
            approval_handler=approval_handler,
            secrets=secrets,
            response_tone=response_tone,
            effort=effort,
            strict_effort=strict_effort,
            architecture=ActorCritic(
                actor=actor,
                critic=critic,
                max_rounds=max_rounds,
                approval_threshold=approval_threshold,
                critique_template=critique_template,
                refine_template=refine_template,
                worker_registry=worker_registry,
                role_to_worker_id=role_to_worker_id,
            ),
        )
        if persistent_subagents and worker_registry is not None:
            coordinator._worker_registry = worker_registry
        return coordinator

    # -----------------------------------------------------------------
    # Blackboard
    # -----------------------------------------------------------------

    @staticmethod
    def blackboard(
        agents: dict[str, Agent],
        *,
        coordinator: Agent | None = None,
        decider: Agent | None = None,
        instructions: str = "",
        model: Model | str | None = None,
        memory: Memory | None = None,
        runtime: Runtime | None = None,
        budget: Budget | None = None,
        permissions: Permissions | None = None,
        hooks: HookRegistry | None = None,
        tools: (
            list[Tool | Callable[..., object]]
            | ToolHost
            | Tool
            | Callable[..., object]
            | None
        ) = None,
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        auto_consolidate: bool = False,
        skills: list[Any] | None = None,
        workspace: Any | str | None = None,
        living_plan: Any = None,
        stop_hooks: list[Any] | None = None,
        max_stop_hook_iterations: int = 15,
        prompt_caching: bool | Mapping[str, Any] | None = None,
        tool_result_summarizer: Model | str | None = None,
        tool_result_summary_threshold: int = 500,
        persist_tool_transcripts: bool = False,
        tool_transcript_max_bytes: int = 50_000,
        snip_window: int = 0,
        auto_compact_at_tokens: int | None = None,
        auto_compact_summariser: Model | str | None = None,
        auto_compact_keep_recent_turns: int = 4,
        retry_policy: Any | None = None,
        auto_extract: bool | None = None,
        approval_handler: Any | None = None,
        secrets: Any | None = None,
        response_tone: str | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        # --- blackboard-specific options ---
        max_rounds: int = 10,
        coordinator_instructions: str | None = None,
        decider_instructions: str | None = None,
        persistent_subagents: bool = True,
    ) -> Agent:
        """Build a blackboard team where ``agents`` collaborate via
        a shared workspace; an optional ``coordinator`` selects who
        acts each round and an optional ``decider`` decides when
        the work is done.

        ``workspace`` adds a persistent shared notebook on top of
        the in-memory blackboard contributions, so notes survive
        across runs and humans can inspect them via the filesystem.
        Each agent's dict key becomes its author identity in the
        notebook.

        ``persistent_subagents=True`` (default) registers each
        contributing agent (plus the coordinator + decider, when
        provided) with stable sessions so each agent's conversation
        memory carries across rounds AND across multiple
        ``Agent.run()`` invocations.

        ``prompt_caching=`` enables provider-native caching on the
        outer Agent that wraps the blackboard architecture (see
        :meth:`Team.supervisor` for accepted shapes). Contributing
        agents cache via their own ``Agent`` ctor.
        """
        coord_ws: Any = workspace
        if workspace is not None:
            _attach_workspace_to_workers(agents)
            from .workspace.resolver import resolve_workspace
            resolved = resolve_workspace(workspace)
            if resolved is not None and hasattr(resolved, "member"):
                coord_ws = resolved.member(
                    "coordinator", teammates=list(agents.keys())
                )
        worker_registry = None
        role_to_worker_id = None
        if persistent_subagents:
            from .agent.worker_registry import build_worker_registry
            bb_workers: dict[str, Agent] = dict(agents)
            if coordinator is not None:
                bb_workers["__coordinator"] = coordinator
            if decider is not None:
                bb_workers["__decider"] = decider
            worker_registry, role_to_worker_id = build_worker_registry(
                bb_workers
            )
        coord_agent = Agent(
            instructions=instructions,
            model=model,
            memory=memory,
            runtime=runtime,
            budget=budget,
            permissions=permissions,
            hooks=hooks,
            tools=tools,
            telemetry=telemetry,
            audit_log=audit_log,
            max_turns=max_turns,
            auto_consolidate=auto_consolidate,
            skills=skills,
            workspace=coord_ws,
            living_plan=living_plan,
            stop_hooks=stop_hooks,
            max_stop_hook_iterations=max_stop_hook_iterations,
            prompt_caching=prompt_caching,
            tool_result_summarizer=tool_result_summarizer,
            tool_result_summary_threshold=(
                tool_result_summary_threshold
            ),
            persist_tool_transcripts=persist_tool_transcripts,
            tool_transcript_max_bytes=tool_transcript_max_bytes,
            snip_window=snip_window,
            auto_compact_at_tokens=auto_compact_at_tokens,
            auto_compact_summariser=auto_compact_summariser,
            auto_compact_keep_recent_turns=(
                auto_compact_keep_recent_turns
            ),
            retry_policy=retry_policy,
            auto_extract=auto_extract,
            approval_handler=approval_handler,
            secrets=secrets,
            response_tone=response_tone,
            effort=effort,
            strict_effort=strict_effort,
            architecture=BlackboardArchitecture(
                agents=agents,
                coordinator=coordinator,
                decider=decider,
                max_rounds=max_rounds,
                coordinator_instructions=coordinator_instructions,
                decider_instructions=decider_instructions,
                worker_registry=worker_registry,
                role_to_worker_id=role_to_worker_id,
            ),
        )
        if persistent_subagents and worker_registry is not None:
            coord_agent._worker_registry = worker_registry
        return coord_agent


# ---------------------------------------------------------------------------
# Standalone-run helper
# ---------------------------------------------------------------------------


async def run_architecture(
    architecture: Architecture,
    prompt: str,
    *,
    instructions: str = "",
    model: Model | str | None = None,
    memory: Memory | None = None,
    runtime: Runtime | None = None,
    budget: Budget | None = None,
    permissions: Permissions | None = None,
    hooks: HookRegistry | None = None,
    tools: (
        list[Tool | Callable[..., object]]
        | ToolHost
        | Tool
        | Callable[..., object]
        | None
    ) = None,
    telemetry: Telemetry | None = None,
    audit_log: AuditLog | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    auto_consolidate: bool = False,
) -> RunResult:
    """Run an :class:`Architecture` once with a minimal Agent shell.

    Useful for testing orchestrators in isolation or for one-shot
    scripts where you don't want to construct an Agent yourself.

    The default ``model`` is the framework's resolver default (set
    via ``model=`` or env / config); pass an explicit model or
    string id to override.

    Example::

        sup = Supervisor(workers={"a": agent_a})
        result = await run_architecture(
            sup, "do the thing", model="gpt-4.1-mini"
        )
    """
    agent = Agent(
        instructions=instructions,
        model=model,
        memory=memory,
        runtime=runtime,
        budget=budget,
        permissions=permissions,
        hooks=hooks,
        tools=tools,
        telemetry=telemetry,
        audit_log=audit_log,
        max_turns=max_turns,
        auto_consolidate=auto_consolidate,
        architecture=architecture,
    )
    return await agent.run(prompt)


# Silence unused-import lints — these names appear only in type
# hints behind ``TYPE_CHECKING`` but the runtime alias above
# (``_ToolsArg``) keeps the imports honest at static-analysis time.
_ = (Any, _ToolsArg)
