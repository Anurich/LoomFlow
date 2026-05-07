"""Team — sibling-style builders for multi-agent architectures.

Coming from LangGraph, CrewAI, AutoGen, or the OpenAI Agents SDK,
you'd expect to construct a multi-agent team like this::

    team = create_supervisor([researcher, writer], model="gpt-4")
    result = await team.invoke(...)

In JeevesAgent the same shape is :class:`Team`::

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

Each method takes the architecture-specific arguments by name plus
``**agent_kwargs`` for any :class:`Agent` config (``memory``,
``permissions``, ``budget``, ``audit_log``, ``hooks``, ``tools``,
``runtime``, ``telemetry``, ``max_turns``, ``auto_consolidate``).

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

from typing import TYPE_CHECKING, Any

from .agent.api import Agent
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
    from .core.protocols import Model


class Team:
    """Namespace for multi-agent team builders.

    Every classmethod returns a fully-built :class:`Agent` whose
    architecture is the corresponding multi-agent strategy. The
    returned Agent has the standard ``run`` / ``stream`` / etc.
    interface — call sites don't change between single-agent and
    team agents.
    """

    @staticmethod
    def supervisor(
        workers: dict[str, Agent],
        *,
        instructions: str = "",
        instructions_template: str | None = None,
        delegate_tool_name: str = "delegate",
        forward_tool_name: str = "forward_message",
        **agent_kwargs: Any,
    ) -> Agent:
        """Build a coordinator Agent that delegates to ``workers``.

        The coordinator can call ``delegate(worker, instructions)``
        to dispatch a subtask, or ``forward_message(worker)`` to
        return a worker's output verbatim. Multiple delegations in
        one turn run in parallel.

        Equivalent to::

            Agent(
                instructions=instructions,
                architecture=Supervisor(
                    workers=workers,
                    instructions_template=instructions_template,
                    delegate_tool_name=delegate_tool_name,
                    forward_tool_name=forward_tool_name,
                ),
                **agent_kwargs,
            )
        """
        return Agent(
            instructions=instructions,
            architecture=Supervisor(
                workers=workers,
                instructions_template=instructions_template,
                delegate_tool_name=delegate_tool_name,
                forward_tool_name=forward_tool_name,
            ),
            **agent_kwargs,
        )

    @staticmethod
    def swarm(
        agents: dict[str, Agent | Handoff],
        entry_agent: str,
        *,
        instructions: str = "",
        max_handoffs: int = 8,
        detect_cycles: bool = True,
        pass_full_history: bool = True,
        handoff_tool_name: str = "handoff",
        **agent_kwargs: Any,
    ) -> Agent:
        """Build a peer-swarm of agents that hand off control via a
        ``handoff`` tool (or per-target ``transfer_to_<name>`` tools
        when peers are wrapped in :class:`Handoff` with an
        ``input_type``).

        ``entry_agent`` is the peer that receives the first message.
        """
        return Agent(
            instructions=instructions,
            architecture=Swarm(
                agents=agents,
                entry_agent=entry_agent,
                max_handoffs=max_handoffs,
                detect_cycles=detect_cycles,
                pass_full_history=pass_full_history,
                handoff_tool_name=handoff_tool_name,
            ),
            **agent_kwargs,
        )

    @staticmethod
    def router(
        routes: list[RouterRoute],
        *,
        instructions: str = "",
        fallback_route: str | None = None,
        require_confidence_above: float = 0.0,
        classifier_prompt: str | None = None,
        **agent_kwargs: Any,
    ) -> Agent:
        """Build a router that classifies once and dispatches to
        ONE specialist :class:`Agent`. Cheaper than Supervisor for
        tasks with clear specialist boundaries (one classifier call
        + one specialist run, no synthesis pass)."""
        return Agent(
            instructions=instructions,
            architecture=Router(
                routes=routes,
                fallback_route=fallback_route,
                require_confidence_above=require_confidence_above,
                classifier_prompt=classifier_prompt,
            ),
            **agent_kwargs,
        )

    @staticmethod
    def debate(
        debaters: list[Agent],
        *,
        judge: Agent | None = None,
        instructions: str = "",
        rounds: int = 2,
        convergence_check: bool = True,
        convergence_similarity: float = 0.85,
        debater_instructions: str | None = None,
        judge_instructions: str | None = None,
        **agent_kwargs: Any,
    ) -> Agent:
        """Build a multi-agent debate where ``debaters`` argue for
        ``rounds`` (with optional convergence early-exit). If
        ``judge`` is provided, the judge synthesizes a final
        answer; otherwise majority vote wins."""
        return Agent(
            instructions=instructions,
            architecture=MultiAgentDebate(
                debaters=debaters,
                judge=judge,
                rounds=rounds,
                convergence_check=convergence_check,
                convergence_similarity=convergence_similarity,
                debater_instructions=debater_instructions,
                judge_instructions=judge_instructions,
            ),
            **agent_kwargs,
        )

    @staticmethod
    def actor_critic(
        actor: Agent,
        critic: Agent,
        *,
        instructions: str = "",
        max_rounds: int = 3,
        approval_threshold: float = 0.9,
        critique_template: str | None = None,
        refine_template: str | None = None,
        **agent_kwargs: Any,
    ) -> Agent:
        """Build an actor-critic pair where the critic reviews the
        actor's output (with structured JSON scoring + rubric) and
        the actor refines below ``approval_threshold``."""
        return Agent(
            instructions=instructions,
            architecture=ActorCritic(
                actor=actor,
                critic=critic,
                max_rounds=max_rounds,
                approval_threshold=approval_threshold,
                critique_template=critique_template,
                refine_template=refine_template,
            ),
            **agent_kwargs,
        )

    @staticmethod
    def blackboard(
        agents: dict[str, Agent],
        *,
        coordinator: Agent | None = None,
        decider: Agent | None = None,
        instructions: str = "",
        max_rounds: int = 10,
        coordinator_instructions: str | None = None,
        decider_instructions: str | None = None,
        **agent_kwargs: Any,
    ) -> Agent:
        """Build a blackboard team where ``agents`` collaborate via
        a shared workspace; an optional ``coordinator`` selects who
        acts each round and an optional ``decider`` decides when
        the work is done."""
        return Agent(
            instructions=instructions,
            architecture=BlackboardArchitecture(
                agents=agents,
                coordinator=coordinator,
                decider=decider,
                max_rounds=max_rounds,
                coordinator_instructions=coordinator_instructions,
                decider_instructions=decider_instructions,
            ),
            **agent_kwargs,
        )


# ---------------------------------------------------------------------------
# Standalone-run helper
# ---------------------------------------------------------------------------


async def run_architecture(
    architecture: Architecture,
    prompt: str,
    *,
    instructions: str = "",
    model: Model | str | None = None,
    tools: list[Any] | None = None,
    **agent_kwargs: Any,
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
        tools=tools,
        architecture=architecture,
        **agent_kwargs,
    )
    return await agent.run(prompt)
