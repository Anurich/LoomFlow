"""Architecture layer.

An :class:`Architecture` is a strategy for driving the agent loop.
The canonical default is :class:`ReAct` — observe / think / act in a
tight loop. Other architectures (Plan-and-Execute, Reflexion,
Self-Refine, Tree of Thoughts, Supervisor, Router, ...) plug into the
same :class:`Agent` by satisfying the :class:`Architecture` protocol.

See ``Subagent.md`` in the repo root for the full architecture
catalogue and design rationale.

Public surface:

* :class:`Architecture` — the protocol architectures implement
* :class:`AgentSession` — mutable per-run state
* :class:`Dependencies` — bundled protocol implementations
* :class:`ReAct` — the canonical default (observe / think / act)
* :func:`resolve_architecture` — string -> Architecture instance
"""

from .actor_critic import ActorCritic
from .base import AgentSession, Architecture, Dependencies
from .blackboard import Blackboard, BlackboardArchitecture, BlackboardEntry
from .debate import MultiAgentDebate
from .plan_and_execute import Plan, PlanAndExecute, PlanStep, StepResult
from .react import ReAct
from .reflexion import Reflexion
from .resolver import resolve_architecture
from .router import Router, RouterRoute
from .self_refine import SelfRefine
from .supervisor import Supervisor
from .swarm import Swarm
from .tree_of_thoughts import ThoughtNode, TreeOfThoughts

__all__ = [
    "ActorCritic",
    "AgentSession",
    "Architecture",
    "Blackboard",
    "BlackboardArchitecture",
    "BlackboardEntry",
    "Dependencies",
    "MultiAgentDebate",
    "Plan",
    "PlanAndExecute",
    "PlanStep",
    "ReAct",
    "Reflexion",
    "Router",
    "RouterRoute",
    "SelfRefine",
    "StepResult",
    "Supervisor",
    "Swarm",
    "ThoughtNode",
    "TreeOfThoughts",
    "resolve_architecture",
]
