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

from .base import AgentSession, Architecture, Dependencies
from .react import ReAct
from .resolver import resolve_architecture

__all__ = [
    "AgentSession",
    "Architecture",
    "Dependencies",
    "ReAct",
    "resolve_architecture",
]
