"""Swarm: peer agents pass control via a ``handoff`` tool.

OpenAI Swarm reference (late 2024, experimental). Anthropic Agent
Teams (Feb 2026) is the production answer that improved on the
original swarm idea by adding lightweight coordination.

⚠ Production warning
---------------------
The 2026 production literature is unanimous: swarm has goal-drift
and deadlock failure modes that hierarchical / graph topologies
don't. Use only for **exploratory or research-mode systems where
flow can't be pre-specified**. For production, prefer
:class:`Supervisor` (clear authority) or :class:`Router` (single
specialist owns the answer).

Pattern
-------

1. **Setup.** N peer :class:`Agent` instances; one designated as
   the entry agent (receives the first user message).
2. **Active turn.** The active agent runs to completion with the
   ``handoff(target, message)`` tool injected via
   :meth:`Agent.run` ``extra_tools``. The model can call it (or
   not) freely during the turn.
3. **Detect handoff.** After the agent's turn ends, Swarm checks
   whether the handoff tool was called. If yes, switch active
   agent to the target and continue. If not, the agent's output
   is the final answer.
4. **Cycle / cap protection.** :data:`max_handoffs` caps total
   handoffs; ``detect_cycles`` watches for ``A→B→A→B`` patterns
   in the recent handoff window.

Replay correctness
------------------
Each peer turn uses a deterministic session id —
``{parent}__swarm_<peer>_<handoff_count>``. Replays of the
parent journal cache the per-turn results.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..core.types import Event
from ..tools.registry import Tool
from .base import AgentSession, Dependencies

if TYPE_CHECKING:
    from ..agent.api import Agent


class Swarm:
    """Peer agents passing control through a ``handoff`` tool."""

    name = "swarm"

    def __init__(
        self,
        *,
        agents: dict[str, Agent],
        entry_agent: str,
        max_handoffs: int = 8,
        detect_cycles: bool = True,
        pass_full_history: bool = True,
        handoff_tool_name: str = "handoff",
    ) -> None:
        if not agents:
            raise ValueError("Swarm requires at least one peer agent")
        if entry_agent not in agents:
            raise ValueError(
                f"entry_agent {entry_agent!r} not in agents "
                f"({', '.join(agents.keys())})"
            )
        if max_handoffs < 0:
            raise ValueError("max_handoffs must be >= 0")
        self._agents = dict(agents)
        self._entry_agent = entry_agent
        self._max_handoffs = max_handoffs
        self._detect_cycles = detect_cycles
        self._pass_full_history = pass_full_history
        self._handoff_tool_name = handoff_tool_name

    def declared_workers(self) -> dict[str, Agent]:
        return dict(self._agents)

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        active_name = self._entry_agent
        handoff_count = 0
        recent_handoffs: deque[tuple[str, str]] = deque(maxlen=4)
        history: list[str] = [prompt]
        last_output = ""

        yield Event.architecture_event(
            session.id,
            "swarm.started",
            entry_agent=active_name,
            num_peers=len(self._agents),
        )

        while True:
            active_agent = self._agents[active_name]
            yield Event.architecture_event(
                session.id,
                "swarm.active",
                agent=active_name,
                handoff_count=handoff_count,
            )

            # Build the input prompt for the active agent.
            if self._pass_full_history:
                active_prompt = "\n\n".join(history)
            else:
                active_prompt = history[-1]

            # The handoff tool records the request via a closure
            # variable; we read it after the agent's turn ends.
            handoff_request: dict[str, str] = {}

            async def _handoff(
                target: str,
                message: str = "",
                # bind closure refs at definition time so each iteration
                # gets a fresh tool that captures THIS iteration's vars
                _request: dict[str, str] = handoff_request,
                _agents: dict[str, Agent] = self._agents,
            ) -> str:
                if target not in _agents:
                    return (
                        f"Error: unknown peer {target!r}. "
                        f"Known: {', '.join(_agents.keys())}"
                    )
                # Last write wins if the model emits multiple handoffs
                # in one turn — that's a model quirk, not our problem.
                _request["target"] = target
                _request["message"] = message
                return f"[handoff requested → {target}]"

            handoff_tool = Tool(
                name=self._handoff_tool_name,
                description=(
                    "Hand off the conversation to another peer "
                    "agent. Pass `target` (peer name) and an "
                    "optional `message` describing context to "
                    "carry over."
                ),
                fn=_handoff,
                input_schema={
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": (
                                "Name of the peer to hand off to. "
                                "Must be one of the configured peers."
                            ),
                        },
                        "message": {
                            "type": "string",
                            "description": (
                                "Optional context to pass along."
                            ),
                        },
                    },
                    "required": ["target"],
                },
            )

            # Run the active agent with the handoff tool injected.
            sub_session_id = (
                f"{session.id}__swarm_{active_name}_{handoff_count}"
            )
            result = await active_agent.run(
                active_prompt,
                session_id=sub_session_id,
                extra_tools=[handoff_tool],
            )
            session.turns += result.turns
            last_output = result.output

            if result.interrupted:
                session.interrupted = True
                session.interruption_reason = (
                    f"swarm:{active_name}:"
                    f"{result.interruption_reason or 'unknown'}"
                )
                session.output = last_output
                yield Event.architecture_event(
                    session.id,
                    "swarm.peer_interrupted",
                    agent=active_name,
                )
                return

            # Did the model call handoff?
            if not handoff_request:
                # No handoff → agent produced the final answer.
                session.output = last_output
                yield Event.architecture_event(
                    session.id,
                    "swarm.completed",
                    agent=active_name,
                    handoffs=handoff_count,
                )
                return

            target = handoff_request["target"]
            message = handoff_request.get("message", "")

            # Record + cycle check
            recent_handoffs.append((active_name, target))
            if self._detect_cycles and _is_cycling(recent_handoffs):
                yield Event.architecture_event(
                    session.id,
                    "swarm.cycle_detected",
                    recent=list(recent_handoffs),
                )
                session.output = last_output
                return

            # Append agent output + transition to history.
            history.append(last_output or "[no output]")
            transition = (
                f"[Handoff: {active_name} → {target}] {message}"
                if message
                else f"[Handoff: {active_name} → {target}]"
            )
            history.append(transition)

            yield Event.architecture_event(
                session.id,
                "swarm.handoff",
                from_agent=active_name,
                to_agent=target,
                message=message,
                handoff_count=handoff_count + 1,
            )

            active_name = target
            handoff_count += 1

            if handoff_count >= self._max_handoffs:
                yield Event.architecture_event(
                    session.id,
                    "swarm.max_handoffs",
                    handoffs=handoff_count,
                )
                # Run the new active agent ONE more time with no
                # handoff tool? That conflicts with the design. The
                # spec says: just emit max_handoffs and use the
                # latest output. We've already captured the
                # PREVIOUS agent's output as last_output; that's
                # what we return.
                session.output = last_output
                return


def _is_cycling(recent: deque[tuple[str, str]]) -> bool:
    """Detect ``A→B→A→B`` repetition in the most recent 4 handoffs.

    Conservative: only fires on exact 2-cycle repetition.
    Triple-cycles (``A→B→C→A→B→C``) and longer aren't caught here;
    the ``max_handoffs`` cap is the backstop.
    """
    if len(recent) < 4:
        return False
    return recent[0] == recent[2] and recent[1] == recent[3]
