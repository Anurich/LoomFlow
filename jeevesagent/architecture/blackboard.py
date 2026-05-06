"""Blackboard: shared state board with a coordinator picking the
next contributor each round.

Classical AI: Erman et al. 1980 (Hearsay-II). Han & Zhang 2025
revived for LLM agents (arXiv:2507.01701). Salemi et al. 2026
(arXiv:2510.01285) reports +13-57% relative improvement on
data-discovery tasks.

Pattern
-------

1. Initialize the blackboard with the user's problem statement.
2. Each round, the **coordinator** reads the blackboard and picks
   one agent to contribute next (or terminates).
3. The chosen agent runs with the blackboard view in its prompt and
   produces a contribution.
4. The contribution is appended to the blackboard.
5. Loop until the coordinator terminates or ``max_rounds`` is hit.
6. The **decider** synthesizes the final answer from the
   blackboard. ``decider=None`` falls back to the last
   ``answer``-kind contribution, or the most recent contribution
   if no answer-kind exists.

Coordinator API
---------------

The coordinator is a :class:`Agent` that produces JSON in this shape::

    {"terminate": bool, "next_agent": str|null, "instruction": str|null}

If ``terminate`` is true, the loop ends. Otherwise ``next_agent``
must name one of the configured agents. If the coordinator emits an
unknown name or malformed JSON, the round is skipped and a warning
event fires (Blackboard does not crash on coordinator misbehavior).

Set ``coordinator=None`` to fall back to round-robin selection —
useful for testing / prototyping but defeats the "contribute when
relevant" feature of the architecture.

Strengths
---------
* Decentralized contribution; agents react to current state rather
  than being forced into a fixed delegation graph.
* Transparent state — the blackboard is the audit log.

Weaknesses
----------
* Coordinator quality is critical. A bad coordinator picks wrong
  agents and the system stalls.
* Blackboard state grows monotonically; long sessions can blow up
  the LLM context.
* "Theoretically interesting but rarely outperforms hierarchical or
  graph in production" (2026 taxonomy guide). Reserve for
  exploratory research / data-discovery problems.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..core.types import Event
from .base import AgentSession, Dependencies

if TYPE_CHECKING:
    from ..agent.api import Agent


DEFAULT_COORDINATOR_INSTRUCTIONS = """\
You coordinate a team of specialist agents on a shared problem.

Read the blackboard state below and decide:
- Should we terminate now? (Has a satisfactory answer been written?)
- If continuing, which agent should contribute next?
- What specific instruction should that agent receive?

Available agents:
{agents}

Output JSON exactly:
{{"terminate": <bool>, "next_agent": <str|null>, "instruction": <str|null>}}

If terminating, set next_agent and instruction to null.
"""


DEFAULT_DECIDER_INSTRUCTIONS = """\
You synthesize the final answer from a multi-agent blackboard
discussion. Read the blackboard state below and produce the best
answer you can. Cite specific contributions when useful;
acknowledge dissent if it matters.
"""


@dataclass
class BlackboardEntry:
    """One contribution on the blackboard."""

    timestamp: datetime
    author: str
    content: str
    kind: str = "contribution"


@dataclass
class Blackboard:
    """Public + per-agent private state for the architecture."""

    public: list[BlackboardEntry] = field(default_factory=list)
    private: dict[str, list[BlackboardEntry]] = field(
        default_factory=dict
    )

    def post(
        self,
        author: str,
        content: str,
        *,
        kind: str = "contribution",
        private_to: str | None = None,
    ) -> BlackboardEntry:
        entry = BlackboardEntry(
            timestamp=datetime.now(UTC),
            author=author,
            content=content,
            kind=kind,
        )
        if private_to:
            self.private.setdefault(private_to, []).append(entry)
        else:
            self.public.append(entry)
        return entry

    def render_for(self, agent_name: str) -> str:
        """Format the blackboard state as a string for ``agent_name``.

        Includes every public entry and the agent's own private
        scratchpad if any.
        """
        public_lines = [
            f"[{e.kind}] {e.author}: {e.content}"
            for e in self.public
        ]
        private_lines = [
            f"[private:{e.kind}] {e.content}"
            for e in self.private.get(agent_name, [])
        ]
        parts = []
        if public_lines:
            parts.append(
                "=== Public board ===\n" + "\n".join(public_lines)
            )
        if private_lines:
            parts.append(
                "=== Your private notes ===\n"
                + "\n".join(private_lines)
            )
        return "\n\n".join(parts) if parts else "(empty)"


@dataclass
class _CoordinatorDecision:
    terminate: bool
    next_agent: str | None
    instruction: str | None
    raw: str


class BlackboardArchitecture:
    """Coordinator + agents + decider, mediated by a shared
    blackboard."""

    name = "blackboard"

    def __init__(
        self,
        *,
        agents: dict[str, Agent],
        coordinator: Agent | None = None,
        decider: Agent | None = None,
        max_rounds: int = 10,
        coordinator_instructions: str | None = None,
        decider_instructions: str | None = None,
    ) -> None:
        if not agents:
            raise ValueError(
                "Blackboard requires at least one agent"
            )
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self._agents = dict(agents)
        self._coordinator = coordinator
        self._decider = decider
        self._max_rounds = max_rounds
        self._coordinator_instructions = (
            coordinator_instructions
            or DEFAULT_COORDINATOR_INSTRUCTIONS
        )
        self._decider_instructions = (
            decider_instructions or DEFAULT_DECIDER_INSTRUCTIONS
        )

    def declared_workers(self) -> dict[str, Agent]:
        workers = dict(self._agents)
        if self._coordinator is not None:
            workers["__coordinator"] = self._coordinator
        if self._decider is not None:
            workers["__decider"] = self._decider
        return workers

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        bb = Blackboard()
        bb.post("user", prompt, kind="problem")

        yield Event.architecture_event(
            session.id,
            "blackboard.started",
            agents=list(self._agents),
            has_coordinator=self._coordinator is not None,
            has_decider=self._decider is not None,
        )

        for round_num in range(1, self._max_rounds + 1):
            status = await deps.budget.allows_step()
            if status.blocked:
                session.interrupted = True
                session.interruption_reason = (
                    f"budget:{status.reason}"
                )
                yield Event.budget_exceeded(session.id, status)
                return

            decision = await self._coordinate(
                session, bb, round_num
            )
            yield Event.architecture_event(
                session.id,
                "blackboard.coordinator_decided",
                round=round_num,
                terminate=decision.terminate,
                next_agent=decision.next_agent,
                raw=decision.raw[:300],
            )

            if decision.terminate:
                break

            if decision.next_agent is None:
                yield Event.architecture_event(
                    session.id,
                    "blackboard.no_contributor",
                    round=round_num,
                )
                continue

            if decision.next_agent not in self._agents:
                bb.post(
                    "system",
                    f"Coordinator picked unknown agent "
                    f"{decision.next_agent!r}; skipping.",
                    kind="error",
                )
                yield Event.architecture_event(
                    session.id,
                    "blackboard.unknown_agent",
                    round=round_num,
                    agent_name=decision.next_agent,
                )
                continue

            picked = self._agents[decision.next_agent]
            view = bb.render_for(decision.next_agent)
            instruction = (
                decision.instruction
                or f"Contribute to the blackboard as {decision.next_agent}."
            )
            agent_prompt = (
                f"You are {decision.next_agent}.\n\n"
                f"Blackboard state:\n{view}\n\n"
                f"Coordinator instruction: {instruction}\n\n"
                f"Produce ONE contribution in plain text. Be "
                f"specific; cite prior contributions when useful."
            )
            yield Event.architecture_event(
                session.id,
                "blackboard.invoking",
                round=round_num,
                agent=decision.next_agent,
            )
            sub_session_id = (
                f"{session.id}__bb_{decision.next_agent}_round_{round_num}"
            )
            result = await picked.run(
                agent_prompt, session_id=sub_session_id
            )
            session.turns += result.turns
            bb.post(
                decision.next_agent,
                result.output,
                kind="contribution",
            )
            yield Event.architecture_event(
                session.id,
                "blackboard.contribution",
                round=round_num,
                agent=decision.next_agent,
                content=result.output[:300],
            )

        # === Synthesize ===
        final = await self._decide_final(session, bb)
        session.output = final
        yield Event.architecture_event(
            session.id,
            "blackboard.completed",
            final=final[:300],
            board_size=len(bb.public),
        )

    # ---- helpers -----------------------------------------------------

    async def _coordinate(
        self,
        session: AgentSession,
        bb: Blackboard,
        round_num: int,
    ) -> _CoordinatorDecision:
        if self._coordinator is None:
            # Round-robin fallback
            names = list(self._agents)
            picked = names[(round_num - 1) % len(names)]
            return _CoordinatorDecision(
                terminate=False,
                next_agent=picked,
                instruction=None,
                raw="(round-robin fallback)",
            )

        coord_prompt = self._coordinator_instructions.format(
            agents="\n".join(
                f"  - {n}: {(a.instructions or '(no description)')[:120]}"
                for n, a in self._agents.items()
            )
        ) + (
            f"\n\nBlackboard state:\n{bb.render_for('__coordinator')}"
        )
        sub_session_id = (
            f"{session.id}__bb_coord_round_{round_num}"
        )
        result = await self._coordinator.run(
            coord_prompt, session_id=sub_session_id
        )
        session.turns += result.turns
        return _parse_coordinator_decision(result.output)

    async def _decide_final(
        self, session: AgentSession, bb: Blackboard
    ) -> str:
        if self._decider is None:
            # Last "answer"-kind entry wins; fall through to last
            # public entry; finally the empty string.
            for entry in reversed(bb.public):
                if entry.kind == "answer":
                    return entry.content
            for entry in reversed(bb.public):
                if entry.kind == "contribution":
                    return entry.content
            return ""

        decide_prompt = self._decider_instructions + (
            f"\n\nFull blackboard state:\n{bb.render_for('__decider')}"
        )
        sub_session_id = f"{session.id}__bb_decider"
        result = await self._decider.run(
            decide_prompt, session_id=sub_session_id
        )
        session.turns += result.turns
        return result.output


def _parse_coordinator_decision(text: str) -> _CoordinatorDecision:
    """Parse a coordinator's JSON output. Robust to markdown fences
    and free-form prose; falls back to "no contributor this round"
    when parsing fails entirely."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    parsed: object
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        parsed = None

    if isinstance(parsed, dict):
        terminate = bool(parsed.get("terminate", False))
        next_agent_raw = parsed.get("next_agent")
        next_agent = (
            str(next_agent_raw)
            if next_agent_raw is not None
            else None
        )
        instruction_raw = parsed.get("instruction")
        instruction = (
            str(instruction_raw)
            if instruction_raw is not None
            else None
        )
        return _CoordinatorDecision(
            terminate=terminate,
            next_agent=next_agent,
            instruction=instruction,
            raw=text,
        )

    # Parse failure → conservative no-op.
    return _CoordinatorDecision(
        terminate=False,
        next_agent=None,
        instruction=None,
        raw=text,
    )
