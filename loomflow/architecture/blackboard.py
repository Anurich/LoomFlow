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
from typing import TYPE_CHECKING, Any

from ..core.types import Event
from .base import AgentSession, Dependencies
from .helpers import SubagentInvocation

if TYPE_CHECKING:
    from ..agent.api import Agent


_BBRegistryT = dict[str, Any]  # worker handle registry
_BBRoleMapT = dict[str, str]   # role → worker_id (incl. __coordinator/__decider)


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
        worker_registry: _BBRegistryT | None = None,
        role_to_worker_id: _BBRoleMapT | None = None,
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
        # Persistent-subagent wiring — when Team.blackboard built us
        # with ``persistent_subagents=True``, every contributing agent
        # (plus the coordinator + decider, which are also registered)
        # runs under its handle's stable session_id. Long-running
        # blackboard sessions then accumulate per-agent memory across
        # rounds + runs instead of restarting each round.
        self._worker_registry = worker_registry
        self._role_to_worker_id = role_to_worker_id

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

            # Coordinator: stream its events through, capture decision.
            decision_holder: list[_CoordinatorDecision] = []
            async for ev in self._coordinate_streaming(
                session, deps, bb, round_num, decision_holder
            ):
                yield ev
            decision = decision_holder[0]
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
            from ..agent.worker_registry import resolve_persistent_session
            sub_session_id, handle = resolve_persistent_session(
                decision.next_agent,
                fallback=(
                    f"{session.id}__bb_{decision.next_agent}"
                    f"_round_{round_num}"
                ),
                registry=self._worker_registry,
                role_to_id=self._role_to_worker_id,
            )
            if handle is not None:
                handle.touch(user_id=deps.context.user_id)
            invocation = SubagentInvocation(
                picked,
                agent_prompt,
                session_id=sub_session_id,
                rollup_into=session,
            )
            async for ev in invocation.events():
                yield ev
            picked_result = invocation.result
            session.turns += int(picked_result.get("turns", 0) or 0)
            output = str(picked_result.get("output", ""))
            bb.post(
                decision.next_agent,
                output,
                kind="contribution",
            )
            yield Event.architecture_event(
                session.id,
                "blackboard.contribution",
                round=round_num,
                agent=decision.next_agent,
                content=output[:300],
            )

        # === Synthesize ===
        final_holder: list[str] = []
        async for ev in self._decide_final_streaming(
            session, deps, bb, final_holder
        ):
            yield ev
        final = final_holder[0]
        session.output = final
        yield Event.architecture_event(
            session.id,
            "blackboard.completed",
            final=final[:300],
            board_size=len(bb.public),
        )

    # ---- helpers -----------------------------------------------------

    async def _coordinate_streaming(
        self,
        session: AgentSession,
        deps: Dependencies,
        bb: Blackboard,
        round_num: int,
        decision_holder: list[_CoordinatorDecision],
    ) -> AsyncIterator[Event]:
        """Run the coordinator (LLM Agent or round-robin fallback)
        and stream its events; write the decision into
        ``decision_holder[0]``."""
        if self._coordinator is None:
            # Round-robin fallback — no LLM call, no events.
            names = list(self._agents)
            picked = names[(round_num - 1) % len(names)]
            decision_holder.append(
                _CoordinatorDecision(
                    terminate=False,
                    next_agent=picked,
                    instruction=None,
                    raw="(round-robin fallback)",
                )
            )
            return

        coord_prompt = self._coordinator_instructions.format(
            agents="\n".join(
                f"  - {n}: {(a.instructions or '(no description)')[:120]}"
                for n, a in self._agents.items()
            )
        ) + (
            f"\n\nBlackboard state:\n{bb.render_for('__coordinator')}"
        )
        from ..agent.worker_registry import resolve_persistent_session
        sub_session_id, coord_handle = resolve_persistent_session(
            "__coordinator",
            fallback=f"{session.id}__bb_coord_round_{round_num}",
            registry=self._worker_registry,
            role_to_id=self._role_to_worker_id,
        )
        if coord_handle is not None:
            coord_handle.touch(user_id=deps.context.user_id)
        invocation = SubagentInvocation(
            self._coordinator,
            coord_prompt,
            session_id=sub_session_id,
            rollup_into=session,
        )
        async for ev in invocation.events():
            yield ev
        coord_result = invocation.result
        session.turns += int(coord_result.get("turns", 0) or 0)
        decision_holder.append(
            _parse_coordinator_decision(
                str(coord_result.get("output", ""))
            )
        )

    async def _decide_final_streaming(
        self,
        session: AgentSession,
        deps: Dependencies,
        bb: Blackboard,
        final_holder: list[str],
    ) -> AsyncIterator[Event]:
        """Run the decider (LLM Agent or fallback) and stream its
        events; write the final answer into ``final_holder[0]``."""
        if self._decider is None:
            # Last "answer"-kind entry wins; fall through to last
            # public entry; finally the empty string.
            for entry in reversed(bb.public):
                if entry.kind == "answer":
                    final_holder.append(entry.content)
                    return
            for entry in reversed(bb.public):
                if entry.kind == "contribution":
                    final_holder.append(entry.content)
                    return
            final_holder.append("")
            return

        decide_prompt = self._decider_instructions + (
            f"\n\nFull blackboard state:\n{bb.render_for('__decider')}"
        )
        from ..agent.worker_registry import resolve_persistent_session
        sub_session_id, dec_handle = resolve_persistent_session(
            "__decider",
            fallback=f"{session.id}__bb_decider",
            registry=self._worker_registry,
            role_to_id=self._role_to_worker_id,
        )
        if dec_handle is not None:
            dec_handle.touch(user_id=deps.context.user_id)
        invocation = SubagentInvocation(
            self._decider,
            decide_prompt,
            session_id=sub_session_id,
            rollup_into=session,
        )
        async for ev in invocation.events():
            yield ev
        decider_result = invocation.result
        session.turns += int(decider_result.get("turns", 0) or 0)
        final_holder.append(str(decider_result.get("output", "")))


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
