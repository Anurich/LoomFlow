"""ActorCritic: generator + adversarial critic, asymmetric by design.

Sutton & Barto 1998 (RL foundations); LLM-era papers Madaan et al.
2023 (Self-Refine — same model both roles), Gou et al. 2023 (CRITIC),
Sun et al. 2025 (CGI — separate critic model). 2026 production
literature recommends ActorCritic for **quality-critical work** —
code generation, security review, important written communications.

The pattern in one line: actor proposes; critic finds problems with a
*different prompt and ideally a different model*; actor revises.

Why a separate :class:`SelfRefine`?
-----------------------------------
:class:`SelfRefine` runs critic + refiner with the parent's same
model and same prompt template. ActorCritic earns its complexity
only when the actor and critic have *different blind spots* —
typically different models. We require both ``actor`` and
``critic`` :class:`Agent` instances; for same-model self-critique,
use :class:`SelfRefine`.

Pattern
-------

1. **Round 0 (actor).** ``actor.run(prompt)`` produces an initial
   output.
2. **For each round up to ``max_rounds``:**

   a. **Critic.** ``critic.run(critique_prompt)`` produces a
      structured critique with an explicit ``score`` 0-1.
   b. **Approval check.** If ``critique.score >=
      approval_threshold``, terminate as approved.
   c. **Refine.** ``actor.run(refine_prompt)`` produces a revised
      output that addresses the critique. The new output replaces
      the old.

3. **Max rounds reached without approval.** Return the current
   output. Best we have.

Replay correctness
------------------
Each actor / critic invocation uses a deterministic session id
(``{parent}__actor_<round>`` / ``{parent}__critic_<round>``) so
replays of the parent reproduce the same sub-sessions.

Tuning
------
* ``max_rounds=3`` is the production sweet spot for code generation.
* ``approval_threshold=0.9`` is strict; lower to 0.85 for friendlier
  convergence.
* Use **different models** for actor and critic. Claude Opus actor +
  GPT-4o critic (or vice versa) is the canonical asymmetry.

Composition
-----------
* Inside :class:`Supervisor`: each worker can be an ActorCritic for
  per-domain quality control (``coder`` worker uses ActorCritic for
  code review).
* Inside :class:`Reflexion`: cross-session learning of which
  critique patterns produce real improvements.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.context import inherit_ambient_memory
from ..core.types import Event
from .base import AgentSession, Dependencies
from .helpers import SubagentInvocation, budget_gate, parse_fenced_json

if TYPE_CHECKING:
    from ..agent.api import Agent


_ACRegistryT = dict[str, Any]  # worker handle registry
_ACRoleMapT = dict[str, str]   # actor|critic → worker_id


DEFAULT_CRITIQUE_TEMPLATE = """\
You are reviewing the output below against the original task. Find
every issue you can: factual errors, missing requirements, edge
cases, security holes, unclear language. Be specific — cite the
section of the output you're criticizing.

Output ONLY a JSON object with this shape:

{{"issues": ["...", "..."], "score": 0.0-1.0, "summary": "..."}}

The score is your confidence the output fully solves the task:
- 1.0 = no issues, ship it
- 0.7-0.9 = mostly correct, minor gaps
- 0.4-0.6 = real problems, must revise
- 0.0-0.3 = wrong or missing core deliverable

Original task:
{prompt}

Output to review:
{output}
"""


DEFAULT_REFINE_TEMPLATE = """\
Revise your previous output based on the critique below. Address
every point in the critique. Output ONLY the revised version, no
preamble or commentary about what changed.

Original task:
{prompt}

Previous output:
{output}

Critique:
{issues_bulleted}
"""


class CriticOutput(BaseModel):
    """Structured critic verdict.

    Parsed from the critic Agent's output. Falls back to a single-
    issue blob with score 0.0 when JSON parsing fails so the loop
    keeps making progress instead of crashing on a malformed reply.
    """

    issues: list[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    summary: str = ""


class ActorCritic:
    """Actor + adversarial critic with optional different models.

    Constructor parameters:

    * ``actor`` (required): the generating :class:`Agent`. Sees the
      original prompt on round 0 and a refine prompt on subsequent
      rounds.
    * ``critic`` (required): the reviewing :class:`Agent`. Sees the
      original prompt + the actor's current output and produces
      structured JSON critique.
    * ``max_rounds``: cap on critique-refine cycles after the
      initial generation. Default 3.
    * ``approval_threshold``: terminate when ``critique.score`` is
      at or above this value. Default 0.9.
    * ``critique_template`` / ``refine_template``: override the
      default prompts. Templates use ``{prompt}``, ``{output}``,
      ``{critique}``, ``{issues_bulleted}``.
    """

    name = "actor-critic"

    def __init__(
        self,
        *,
        actor: Agent,
        critic: Agent,
        max_rounds: int = 3,
        approval_threshold: float = 0.9,
        critique_template: str | None = None,
        refine_template: str | None = None,
        worker_registry: _ACRegistryT | None = None,
        role_to_worker_id: _ACRoleMapT | None = None,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if not 0.0 <= approval_threshold <= 1.0:
            raise ValueError(
                "approval_threshold must be in [0.0, 1.0]"
            )
        self._actor = actor
        self._critic = critic
        self._max_rounds = max_rounds
        self._threshold = approval_threshold
        self._critique_template = (
            critique_template or DEFAULT_CRITIQUE_TEMPLATE
        )
        self._refine_template = (
            refine_template or DEFAULT_REFINE_TEMPLATE
        )
        # Persistent-subagent wiring — when Team.actor_critic built us
        # with ``persistent_subagents=True``, actor and critic each
        # run under their handle's stable session_id. All rounds within
        # a single Agent.run() AND across multiple runs reuse the same
        # session, so the critic remembers what it already flagged and
        # the actor remembers what it already refined.
        self._worker_registry = worker_registry
        self._role_to_worker_id = role_to_worker_id

    def declared_workers(self) -> dict[str, Agent]:
        return {"actor": self._actor, "critic": self._critic}

    async def _stream_guarded(
        self,
        session: AgentSession,
        deps: Dependencies,
        invocation: SubagentInvocation,
        handle: Any,
        role: str,
        round_num: int,
        rejection: list[str],
    ) -> AsyncIterator[Event]:
        """Stream ``invocation`` under the persistent-worker session
        guard (cross-user check + lock + touch via the shared
        :func:`acquire_worker_session`). On cross-user rejection the
        error is appended to ``rejection``, the session is marked
        interrupted, and the rejection event is emitted — callers
        check ``rejection`` and stop."""
        if handle is None:
            async for ev in invocation.events():
                yield ev
            return
        from ..agent.worker_registry import (
            CrossUserWorkerError,
            acquire_worker_session,
        )
        try:
            async with acquire_worker_session(
                handle, deps.context.user_id
            ):
                async for ev in invocation.events():
                    yield ev
        except CrossUserWorkerError as exc:
            rejection.append(str(exc))
            session.interrupted = True
            session.interruption_reason = (
                f"{role}:round_{round_num}:cross_user_worker"
            )
            yield Event.architecture_event(
                session.id,
                "actor_critic.cross_user_rejected",
                round=round_num,
                agent=role,
                error=str(exc),
            )

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # Memory propagation — actor + critic + refine spawns all
        # inherit the coordinator's memory. See base helper docstring.
        with inherit_ambient_memory(deps.memory):
            async for ev in self._run_inner(session, deps, prompt):
                yield ev

    async def _run_inner(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # === Round 0: initial generation by actor ===
        yield Event.architecture_event(
            session.id,
            "actor_critic.actor_started",
            round=0,
            phase="generate",
        )
        from ..agent.worker_registry import resolve_persistent_session
        actor_sid_0, actor_handle_0 = resolve_persistent_session(
            "actor",
            fallback=f"{session.id}__actor_0",
            registry=self._worker_registry,
            role_to_id=self._role_to_worker_id,
        )
        actor_inv = SubagentInvocation(
            self._actor,
            prompt,
            session_id=actor_sid_0,
            rollup_into=session,
        )
        rejection: list[str] = []
        async for ev in self._stream_guarded(
            session, deps, actor_inv, actor_handle_0,
            "actor", 0, rejection,
        ):
            yield ev
        if rejection:
            return
        actor_result = actor_inv.result
        current_output = str(actor_result.get("output", ""))
        session.output = current_output
        session.turns += int(actor_result.get("turns", 0) or 0)

        if bool(actor_result.get("interrupted", False)):
            session.interrupted = True
            session.interruption_reason = (
                f"actor:round_0:"
                f"{actor_result.get('interruption_reason') or 'unknown'}"
            )
            return

        yield Event.architecture_event(
            session.id,
            "actor_critic.actor_completed",
            round=0,
            phase="generate",
        )

        # === Critique → refine loop ===
        for round_num in range(1, self._max_rounds + 1):
            blocked, gate_events = await budget_gate(deps, session)
            for gate_event in gate_events:
                yield gate_event
            if blocked:
                return

            # --- Critic ---
            yield Event.architecture_event(
                session.id,
                "actor_critic.critic_started",
                round=round_num,
            )
            critique_prompt = self._critique_template.format(
                prompt=prompt,
                output=current_output,
            )
            critic_sid, critic_handle = resolve_persistent_session(
                "critic",
                fallback=f"{session.id}__critic_{round_num}",
                registry=self._worker_registry,
                role_to_id=self._role_to_worker_id,
            )
            critic_inv = SubagentInvocation(
                self._critic,
                critique_prompt,
                session_id=critic_sid,
                rollup_into=session,
            )
            rejection = []
            async for ev in self._stream_guarded(
                session, deps, critic_inv, critic_handle,
                "critic", round_num, rejection,
            ):
                yield ev
            if rejection:
                return
            critic_result = critic_inv.result
            session.turns += int(critic_result.get("turns", 0) or 0)

            if bool(critic_result.get("interrupted", False)):
                # Critic interrupted; treat current output as best
                # we have and stop.
                session.interrupted = True
                session.interruption_reason = (
                    f"critic:round_{round_num}:"
                    f"{critic_result.get('interruption_reason') or 'unknown'}"
                )
                return

            critique = _parse_critique(
                str(critic_result.get("output", ""))
            )
            yield Event.architecture_event(
                session.id,
                "actor_critic.critique",
                round=round_num,
                score=critique.score,
                issues=critique.issues,
                summary=critique.summary,
            )

            if critique.score >= self._threshold:
                yield Event.architecture_event(
                    session.id,
                    "actor_critic.approved",
                    round=round_num,
                    score=critique.score,
                )
                return

            if round_num >= self._max_rounds:
                yield Event.architecture_event(
                    session.id,
                    "actor_critic.max_rounds_reached",
                    rounds=round_num,
                    final_score=critique.score,
                )
                return

            # --- Refine via actor ---
            yield Event.architecture_event(
                session.id,
                "actor_critic.actor_started",
                round=round_num,
                phase="refine",
            )
            issues_bulleted = "\n".join(
                f"- {issue}" for issue in critique.issues
            ) or "(no specific issues listed; general improvement)"
            refine_prompt = self._refine_template.format(
                prompt=prompt,
                output=current_output,
                critique=critique.summary or "",
                issues_bulleted=issues_bulleted,
            )
            refine_sid, refine_handle = resolve_persistent_session(
                "actor",
                fallback=f"{session.id}__actor_{round_num}",
                registry=self._worker_registry,
                role_to_id=self._role_to_worker_id,
            )
            refine_inv = SubagentInvocation(
                self._actor,
                refine_prompt,
                session_id=refine_sid,
                rollup_into=session,
            )
            rejection = []
            async for ev in self._stream_guarded(
                session, deps, refine_inv, refine_handle,
                "actor", round_num, rejection,
            ):
                yield ev
            if rejection:
                return
            refine_result = refine_inv.result
            session.turns += int(refine_result.get("turns", 0) or 0)

            if bool(refine_result.get("interrupted", False)):
                session.interrupted = True
                session.interruption_reason = (
                    f"actor:round_{round_num}:"
                    f"{refine_result.get('interruption_reason') or 'unknown'}"
                )
                return

            current_output = str(refine_result.get("output", ""))
            session.output = current_output

            yield Event.architecture_event(
                session.id,
                "actor_critic.actor_completed",
                round=round_num,
                phase="refine",
            )


# ---------------------------------------------------------------------------
# Critique parser
# ---------------------------------------------------------------------------


_SCORE_RE = re.compile(
    r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
)


def _parse_critique(text: str) -> CriticOutput:
    """Best-effort parse of critic output.

    Tries: (1) raw JSON, (2) JSON inside markdown code fences,
    (3) regex fallback that extracts a score and uses the full
    text as a single issue.

    Returns a default ``CriticOutput`` (score 0.0, single issue =
    raw text) when parsing fails entirely so the loop keeps making
    progress on the next refine pass instead of crashing.
    """
    # (1)+(2) Fence-tolerant JSON parse via the shared helper.
    parsed: object = parse_fenced_json(text)

    if isinstance(parsed, dict):
        try:
            issues_raw = parsed.get("issues", []) or []
            issues = [str(i) for i in issues_raw if i]
            score_raw = parsed.get("score", 0.0)
            score = max(0.0, min(1.0, float(score_raw)))
            summary = str(parsed.get("summary", ""))
            return CriticOutput(
                issues=issues, score=score, summary=summary
            )
        except (TypeError, ValueError):
            pass

    # (3) Regex fallback for the score; whole text becomes one issue.
    match = _SCORE_RE.search(text)
    score = 0.0
    if match is not None:
        try:
            score = max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            score = 0.0
    return CriticOutput(
        issues=[text.strip()] if text.strip() else [],
        score=score,
        summary="",
    )
