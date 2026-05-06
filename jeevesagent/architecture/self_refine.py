"""Self-Refine: iterative refinement via critique.

Madaan et al. 2023 — `Self-Refine: Iterative Refinement with
Self-Feedback <https://arxiv.org/abs/2303.17651>`_. The same model
plays generator, critic, and refiner across rounds.

Pattern:

1. **Round 0 (generator).** Run the base architecture (default
   :class:`~jeevesagent.architecture.ReAct`) to produce an initial
   output.
2. **Round 1..max_rounds:**

   a. **Critic.** Review the current output. Output a critique. If
      the critique contains ``stop_phrase`` (default ``"no issues"``)
      anywhere, terminate — the model considers the output good
      enough.
   b. **Refiner.** Produce a revised output that addresses the
      critique. The new output replaces the old.

When the same model plays all three roles, gains are real but
modest (~5-15% on most tasks). For asymmetric blind-spot coverage,
use ``ActorCritic`` with a different model for the critic — that
ships in a later release.

Strengths
---------
* Simple, well-defined; one new architecture wraps any base.
* Cheap relative to multi-agent debate or actor-critic with
  separate models.
* Composes with :class:`~jeevesagent.architecture.ReAct` (default
  ``base``) and any other architecture that satisfies the protocol.

Weaknesses
----------
* Same-model self-critique shares blind spots — bounded gains.
* 2-3× token cost vs single-pass.
* Latency adds up — sequential critic / refiner per round.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..core.types import Event, Message, Role
from .base import AgentSession, Architecture, Dependencies
from .helpers import add_usage, text_only_model_call
from .react import ReAct

if TYPE_CHECKING:
    from ..agent.api import Agent


DEFAULT_CRITIC_PROMPT = """\
You are a careful reviewer. Read the original task and the proposed
output. Identify every issue you can find: factual errors, missing
edge cases, unclear language, inconsistent reasoning, gaps versus
what the task asked for.

Be specific. Cite the relevant section of the output.

If you find no issues at all and the output fully satisfies the
task, respond with exactly:

  no issues

Otherwise, list the issues as a bulleted critique."""


DEFAULT_REFINER_PROMPT = """\
You are revising your previous output based on a critique. Address
every point in the critique. Preserve what was correct; fix what
was wrong; add what was missing. Output ONLY the revised version,
not commentary about the changes."""


class SelfRefine:
    """Wrap a base architecture with iterative critique / refine.

    ``base`` defaults to :class:`ReAct`; the round-0 generator runs
    the base architecture's full strategy. Subsequent rounds are
    text-only model calls — no tools, just critique and rewrite.
    """

    name = "self-refine"

    def __init__(
        self,
        *,
        base: Architecture | None = None,
        max_rounds: int = 3,
        critic_prompt: str | None = None,
        refiner_prompt: str | None = None,
        stop_phrase: str = "no issues",
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self._base: Architecture = base if base is not None else ReAct()
        self._max_rounds = max_rounds
        self._critic_prompt = critic_prompt or DEFAULT_CRITIC_PROMPT
        self._refiner_prompt = refiner_prompt or DEFAULT_REFINER_PROMPT
        self._stop_phrase = stop_phrase

    def declared_workers(self) -> dict[str, Agent]:
        return {}

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # === Round 0: initial generation via base architecture ===
        yield Event.architecture_event(
            session.id,
            "self_refine.round_started",
            round=0,
            role="generator",
        )
        async for event in self._base.run(session, deps, prompt):
            yield event

        # If the base architecture interrupted itself (max_turns,
        # budget, ...) don't refine — refining a partial result would
        # waste tokens and confuse downstream consumers.
        if session.interrupted:
            return

        # === Refinement rounds ===
        for round_num in range(1, self._max_rounds + 1):
            status = await deps.budget.allows_step()
            if status.blocked:
                session.interrupted = True
                session.interruption_reason = f"budget:{status.reason}"
                yield Event.budget_exceeded(session.id, status)
                return
            if status.warn:
                yield Event.budget_warning(session.id, status)

            # --- Critic ---
            yield Event.architecture_event(
                session.id,
                "self_refine.round_started",
                round=round_num,
                role="critic",
            )
            critic_messages = [
                Message(role=Role.SYSTEM, content=self._critic_prompt),
                Message(
                    role=Role.USER,
                    content=(
                        f"Original task:\n{prompt}\n\n"
                        f"Output to review:\n{session.output}"
                    ),
                ),
            ]
            critique, critic_usage = await text_only_model_call(
                deps, f"self_refine_critic_{round_num}", critic_messages
            )
            await deps.budget.consume(
                tokens_in=critic_usage.input_tokens,
                tokens_out=critic_usage.output_tokens,
                cost_usd=critic_usage.cost_usd,
            )
            session.cumulative_usage = add_usage(
                session.cumulative_usage, critic_usage
            )
            session.turns += 1

            yield Event.architecture_event(
                session.id,
                "self_refine.critique",
                round=round_num,
                critique=critique,
            )

            if self._stop_phrase.lower() in critique.lower():
                yield Event.architecture_event(
                    session.id,
                    "self_refine.converged",
                    round=round_num,
                )
                return

            # --- Refiner ---
            yield Event.architecture_event(
                session.id,
                "self_refine.round_started",
                round=round_num,
                role="refiner",
            )
            refiner_messages = [
                Message(role=Role.SYSTEM, content=self._refiner_prompt),
                Message(
                    role=Role.USER,
                    content=(
                        f"Original task:\n{prompt}\n\n"
                        f"Previous output:\n{session.output}\n\n"
                        f"Critique to address:\n{critique}\n\n"
                        f"Produce the revised output."
                    ),
                ),
            ]
            refined, refiner_usage = await text_only_model_call(
                deps, f"self_refine_refiner_{round_num}", refiner_messages
            )
            await deps.budget.consume(
                tokens_in=refiner_usage.input_tokens,
                tokens_out=refiner_usage.output_tokens,
                cost_usd=refiner_usage.cost_usd,
            )
            session.cumulative_usage = add_usage(
                session.cumulative_usage, refiner_usage
            )
            session.turns += 1

            session.output = refined
            yield Event.architecture_event(
                session.id,
                "self_refine.refined",
                round=round_num,
                output=refined,
            )

        # max_rounds reached without convergence — current output is
        # the best we have; let it stand.
        yield Event.architecture_event(
            session.id,
            "self_refine.max_rounds_reached",
            rounds=self._max_rounds,
        )
