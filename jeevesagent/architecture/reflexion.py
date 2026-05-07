"""Reflexion: verbal reinforcement learning via memory.

Shinn et al. 2023 — `Reflexion: Language Agents with Verbal
Reinforcement Learning <https://arxiv.org/abs/2303.11366>`_. After
each attempt, an evaluator scores the output. Below threshold, a
reflector produces a single-sentence "lesson" — written advice the
agent can read on its next attempt.

Lesson storage modes
--------------------

Two storage modes for the persisted lessons:

* **Monotonic block (legacy default).** Every lesson is appended
  to ``memory.<lessons_block_name>`` and shown to the agent on
  every subsequent attempt. Simple but bloats context as lessons
  accumulate.
* **Selective recall (recommended).** Pass ``lesson_store=`` a
  :class:`VectorStore`. Lessons are stored as embedded chunks;
  before each attempt, only the **top-k most relevant lessons**
  for the current task are retrieved and surfaced. Avoids
  context bloat and keeps tutorial advice scoped to where it
  applies. Pair with :class:`InMemoryVectorStore` for in-process,
  or :class:`PostgresVectorStore` for cross-session learning.

Pattern
-------

For each attempt up to ``max_attempts``:

1. **Recall** (selective-recall mode only): query ``lesson_store``
   with the current prompt; write the top-k results into the
   working memory block for this attempt.
2. **Run base architecture** (default
   :class:`~jeevesagent.architecture.ReAct`).
3. **Evaluate.** A text-only model call scores the output (0-1).
4. **Threshold check.** If ``score >= threshold``, terminate.
5. **Max-attempts check.** If we've hit the cap, terminate.
6. **Reflect.** A text-only model call produces a single sentence
   identifying what went wrong.
7. **Persist.** Append (legacy) or add to ``lesson_store``
   (selective-recall) — keyed by the failing prompt so future
   recall can find it.
8. **Reset.** Clear ``session.messages`` so the base re-seeds its
   context. Cumulative usage and turn count carry across attempts.

Strengths
---------
* **Cross-session learning** when paired with a persistent
  memory backend (legacy) or a persistent vector store
  (selective recall).
* **Wraps any base** that reads ``memory.working()``.
* **Cheap**: 1 evaluator + 1 reflector call per failed attempt.

Weaknesses
----------
* **Same-model evaluation.** Self-grading is biased; the score
  may not match human judgment.
* **Score parsing is best-effort.** Falls back to 0.0 on parse
  failure (treated as a failed attempt).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..core.types import Event, Message, Role
from ..loader.base import Chunk
from .base import AgentSession, Architecture, Dependencies
from .helpers import add_usage, parse_score, text_only_model_call
from .react import ReAct

if TYPE_CHECKING:
    from ..agent.api import Agent
    from ..vectorstore.base import VectorStore


DEFAULT_EVALUATOR_PROMPT = """\
You are an evaluator scoring an agent's output against a task.

Score the output from 0.0 (completely failed) to 1.0 (fully
successful). Be calibrated:
- 1.0 = task is fully solved with no issues
- 0.7-0.9 = mostly correct, minor gaps
- 0.4-0.6 = partially correct, significant gaps
- 0.0-0.3 = wrong or missing key components

Output exactly one line in this format:
score: <number between 0 and 1>

Then on subsequent lines, briefly justify the score. The first line
must match the score format exactly so it can be parsed."""


DEFAULT_REFLECTOR_PROMPT = """\
You are a reflector that produces lessons for an agent that just
fell short on a task.

Read the original task and the agent's failed attempt. Produce ONE
sentence describing the most important thing the agent should do
differently next time. Be specific and concrete:
- Bad: "Be more careful."
- Good: "When asked to extract dates, always normalize to ISO 8601
  format before returning."

Output ONLY the single sentence — no preamble, no list."""


class Reflexion:
    """Wrap a base architecture with evaluator + reflector + lesson
    memory.

    See module docstring for the full mechanism. Constructor
    parameters:

    * ``base`` — architecture to retry. Default :class:`ReAct`.
    * ``max_attempts`` — cap on retries within a single run.
      Default 3.
    * ``threshold`` — minimum evaluator score to terminate as
      success. Default 0.8.
    * ``evaluator_prompt`` / ``reflector_prompt`` — override the
      default system prompts.
    * ``lessons_block_name`` — memory working-block name for
      persisted lessons. Default ``"reflexion_lessons"``. Multiple
      Reflexion-wrapped agents in the same memory should pick
      distinct names.
    * ``lesson_store`` — optional :class:`VectorStore` enabling
      selective recall. When set, lessons are stored as embedded
      chunks and only the top-``top_k_lessons`` most relevant
      lessons are surfaced on each attempt (instead of all past
      lessons). Avoids context bloat as lessons accumulate.
    * ``top_k_lessons`` — how many lessons to recall per attempt
      (selective-recall mode only). Default 5.
    """

    name = "reflexion"

    def __init__(
        self,
        *,
        base: Architecture | None = None,
        max_attempts: int = 3,
        threshold: float = 0.8,
        evaluator_prompt: str | None = None,
        reflector_prompt: str | None = None,
        lessons_block_name: str = "reflexion_lessons",
        lesson_store: VectorStore | None = None,
        top_k_lessons: int = 5,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if top_k_lessons < 1:
            raise ValueError("top_k_lessons must be >= 1")
        self._base: Architecture = base if base is not None else ReAct()
        self._max_attempts = max_attempts
        self._threshold = threshold
        self._evaluator_prompt = evaluator_prompt or DEFAULT_EVALUATOR_PROMPT
        self._reflector_prompt = reflector_prompt or DEFAULT_REFLECTOR_PROMPT
        self._lessons_block = lessons_block_name
        self._lesson_store = lesson_store
        self._top_k = top_k_lessons

    def declared_workers(self) -> dict[str, Agent]:
        return {}

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        for attempt in range(1, self._max_attempts + 1):
            yield Event.architecture_event(
                session.id,
                "reflexion.attempt_started",
                attempt=attempt,
                max_attempts=self._max_attempts,
            )

            # Selective recall: when a lesson_store is configured,
            # query for lessons relevant to THIS prompt and write
            # them to the working memory block. The block is
            # rewritten (not appended) so the agent only sees the
            # top-k relevant lessons, not the full lesson history.
            if self._lesson_store is not None:
                hits = await self._lesson_store.search(
                    prompt, k=self._top_k
                )
                if hits:
                    bullets = "\n".join(
                        f"- {r.chunk.content}" for r in hits
                    )
                    await deps.memory.update_block(
                        self._lessons_block, bullets
                    )
                    yield Event.architecture_event(
                        session.id,
                        "reflexion.lessons_recalled",
                        attempt=attempt,
                        n_recalled=len(hits),
                    )

            # Each attempt is a fresh seed: clear messages so the
            # base re-runs seed_context, which will pick up lessons
            # from memory.working() automatically.
            session.messages = []

            async for event in self._base.run(session, deps, prompt):
                yield event

            if session.interrupted:
                # Base architecture interrupted itself (max_turns,
                # budget). Don't reflect on a partial output.
                return

            # --- Evaluate ---
            score = await self._evaluate(deps, session, prompt, attempt)
            yield Event.architecture_event(
                session.id,
                "reflexion.evaluated",
                attempt=attempt,
                score=score,
            )

            if score >= self._threshold:
                yield Event.architecture_event(
                    session.id,
                    "reflexion.threshold_met",
                    attempt=attempt,
                    score=score,
                )
                return

            if attempt >= self._max_attempts:
                yield Event.architecture_event(
                    session.id,
                    "reflexion.max_attempts_reached",
                    final_score=score,
                    attempts=attempt,
                )
                return

            # --- Reflect → produce a lesson ---
            lesson = await self._reflect(
                deps, session, prompt, attempt, score
            )
            yield Event.architecture_event(
                session.id,
                "reflexion.lesson_produced",
                attempt=attempt,
                lesson=lesson,
            )

            # --- Persist the lesson.
            #
            # Selective-recall mode: write to the vector store with
            # the failing prompt as metadata so future recalls of
            # similar prompts surface this lesson. The store handles
            # the embedding internally.
            #
            # Legacy mode: append to the memory working block.
            if self._lesson_store is not None:
                await self._lesson_store.add(
                    [
                        Chunk(
                            content=lesson,
                            metadata={
                                "attempt": attempt,
                                "score": score,
                                "prompt_excerpt": prompt[:200],
                            },
                        )
                    ]
                )
                persist_target = "lesson_store"
            else:
                await deps.memory.append_block(
                    self._lessons_block, f"- {lesson}"
                )
                persist_target = self._lessons_block
            yield Event.architecture_event(
                session.id,
                "reflexion.lesson_persisted",
                attempt=attempt,
                block=persist_target,
            )

    # ---- helpers ---------------------------------------------------------

    async def _evaluate(
        self,
        deps: Dependencies,
        session: AgentSession,
        prompt: str,
        attempt: int,
    ) -> float:
        msgs = [
            Message(role=Role.SYSTEM, content=self._evaluator_prompt),
            Message(
                role=Role.USER,
                content=(
                    f"Task:\n{prompt}\n\n"
                    f"Agent output (attempt {attempt}):\n{session.output}"
                ),
            ),
        ]
        text, usage = await text_only_model_call(
            deps, f"reflexion_eval_{attempt}", msgs
        )
        await deps.budget.consume(
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )
        session.cumulative_usage = add_usage(
            session.cumulative_usage, usage
        )
        session.turns += 1
        return parse_score(text)

    async def _reflect(
        self,
        deps: Dependencies,
        session: AgentSession,
        prompt: str,
        attempt: int,
        score: float,
    ) -> str:
        msgs = [
            Message(role=Role.SYSTEM, content=self._reflector_prompt),
            Message(
                role=Role.USER,
                content=(
                    f"Task:\n{prompt}\n\n"
                    f"Failed attempt (score {score:.2f}):\n{session.output}\n\n"
                    f"Produce one sentence of advice for the next attempt."
                ),
            ),
        ]
        text, usage = await text_only_model_call(
            deps, f"reflexion_reflect_{attempt}", msgs
        )
        await deps.budget.consume(
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )
        session.cumulative_usage = add_usage(
            session.cumulative_usage, usage
        )
        session.turns += 1
        return text.strip()


# ---------------------------------------------------------------------------
# Score-parsing alias (kept for backwards compat with tests that
# import ``_parse_score`` from this module).
# ---------------------------------------------------------------------------

_parse_score = parse_score
