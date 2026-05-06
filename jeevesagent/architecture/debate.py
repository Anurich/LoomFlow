"""Multi-Agent Debate: N debaters argue, judge synthesizes.

Du et al. 2023 — `Improving Factuality and Reasoning in Language
Models through Multiagent Debate <https://arxiv.org/abs/2305.14325>`_.
Liang et al. 2023 (divergent thinking via debate). Production
patterns in AutoGen GroupChat, CAMEL.

Reserve for **high-stakes contested questions where a wrong answer
is expensive**. The 2026 production literature is cautious — debate
adds 3-5× cost over single-agent and works best on a narrow set of
decision-style questions where blind-spot triangulation matters.

Pattern
-------

1. **Round 0 (independent).** All debaters answer the original
   question simultaneously, with no awareness of each other. Run
   in parallel.
2. **Rounds 1..K (debate).** Each debater receives the original
   question + the full transcript so far. They defend or update
   their position. All debaters in a round run in parallel.
3. **Optional convergence check.** If all debaters in a round
   produce exactly-matching answers (after whitespace normalize),
   terminate early. Defaults on; disable for adversarial-only
   debates where you want full rounds.
4. **Judge synthesis.** A separate ``judge`` :class:`Agent`
   receives the full transcript and produces the final answer.
   ``judge=None`` falls back to majority vote (modal answer
   across the final round; tie-broken by first appearance).

Deterministic session ids
-------------------------
Each debater invocation uses
``{parent}__debater_<i>_round_<r>``; the judge uses
``{parent}__judge``. Replays of the parent journal cache the
sub-results and don't re-execute debaters.

Strengths
---------
* **Surfaces blind spots through disagreement** — different priors
  produce different errors; debate transcribes them.
* **Strong on factuality.** TruthfulQA improvement over single-agent
  is well-documented (Du et al. 2023, +12% on 3 debaters / 2 rounds).
* **Heterogeneous-model friendly.** Use Claude + GPT + Llama for
  genuine prior diversity rather than 3× of the same model.

Weaknesses
----------
* **3-5× cost.** N debaters × K rounds + judge. Real money.
* **Sequential rounds.** Even with parallel debaters per round, you
  still wait round-by-round.
* **Groupthink risk.** Same model → same priors → no real
  disagreement. Differentiate models or personas.
* **Judge quality is critical.** Bad judge = bad final answer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import anyio

from ..core.types import Event
from .base import AgentSession, Dependencies

if TYPE_CHECKING:
    from ..agent.api import Agent


DEFAULT_DEBATER_INSTRUCTIONS = """\
You are participating in a structured debate. Other debaters have
proposed answers (shown below). Your task on this round:

1. State your position clearly.
2. Address each other debater's argument: where you agree, where
   you disagree, and why.
3. If a counter-argument is convincing, update your position
   openly — don't be stubborn for its own sake.
4. Cite specifics; avoid hand-waving.

End with a clear final answer for THIS round.
"""


DEFAULT_JUDGE_INSTRUCTIONS = """\
You are an impartial judge synthesizing a multi-agent debate. Read
the original question and the full debate transcript below, then
output the answer you believe is best supported by the arguments.

Output the final answer as plain text. Be decisive — pick the
strongest position even when debaters didn't fully agree.
"""


class MultiAgentDebate:
    """N debaters + optional judge orchestration."""

    name = "debate"

    def __init__(
        self,
        *,
        debaters: list[Agent],
        judge: Agent | None = None,
        rounds: int = 2,
        convergence_check: bool = True,
        debater_instructions: str | None = None,
        judge_instructions: str | None = None,
    ) -> None:
        if len(debaters) < 2:
            raise ValueError("Debate requires at least 2 debaters")
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        self._debaters = list(debaters)
        self._judge = judge
        self._rounds = rounds
        self._convergence_check = convergence_check
        self._debater_instructions = (
            debater_instructions or DEFAULT_DEBATER_INSTRUCTIONS
        )
        self._judge_instructions = (
            judge_instructions or DEFAULT_JUDGE_INSTRUCTIONS
        )

    def declared_workers(self) -> dict[str, Agent]:
        workers: dict[str, Agent] = {
            f"debater_{i}": d for i, d in enumerate(self._debaters)
        }
        if self._judge is not None:
            workers["judge"] = self._judge
        return workers

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # Each entry is {debater_name: response} for that round.
        history: list[dict[str, str]] = []

        # === Round 0: independent answers ===
        yield Event.architecture_event(
            session.id,
            "debate.round_started",
            round=0,
            phase="independent",
            num_debaters=len(self._debaters),
        )
        round0 = await self._run_round_parallel(
            session, prompt, history=[], round_num=0
        )
        history.append(round0)
        for name, resp in round0.items():
            yield Event.architecture_event(
                session.id,
                "debate.response",
                round=0,
                debater=name,
                response=resp[:300],
            )

        if self._convergence_check and _converged(round0):
            yield Event.architecture_event(
                session.id,
                "debate.converged",
                round=0,
            )
            session.output = next(iter(round0.values()))
            return

        # === Debate rounds ===
        for r in range(1, self._rounds + 1):
            status = await deps.budget.allows_step()
            if status.blocked:
                session.interrupted = True
                session.interruption_reason = (
                    f"budget:{status.reason}"
                )
                yield Event.budget_exceeded(session.id, status)
                return
            if status.warn:
                yield Event.budget_warning(session.id, status)

            yield Event.architecture_event(
                session.id,
                "debate.round_started",
                round=r,
                phase="debate",
            )
            round_responses = await self._run_round_parallel(
                session, prompt, history=history, round_num=r
            )
            history.append(round_responses)
            for name, resp in round_responses.items():
                yield Event.architecture_event(
                    session.id,
                    "debate.response",
                    round=r,
                    debater=name,
                    response=resp[:300],
                )

            if self._convergence_check and _converged(round_responses):
                yield Event.architecture_event(
                    session.id,
                    "debate.converged",
                    round=r,
                )
                break

        # === Synthesis ===
        if self._judge is None:
            final = _majority_vote(history[-1])
            session.output = final
            yield Event.architecture_event(
                session.id,
                "debate.synthesized",
                method="majority_vote",
                final=final[:300],
            )
            return

        yield Event.architecture_event(
            session.id,
            "debate.judging",
        )
        judge_prompt = self._build_judge_prompt(prompt, history)
        judge_result = await self._judge.run(
            judge_prompt,
            session_id=f"{session.id}__judge",
        )
        session.turns += judge_result.turns

        if judge_result.interrupted:
            session.interrupted = True
            session.interruption_reason = (
                f"judge:{judge_result.interruption_reason or 'unknown'}"
            )
            return

        session.output = judge_result.output
        yield Event.architecture_event(
            session.id,
            "debate.synthesized",
            method="judge",
            final=judge_result.output[:300],
        )

    # ---- helpers ---------------------------------------------------------

    async def _run_round_parallel(
        self,
        session: AgentSession,
        prompt: str,
        history: list[dict[str, str]],
        round_num: int,
    ) -> dict[str, str]:
        """Run all debaters for a single round in parallel.

        Each debater sees the same round-specific prompt; results
        come back keyed by debater name.
        """
        responses: dict[str, str] = {}

        async with anyio.create_task_group() as tg:
            for i, debater in enumerate(self._debaters):
                name = f"debater_{i}"
                debater_prompt = self._build_debater_prompt(
                    prompt, history, name, round_num
                )
                tg.start_soon(
                    self._run_one_debater,
                    debater,
                    name,
                    debater_prompt,
                    session,
                    round_num,
                    responses,
                )

        return responses

    async def _run_one_debater(
        self,
        debater: Agent,
        name: str,
        debater_prompt: str,
        session: AgentSession,
        round_num: int,
        responses: dict[str, str],
    ) -> None:
        sub_session_id = (
            f"{session.id}__{name}_round_{round_num}"
        )
        result = await debater.run(
            debater_prompt, session_id=sub_session_id
        )
        responses[name] = result.output
        # Accumulate turns into the parent for accurate accounting.
        session.turns += result.turns

    def _build_debater_prompt(
        self,
        original_prompt: str,
        history: list[dict[str, str]],
        debater_name: str,
        round_num: int,
    ) -> str:
        if round_num == 0:
            return (
                f"{original_prompt}\n\n"
                f"Provide your independent answer with reasoning."
            )
        transcript_lines = [
            self._debater_instructions,
            "",
            f"Original question:\n{original_prompt}",
            "",
            "Debate transcript so far:",
        ]
        for r_idx, round_dict in enumerate(history):
            transcript_lines.append(f"\n=== Round {r_idx} ===")
            for n, resp in round_dict.items():
                marker = " (you)" if n == debater_name else ""
                transcript_lines.append(f"{n}{marker}: {resp}")
        transcript_lines.append(
            f"\nIt is now round {round_num}. Defend or update "
            f"your position with reasoning."
        )
        return "\n".join(transcript_lines)

    def _build_judge_prompt(
        self,
        original_prompt: str,
        history: list[dict[str, str]],
    ) -> str:
        lines = [
            self._judge_instructions,
            "",
            f"Original question:\n{original_prompt}",
            "",
            "Debate transcript:",
        ]
        for r_idx, round_dict in enumerate(history):
            lines.append(f"\n=== Round {r_idx} ===")
            for n, resp in round_dict.items():
                lines.append(f"{n}: {resp}")
        lines.append("\nProduce the final answer.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convergence + majority vote
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Whitespace-normalize for naive convergence detection."""
    return " ".join(text.split()).strip().lower()


def _converged(round_responses: dict[str, str]) -> bool:
    """All debaters in this round agree (after whitespace normalize).

    Naive but conservative — false negatives are fine (debate
    continues an extra round); false positives would prematurely
    end debate. Production users wanting semantic convergence can
    write a custom architecture that subclasses
    :class:`MultiAgentDebate` and overrides this check, or pass
    ``convergence_check=False`` to disable.
    """
    if not round_responses:
        return False
    normalized = {_normalize(r) for r in round_responses.values()}
    return len(normalized) == 1


def _majority_vote(round_responses: dict[str, str]) -> str:
    """Pick the modal answer; ties broken by first appearance.

    Whitespace-normalized for matching, but the original (unnormalized)
    string is returned so casing / formatting is preserved.
    """
    if not round_responses:
        return ""
    normalized_to_original: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for resp in round_responses.values():
        norm = _normalize(resp)
        counts[norm] += 1
        if norm not in normalized_to_original:
            normalized_to_original[norm] = resp
    winner_norm, _votes = counts.most_common(1)[0]
    return normalized_to_original[winner_norm]
