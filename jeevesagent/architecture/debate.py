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
from anyio.streams.memory import MemoryObjectSendStream

from ..core.types import Event
from .base import AgentSession, Dependencies
from .helpers import SubagentInvocation

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
        convergence_similarity: float = 0.85,
        debater_instructions: str | None = None,
        judge_instructions: str | None = None,
    ) -> None:
        if len(debaters) < 2:
            raise ValueError("Debate requires at least 2 debaters")
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        if not 0.0 <= convergence_similarity <= 1.0:
            raise ValueError(
                "convergence_similarity must be in [0.0, 1.0]"
            )
        self._debaters = list(debaters)
        self._judge = judge
        self._rounds = rounds
        self._convergence_check = convergence_check
        # 0.85 = "essentially the same answer, possibly different
        # wording" — empirically the sweet spot for debate convergence.
        # 1.0 reproduces the legacy strict-equality behaviour. Lower
        # values are more aggressive (cuts cost, risks premature exit).
        self._convergence_similarity = convergence_similarity
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
        round0: dict[str, str] = {}
        async for ev in self._run_round_parallel(
            session, prompt, history=[], round_num=0, responses=round0
        ):
            yield ev
        history.append(round0)
        for name, resp in round0.items():
            yield Event.architecture_event(
                session.id,
                "debate.response",
                round=0,
                debater=name,
                response=resp[:300],
            )

        if self._convergence_check and _converged(
            round0, threshold=self._convergence_similarity
        ):
            yield Event.architecture_event(
                session.id,
                "debate.converged",
                round=0,
                threshold=self._convergence_similarity,
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
            round_responses: dict[str, str] = {}
            async for ev in self._run_round_parallel(
                session, prompt, history=history, round_num=r,
                responses=round_responses,
            ):
                yield ev
            history.append(round_responses)
            for name, resp in round_responses.items():
                yield Event.architecture_event(
                    session.id,
                    "debate.response",
                    round=r,
                    debater=name,
                    response=resp[:300],
                )

            if self._convergence_check and _converged(
                round_responses,
                threshold=self._convergence_similarity,
            ):
                yield Event.architecture_event(
                    session.id,
                    "debate.converged",
                    round=r,
                    threshold=self._convergence_similarity,
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
        judge_inv = SubagentInvocation(
            self._judge,
            judge_prompt,
            session_id=f"{session.id}__judge",
        )
        async for ev in judge_inv.events():
            yield ev
        judge_result = judge_inv.result
        session.turns += int(judge_result.get("turns", 0) or 0)

        if bool(judge_result.get("interrupted", False)):
            session.interrupted = True
            session.interruption_reason = (
                f"judge:{judge_result.get('interruption_reason') or 'unknown'}"
            )
            return

        judge_output = str(judge_result.get("output", ""))
        session.output = judge_output
        yield Event.architecture_event(
            session.id,
            "debate.synthesized",
            method="judge",
            final=judge_output[:300],
        )

    # ---- helpers ---------------------------------------------------------

    async def _run_round_parallel(
        self,
        session: AgentSession,
        prompt: str,
        history: list[dict[str, str]],
        round_num: int,
        responses: dict[str, str],
    ) -> AsyncIterator[Event]:
        """Run all debaters for a single round in parallel and yield
        their streaming events.

        ``responses`` is mutated in place — each debater's final
        output is written to ``responses[debater_name]``. Events
        from each debater's own iteration (model chunks, tool calls,
        nested architecture events) flow through this generator so
        token-level streaming surfaces in ``agent.stream(...)``.
        """
        send, receive = anyio.create_memory_object_stream[Event](
            max_buffer_size=128
        )

        async def _dispatch_all() -> None:
            async with send:
                async with anyio.create_task_group() as inner_tg:
                    for i, debater in enumerate(self._debaters):
                        name = f"debater_{i}"
                        debater_prompt = self._build_debater_prompt(
                            prompt, history, name, round_num
                        )
                        inner_tg.start_soon(
                            _run_one_debater_streaming,
                            debater,
                            name,
                            debater_prompt,
                            session,
                            round_num,
                            responses,
                            send.clone(),
                        )

        async with anyio.create_task_group() as outer_tg:
            outer_tg.start_soon(_dispatch_all)
            async with receive:
                async for ev in receive:
                    yield ev

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


async def _run_one_debater_streaming(
    debater: Agent,
    name: str,
    debater_prompt: str,
    session: AgentSession,
    round_num: int,
    responses: dict[str, str],
    send: MemoryObjectSendStream[Event],
) -> None:
    """Single-debater worker for parallel dispatch: run the debater
    via :class:`SubagentInvocation`, forward its events into ``send``,
    write final output into ``responses[name]``."""
    async with send:
        sub_session_id = f"{session.id}__{name}_round_{round_num}"
        invocation = SubagentInvocation(
            debater, debater_prompt, session_id=sub_session_id
        )
        async for ev in invocation.events():
            await send.send(ev)
        responses[name] = str(invocation.result.get("output", ""))
        session.turns += int(invocation.result.get("turns", 0) or 0)


# ---------------------------------------------------------------------------
# Convergence + majority vote
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Whitespace-normalize for naive convergence detection."""
    return " ".join(text.split()).strip().lower()


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity in [0, 1].

    Whitespace-normalized + lowercased on both sides. Empty/empty
    is 1.0 (vacuously equal); empty/non-empty is 0.0. No external
    dependencies — simple, transparent, deterministic.
    """
    tokens_a = set(_normalize(a).split())
    tokens_b = set(_normalize(b).split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def _converged(
    round_responses: dict[str, str], *, threshold: float = 0.85
) -> bool:
    """Pairwise Jaccard similarity ≥ ``threshold`` for every pair.

    With ``threshold=1.0`` this reduces to strict (whitespace-
    normalized) equality — the legacy v0.5 behaviour. The default
    ``0.85`` allows minor wording differences ("the answer is 42"
    vs "answer: 42"). Lower values terminate debate earlier and
    cut cost; the trade-off is risking premature exit on disputes
    where debaters technically disagree but use overlapping
    vocabulary.
    """
    if not round_responses:
        return False
    if len(round_responses) == 1:
        return True
    responses = list(round_responses.values())
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            if _jaccard(responses[i], responses[j]) < threshold:
                return False
    return True


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
