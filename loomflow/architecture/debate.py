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
3. **Optional convergence check.** If every debater in a round
   gives the same final answer, terminate early. Defaults on;
   disable for adversarial-only debates where you want full rounds.
4. **Judge synthesis.** A separate ``judge`` :class:`Agent`
   receives the full transcript and produces the final answer.
   ``judge=None`` falls back to majority vote over the final round
   (Du et al.): the most common final answer wins, ties going to the
   lowest-numbered debater.

Debaters are asked to end with a ``Final answer: ...`` line; the vote
and the convergence check compare those answers (normalised: case,
whitespace, surrounding punctuation, ``**bold**`` and ``\boxed{}``).
A reply without one is compared as a whole — by Jaccard similarity of
its words for convergence (``convergence_similarity``).

``rounds`` counts debate rounds *after* the independent round 0, so
each debater answers ``rounds + 1`` times; Du et al.'s ``rounds=2``
(one round of debate) is ``rounds=1`` here.

A debater that raises or is interrupted (its own ``max_turns``,
budget) sits that round out (``debate.debater_failed``); the debate
fails only when no debater answers. Each debater's and the judge's
spend is charged to the coordinator's budget.

Session ids
-----------
Each debater invocation uses ``{parent}__debater_<i>_round_<r>``; the
judge uses ``{parent}__judge``. When the debaters run on a durable
runtime, resuming a crashed debate replays their finished work.

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

import re
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import anyio
from anyio.streams.memory import MemoryObjectSendStream

from ..core.context import inherit_ambient_memory
from ..core.streaming import relay
from ..core.types import Event
from .base import AgentSession, Dependencies
from .helpers import (
    SubagentInvocation,
    budget_gate,
    consume_worker_usage,
    usage_from_result_dict,
)

if TYPE_CHECKING:
    from ..agent.api import Agent


_DebateRegistryT = dict[str, Any]  # worker handle registry
_DebateRoleMapT = dict[str, str]   # debater_<i>|judge → worker_id


DEFAULT_DEBATER_INSTRUCTIONS = """\
You are participating in a structured debate. Other debaters have
proposed answers (shown below). Your task on this round:

1. State your position clearly.
2. Address each other debater's argument: where you agree, where
   you disagree, and why.
3. If a counter-argument is convincing, update your position
   openly — don't be stubborn for its own sake.
4. Cite specifics; avoid hand-waving.

End with a line of the form "Final answer: <your answer>" for THIS
round.
"""

_ROUND0_SUFFIX = (
    "Provide your independent answer with reasoning. End with a line of "
    'the form "Final answer: <your answer>".'
)


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
        worker_registry: _DebateRegistryT | None = None,
        role_to_worker_id: _DebateRoleMapT | None = None,
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
        # Persistent-subagent wiring — when Team.debate built us with
        # ``persistent_subagents=True``, debater_<i> and judge each
        # run under their registered handle's stable session_id, so
        # the same debaters carry conversation memory across multiple
        # Agent.run() invocations (e.g. follow-up questions on the
        # same topic).
        self._worker_registry = worker_registry
        self._role_to_worker_id = role_to_worker_id

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
        # Memory propagation: install the coordinator's memory as
        # ambient for the entire debate. Each spawn site below
        # (debaters via parallel task-group, judge via SubagentInvocation)
        # inherits this contextvar through anyio's task-context
        # inheritance + async-generator semantics. Workers with
        # ``_memory_was_explicit=True`` ignore the ambient; the rest
        # pick it up via :meth:`Agent._resolve_run_memory`.
        with inherit_ambient_memory(deps.memory):
            async for ev in self._run_inner(session, deps, prompt):
                yield ev

    async def _run_inner(
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
            session, deps, prompt, history=[], round_num=0, responses=round0,
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
            session.output = _majority_vote(round0)
            return

        # === Debate rounds ===
        for r in range(1, self._rounds + 1):
            blocked, gate_events = await budget_gate(deps, session)
            for gate_event in gate_events:
                yield gate_event
            if blocked:
                return

            yield Event.architecture_event(
                session.id,
                "debate.round_started",
                round=r,
                phase="debate",
            )
            round_responses: dict[str, str] = {}
            async for ev in self._run_round_parallel(
                session, deps, prompt, history=history, round_num=r,
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
        from ..agent.worker_registry import (
            CrossUserWorkerError,
            acquire_worker_session,
            resolve_persistent_session,
        )
        judge_sid, judge_handle = resolve_persistent_session(
            "judge",
            fallback=f"{session.id}__judge",
            registry=self._worker_registry,
            role_to_id=self._role_to_worker_id,
        )
        judge_inv = SubagentInvocation(
            self._judge,
            judge_prompt,
            session_id=judge_sid,
            rollup_into=session,
        )
        if judge_handle is not None:
            try:
                async with acquire_worker_session(
                    judge_handle, deps.context.user_id
                ):
                    async for ev in judge_inv.events():
                        yield ev
            except CrossUserWorkerError as exc:
                # Cross-tenant judge can't synthesize — degrade to
                # majority vote over the final round.
                yield Event.architecture_event(
                    session.id,
                    "debate.cross_user_rejected",
                    debater="judge",
                    error=str(exc),
                )
                final = _majority_vote(history[-1])
                session.output = final
                yield Event.architecture_event(
                    session.id,
                    "debate.synthesized",
                    method="majority_vote",
                    final=final[:300],
                )
                return
        else:
            async for ev in judge_inv.events():
                yield ev
        judge_result = judge_inv.result
        session.turns += int(judge_result.get("turns", 0) or 0)
        await consume_worker_usage(
            deps, self._judge, usage_from_result_dict(judge_result)
        )

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
        deps: Dependencies,
        prompt: str,
        history: list[dict[str, str]],
        round_num: int,
        responses: dict[str, str],
    ) -> AsyncIterator[Event]:
        """Run all debaters for a single round in parallel and yield
        their streaming events.

        ``responses`` is filled in debater order (not completion
        order) with each answering debater's final output. Events
        from each debater's own iteration (model chunks, tool calls,
        nested architecture events) flow through this generator so
        token-level streaming surfaces in ``agent.stream(...)``.
        Raises when no debater answered.
        """
        answered: dict[str, str] = {}
        failures: dict[str, BaseException] = {}
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
                            deps,
                            round_num,
                            answered,
                            failures,
                            send.clone(),
                            self._worker_registry,
                            self._role_to_worker_id,
                        )

        # The debaters' task group lives in its own task
        # (core/streaming.py), so no cancel scope is held across ``yield``.
        async with aclosing(relay(_dispatch_all, receive)) as events:
            async for ev in events:
                yield ev
        for i in range(len(self._debaters)):
            name = f"debater_{i}"
            if name in answered:
                responses[name] = answered[name]
        if not responses:
            first = next(iter(failures.values()), None)
            raise RuntimeError(
                f"debate round {round_num}: no debater answered"
                + (f" ({first})" if first is not None else "")
            ) from first

    def _build_debater_prompt(
        self,
        original_prompt: str,
        history: list[dict[str, str]],
        debater_name: str,
        round_num: int,
    ) -> str:
        if round_num == 0:
            return f"{original_prompt}\n\n{_ROUND0_SUFFIX}"
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
    deps: Dependencies,
    round_num: int,
    answered: dict[str, str],
    failures: dict[str, BaseException],
    send: MemoryObjectSendStream[Event],
    worker_registry: _DebateRegistryT | None,
    role_to_worker_id: _DebateRoleMapT | None,
) -> None:
    """Single-debater worker for parallel dispatch: run the debater
    via :class:`SubagentInvocation`, forward its events into ``send``,
    record its final output in ``answered[name]`` and charge its spend
    to the coordinator's budget.

    When ``worker_registry`` is non-None and the debater has a
    registered handle, use the handle's stable session_id so the
    debater carries memory across rounds AND across multiple
    ``Agent.run()`` invocations. :func:`acquire_worker_session`
    does the cross-user check + per-handle lock + touch with the
    RUN's real ``user_id``. A debater that raises, is interrupted, or
    is refused as cross-user sits the round out: an event says why
    and the round goes on without it.
    """
    async with send:
        # Lazy import — circular if hoisted to module scope (the
        # worker_registry module imports Agent which imports us
        # through architecture/__init__.py).
        from ..agent.worker_registry import (
            CrossUserWorkerError,
            acquire_worker_session,
            resolve_persistent_session,
        )
        sub_session_id, handle = resolve_persistent_session(
            name,
            fallback=f"{session.id}__{name}_round_{round_num}",
            registry=worker_registry,
            role_to_id=role_to_worker_id,
        )
        invocation = SubagentInvocation(
            debater,
            debater_prompt,
            session_id=sub_session_id,
            rollup_into=session,
        )

        async def failed(event: str, exc: BaseException | None, error: str) -> None:
            if exc is not None:
                failures[name] = exc
            await send.send(
                Event.architecture_event(
                    session.id, event, round=round_num, debater=name, error=error
                )
            )

        try:
            if handle is not None:
                async with acquire_worker_session(handle, deps.context.user_id):
                    async for ev in invocation.events():
                        await send.send(ev)
            else:
                async for ev in invocation.events():
                    await send.send(ev)
        except CrossUserWorkerError as exc:
            await failed("debate.cross_user_rejected", exc, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — one debater's failure isn't the debate's
            await failed("debate.debater_failed", exc, f"{type(exc).__name__}: {exc}")
            return
        result = invocation.result
        session.turns += int(result.get("turns", 0) or 0)
        await consume_worker_usage(deps, debater, usage_from_result_dict(result))
        if result.get("interrupted"):
            await failed(
                "debate.debater_failed",
                None,
                f"interrupted: {result.get('interruption_reason') or 'unknown'}",
            )
            return
        answered[name] = str(result.get("output", ""))


# ---------------------------------------------------------------------------
# Convergence + majority vote
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Whitespace-normalize for naive convergence detection."""
    return " ".join(text.split()).strip().lower()


_FINAL_ANSWER_RE = re.compile(
    r"\bfinal\s+answer\b\s*(?:\*\*|__)?\s*(?:[:=\-–—]|\bis\b)\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def final_answer(text: str) -> str | None:
    """The normalised answer from a reply's last ``Final answer: ...``
    line (or ``\\boxed{...}``), or ``None`` when it has neither."""
    matches = _FINAL_ANSWER_RE.findall(text)
    if matches:
        answer = matches[-1]
    else:
        boxed = _BOXED_RE.findall(text)
        if not boxed:
            return None
        answer = boxed[-1]
    boxed_inner = _BOXED_RE.findall(answer)
    if boxed_inner:
        answer = boxed_inner[-1]
    answer = answer.replace("**", "").replace("__", "")
    return _normalize(answer).strip(" .,;:!\"'`$")


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
    """True when the round agrees: every reply has the same final
    answer (see :func:`final_answer`), or — when some reply has none —
    every pair's word-level Jaccard similarity is at least
    ``threshold`` (``1.0`` = whitespace-normalised equality).

    Comparing answers rather than whole replies keeps two long,
    similarly worded arguments for *different* answers ("Final
    answer: 42" / "Final answer: 48") from counting as agreement.
    """
    if not round_responses:
        return False
    if len(round_responses) == 1:
        return True
    responses = list(round_responses.values())
    answers = [final_answer(r) for r in responses]
    if all(a is not None for a in answers):
        return len(set(answers)) == 1
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            if _jaccard(responses[i], responses[j]) < threshold:
                return False
    return True


def _majority_vote(round_responses: dict[str, str]) -> str:
    """The reply whose final answer is the most common; ties go to the
    answer given first in debater order.

    Replies are grouped by :func:`final_answer` (whole normalised text
    when a reply has none); the first reply in the winning group is
    returned unchanged.
    """
    if not round_responses:
        return ""
    first_reply: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for resp in round_responses.values():
        key = final_answer(resp) or _normalize(resp)
        counts[key] += 1
        first_reply.setdefault(key, resp)
    # Counter.most_common keeps insertion order among equal counts.
    winner, _votes = counts.most_common(1)[0]
    return first_reply[winner]
