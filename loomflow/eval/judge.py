"""LLM-as-judge metric.

:class:`LLMJudge` asks a (separate!) model to grade the agent's output
against the case input, optional ground truth, and an optional rubric.
The judge must reply with an explicit ``score: <0-1>`` line — free-form
numbers in prose are deliberately NOT accepted (same discipline as
:func:`loomflow.architecture.helpers.parse_score`, but stricter: this
judge requires the labelled line). On a parse failure the judge retries
once with corrective feedback; on a second failure it warns and returns
the neutral score 0.5 rather than silently reporting 0.0 (which would
conflate "judge misbehaved" with "agent failed").

Anti-pattern guard: grading a model's output with the *same model
instance* that produced it inflates scores (self-preference bias).
:class:`~loomflow.eval.EvalHarness` warns when it detects
``judge.model is agent.model``.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

from ..core.protocols import Model
from ..core.types import Event, Message, Role, RunResult
from .dataset import Case

__all__ = ["LLMJudge"]

# An explicit, labelled score line: ``score: 0.7`` / ``Score = 1``.
# Anchored per-line; prose numbers never match.
_SCORE_LINE_RE = re.compile(r"(?im)^\s*score\s*[:=]\s*([01](?:\.\d+)?|\.\d+)\s*$")

_DEFAULT_RUBRIC = (
    "Judge whether the agent's answer correctly and completely addresses "
    "the task. Penalise factual errors, ignored instructions, and "
    "irrelevant content. 1.0 = fully correct and complete; 0.0 = wrong "
    "or off-task."
)

_RETRY_NUDGE = (
    "Your previous reply did not contain a parseable score line. Reply "
    "again, ending with exactly one line of the form 'score: <number "
    "between 0 and 1>'."
)


def _parse_score_line(text: str) -> float | None:
    """Extract the score from an explicit ``score:`` line, or None."""
    matches = _SCORE_LINE_RE.findall(text)
    if not matches:
        return None
    try:
        value = float(matches[-1])  # last line wins if the judge rambled
    except ValueError:  # pragma: no cover - regex guarantees a float
        return None
    return max(0.0, min(1.0, value))


async def _complete_text(model: Model, messages: list[Message]) -> str:
    """Call the judge model, preferring ``complete()`` over draining ``stream()``."""
    complete = getattr(model, "complete", None)
    if callable(complete):
        text, _calls, _usage, _finish = await complete(messages)
        return str(text)
    parts: list[str] = []
    async for chunk in model.stream(messages):
        if chunk.kind == "text" and chunk.text:
            parts.append(chunk.text)
    return "".join(parts)


class LLMJudge:
    """Grade agent output with a judge model against a rubric.

    ``model`` is any loomflow :class:`~loomflow.core.protocols.Model`.
    ``rubric`` overrides the default grading instructions. The judge
    prompt carries the case input, the agent's output, and — when the
    case has ground truth — the expected answer as reference.

    Parsing contract: the judge must emit an explicit ``score: X``
    line with ``X`` in ``[0, 1]``. One retry with corrective feedback
    on parse failure; a second failure yields ``neutral_score``
    (default 0.5) plus a :class:`UserWarning`.
    """

    name = "llm_judge"

    def __init__(
        self,
        model: Model,
        rubric: str | None = None,
        *,
        neutral_score: float = 0.5,
    ) -> None:
        self.model = model
        self.rubric = rubric if rubric is not None else _DEFAULT_RUBRIC
        self.neutral_score = neutral_score

    def _messages(self, case: Case, result: RunResult) -> list[Message]:
        reference = (
            f"\n\nReference (expected) answer:\n{case.expected}"
            if case.expected is not None
            else ""
        )
        system = (
            "You are an impartial evaluator grading an AI agent's answer.\n"
            f"Rubric: {self.rubric}\n"
            "After your (brief) reasoning, end your reply with exactly one "
            "line of the form 'score: <number between 0 and 1>'."
        )
        user = (
            f"Task given to the agent:\n{case.input}\n\nAgent's answer:\n{result.output}{reference}"
        )
        return [
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content=user),
        ]

    async def score(self, case: Case, result: RunResult, events: list[Event]) -> float:
        messages = self._messages(case, result)
        reply = await _complete_text(self.model, messages)
        parsed = _parse_score_line(reply)
        if parsed is not None:
            return parsed
        # One corrective retry, carrying the failed reply for context.
        retry_messages = [
            *messages,
            Message(role=Role.ASSISTANT, content=reply),
            Message(role=Role.USER, content=_RETRY_NUDGE),
        ]
        reply = await _complete_text(self.model, retry_messages)
        parsed = _parse_score_line(reply)
        if parsed is not None:
            return parsed
        warnings.warn(
            f"LLMJudge could not parse a 'score:' line from the judge model "
            f"after one retry (case {case.id}); returning neutral score "
            f"{self.neutral_score}. Last reply: {reply[:200]!r}",
            UserWarning,
            stacklevel=2,
        )
        return self.neutral_score


def warn_if_same_model(judge: Any, agent_model: Any) -> None:
    """Warn on the same-model-judge anti-pattern (shared *instance*).

    Called by :class:`~loomflow.eval.EvalHarness` at construction for
    every metric that exposes a ``model`` attribute. Identity (not
    equality) comparison: two separate connections to the same
    provider model are fine-ish; literally reusing the agent's Model
    object is the unambiguous smell we can detect.
    """
    judge_model = getattr(judge, "model", None)
    if judge_model is not None and agent_model is not None and judge_model is agent_model:
        warnings.warn(
            "LLMJudge is using the SAME Model instance as the agent under "
            "evaluation (same-model-judge anti-pattern): self-grading "
            "inflates scores via self-preference bias. Use a different "
            "model (ideally a different family) as the judge.",
            UserWarning,
            stacklevel=3,
        )
