"""LLM-as-judge metric.

:class:`LLMJudge` asks a (separate!) model to grade the agent's output
against the case input, optional ground truth, and an optional rubric.
The judge must reply with an explicit ``score: <0-1>`` line — free-form
numbers in prose are deliberately NOT accepted (same discipline as
:func:`loomflow.architecture.helpers.parse_score`, but stricter: this
judge requires the labelled line). On a parse failure the judge retries
once with corrective feedback; on a second failure it raises
:class:`JudgeParseError`, which the harness records as a metric error
for the case — "the judge misbehaved" is neither a score of 0.0 nor a
score of 0.5, and a gate must not pass on it.

What the judge model costs is counted: each call's usage is added to
the judge's own ``usage`` total and, inside an
:class:`~loomflow.eval.EvalHarness` run, to the case being scored (so
``CaseResult.cost_usd`` and ``max_cost_usd`` include judge spend). A
custom metric that calls a model reports its spend the same way with
:func:`record_metric_usage`.

Anti-pattern guard: grading a model's output with the *same model
instance* that produced it inflates scores (self-preference bias).
:class:`~loomflow.eval.EvalHarness` warns when it detects
``judge.model is agent.model``.
"""

from __future__ import annotations

import re
import warnings
from contextvars import ContextVar
from typing import Any

from ..core.protocols import Model
from ..core.types import Event, Message, Role, RunResult, Usage
from .dataset import Case

__all__ = ["JudgeParseError", "LLMJudge", "record_metric_usage"]


class JudgeParseError(ValueError):
    """The judge model's reply had no parseable ``score:`` line, even
    after the corrective retry."""


# Usage of the metric calls made while one case is being scored. The
# harness installs a fresh list per case (each case runs in its own
# task, so concurrent cases never share one).
_metric_usage_var: ContextVar[list[Usage] | None] = ContextVar(
    "loomflow_eval_metric_usage", default=None
)


def record_metric_usage(usage: Usage | None) -> None:
    """Report what a metric's own model call cost.

    Call it from a custom metric's ``score`` after each model call;
    inside an :class:`~loomflow.eval.EvalHarness` run the usage is
    counted in the case's ``cost_usd`` and against ``max_cost_usd``.
    Outside a harness run it does nothing. :class:`LLMJudge` calls it
    itself.
    """
    sink = _metric_usage_var.get()
    if sink is not None and usage is not None:
        sink.append(usage)


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        cached_input_tokens=a.cached_input_tokens + b.cached_input_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cost_usd=a.cost_usd + b.cost_usd,
    )

# An explicit, labelled score line: ``score: 0.7`` / ``Score = 1`` —
# also ``**Score:** 0.8``, ``Final score: 0.8.``, ``score: 4/5`` and
# ``score: 80%``. Anchored per-line; prose numbers never match.
_SCORE_LINE_RE = re.compile(
    r"(?im)^[\s*_#>-]*(?:final\s+)?score[*_\s]*[:=][*_\s]*"
    r"(-?\d*\.?\d+)\s*(?:(%)|/\s*(\d+(?:\.\d+)?))?[*_\s.]*$"
)

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
    """Extract the score from an explicit ``score:`` line, or None.

    A fraction or percentage is scaled to 0-1. A bare value outside
    [0, 1] is not a score on the requested scale, so it reads as
    unparseable (the judge is asked again) rather than being clamped
    into a perfect or zero score."""
    matches = _SCORE_LINE_RE.findall(text)
    if not matches:
        return None
    number, percent, denominator = matches[-1]  # last line wins if the judge rambled
    try:
        value = float(number)
        if percent:
            value /= 100
        elif denominator:
            value = value / float(denominator) if float(denominator) > 0 else -1.0
    except ValueError:  # pragma: no cover - regex guarantees a float
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return value


async def _complete_text(model: Model, messages: list[Message]) -> tuple[str, Usage | None]:
    """Call the judge model, preferring ``complete()`` over draining
    ``stream()``. Returns the reply and the call's usage (None when the
    model didn't report one)."""
    complete = getattr(model, "complete", None)
    if callable(complete):
        text, _calls, usage, _finish = await complete(messages)
        return str(text), usage if isinstance(usage, Usage) else None
    parts: list[str] = []
    stream_usage: Usage | None = None
    async for chunk in model.stream(messages):
        if chunk.kind == "text" and chunk.text:
            parts.append(chunk.text)
        chunk_usage = getattr(chunk, "usage", None)
        if isinstance(chunk_usage, Usage):
            stream_usage = chunk_usage
    return "".join(parts), stream_usage


class LLMJudge:
    """Grade agent output with a judge model against a rubric.

    ``model`` is any loomflow :class:`~loomflow.core.protocols.Model`.
    ``rubric`` overrides the default grading instructions. The judge
    prompt carries the case input, the agent's output, and — when the
    case has ground truth — the expected answer as reference.

    ``name`` is the metric name the scores are reported under (default
    ``"llm_judge"``). Give each judge its own when a harness has more
    than one — ``LLMJudge(m, rubric=..., name="accuracy")`` — since a
    harness refuses two metrics with the same name.

    Parsing contract: the judge must emit an explicit ``score: X``
    line with ``X`` in ``[0, 1]``. One retry with corrective feedback
    on parse failure; a second failure raises :class:`JudgeParseError`
    (the harness records it as a metric error and the case fails).
    Passing ``neutral_score`` opts back into the old behaviour: that
    score is returned, with a :class:`UserWarning`, instead of raising.

    ``usage`` is the running total of what this judge's model calls
    have cost.
    """

    name: str = "llm_judge"

    def __init__(
        self,
        model: Model,
        rubric: str | None = None,
        *,
        name: str | None = None,
        neutral_score: float | None = None,
    ) -> None:
        if name is not None:
            if not name:
                raise ValueError("LLMJudge name must not be empty")
            self.name = name
        if neutral_score is not None and not 0.0 <= neutral_score <= 1.0:
            raise ValueError(f"neutral_score must be in [0, 1], got {neutral_score!r}")
        self.model = model
        self.rubric = rubric if rubric is not None else _DEFAULT_RUBRIC
        self.neutral_score = neutral_score
        self.usage = Usage()

    async def _ask(self, messages: list[Message]) -> str:
        reply, usage = await _complete_text(self.model, messages)
        if usage is not None:
            self.usage = _add_usage(self.usage, usage)
            record_metric_usage(usage)
        return reply

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
        reply = await self._ask(messages)
        parsed = _parse_score_line(reply)
        if parsed is not None:
            return parsed
        # One corrective retry, carrying the failed reply for context.
        retry_messages = [
            *messages,
            Message(role=Role.ASSISTANT, content=reply),
            Message(role=Role.USER, content=_RETRY_NUDGE),
        ]
        reply = await self._ask(retry_messages)
        parsed = _parse_score_line(reply)
        if parsed is not None:
            return parsed
        if self.neutral_score is None:
            raise JudgeParseError(
                f"judge {self.name!r} gave no parseable 'score:' line after one "
                f"retry (case {case.id}). Last reply: {reply[:200]!r}"
            )
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
