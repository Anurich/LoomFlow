"""LLM-backed DecisionModel — the no-waitlist fallback.

Constrains any loomflow :class:`~loomflow.Model` to the System One
output shapes: one completion call answers every question with
probability distributions, which are normalised and coerced into the
same frozen result types :class:`~loomflow.decisions.JevModel`
returns. Code written against the fallback swaps to Jev by changing
one spec string — and threshold logic behaves identically because the
result semantics (including Noul's bare P(yes)) are reproduced
exactly.

Not calibrated the way Jev claims to be — an LLM's self-reported
probabilities are directional, not statistical. Good enough to build
and test every seam today; revisit thresholds when real traffic moves
onto Jev.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.protocols import Model
from ..core.types import Message, Role, Usage
from .base import DecisionError, DecisionState
from .types import (
    Choice,
    ChoiceDecision,
    DecisionAnswer,
    Decisions,
    NoulDecision,
    Question,
    Score,
    ScoreDecision,
)

__all__ = ["LLMDecisionModel"]

_SYSTEM_PROMPT = """\
You are a decision engine. You do NOT converse or explain. You
evaluate the STATE against each QUESTION and output probability
distributions as JSON.

Rules:
- Output ONLY a JSON object, no prose, no code fences.
- Probabilities are your honest degree of belief. Spread them when
  uncertain; concentrate them when the state is clear.
- The JSON must have exactly this shape:
  {"answers": {<question name>: <answer>, ...}}
  where <answer> is, per question type:
  - choice:  {"probabilities": {<option key>: <p>, ...}} over the
    listed option keys (they should sum to ~1)
  - score:   {"level_probabilities": [<p per level, in order>]}
    (one entry per listed level, summing to ~1)
  - noul:    {"yes_probability": <p that the answer is yes>}

QUESTIONS:
{questions}
"""


class LLMDecisionModel:
    """Any loomflow ``Model``, constrained to decision output."""

    def __init__(self, model: Model) -> None:
        self._model = model
        self.name = f"llm-decisions:{getattr(model, 'name', 'model')}"

    async def decide(
        self,
        state: DecisionState,
        *,
        questions: Mapping[str, Question],
    ) -> Decisions:
        if not questions:
            raise DecisionError("decide() requires at least one question")
        system = _SYSTEM_PROMPT.replace(
            "{questions}", _render_questions(questions)
        )
        messages = [
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content=_render_state(state)),
        ]
        total = Usage()
        last_error = ""
        # One retry with parse feedback, mirroring the framework's
        # validate-with-retry philosophy for structured outputs.
        for _attempt in range(2):
            text, usage = await self._complete(messages)
            total = _add_usage(total, usage)
            try:
                answers = _coerce(text, questions)
            except DecisionError as exc:
                last_error = str(exc)
                messages = messages + [
                    Message(role=Role.ASSISTANT, content=text),
                    Message(
                        role=Role.USER,
                        content=(
                            f"Invalid: {exc}. Reply again with ONLY "
                            "the JSON object, exactly matching the "
                            "required shape."
                        ),
                    ),
                ]
                continue
            return Decisions(answers=answers, usage=total)
        raise DecisionError(
            f"model failed to produce valid decisions after retry: "
            f"{last_error}"
        )

    async def _complete(
        self, messages: list[Message]
    ) -> tuple[str, Usage]:
        """One text answer from the wrapped model.

        The ``Model`` protocol only REQUIRES ``stream``; ``complete``
        is an optional fast path adapters may expose (same posture as
        ``text_only_model_call``) — use it when present, else drain
        the stream.
        """
        if hasattr(self._model, "complete"):
            text, _calls, usage, _finish = await self._model.complete(
                messages
            )
            return text, usage
        parts: list[str] = []
        usage = Usage()
        async for chunk in self._model.stream(messages):
            if chunk.kind == "text" and chunk.text:
                parts.append(chunk.text)
            # Usage rides the "finish" chunk (ModelChunk is
            # kind-discriminated; there is no separate usage kind).
            if chunk.usage is not None:
                usage = chunk.usage
        return "".join(parts), usage


def _render_questions(questions: Mapping[str, Question]) -> str:
    lines: list[str] = []
    for name, q in questions.items():
        if isinstance(q, Choice):
            opts = "; ".join(
                f"{key}: {desc}" for key, desc in q.criteria.items()
            )
            lines.append(
                f"- {name} (choice): {q.instructions}\n  options: {opts}"
            )
        elif isinstance(q, Score):
            levels = "; ".join(
                f"[{i}] {level}" for i, level in enumerate(q.criteria)
            )
            lines.append(
                f"- {name} (score): {q.instructions}\n  levels: {levels}"
            )
        else:
            clar = ""
            if q.criteria:
                clar = " (" + "; ".join(
                    f"{k}: {v}" for k, v in q.criteria.items()
                ) + ")"
            lines.append(f"- {name} (noul): {q.instructions}{clar}")
    return "\n".join(lines)


def _render_state(state: DecisionState) -> str:
    if isinstance(state, str):
        return f"STATE:\n{state}"
    if isinstance(state, Mapping):
        return "STATE:\n" + json.dumps(state, default=str, indent=2)
    return "STATE:\n" + "\n".join(str(part) for part in state)


def _coerce(
    text: str, questions: Mapping[str, Question]
) -> dict[str, DecisionAnswer]:
    payload = _parse_json(text)
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise DecisionError("missing top-level 'answers' object")
    out: dict[str, DecisionAnswer] = {}
    for name, question in questions.items():
        raw = raw_answers.get(name)
        if not isinstance(raw, Mapping):
            raise DecisionError(f"missing answer for question {name!r}")
        if isinstance(question, Choice):
            out[name] = _coerce_choice(name, question, raw)
        elif isinstance(question, Score):
            out[name] = _coerce_score(name, question, raw)
        else:
            out[name] = _coerce_noul(name, raw)
    return out


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    # Models fenced-code despite instructions often enough to be
    # worth tolerating (same posture as the agent loop's parser).
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise DecisionError("no JSON object in response")
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise DecisionError(f"malformed JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DecisionError("response JSON is not an object")
    return parsed


def _coerce_choice(
    name: str, question: Choice, raw: Mapping[str, Any]
) -> ChoiceDecision:
    probs_raw = raw.get("probabilities")
    if not isinstance(probs_raw, Mapping):
        raise DecisionError(
            f"choice answer {name!r} needs a 'probabilities' object"
        )
    # Only declared option keys count; extras are ignored, missing
    # options get 0. Normalise so downstream thresholds see a
    # distribution regardless of how sloppily the LLM summed.
    probs: dict[str, float] = {}
    for key in question.criteria:
        try:
            probs[key] = max(0.0, float(probs_raw.get(key, 0.0)))
        except (TypeError, ValueError) as exc:
            raise DecisionError(
                f"choice answer {name!r} has a non-numeric "
                f"probability for {key!r}"
            ) from exc
    total = sum(probs.values())
    if total <= 0:
        raise DecisionError(
            f"choice answer {name!r} assigns no probability to any "
            "declared option"
        )
    probs = {k: v / total for k, v in probs.items()}
    choice = max(probs, key=lambda k: probs[k])
    return ChoiceDecision(
        choice=choice, probabilities=probs, confidence=probs[choice]
    )


def _coerce_score(
    name: str, question: Score, raw: Mapping[str, Any]
) -> ScoreDecision:
    levels_raw = raw.get("level_probabilities")
    if not isinstance(levels_raw, Sequence) or isinstance(
        levels_raw, str | bytes
    ):
        raise DecisionError(
            f"score answer {name!r} needs a 'level_probabilities' list"
        )
    if len(levels_raw) != len(question.criteria):
        raise DecisionError(
            f"score answer {name!r} has {len(levels_raw)} "
            f"probabilities for {len(question.criteria)} levels"
        )
    try:
        probs = [max(0.0, float(p)) for p in levels_raw]
    except (TypeError, ValueError) as exc:
        raise DecisionError(
            f"score answer {name!r} has a non-numeric probability"
        ) from exc
    total = sum(probs)
    if total <= 0:
        raise DecisionError(
            f"score answer {name!r} assigns no probability to any level"
        )
    probs = [p / total for p in probs]
    # Probability-weighted expectation — the score "can fall between
    # two levels", matching the wire API's semantics.
    score = sum(i * p for i, p in enumerate(probs))
    return ScoreDecision(
        score=score,
        legend=list(question.criteria),
        probabilities=probs,
        confidence=max(probs),
    )


def _coerce_noul(name: str, raw: Mapping[str, Any]) -> NoulDecision:
    p_raw = raw.get("yes_probability")
    try:
        p = float(p_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DecisionError(
            f"noul answer {name!r} needs a numeric 'yes_probability'"
        ) from exc
    return NoulDecision(noul=min(1.0, max(0.0, p)))


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        cached_input_tokens=a.cached_input_tokens
        + b.cached_input_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cost_usd=a.cost_usd + b.cost_usd,
    )
