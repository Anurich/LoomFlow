"""Jev adapter — TypeSafe AI's System One model over the wire.

Lazy import of the vendor SDK; install with
``pip install 'loomflow[typesafe]'`` and set ``TYPESAFE_API_KEY``
(explicit ``api_key=`` and the Secrets backend also work, in the
usual precedence). Early-access API as of September 2026.

The adapter is deliberately thin: our question types mirror the SDK's
(:class:`Choice` / :class:`Score` / :class:`Noul` with the same
constructor fields), so mapping is field-for-field, and responses are
read duck-typed so SDK point releases don't break us.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ..core.types import Usage
from ..model._pricing import estimate_cost
from .base import DecisionError, DecisionState
from .types import (
    Choice,
    ChoiceDecision,
    DecisionAnswer,
    Decisions,
    Noul,
    NoulDecision,
    Question,
    Score,
    ScoreDecision,
)

__all__ = ["JevModel"]


class JevModel:
    """System One decisions served by TypeSafe's ``/v1/systemone``."""

    def __init__(
        self,
        model: str = "jev-latest",
        *,
        api_key: str | None = None,
        client: Any | None = None,
        secrets: Any | None = None,
    ) -> None:
        self.name = model
        if client is not None:
            self._client = client
            return
        try:
            from typesafe_sdk import (  # type: ignore[import-not-found, import-untyped]
                AsyncTypeSafeClient,
            )
        except ImportError as exc:  # pragma: no cover — depends on env
            raise ImportError(
                "TypeSafe SDK not installed. "
                "Install with: pip install 'loomflow[typesafe]'"
            ) from exc
        # Key precedence: explicit arg → Secrets backend → env. The
        # SDK also self-reads TYPESAFE_API_KEY, but resolving here
        # keeps vault-backed Secrets working and errors early.
        resolved_key = api_key
        if resolved_key is None and secrets is not None:
            resolved_key = secrets.lookup_sync("TYPESAFE_API_KEY")
        if resolved_key is None:
            resolved_key = os.environ.get("TYPESAFE_API_KEY")
        self._client = AsyncTypeSafeClient(api_key=resolved_key)

    async def decide(
        self,
        state: DecisionState,
        *,
        questions: Mapping[str, Question],
    ) -> Decisions:
        if not questions:
            raise DecisionError("decide() requires at least one question")
        sdk_questions = {
            name: _to_sdk_question(q) for name, q in questions.items()
        }
        response = await self._client.system_one(
            state=state, questions=sdk_questions
        )
        raw_answers = getattr(response, "answers", None) or {}
        answers: dict[str, DecisionAnswer] = {}
        for name, question in questions.items():
            raw = (
                raw_answers.get(name)
                if isinstance(raw_answers, Mapping)
                else getattr(raw_answers, name, None)
            )
            if raw is None:
                raise DecisionError(
                    f"Jev response missing answer for question {name!r}"
                )
            answers[name] = _parse_answer(name, question, raw)

        u = getattr(response, "usage", None)
        in_tok = int(getattr(u, "input_tokens", 0) or 0)
        out_tok = int(getattr(u, "output_tokens", 0) or 0)
        usage = Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost(self.name, in_tok, out_tok),
        )
        return Decisions(answers=answers, usage=usage)


def _to_sdk_question(q: Question) -> Any:
    """Field-for-field translation into the vendor SDK's types.

    When the SDK isn't installed (an injected fake ``client=`` in
    tests), our frozen dataclasses pass through verbatim — they carry
    the same field names, so a fake sees identical shapes.
    """
    try:
        from typesafe_sdk import (  # type: ignore[import-not-found, import-untyped]
            Choice as SdkChoice,
        )
        from typesafe_sdk import (  # type: ignore[import-not-found, import-untyped]
            Noul as SdkNoul,
        )
        from typesafe_sdk import (  # type: ignore[import-not-found, import-untyped]
            Score as SdkScore,
        )
    except ImportError:
        return q

    if isinstance(q, Choice):
        return SdkChoice(
            instructions=q.instructions, criteria=dict(q.criteria)
        )
    if isinstance(q, Score):
        return SdkScore(
            instructions=q.instructions, criteria=list(q.criteria)
        )
    if isinstance(q, Noul):
        if q.criteria is not None:
            return SdkNoul(
                instructions=q.instructions, criteria=dict(q.criteria)
            )
        return SdkNoul(instructions=q.instructions)
    raise DecisionError(f"unknown question type: {type(q).__name__}")


def _parse_answer(
    name: str, question: Question, raw: Any
) -> DecisionAnswer:
    """Duck-typed read of one SDK answer into our frozen types."""
    if isinstance(question, Choice):
        return ChoiceDecision(
            choice=str(getattr(raw, "choice", "")),
            probabilities=dict(getattr(raw, "probabilities", {}) or {}),
            confidence=float(getattr(raw, "confidence", 0.0) or 0.0),
        )
    if isinstance(question, Score):
        legend = getattr(raw, "legend", None) or list(question.criteria)
        return ScoreDecision(
            score=float(getattr(raw, "score", 0.0) or 0.0),
            legend=list(legend),
            probabilities=list(getattr(raw, "probabilities", []) or []),
            confidence=float(getattr(raw, "confidence", 0.0) or 0.0),
        )
    if isinstance(question, Noul):
        noul = getattr(raw, "noul", None)
        if noul is None:
            raise DecisionError(
                f"Jev answer for Noul question {name!r} has no "
                "'noul' probability"
            )
        return NoulDecision(noul=float(noul))
    raise DecisionError(f"unknown question type: {type(question).__name__}")
