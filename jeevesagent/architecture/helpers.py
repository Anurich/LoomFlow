"""Cross-architecture helpers.

Small utilities multiple architectures need. Putting them here keeps
each architecture's module focused on its strategy and avoids
circular re-implementation:

* :func:`text_only_model_call` — run a single model call with no
  tools, collecting the response text and usage. Used by Self-Refine
  (critic / refiner), Reflexion (evaluator / reflector),
  Plan-and-Execute (planner / replanner), Router (classifier), and
  any other architecture that needs a one-shot structured LLM call.
* :func:`add_usage` — sum two :class:`Usage` records.
* :func:`parse_score` — extract a 0-1 confidence number from
  free-form evaluator output. Used by Reflexion and Tree of Thoughts;
  any architecture with an evaluator step.
"""

from __future__ import annotations

import re

from ..core.types import Message, Usage
from .base import Dependencies


async def text_only_model_call(
    deps: Dependencies,
    step_name: str,
    messages: list[Message],
) -> tuple[str, Usage]:
    """Run a single text-only model call through ``runtime.step``.

    Returns ``(text, usage)``. The call is journaled so replays are
    deterministic, but no tools are exposed — used for one-shot
    structured prompts (critique, evaluation, classification,
    planning).
    """
    text_parts: list[str] = []
    usage = Usage()

    chunks = deps.runtime.stream_step(
        step_name,
        deps.model.stream,
        messages,
        tools=None,
    )
    async for chunk in chunks:
        if chunk.kind == "text" and chunk.text is not None:
            text_parts.append(chunk.text)
        elif chunk.kind == "finish" and chunk.usage is not None:
            usage = chunk.usage

    return "".join(text_parts), usage


def add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cost_usd=a.cost_usd + b.cost_usd,
    )


_SCORE_LINE_RE = re.compile(
    r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
)
_FALLBACK_NUMBER_RE = re.compile(r"\b(0\.\d+|1\.0+|0|1)\b")


def parse_score(text: str) -> float:
    """Extract a 0-1 score from free-form evaluator output.

    Prefers the ``score: X`` (or ``score=X``) pattern; falls back to
    any plausible number in the text. Clamps to ``[0.0, 1.0]``.
    Returns 0.0 on parse failure (treated as a failed evaluation —
    let the caller decide what that means).

    Used by :class:`~jeevesagent.architecture.Reflexion` (attempt
    score) and :class:`~jeevesagent.architecture.TreeOfThoughts`
    (per-thought evaluation).
    """
    match = _SCORE_LINE_RE.search(text)
    if match is None:
        match = _FALLBACK_NUMBER_RE.search(text)
    if match is None:
        return 0.0
    try:
        value = float(match.group(1))
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, value))
