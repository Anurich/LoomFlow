"""ScriptedDecisions — the zero-key test fake for DecisionModel.

Same philosophy as :class:`~loomflow.ScriptedModel`: deterministic,
records every call, no network. Construct with per-question answers
(used for every call) and/or a queue of per-call answer maps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..core.types import Usage
from .base import DecisionError, DecisionState
from .types import DecisionAnswer, Decisions, Question

__all__ = ["ScriptedDecisions"]


@dataclass
class ScriptedDecisions:
    """Canned answers, in-order or as a standing default.

    * ``answers`` — a standing name→answer map applied to every call.
    * ``script`` — a queue of per-call maps; each call pops one and
      overlays it on ``answers``. When the queue runs dry, calls fall
      back to ``answers`` alone.

    Every question in a call must resolve to an answer, else
    :class:`DecisionError` — an unteed question in a test is a bug.
    Calls are recorded on ``calls`` as ``(state, question_names)``.
    """

    answers: Mapping[str, DecisionAnswer] = field(default_factory=dict)
    script: list[Mapping[str, DecisionAnswer]] = field(
        default_factory=list
    )
    usage: Usage = field(default_factory=Usage)
    name: str = "scripted-decisions"
    calls: list[tuple[DecisionState, tuple[str, ...]]] = field(
        default_factory=list
    )

    async def decide(
        self,
        state: DecisionState,
        *,
        questions: Mapping[str, Question],
    ) -> Decisions:
        self.calls.append((state, tuple(questions)))
        overlay: Mapping[str, DecisionAnswer] = (
            self.script.pop(0) if self.script else {}
        )
        out: dict[str, DecisionAnswer] = {}
        for name in questions:
            answer = overlay.get(name, self.answers.get(name))
            if answer is None:
                raise DecisionError(
                    f"ScriptedDecisions has no answer for {name!r}"
                )
            out[name] = answer
        return Decisions(answers=out, usage=self.usage)
