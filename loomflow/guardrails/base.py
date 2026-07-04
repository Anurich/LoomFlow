"""Guardrail protocol + verdict types + ordered composition (G13).

A :class:`Guardrail` inspects a piece of text at one of three trust
boundaries and returns a :class:`GuardVerdict`:

* ``"input"`` — the user prompt, before it seeds the run (before any
  model call).
* ``"output"`` — the final ``session.output``, after the architecture
  completes and before output-schema validation.
* ``"tool_result"`` — a tool's text output, before it enters
  conversation history (and BEFORE the ``tool_result_max_chars``
  truncation, so injected delimiters survive for all but pathological
  outputs).

Verdict actions:

* ``"allow"`` — pass the text through unchanged.
* ``"annotate"`` — replace the text with ``transformed`` (delimiter
  wrapping, PII redaction, ...). ``reason`` is optional; when set it
  marks a *detection* (not just a mechanical transform) and the
  framework emits a ``guardrail.triggered`` event.
* ``"block"`` — stop. The framework substitutes a refusal /
  blocked-marker text and, for the input/output stages, marks the run
  ``interrupted`` with reason ``guardrail:<name>``.

Guards compose in order: each guard sees the previous guard's
transformed text. The whole layer is zero-cost when no guardrails are
configured (``Dependencies.fast_guardrails`` short-circuits every call
site).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..core.context import RunContext

__all__ = [
    "VALID_STAGES",
    "GuardAction",
    "Guardrail",
    "GuardrailOutcome",
    "GuardrailTrigger",
    "GuardVerdict",
    "apply_guardrails",
]

GuardAction = Literal["allow", "annotate", "block"]

#: The three trust boundaries a guard can subscribe to.
VALID_STAGES: frozenset[str] = frozenset(
    {"input", "output", "tool_result"}
)


@dataclass
class GuardVerdict:
    """Outcome of a single guardrail check.

    ``transformed`` is the replacement text when ``action ==
    "annotate"`` (``None`` means keep the input unchanged). ``reason``
    explains a block, or marks an annotate as a *detection* — an
    annotate with a reason emits a ``guardrail.triggered`` event;
    an annotate without one (plain mechanical wrapping) does not.
    """

    action: GuardAction
    transformed: str | None = None
    reason: str | None = None


@runtime_checkable
class Guardrail(Protocol):
    """Protocol every guardrail implements.

    ``name`` identifies the guard in events / interruption reasons /
    refusal texts. ``stages`` is the subset of :data:`VALID_STAGES`
    the guard subscribes to — the framework only invokes ``check``
    for matching stages.
    """

    name: str
    stages: frozenset[str]

    async def check(
        self,
        text: str,
        *,
        stage: str,
        context: RunContext | None = None,
    ) -> GuardVerdict:
        """Inspect ``text`` at ``stage``; return a verdict."""
        ...


@dataclass
class GuardrailTrigger:
    """One event-worthy guardrail firing (annotate-with-detection or
    block). Collected by :func:`apply_guardrails`; the caller turns
    each into an ``Event.architecture_event(..., "guardrail.triggered",
    ...)``."""

    guard: str
    stage: str
    action: str
    reason: str | None = None


@dataclass
class GuardrailOutcome:
    """Composite result of running an ordered guard sequence.

    ``text`` is the (possibly transformed) text after every annotate.
    On ``blocked``, ``guard`` / ``reason`` identify the blocking guard
    and ``text`` is left at the last pre-block transform — callers
    substitute their own refusal / blocked-marker text.
    """

    text: str
    blocked: bool = False
    guard: str | None = None
    reason: str | None = None
    triggered: list[GuardrailTrigger] = field(default_factory=list)


async def apply_guardrails(
    guards: Sequence[Guardrail],
    text: str,
    *,
    stage: str,
    context: RunContext | None = None,
) -> GuardrailOutcome:
    """Run ``guards`` in order against ``text`` for ``stage``.

    Ordered composition: each guard sees the previous guard's
    transformed text. Guards whose ``stages`` don't include ``stage``
    are skipped. The first ``block`` verdict stops the chain.

    ``triggered`` records every block and every annotate that carried
    a ``reason`` (detection). Plain annotates (reason ``None`` — e.g.
    InjectionGuard's unconditional delimiter wrapping) transform the
    text but do NOT trigger an event; per-tool-call events for
    behaviour that fires on 100% of tool results would be pure noise.
    """
    outcome = GuardrailOutcome(text=text)
    for guard in guards:
        if stage not in guard.stages:
            continue
        verdict = await guard.check(
            outcome.text, stage=stage, context=context
        )
        if verdict.action == "block":
            outcome.blocked = True
            outcome.guard = guard.name
            outcome.reason = verdict.reason
            outcome.triggered.append(
                GuardrailTrigger(
                    guard=guard.name,
                    stage=stage,
                    action="block",
                    reason=verdict.reason,
                )
            )
            return outcome
        if verdict.action == "annotate":
            if verdict.transformed is not None:
                outcome.text = verdict.transformed
            if verdict.reason is not None:
                outcome.triggered.append(
                    GuardrailTrigger(
                        guard=guard.name,
                        stage=stage,
                        action="annotate",
                        reason=verdict.reason,
                    )
                )
        # "allow" (and any unknown action, defensively) — continue.
    return outcome
