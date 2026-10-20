"""Guardrails (G13) — input / output / tool-result trust boundaries.

Opt-in, ordered, composable text guards::

    from loomflow.guardrails import InjectionGuard, PIIGuard

    agent = Agent(
        "...",
        model="...",
        guardrails=[InjectionGuard(), PIIGuard()],
    )

With output-stage guards wired, streamed text goes through them before
it is sent (see :class:`GuardedStream`).

Deliberately NOT re-exported from the top-level ``loomflow``
namespace — guardrails are a security capability you import
explicitly.
"""

from .base import (
    Guardrail,
    GuardrailOutcome,
    GuardrailTrigger,
    GuardVerdict,
    apply_guardrails,
)
from .builtin import (
    InjectionGuard,
    ModerationGuard,
    PIIGuard,
    RegexGuard,
)
from .streaming import GuardedStream

__all__ = [
    "GuardedStream",
    "Guardrail",
    "GuardrailOutcome",
    "GuardrailTrigger",
    "GuardVerdict",
    "InjectionGuard",
    "ModerationGuard",
    "PIIGuard",
    "RegexGuard",
    "apply_guardrails",
]
