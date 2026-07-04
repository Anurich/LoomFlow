"""Guardrails (G13) — input / output / tool-result trust boundaries.

Opt-in, ordered, composable text guards::

    from loomflow.guardrails import InjectionGuard, PIIGuard

    agent = Agent(
        "...",
        model="...",
        guardrails=[InjectionGuard(), PIIGuard()],
    )

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

__all__ = [
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
