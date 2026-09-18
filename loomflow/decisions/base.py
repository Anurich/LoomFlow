"""DecisionModel protocol + resolver.

A :class:`DecisionModel` is a sibling of :class:`~loomflow.Model` —
not a subtype. A System One model cannot ``complete()`` or
``stream()`` text; it evaluates *questions* against *state* and
returns typed, probabilistic answers. Keeping the protocols separate
means neither can be passed where the other is expected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from ..core.errors import ConfigError, LoomError
from .types import Decisions, Question

__all__ = [
    "DecisionError",
    "DecisionModel",
    "DecisionState",
    "resolve_decision_model",
]


# State accepted by ``decide()`` — free text, a JSON-able mapping, or
# a list of text parts. Text-only (the wire API takes no media).
DecisionState = str | Mapping[str, Any] | Sequence[str]


class DecisionError(LoomError):
    """A decision backend failed to produce a valid, typed answer."""


@runtime_checkable
class DecisionModel(Protocol):
    """Anything that evaluates questions against state.

    Implementations: :class:`~loomflow.decisions.JevModel` (TypeSafe
    Jev over the wire), :class:`~loomflow.decisions.LLMDecisionModel`
    (any loomflow ``Model`` constrained to the same output shapes),
    :class:`~loomflow.decisions.ScriptedDecisions` (test fake).
    """

    name: str

    async def decide(
        self,
        state: DecisionState,
        *,
        questions: Mapping[str, Question],
    ) -> Decisions: ...


def resolve_decision_model(
    spec: DecisionModel | str | Any | None,
    *,
    secrets: Any | None = None,
) -> DecisionModel | None:
    """Resolve a ``decider=`` spec to a concrete :class:`DecisionModel`.

    Accepted shapes:

    * ``None`` → ``None`` (seam keeps its current LLM behaviour)
    * a :class:`DecisionModel` instance → itself
    * ``"jev"`` / ``"jev-*"`` → :class:`JevModel` (needs
      ``TYPESAFE_API_KEY`` via the Secrets backend or env)
    * a loomflow ``Model`` instance → :class:`LLMDecisionModel`
      wrapping it (works today, swap to ``"jev"`` later)
    * any other model-spec string (``"claude-haiku-4-5"``, ``"echo"``,
      …) → resolved through the normal model resolver, then wrapped
      in :class:`LLMDecisionModel`
    """
    if spec is None:
        return None
    # A DecisionModel instance passes straight through. Duck-typed
    # (like ToolHost coercion) so test fakes need no registration.
    if hasattr(spec, "decide"):
        return spec  # type: ignore[return-value]
    if isinstance(spec, str):
        if spec == "jev" or spec.startswith("jev-"):
            from .jev import JevModel

            return JevModel(
                "jev-latest" if spec == "jev" else spec,
                secrets=secrets,
            )
        # Any other string is a normal model spec → LLM fallback.
        # Lazy import: agent.api imports architectures which may
        # import this module — same cycle-safety as resolve_role_model.
        from ..agent.api import _resolve_model
        from .llm import LLMDecisionModel

        return LLMDecisionModel(_resolve_model(spec, secrets=secrets))
    # A Model instance (has complete) → LLM fallback around it.
    if hasattr(spec, "complete"):
        from .llm import LLMDecisionModel

        return LLMDecisionModel(spec)
    raise ConfigError(
        f"decider= expects a DecisionModel, a Model, a model-spec "
        f"string, 'jev', or None; got {type(spec).__name__}: {spec!r}"
    )
