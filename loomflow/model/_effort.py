"""Reasoning-effort translation between Loom's unified enum and
each provider's native shape.

Providers have not converged on a single API:

* **OpenAI** (o1/o3/o4/GPT-5) — ``reasoning_effort: "minimal" |
  "low" | "medium" | "high"`` on ``chat.completions.create``.
* **Anthropic** Claude 4.6+ — ``output_config.effort`` + adaptive
  ``thinking`` block. Opus 4.7 added ``xhigh`` and ``max``.
* **Anthropic** Claude 3.7 / 4 / 4.5 — ``thinking.budget_tokens``
  integer (min 1024). No string enum.
* **LiteLLM** — normalises to OpenAI shape.

This module centralises the mapping so each adapter stays small.
Adapters that don't support reasoning effort log a one-time
warning per ``(model, effort)`` pair and drop the kwarg, unless
the agent passed ``strict_effort=True`` (then we raise).
"""

from __future__ import annotations

import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Shared warning + raise helper
# ---------------------------------------------------------------------------


class EffortNotSupportedError(ValueError):
    """Raised when a model doesn't support reasoning effort AND the
    caller opted into ``strict_effort=True``. Default behaviour is
    warn-and-drop; this only fires when strict mode is on."""


# Track which (model_name, effort) pairs have already produced a
# warning so we don't spam the logs on every call. Module-global
# is fine — the set never grows past a handful of entries in
# practice (one per model the user touches).
_WARNED: set[tuple[str, str]] = set()


def _unsupported(
    model_name: str,
    effort: str,
    *,
    strict: bool,
    reason: str,
) -> None:
    """Either raise (strict mode) or warn-once (default)."""
    if strict:
        raise EffortNotSupportedError(
            f"Model {model_name!r} does not support effort="
            f"{effort!r}: {reason}. Pass strict_effort=False on "
            f"the Agent to downgrade this to a warning + drop."
        )
    key = (model_name, effort)
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(
        f"Model {model_name!r} does not support effort={effort!r}: "
        f"{reason}. The kwarg has been dropped for this and future "
        f"calls; emitted once per (model, effort) pair.",
        UserWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

# Models that accept ``reasoning_effort``. Prefix-matched against
# ``model.startswith(prefix)`` so future minor versions land
# automatically (e.g. "o3-mini-2025-01-01" still matches "o3-").
_OPENAI_REASONING_PREFIXES = (
    "o1",
    "o3",
    "o4",
    "gpt-5",
)


def openai_kwargs(
    effort: str | None,
    model_name: str,
    *,
    strict: bool,
) -> dict[str, Any]:
    """Translate ``effort`` into OpenAI request kwargs.

    Returns ``{}`` when effort is None or when the model doesn't
    support reasoning effort. Returns ``{"reasoning_effort": ...}``
    otherwise. The caller merges the dict into its create() call.
    """
    if effort is None:
        return {}
    if not _model_supports_openai_effort(model_name):
        _unsupported(
            model_name, effort,
            strict=strict,
            reason="OpenAI accepts reasoning_effort only on o1/o3/o4/GPT-5",
        )
        return {}
    # Loom's ``xhigh`` and ``max`` don't exist on OpenAI — clamp
    # them to ``high`` (the highest legal OpenAI value).
    mapped = {
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
        "max": "high",
    }.get(effort, "medium")
    return {"reasoning_effort": mapped}


def _model_supports_openai_effort(model_name: str) -> bool:
    name = model_name.lower()
    return any(name.startswith(p) for p in _OPENAI_REASONING_PREFIXES)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

# Three regimes:
#  1. Opus 4.7 / Mythos — adaptive-only; supports xhigh + max.
#  2. Opus 4.6 / Sonnet 4.6 — adaptive + output_config.effort enum.
#  3. Sonnet 3.7 / 4 / 4.5 — thinking={enabled, budget_tokens=N}.
#
# Anything else (Haiku, older Claudes) — drop with warning.


def _anthropic_regime(model_name: str) -> str | None:
    """Return ``"4.7"`` / ``"4.6"`` / ``"legacy"`` / ``None``."""
    name = model_name.lower()
    # Opus 4.7 / Mythos — adaptive-only.
    if "opus-4-7" in name or "mythos" in name:
        return "4.7"
    # Opus 4.6 / Sonnet 4.6 — adaptive + effort enum.
    if "opus-4-6" in name or "sonnet-4-6" in name:
        return "4.6"
    # Sonnet 3.7 / 4 / 4.5 — legacy thinking-budget.
    if any(
        marker in name
        for marker in ("sonnet-3-7", "sonnet-4-5", "opus-4-5", "sonnet-4")
    ):
        return "legacy"
    return None


# Token-budget mapping for the legacy ``thinking.budget_tokens``
# regime. Tuned to roughly match the effort dial — these are not
# provider-documented constants, just defensible defaults.
_ANTHROPIC_LEGACY_BUDGET = {
    "minimal": 1024,    # the Anthropic minimum
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}


def anthropic_kwargs(
    effort: str | None,
    model_name: str,
    *,
    strict: bool,
) -> dict[str, Any]:
    """Translate ``effort`` into Anthropic request kwargs.

    Returns the merge-into-create dict. Empty when effort is None
    or model doesn't support thinking.
    """
    if effort is None:
        return {}
    regime = _anthropic_regime(model_name)
    if regime is None:
        _unsupported(
            model_name, effort,
            strict=strict,
            reason="Anthropic supports thinking only on Sonnet 3.7+/4+/4.5+/4.6+ and Opus 4.5/4.6/4.7",
        )
        return {}

    if regime == "4.7":
        # Adaptive-only; xhigh + max are legal here.
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }

    if regime == "4.6":
        # Adaptive + effort enum. xhigh/max → clamp to high (4.6
        # doesn't accept them).
        mapped = {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
            "max": "high",
        }.get(effort, "medium")
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": mapped},
        }

    # regime == "legacy" — Sonnet 3.7 / 4 / 4.5: budget_tokens int.
    budget = _ANTHROPIC_LEGACY_BUDGET.get(effort, 4096)
    return {"thinking": {"type": "enabled", "budget_tokens": budget}}


# ---------------------------------------------------------------------------
# Test-only — clear the warning cache between unit tests so the
# one-time-per-pair logic doesn't bleed across tests.
# ---------------------------------------------------------------------------


def _reset_warned_cache_for_tests() -> None:
    """Internal: only intended for use by ``tests/test_effort.py``."""
    _WARNED.clear()
