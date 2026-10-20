"""Reasoning-effort translation between Loom's unified enum and
each provider's native shape.

Providers have not converged on a single API:

* **OpenAI** (o-series, GPT-5.x, GPT-6.x) — ``reasoning_effort`` on
  ``chat.completions.create``; each family accepts its own subset of
  ``minimal`` / ``low`` / ``medium`` / ``high`` / ``xhigh`` / ``max``.
* **Anthropic**, adaptive-thinking models (Claude 4.6 and later) —
  ``thinking: {"type": "adaptive"}`` plus ``output_config.effort``
  (``low`` / ``medium`` / ``high``, and ``xhigh`` / ``max`` where the
  model has them). Opus 4.5 takes ``output_config.effort`` together
  with a ``thinking.budget_tokens`` budget.
* **Anthropic**, extended-thinking-only models (Haiku / Sonnet 4.5,
  Claude 4.0 / 4.1, Sonnet 3.7) — ``thinking.budget_tokens`` only.
* **LiteLLM** — normalises to OpenAI shape.

A level a model doesn't accept is clamped to the nearest one it does
(``minimal`` → ``low`` on Claude). Sources: the Anthropic effort and
thinking docs, and the OpenAI reasoning guide and model pages.

This module centralises the mapping so each adapter stays small.
Adapters that don't support reasoning effort log a one-time
warning per ``(model, effort)`` pair and drop the kwarg, unless
the agent passed ``strict_effort=True`` (then we raise). An effort
string that isn't one of :data:`EFFORT_LEVELS` is a ``ValueError``.
"""

from __future__ import annotations

import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Shared warning + raise helper
# ---------------------------------------------------------------------------


#: The effort levels loomflow accepts, lowest first.
EFFORT_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")


def validate_effort(effort: str | None) -> None:
    """Raise ``ValueError`` unless ``effort`` is ``None`` or one of
    :data:`EFFORT_LEVELS` (a typo like ``"hgih"`` must not be dropped
    silently)."""
    if effort is not None and effort not in EFFORT_LEVELS:
        raise ValueError(
            f"unknown effort {effort!r}; expected one of {', '.join(EFFORT_LEVELS)}"
        )


def _clamp(effort: str, allowed: tuple[str, ...]) -> str:
    """The highest level in ``allowed`` not above ``effort`` (or the
    lowest allowed level when ``effort`` is below all of them)."""
    rank = EFFORT_LEVELS.index(effort)
    below = [lvl for lvl in allowed if EFFORT_LEVELS.index(lvl) <= rank]
    if below:
        return max(below, key=EFFORT_LEVELS.index)
    return min(allowed, key=EFFORT_LEVELS.index)


def _prefix_lookup(table: dict[str, Any], name: str) -> Any:
    best = None
    for key in table:
        if name.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return None if best is None else table[best]


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

# The reasoning-effort levels each OpenAI family accepts (OpenAI model
# pages), matched by longest prefix so dated snapshots land.
_OPENAI_LEVELS: dict[str, tuple[str, ...]] = {
    "o1": ("low", "medium", "high"),
    "o3": ("low", "medium", "high"),
    "o4": ("low", "medium", "high"),
    "gpt-5": ("minimal", "low", "medium", "high"),
    "gpt-5.1": ("low", "medium", "high"),
    "gpt-5.2": ("low", "medium", "high", "xhigh"),
    "gpt-5.3": ("low", "medium", "high", "xhigh"),
    "gpt-5.4": ("low", "medium", "high", "xhigh"),
    "gpt-5.5": ("low", "medium", "high", "xhigh"),
    "gpt-5.6": ("low", "medium", "high", "xhigh", "max"),
    "gpt-6": ("low", "medium", "high", "xhigh", "max"),
}
# Older name kept for importers.
_OPENAI_REASONING_PREFIXES = tuple(_OPENAI_LEVELS)


def openai_kwargs(
    effort: str | None,
    model_name: str,
    *,
    strict: bool,
) -> dict[str, Any]:
    """Translate ``effort`` into OpenAI request kwargs.

    Returns ``{}`` when effort is None or when the model doesn't
    support reasoning effort. Returns ``{"reasoning_effort": ...}``
    otherwise, clamped to a level the model accepts. The caller
    merges the dict into its create() call.
    """
    validate_effort(effort)
    if effort is None:
        return {}
    levels = _prefix_lookup(_OPENAI_LEVELS, model_name.lower())
    if levels is None:
        _unsupported(
            model_name, effort,
            strict=strict,
            reason="OpenAI accepts reasoning_effort only on o-series, GPT-5.x and GPT-6.x models",
        )
        return {}
    return {"reasoning_effort": _clamp(effort, levels)}


def _model_supports_openai_effort(model_name: str) -> bool:
    return _prefix_lookup(_OPENAI_LEVELS, model_name.lower()) is not None


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

# Per-model effort support (Anthropic effort + thinking docs):
# ``(thinking mode, effort levels or None)``. "adaptive" models take
# ``thinking: {"type": "adaptive"}`` + ``output_config.effort``;
# "extended" models only ``thinking: {"type": "enabled",
# "budget_tokens": N}`` (Opus 4.5 also takes ``output_config.effort``).
# Longest-prefix match on the model name.
_ALL = ("low", "medium", "high", "xhigh", "max")
_NO_XHIGH = ("low", "medium", "high", "max")
_ANTHROPIC_MODELS: dict[str, tuple[str, tuple[str, ...] | None]] = {
    "claude-fable": ("adaptive", _ALL),
    "claude-mythos-preview": ("adaptive", _NO_XHIGH),
    "claude-mythos": ("adaptive", _ALL),
    "claude-opus-5": ("adaptive", _ALL),
    "claude-opus-4-8": ("adaptive", _ALL),
    "claude-opus-4-7": ("adaptive", _ALL),
    "claude-opus-4-6": ("adaptive", _NO_XHIGH),
    "claude-opus-4-5": ("extended", ("low", "medium", "high")),
    "claude-opus-4": ("extended", None),
    "claude-sonnet-5": ("adaptive", _ALL),
    "claude-sonnet-4-6": ("adaptive", _NO_XHIGH),
    "claude-sonnet-4": ("extended", None),
    "claude-sonnet-3-7": ("extended", None),
    "claude-3-7-sonnet": ("extended", None),
    "claude-haiku-5": ("adaptive", _ALL),
    "claude-haiku-4-5": ("extended", None),
}


def _anthropic_regime(model_name: str) -> str | None:
    """``"adaptive"`` / ``"extended"`` / ``None`` (no thinking)."""
    entry = _prefix_lookup(_ANTHROPIC_MODELS, model_name.lower())
    return None if entry is None else str(entry[0])


# Token-budget mapping for the extended-thinking regime. Tuned to
# roughly match the effort dial — these are not provider-documented
# constants, just defensible defaults (1024 is the API minimum).
_ANTHROPIC_LEGACY_BUDGET = {
    "minimal": 1024,
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
    validate_effort(effort)
    if effort is None:
        return {}
    entry = _prefix_lookup(_ANTHROPIC_MODELS, model_name.lower())
    if entry is None:
        _unsupported(
            model_name, effort,
            strict=strict,
            reason="this Claude model supports neither effort nor thinking",
        )
        return {}
    mode, levels = entry
    if mode == "adaptive":
        assert levels is not None
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": _clamp(effort, levels)},
        }
    budget = _ANTHROPIC_LEGACY_BUDGET[effort]
    kwargs: dict[str, Any] = {
        "thinking": {"type": "enabled", "budget_tokens": budget}
    }
    if levels is not None:  # Opus 4.5: effort composes with the budget
        kwargs["output_config"] = {"effort": _clamp(effort, levels)}
    return kwargs


# ---------------------------------------------------------------------------
# Test-only — clear the warning cache between unit tests so the
# one-time-per-pair logic doesn't bleed across tests.
# ---------------------------------------------------------------------------


def _reset_warned_cache_for_tests() -> None:
    """Internal: only intended for use by ``tests/test_effort.py``."""
    _WARNED.clear()
