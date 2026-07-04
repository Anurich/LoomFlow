"""Token-budgeted + decaying memory injection helpers (G7).

Seed-time memory injection historically used item-count limits
(5 facts / 3 episodes / all working blocks) with no token cap and no
recency weighting — one oversized episode could blow the context.
This module provides the pure scoring/packing primitives; the ReAct
seed assembly (:func:`loomflow.architecture.react._build_seed_messages`)
wires them in when ``Tuning(memory_token_budget=...)`` is set.

Design contract (documented here, enforced by the caller):

* **Working blocks are pinned** — they are always injected FIRST and
  COUNT against the budget, but are never dropped or truncated. Only
  the remaining allowance is offered to facts + episodes.
* **Score = relevance x decay** where decay is
  ``0.5 ** (age_days / half_life_days)``. Decay is opt-in
  (``half_life_days=None`` → decay 1.0); items without a timestamp
  never decay.
* **Greedy fill** best-score-first; the first item that does not fit
  fully is truncated to the remaining allowance with a
  :data:`TRUNCATION_MARKER` and packing stops. At least one item is
  always returned (truncated if need be) so a tiny budget still
  surfaces the single most relevant memory.

Everything here is dependency-free (stdlib only) so the architecture
layer can import it without touching heavier memory backends.
"""

from __future__ import annotations

from datetime import UTC, datetime

TRUNCATION_MARKER = "…[truncated]"
"""Suffix appended to the tail item when it only partially fits."""


def estimate_tokens(text: str) -> int:
    """Chars/4 heuristic — same estimator family the framework uses
    for tool-result caps and tool-def budgeting. Minimum 1 so empty
    strings still cost something (they occupy a list slot)."""
    return max(1, len(text) // 4)


def _age_days(now: datetime, ts: datetime) -> float:
    """Age in days, clamped at zero (future timestamps don't boost).

    Naive/aware mismatches are reconciled by assuming UTC for the
    naive side — memory backends store UTC by convention, and a
    wrong-but-bounded decay beats a ``TypeError`` mid-seed.
    """
    if ts.tzinfo is None and now.tzinfo is not None:
        ts = ts.replace(tzinfo=UTC)
    elif ts.tzinfo is not None and now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return max(0.0, (now - ts).total_seconds() / 86400.0)


def decay_factor(
    ts: datetime | None,
    *,
    half_life_days: float | None,
    now: datetime,
) -> float:
    """Exponential recency decay: ``0.5 ** (age_days / half_life)``.

    Returns 1.0 (no decay) when decay is disabled
    (``half_life_days`` is ``None`` or non-positive) or the item has
    no timestamp.
    """
    if half_life_days is None or half_life_days <= 0 or ts is None:
        return 1.0
    return float(0.5 ** (_age_days(now, ts) / half_life_days))


def budget_items(
    items: list[tuple[str, float, datetime | None]],
    *,
    budget_tokens: int,
    half_life_days: float | None,
    now: datetime,
) -> list[str]:
    """Pack ``(text, relevance, timestamp)`` items into a token budget.

    Items are scored ``relevance * decay`` (see :func:`decay_factor`),
    sorted best-first (original order is the tie-break, so equal
    scores preserve caller ordering), then greedily packed. The first
    item that doesn't fully fit is truncated to the remaining
    allowance and suffixed with :data:`TRUNCATION_MARKER`; packing
    stops there. The top-scored item is ALWAYS included — truncated
    to the budget when even it doesn't fit — so a small budget still
    injects the single best memory rather than nothing.

    Returns the selected (possibly tail-truncated) texts, best first.
    """
    if not items:
        return []

    scored: list[tuple[float, int, str]] = []
    for idx, (text, relevance, ts) in enumerate(items):
        score = relevance * decay_factor(
            ts, half_life_days=half_life_days, now=now
        )
        scored.append((score, idx, text))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))

    out: list[str] = []
    remaining = budget_tokens
    for _score, _idx, text in scored:
        cost = estimate_tokens(text)
        if cost <= remaining:
            out.append(text)
            remaining -= cost
            continue
        # Partial fit (or first item over a too-small budget):
        # truncate to the remaining allowance and stop packing.
        if remaining > 0 or not out:
            keep_chars = max(0, remaining) * 4
            out.append(text[:keep_chars].rstrip() + TRUNCATION_MARKER)
        break
    return out


__all__ = [
    "TRUNCATION_MARKER",
    "budget_items",
    "decay_factor",
    "estimate_tokens",
]
