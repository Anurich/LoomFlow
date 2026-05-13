"""Cost estimation for model calls.

Every model adapter that knows the (input_tokens, output_tokens) of
a call routes through :func:`estimate_cost` to attach a USD figure
to :class:`~loomflow.core.types.Usage`. Without this, every
``result.cost_usd`` is ``0.0`` and the ``StandardBudget(max_cost_usd=)``
cap is unenforceable.

Snapshots, not facts. Provider pricing changes; the table below
captures rates as of **May 2026** for the models loomflow's adapters
target by default. Two ways to keep up:

* Override at the call site — adapters accept a ``cost_per_mtoken``
  override kwarg (not implemented yet — opens room for users with
  negotiated rates / enterprise discounts).
* Update :data:`PRICING_PER_MTOKEN` and ship a patch release.

Models the table doesn't recognise fall through to a longest-prefix
match (so ``gpt-4.1-mini-2026-05-13`` still gets the
``gpt-4.1-mini`` rate). If even that misses, the call returns
``0.0`` and emits a one-time warning per unknown model — quiet
enough for production noise, loud enough to surface a typo.
"""

from __future__ import annotations

import warnings

# Prices in USD per **1 million tokens**, as ``(input, output)``.
# Cached input tokens (OpenAI 50%, Anthropic ~10%) are NOT yet
# discounted here — treat the figures as upper bounds.
PRICING_PER_MTOKEN: dict[str, tuple[float, float]] = {
    # ----- OpenAI ---------------------------------------------------------
    "gpt-4.1":          (2.00,   8.00),
    "gpt-4.1-mini":     (0.40,   1.60),
    "gpt-4.1-nano":     (0.10,   0.40),
    "gpt-4o":           (2.50,  10.00),
    "gpt-4o-mini":      (0.15,   0.60),
    "gpt-4-turbo":      (10.00, 30.00),
    "gpt-4":            (30.00, 60.00),
    "gpt-3.5-turbo":    (0.50,   1.50),
    # Reasoning models
    "o1":               (15.00, 60.00),
    "o1-preview":       (15.00, 60.00),
    "o1-mini":          (3.00,  12.00),
    "o3":               (10.00, 40.00),
    "o3-mini":          (1.10,   4.40),
    "o4-mini":          (1.10,   4.40),

    # ----- Anthropic ------------------------------------------------------
    "claude-opus-4-7":  (15.00, 75.00),
    "claude-opus-4-6":  (15.00, 75.00),
    "claude-opus-4-5":  (15.00, 75.00),
    "claude-opus-4-1":  (15.00, 75.00),
    "claude-opus-4-0":  (15.00, 75.00),
    "claude-opus":      (15.00, 75.00),  # generic fallback
    "claude-sonnet-4-6":(3.00,  15.00),
    "claude-sonnet-4-5":(3.00,  15.00),
    "claude-sonnet-4-0":(3.00,  15.00),
    "claude-sonnet":    (3.00,  15.00),
    "claude-haiku-4-5": (1.00,   5.00),
    "claude-haiku-4-0": (0.80,   4.00),
    "claude-haiku":     (1.00,   5.00),
    "claude-3-5-sonnet":(3.00,  15.00),
    "claude-3-5-haiku": (0.80,   4.00),
    "claude-3-opus":    (15.00, 75.00),
    "claude-3-haiku":   (0.25,   1.25),

    # ----- LiteLLM-routed common providers --------------------------------
    "mistral-large":    (3.00,   9.00),
    "mistral-medium":   (2.70,   8.10),
    "mistral-small":    (1.00,   3.00),
    "command-r-plus":   (3.00,  15.00),
    "command-r":        (0.15,   0.60),
    "gemini-1.5-pro":   (1.25,   5.00),
    "gemini-1.5-flash": (0.075,  0.30),
    "gemini-2.0-flash": (0.10,   0.40),
}


# Track models we've already warned about so the log isn't flooded.
_WARNED_UNKNOWN: set[str] = set()


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Return the USD cost of ``input_tokens`` + ``output_tokens``
    for ``model``, or ``0.0`` for unknown models.

    Lookup order:

    1. Exact match against :data:`PRICING_PER_MTOKEN`.
    2. **Longest-prefix** match (so ``gpt-4.1-mini-2026-05-13``
       still hits the ``gpt-4.1-mini`` rate).
    3. Miss → return ``0.0`` and warn once per unknown model.

    The longest-prefix step keeps the table small while tolerating
    OpenAI's date-suffixed snapshot model IDs.
    """
    if not model:
        return 0.0
    pricing = PRICING_PER_MTOKEN.get(model)
    if pricing is None:
        pricing = _longest_prefix_match(model)
    if pricing is None:
        if model not in _WARNED_UNKNOWN:
            _WARNED_UNKNOWN.add(model)
            warnings.warn(
                f"cost estimation: unknown model {model!r}. Add it to "
                "loomflow.model._pricing.PRICING_PER_MTOKEN to track "
                "spend; the call will be reported as $0.00 in usage.",
                stacklevel=3,
            )
        return 0.0
    in_price_per_mtok, out_price_per_mtok = pricing
    return (
        (input_tokens * in_price_per_mtok) +
        (output_tokens * out_price_per_mtok)
    ) / 1_000_000.0


def _longest_prefix_match(model: str) -> tuple[float, float] | None:
    """Find the longest key in the pricing table that ``model`` starts
    with. Returns ``None`` if no key is a prefix.

    Examples (assuming the table):

    * ``"gpt-4.1-mini-2026-05-13"`` → matches ``"gpt-4.1-mini"``
    * ``"claude-opus-4-7-20251022"`` → matches ``"claude-opus-4-7"``
    * ``"foo-bar"`` → no match
    """
    best: str | None = None
    for key in PRICING_PER_MTOKEN:
        if model.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return PRICING_PER_MTOKEN[best] if best is not None else None
