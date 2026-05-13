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


# ---------------------------------------------------------------------------
# Cache-read / cache-write multipliers, per provider
# ---------------------------------------------------------------------------
#
# Cache-read: cached prompt tokens cost this fraction of the base input
# rate. OpenAI gives 50%; Anthropic and Gemini give 90%.
#
# Cache-write: writing tokens to the cache costs this multiple of the
# base input rate. OpenAI doesn't charge separately for writes (so we
# never bill cache_write_tokens for OpenAI). Anthropic charges 1.25x
# for 5-minute TTL and 2x for 1-hour TTL.

_CACHE_READ_MULTIPLIER: dict[str, float] = {
    "openai": 0.5,
    "anthropic": 0.1,
    "gemini": 0.1,
    "litellm": 0.5,   # routed; conservative
}

_CACHE_WRITE_MULTIPLIER: dict[str, dict[str, float]] = {
    "openai": {"5m": 0.0, "1h": 0.0},     # OpenAI doesn't bill writes
    "anthropic": {"5m": 1.25, "1h": 2.0},
    "gemini": {"5m": 0.0, "1h": 0.0},     # cache storage billed separately
    "litellm": {"5m": 1.25, "1h": 2.0},   # assume Anthropic-style
}


def _provider_for(model: str) -> str:
    """Map a model name onto its provider family for cache-rate
    lookup. Detection is name-prefix based — the same heuristic used
    by ``_resolve_model`` in :mod:`loomflow.agent.api`.
    """
    if not model:
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if model.startswith("gemini-"):
        return "gemini"
    return "openai"  # safe default for the longest-prefix fallback path


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_ttl: str = "5m",
) -> float:
    """Return the USD cost of a model call given its token buckets.

    Argument semantics (Anthropic-style **separate buckets**):

    * ``input_tokens`` — full-rate (cache miss / caching disabled).
    * ``cached_input_tokens`` — cache hits, at the provider's
      discount multiplier (OpenAI 0.5x, Anthropic / Gemini 0.1x).
    * ``cache_write_tokens`` — tokens being written into cache on
      this call. Anthropic only (1.25x for 5m TTL, 2x for 1h);
      OpenAI doesn't bill writes.
    * ``output_tokens`` — completion at the model's output rate.
    * ``cache_ttl`` — ``"5m"`` (default) or ``"1h"``. Affects only
      the cache-write rate.

    Lookup order:

    1. Exact match against :data:`PRICING_PER_MTOKEN`.
    2. **Longest-prefix** match (``gpt-4.1-mini-2026-05-13`` →
       ``gpt-4.1-mini``).
    3. Miss → return ``0.0`` and warn once per unknown model.
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
    in_rate, out_rate = pricing

    provider = _provider_for(model)
    read_mult = _CACHE_READ_MULTIPLIER.get(provider, 0.5)
    write_mult = _CACHE_WRITE_MULTIPLIER.get(provider, {}).get(
        cache_ttl, 1.25
    )

    total = (
        input_tokens * in_rate
        + cached_input_tokens * (in_rate * read_mult)
        + cache_write_tokens * (in_rate * write_mult)
        + output_tokens * out_rate
    )
    return total / 1_000_000.0


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
