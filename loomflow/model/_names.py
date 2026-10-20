"""Canonical model names for capability and price lookups.

Gateways such as OpenRouter name models ``vendor/model``
(``anthropic/claude-haiku-5.5``, ``openai/gpt-5-mini``), add
``~vendor/...`` aliases and ``:variant`` suffixes (``:free``,
``:nitro``), and write Claude versions with dots. loomflow's tables —
which models take native JSON outputs, effort, ``max_completion_tokens``,
which tokenizer, context windows, prices — are keyed by the vendors'
own ids, so lookups go through :func:`canonical_model_name` first. The
model name sent on the wire is never changed.
"""

from __future__ import annotations

__all__ = ["canonical_model_name"]

# Vendor prefixes whose models keep the vendor's own capabilities and
# prices behind a gateway.
GATEWAY_VENDORS = ("anthropic/", "openai/", "deepseek/", "google/")


def canonical_model_name(name: str, *, strip_variant: bool = True) -> str:
    """The vendor's id for a gateway model name: ``anthropic/claude-haiku-5.5``
    → ``claude-haiku-5-5``, ``openai/gpt-5-mini:free`` → ``gpt-5-mini``.
    Names without a known vendor prefix come back unchanged.
    ``strip_variant=False`` keeps a ``:variant`` suffix (price lookups:
    a variant may be priced differently)."""
    stripped = name.strip()
    if stripped.startswith("~"):
        stripped = stripped[1:]
    lowered = stripped.lower()
    vendor = next((v for v in GATEWAY_VENDORS if lowered.startswith(v)), None)
    if vendor is None:
        return name
    model = stripped[len(vendor) :]
    if strip_variant:
        model = model.split(":", 1)[0]
    if model.lower().startswith("claude-"):
        model = model.replace(".", "-")
    return model
