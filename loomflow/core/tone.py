"""Response-tone presets + resolver.

A ``response_tone`` is a one-line style directive appended to the
agent's system prompt to steer *how* the model phrases its answer —
not *what* it answers. Tone is orthogonal to instructions and
persona:

* **Instructions** (``Agent(instructions=...)``) — what the agent
  is supposed to do.
* **Persona** (part of instructions: "You are a tax lawyer...")
  — who the agent is.
* **Tone** (``response_tone=``) — how the agent phrases its output.

The framework ships a small set of well-tested presets covering
the most common asks, plus accepts free-form strings for anything
else. Unknown strings are passed verbatim to the model — the
preset map is a convenience, not a gatekeeper.

Resolution order in the agent loop:

    per-call > agent default > workflow ambient > None (no tone)

When the resolved tone is ``None``, nothing is appended — the
feature is invisible until the user opts in.
"""

from __future__ import annotations

# Each preset is intentionally ONE sentence. We provide the dial;
# the model handles the details. Longer prompt fragments dilute the
# effect and chew through user-prompt-space budget.
RESPONSE_TONES: dict[str, str] = {
    "casual":
        "Respond in a warm, conversational tone. Use plain language "
        "and contractions; avoid jargon and stiff phrasing.",
    "professional":
        "Respond in a neutral, polished, professional tone. Be clear "
        "and structured; avoid slang and overly casual phrasing.",
    "technical":
        "Respond in a precise technical tone. Use exact terminology; "
        "show reasoning step by step; favor specificity over "
        "generality.",
    "legal":
        "Respond in a formal legal tone. Use precise legal "
        "terminology; qualify claims explicitly; avoid casual or "
        "persuasive language.",
    "finance":
        "Respond in an analytical financial tone. Use numbers, "
        "percentages, and timeframes; distinguish known data from "
        "estimates; cite figures when possible.",
    "executive":
        "Respond as an executive briefing: lead with the headline / "
        "decision, then 3-5 bullets of supporting detail. Concise; "
        "action-oriented.",
    "academic":
        "Respond in a formal academic tone. Use measured, hedged "
        "language where uncertainty exists; structure claims with "
        "supporting evidence; avoid colloquialisms.",
}


def resolve_response_tone(spec: str | None) -> str | None:
    """Map a tone spec to the directive that gets appended to the
    system prompt.

    * ``None`` → ``None``. Caller skips injection entirely.
    * Preset name (case-insensitive, e.g. ``"Legal"``) → the
      preset's one-sentence directive.
    * Anything else → returned verbatim. Free-form strings let
      callers pin a custom voice without registering it as a
      preset first. The framework treats the value as opaque —
      whatever the model does with it is the caller's choice.

    Returns the resolved directive string the agent loop should
    append, or ``None`` when no tone was requested.
    """
    if spec is None:
        return None
    stripped = spec.strip()
    if not stripped:
        return None
    return RESPONSE_TONES.get(stripped.lower(), stripped)


# ---------------------------------------------------------------------------
# System-prompt augmentation
# ---------------------------------------------------------------------------

# Suffix template — applied AFTER any schema directive so the tone
# guidance is the last thing the model reads before the user turn.
# Empirically, late-system-prompt instructions get the most weight.
_TONE_DIRECTIVE_TEMPLATE = "\n\n---\nResponse style: {directive}"


def append_tone_directive(base_instructions: str, tone: str | None) -> str:
    """Append a tone directive to a system prompt if a tone was
    requested; return ``base_instructions`` unchanged otherwise.

    Centralised so both ``Agent._loop`` (per-run) and any future
    consumer hit the same formatting. Idempotent for ``tone=None``.
    """
    directive = resolve_response_tone(tone)
    if directive is None:
        return base_instructions
    return base_instructions.rstrip() + _TONE_DIRECTIVE_TEMPLATE.format(
        directive=directive
    )
