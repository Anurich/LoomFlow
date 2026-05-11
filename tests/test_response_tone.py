"""``response_tone=`` — system-prompt style steering.

Covers:

* Preset names map to the correct one-sentence directives.
* Free-form strings pass through verbatim (the preset map is a
  convenience, not a gatekeeper).
* ``None`` (default) is a no-op — no directive in the system
  prompt at all.
* Resolution precedence at run time:
  per-call > agent default > workflow ambient > None.
* Tone is appended AFTER any schema directive so it's the last
  thing the model reads.

All tests use ``_CapturingScripted`` so we can assert on the exact
system-prompt text the model received.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from loomflow import Agent, Role, Workflow
from loomflow.core.tone import (
    RESPONSE_TONES,
    append_tone_directive,
    resolve_response_tone,
)
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Capturing scripted model — same helper as test_structured_output.py
# ---------------------------------------------------------------------------


class _CapturingScripted(ScriptedModel):
    """Records the messages it was called with so tests can verify
    the system-prompt augmentation."""

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.captured: list[list[Any]] = []

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.captured.append(list(messages))
        return await super().complete(messages, **kwargs)


def _system_text(captured: list[list[Any]]) -> str:
    """Pull the system-message text from the first captured call."""
    return next(
        (m.content for m in captured[0] if m.role == Role.SYSTEM),
        "",
    )


# ---------------------------------------------------------------------------
# Pure resolver tests — no agent needed
# ---------------------------------------------------------------------------


def test_resolve_response_tone_returns_none_for_none() -> None:
    assert resolve_response_tone(None) is None


def test_resolve_response_tone_returns_none_for_empty_string() -> None:
    """Empty / whitespace-only spec is treated the same as None —
    accidentally passing an empty string from a config doesn't
    inject a meaningless 'respond in style: ' directive."""
    assert resolve_response_tone("") is None
    assert resolve_response_tone("   ") is None


def test_resolve_response_tone_maps_known_preset() -> None:
    """A preset name should resolve to that preset's one-sentence
    directive — case-insensitively, so users can write ``"Legal"``
    or ``"LEGAL"`` interchangeably."""
    assert resolve_response_tone("legal") == RESPONSE_TONES["legal"]
    assert resolve_response_tone("LEGAL") == RESPONSE_TONES["legal"]
    assert resolve_response_tone("Casual") == RESPONSE_TONES["casual"]


def test_resolve_response_tone_passes_through_freeform() -> None:
    """An unknown string is treated as free-form passthrough — the
    preset map is convenience, not a gatekeeper. Users can pin a
    custom voice without registering it as a preset first."""
    custom = "Respond like a friendly doctor — warm but precise."
    assert resolve_response_tone(custom) == custom


def test_append_tone_directive_appends_after_base_instructions() -> None:
    """The tone directive is appended; the base instructions stay
    intact at the front (the model reads top-to-bottom and we want
    persona/role first, style guidance last)."""
    out = append_tone_directive("You are an assistant.", "legal")
    assert out.startswith("You are an assistant.")
    assert RESPONSE_TONES["legal"] in out
    # No directive added when tone is None.
    assert append_tone_directive(
        "base", None
    ) == "base"


# ---------------------------------------------------------------------------
# Agent-level: per-call wins over agent default
# ---------------------------------------------------------------------------


async def test_agent_default_tone_appears_in_system_prompt() -> None:
    """``Agent(response_tone='legal')`` should append the legal
    directive to the system prompt for every run."""
    model = _CapturingScripted([ScriptedTurn(text="ok")])
    agent = Agent("base", model=model, response_tone="legal")

    await agent.run("hello")

    sys_text = _system_text(model.captured)
    assert "base" in sys_text
    assert RESPONSE_TONES["legal"] in sys_text


async def test_per_call_response_tone_overrides_agent_default() -> None:
    """When both ``Agent(response_tone=...)`` and
    ``run(response_tone=...)`` are set, the per-call value wins —
    same precedence pattern as ``output_schema=``."""
    model = _CapturingScripted([ScriptedTurn(text="ok")])
    agent = Agent("base", model=model, response_tone="legal")

    await agent.run("hello", response_tone="casual")

    sys_text = _system_text(model.captured)
    assert RESPONSE_TONES["casual"] in sys_text
    assert RESPONSE_TONES["legal"] not in sys_text


async def test_no_tone_set_means_no_directive() -> None:
    """No tone set anywhere → no directive in the system prompt.
    The feature is invisible until the user opts in."""
    model = _CapturingScripted([ScriptedTurn(text="ok")])
    agent = Agent("base", model=model)  # no response_tone

    await agent.run("hello")

    sys_text = _system_text(model.captured)
    assert "Response style:" not in sys_text


async def test_freeform_tone_string_passes_through_to_prompt() -> None:
    """A free-form spec ends up verbatim in the system prompt —
    same convenience users get for custom personas, just on the
    tone axis."""
    custom = "Respond in the voice of a 1920s newspaper editor."
    model = _CapturingScripted([ScriptedTurn(text="ok")])
    agent = Agent("base", model=model, response_tone=custom)

    await agent.run("hello")

    sys_text = _system_text(model.captured)
    assert custom in sys_text


# ---------------------------------------------------------------------------
# Workflow propagation: ambient flows into nested agents
# ---------------------------------------------------------------------------


async def test_workflow_response_tone_propagates_to_nested_agent() -> None:
    """``Workflow(response_tone='finance')`` should be picked up
    by nested Agent steps that didn't set their own tone — same
    contextvar propagation pattern as ``Workflow(memory=...)``."""
    model = _CapturingScripted([ScriptedTurn(text="42")])
    agent = Agent("base", model=model)  # no per-agent tone

    wf = Workflow.chain([agent], response_tone="finance")
    await wf.run("hi")

    sys_text = _system_text(model.captured)
    assert RESPONSE_TONES["finance"] in sys_text


async def test_agent_tone_overrides_workflow_ambient() -> None:
    """Explicit tone on the Agent should beat the workflow ambient.
    Same precedence rule as memory propagation."""
    model = _CapturingScripted([ScriptedTurn(text="42")])
    agent = Agent("base", model=model, response_tone="legal")

    wf = Workflow.chain([agent], response_tone="finance")
    await wf.run("hi")

    sys_text = _system_text(model.captured)
    assert RESPONSE_TONES["legal"] in sys_text
    assert RESPONSE_TONES["finance"] not in sys_text


async def test_workflow_response_tone_does_not_leak_between_runs() -> None:
    """The ambient contextvar must reset after each workflow run
    so a later workflow without ``response_tone=`` doesn't inherit
    the prior workflow's tone."""
    model = _CapturingScripted(
        [ScriptedTurn(text="a"), ScriptedTurn(text="b")]
    )
    agent = Agent("base", model=model)

    # First workflow sets tone.
    wf_legal = Workflow.chain([agent], response_tone="legal")
    await wf_legal.run("first")

    # Second workflow does NOT set tone.
    wf_plain = Workflow.chain([agent])
    await wf_plain.run("second")

    # Captured 2 calls. First should have legal; second should not.
    assert RESPONSE_TONES["legal"] in _system_text([model.captured[0]])
    assert "Response style:" not in _system_text([model.captured[1]])


# ---------------------------------------------------------------------------
# Interaction with output_schema — both directives present
# ---------------------------------------------------------------------------


async def test_tone_appended_after_schema_directive() -> None:
    """When both ``response_tone`` and ``output_schema`` are set
    AND the model lacks native structured output (so the schema
    directive does get added), tone is appended AFTER the schema
    so it's the last thing the model reads. Late-in-system-prompt
    instructions get the most weight."""
    from pydantic import BaseModel

    class Out(BaseModel):
        ans: str

    payload = {"ans": "42"}
    model = _CapturingScripted([ScriptedTurn(text=json.dumps(payload))])
    # Scripted fake doesn't claim native structured-output → schema
    # directive will be injected by the agent loop.

    agent = Agent("base", model=model, response_tone="casual")
    await agent.run("hello", output_schema=Out)

    sys_text = _system_text(model.captured)
    # Both present.
    assert "STRUCTURED OUTPUT REQUIRED" in sys_text
    assert RESPONSE_TONES["casual"] in sys_text
    # Tone comes AFTER schema directive in the text.
    schema_idx = sys_text.index("STRUCTURED OUTPUT REQUIRED")
    tone_idx = sys_text.index(RESPONSE_TONES["casual"])
    assert tone_idx > schema_idx
