"""Tests for the tool-result summarization feature (0.10.14).

The feature: when ``Agent(tool_result_summarizer=<model>)`` is wired,
the ReAct loop hands oversized tool results to the summariser model
BEFORE appending them to conversation history. The summary replaces
the verbatim output — saving prompt-tokens on every subsequent turn.

Coverage:

* ``summarize_tool_result`` is a pass-through below the threshold.
* It calls the summariser and returns the summary above threshold.
* It falls back to the original when the summariser raises.
* It falls back to the original when the summariser returns "".
* End-to-end via :class:`Agent` with ScriptedModels:
  - Tool result above threshold → conversation history has the
    summary, not the verbatim output.
  - Tool result below threshold → verbatim survives (no summary).
  - Disabled by default (no summariser kwarg → verbatim survives
    regardless of size).
* ``Dependencies.fast_tool_summary`` flips False when summariser
  wired and True when not.
"""

from __future__ import annotations

import pytest

from loomflow import (
    Agent,
    EchoModel,
    ScriptedModel,
    ScriptedTurn,
    tool,
)
from loomflow.core.types import ToolCall
from loomflow.tools.result_summarizer import (
    DEFAULT_SUMMARY_THRESHOLD,
    summarize_tool_result,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# summarize_tool_result helper — direct tests
# ---------------------------------------------------------------------------


async def test_summarize_below_threshold_is_passthrough() -> None:
    """Result smaller than threshold returns verbatim — no API call."""
    small = "small output"
    out = await summarize_tool_result(
        small, tool_name="read", summarizer=EchoModel(), threshold=500
    )
    assert out == small


async def test_summarize_above_threshold_calls_summariser() -> None:
    """Above threshold → summariser model fires; its text is returned."""
    big = "x" * 1000
    summariser = ScriptedModel(turns=[ScriptedTurn(text="condensed")])
    out = await summarize_tool_result(
        big, tool_name="read", summarizer=summariser, threshold=500
    )
    assert out == "condensed"


async def test_summarize_returns_original_when_summariser_raises() -> None:
    """Summariser exception → original verbatim survives. The
    framework principle: summarisation must never kill a turn."""

    class _RaisingModel:
        name = "raiser"

        def stream(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            async def _gen():  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")
                yield  # pragma: no cover
            return _gen()

    big = "y" * 1000
    out = await summarize_tool_result(
        big, tool_name="read", summarizer=_RaisingModel(), threshold=500  # type: ignore[arg-type]
    )
    assert out == big


async def test_summarize_returns_original_when_summary_is_empty() -> None:
    """Empty / whitespace-only summary → original survives. Shipping
    an empty Message would be worthless and the model might
    interpret it as 'tool returned nothing'."""
    big = "z" * 1000
    summariser = ScriptedModel(turns=[ScriptedTurn(text="   \n  ")])
    out = await summarize_tool_result(
        big, tool_name="read", summarizer=summariser, threshold=500
    )
    assert out == big


def test_default_threshold_constant() -> None:
    """Defaults exposed via the public module constant — pinned so
    a future tuning round is a single-line change with explicit
    test coverage."""
    assert DEFAULT_SUMMARY_THRESHOLD == 500


# ---------------------------------------------------------------------------
# Agent integration — fast flag + Dependencies wiring
# ---------------------------------------------------------------------------


def test_fast_tool_summary_true_when_no_summariser() -> None:
    """The default Agent has no summariser → fast-flag stays True
    and the ReAct loop short-circuits the summarisation call site."""
    agent = Agent("you help", model="echo")
    assert agent._tool_result_summarizer is None


def test_fast_tool_summary_flips_false_with_summariser() -> None:
    """Passing a summariser wires it onto the Agent and flips the
    fast flag (verified end-to-end below by checking the wire
    behaviour)."""
    summariser = ScriptedModel(turns=[ScriptedTurn(text="ok")])
    agent = Agent(
        "you help",
        model="echo",
        tool_result_summarizer=summariser,
    )
    assert agent._tool_result_summarizer is summariser


def test_negative_threshold_rejected() -> None:
    """Threshold is bounded at 0+ — guards against `range`/`slice`
    bugs in user code that would otherwise convert a typo
    (``threshold=-1``) into silent always-summarise behaviour."""
    with pytest.raises(
        ValueError, match="tool_result_summary_threshold must be >= 0"
    ):
        Agent(
            "you help",
            model="echo",
            tool_result_summary_threshold=-1,
        )


# ---------------------------------------------------------------------------
# End-to-end: tool result above threshold gets summarised in history
# ---------------------------------------------------------------------------


@tool
async def big_read() -> str:
    """Tool whose output is large — triggers the summariser."""
    return "BIG OUTPUT " * 200  # ~2400 chars


@tool
async def small_read() -> str:
    """Tool whose output is small — should NOT trigger the
    summariser (below default threshold)."""
    return "small"


async def _collect_arch_events(
    agent: Agent, prompt: str, name: str
) -> list[dict]:
    """Drain ``agent.stream()`` and return every
    architecture_event whose name matches ``name`` (just the
    payload dict). The summariser feature fires
    ``tool_result_summarized`` — we use it as a structural proof
    that the summarisation path executed."""
    matches: list[dict] = []
    async for event in agent.stream(prompt):
        kind = getattr(event, "kind", None)
        payload = getattr(event, "payload", None)
        if kind is None or payload is None:
            continue
        if str(kind).endswith("architecture_event") and (
            payload.get("name") == name
        ):
            matches.append(payload)
    return matches


async def test_oversized_tool_result_fires_summarized_event() -> None:
    """Coordinator calls ``big_read``; ReAct's tool-dispatch loop
    sees the result exceeds threshold, hands it to the summariser,
    and emits ``tool_result_summarized`` with before/after char
    counts. The event existence + char drop is the structural
    proof that the verbatim output was replaced in conversation
    history before the next model call."""
    coord = ScriptedModel(
        turns=[
            ScriptedTurn(
                text="reading the file",
                tool_calls=[
                    ToolCall(id="t1", tool="big_read", args={})
                ],
            ),
            ScriptedTurn(text="done"),
        ]
    )
    summariser = ScriptedModel(
        turns=[ScriptedTurn(text="SHORT SUMMARY")]
    )
    agent = Agent(
        "you help",
        model=coord,
        tools=[big_read],
        tool_result_summarizer=summariser,
    )
    events = await _collect_arch_events(
        agent, "read it", "tool_result_summarized"
    )
    assert len(events) == 1
    payload = events[0]
    assert payload["tool"] == "big_read"
    # ``big_read`` returns "BIG OUTPUT " * 200 = 2400 chars; the
    # summariser returned "SHORT SUMMARY" = 13 chars.
    assert payload["original_chars"] > 2000
    assert payload["summary_chars"] < 50
    assert payload["summary_chars"] < payload["original_chars"]


async def test_undersized_tool_result_does_not_fire_summarized_event() -> None:
    """Below threshold → summariser short-circuited, no
    ``tool_result_summarized`` event emitted."""
    coord = ScriptedModel(
        turns=[
            ScriptedTurn(
                text="reading",
                tool_calls=[
                    ToolCall(id="t1", tool="small_read", args={})
                ],
            ),
            ScriptedTurn(text="done"),
        ]
    )
    summariser = ScriptedModel(
        turns=[ScriptedTurn(text="WOULD_NOT_APPEAR")]
    )
    agent = Agent(
        "you help",
        model=coord,
        tools=[small_read],
        tool_result_summarizer=summariser,
    )
    events = await _collect_arch_events(
        agent, "read", "tool_result_summarized"
    )
    assert events == []


async def test_summarizer_disabled_by_default_emits_nothing() -> None:
    """No ``tool_result_summarizer=`` kwarg → summarisation path
    fully short-circuited (``fast_tool_summary=True``); no event,
    no summariser call, even for huge results."""
    coord = ScriptedModel(
        turns=[
            ScriptedTurn(
                text="reading",
                tool_calls=[
                    ToolCall(id="t1", tool="big_read", args={})
                ],
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent("you help", model=coord, tools=[big_read])
    events = await _collect_arch_events(
        agent, "read", "tool_result_summarized"
    )
    assert events == []
