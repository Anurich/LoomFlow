"""Tests for auto-compact (0.10.19).

Auto-compact is the third tier of context-budget defence. It fires
inside the Ralph loop between architecture iterations when
``session.messages`` accumulates past ``auto_compact_at_tokens``.
The older half (up to ``keep_recent_turns`` user-anchored groups
from the end) is summarised into a single system message; the
recent tail stays verbatim.

Coverage:

* Helper unit tests: ``context_window_for`` substring matching,
  ``_split_at_user_anchor`` slicing logic, ``maybe_auto_compact``
  under threshold (no-op) / over threshold (compacted) / no
  anchor (no-op) / summariser failure (no-op).
* Agent kwarg validation: negative threshold rejected,
  keep_recent_turns >= 1 enforced, default disabled.
* End-to-end is exercised by the helper tests; wiring is
  trivially mypy-verified plus the manual smoke of "kwargs flow
  through to _auto_compact_* attributes."
"""

from __future__ import annotations

import warnings

import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn
from loomflow.agent import auto_compact as _auto_compact_mod
from loomflow.agent.auto_compact import (
    DEFAULT_AUTO_COMPACT_PCT,
    _split_at_user_anchor,
    context_window_for,
    maybe_auto_compact,
)
from loomflow.core.types import Message, Role

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# context_window_for — substring lookup
# ---------------------------------------------------------------------------


def test_context_window_known_claude() -> None:
    assert context_window_for("claude-opus-4-7") >= 200_000
    assert context_window_for("claude-haiku-4-5-20251001") >= 200_000


def test_context_window_known_openai() -> None:
    assert context_window_for("gpt-4.1-mini") == 1_000_000
    assert context_window_for("gpt-4o") == 128_000


def test_context_window_unknown_model_falls_back() -> None:
    """Anything not in the table returns the conservative
    8k default — users can override via auto_compact_at_tokens
    if their model has a known larger window."""
    # Unknown names also warn-once (covered separately); suppress
    # here so this test stays focused on the fallback value and
    # survives a future strict-warnings filter.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = context_window_for("ollama/some-weird-thing-7b")
    assert out == 8_192


def test_default_pct_constant_is_sensible() -> None:
    """Pinned via constant so a future tuning round is one
    line."""
    assert 0.0 < DEFAULT_AUTO_COMPACT_PCT < 1.0


# ---------------------------------------------------------------------------
# context_window_for — warn-once-per-process fallback
# ---------------------------------------------------------------------------


@pytest.fixture
def _clear_warned_models() -> None:
    """Reset the module-level warn-once dedupe set for isolation.

    The set persists for the life of the process, so a test that
    asserts "warns" must start from an empty set (otherwise a prior
    test that already warned about the same name would suppress it).
    """
    _auto_compact_mod._warned_unknown_models.clear()  # noqa: SLF001


def test_unknown_model_warns_once_and_returns_default(
    _clear_warned_models: None,
) -> None:
    """First lookup of an unrecognised name → UserWarning + the
    conservative 8k fallback. The squeeze is never silent."""
    with pytest.warns(UserWarning, match="unknown model"):
        out = context_window_for("totally-made-up-model-xyz")
    assert out == 8_192


def test_unknown_model_warns_only_once_per_name(
    _clear_warned_models: None,
) -> None:
    """Second + subsequent lookups of the SAME name must NOT warn —
    a hot loop calling ``context_window_for`` shouldn't spam stderr."""
    name = "another-made-up-model-abc"
    # Prime the set via a first call (this one warns; we suppress).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        context_window_for(name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = context_window_for(name)
    assert out == 8_192
    assert caught == []


def test_known_model_never_warns(_clear_warned_models: None) -> None:
    """A name that matches the table returns its real window and
    emits no warning at all."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = context_window_for("gpt-4o")
    assert out == 128_000
    assert caught == []


# ---------------------------------------------------------------------------
# _split_at_user_anchor — slicing logic
# ---------------------------------------------------------------------------


def _msg(role: Role, content: str) -> Message:
    return Message(role=role, content=content)


def test_split_too_few_turns_keeps_everything() -> None:
    msgs = [
        _msg(Role.USER, "q1"),
        _msg(Role.ASSISTANT, "a1"),
    ]
    older, recent = _split_at_user_anchor(msgs, keep_last_n_turns=4)
    assert older == []
    assert recent == msgs


def test_split_at_user_boundary() -> None:
    msgs = [
        _msg(Role.USER, "q1"),
        _msg(Role.ASSISTANT, "a1"),
        _msg(Role.USER, "q2"),
        _msg(Role.ASSISTANT, "a2"),
        _msg(Role.USER, "q3"),
        _msg(Role.ASSISTANT, "a3"),
    ]
    older, recent = _split_at_user_anchor(msgs, keep_last_n_turns=1)
    # keep_last_n=1 → keep just q3+a3, drop everything before.
    assert [m.content for m in older] == ["q1", "a1", "q2", "a2"]
    assert [m.content for m in recent] == ["q3", "a3"]


def test_split_no_user_messages_keeps_everything() -> None:
    msgs = [
        _msg(Role.ASSISTANT, "noop"),
    ]
    older, recent = _split_at_user_anchor(msgs, keep_last_n_turns=2)
    assert older == []
    assert recent == msgs


# ---------------------------------------------------------------------------
# maybe_auto_compact — full flow
# ---------------------------------------------------------------------------


async def test_no_compact_below_threshold() -> None:
    """``current_token_count <= at_tokens`` → returns ``(None,
    "")`` and never calls the summariser."""

    class _NeverCalledModel:
        name = "never"

        def stream(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("summariser must not be called")

    msgs = [_msg(Role.USER, "q1"), _msg(Role.ASSISTANT, "a1")]
    new_msgs, summary = await maybe_auto_compact(
        msgs,
        summariser=_NeverCalledModel(),  # type: ignore[arg-type]
        at_tokens=10_000,
        current_token_count=500,
    )
    assert new_msgs is None
    assert summary == ""


async def test_compact_replaces_older_with_summary() -> None:
    """Over threshold → summariser fires, older half replaced
    with a single system message carrying the summary."""
    summariser = ScriptedModel(
        turns=[ScriptedTurn(text="COMPACTED SUMMARY")]
    )
    msgs = [
        _msg(Role.SYSTEM, "be helpful"),
        _msg(Role.USER, "q1"),
        _msg(Role.ASSISTANT, "a1"),
        _msg(Role.USER, "q2"),
        _msg(Role.ASSISTANT, "a2"),
        _msg(Role.USER, "q3"),
        _msg(Role.ASSISTANT, "a3"),
    ]
    new_msgs, summary = await maybe_auto_compact(
        msgs,
        summariser=summariser,
        at_tokens=10,
        current_token_count=1000,
        keep_recent_turns=1,
    )
    assert new_msgs is not None
    assert summary == "COMPACTED SUMMARY"
    # Expected shape: leading system + summary system + last
    # turn group (q3, a3).
    roles = [m.role for m in new_msgs]
    assert roles == [
        Role.SYSTEM, Role.SYSTEM, Role.USER, Role.ASSISTANT,
    ]
    assert "[auto-compacted summary" in new_msgs[1].content
    assert "COMPACTED SUMMARY" in new_msgs[1].content
    assert new_msgs[2].content == "q3"
    assert new_msgs[3].content == "a3"


async def test_compact_strips_summariser_preamble() -> None:
    """Sometimes the model ignores the instruction and prefixes
    with 'Here is the summary:' — the helper strips that so the
    output is just the body."""
    summariser = ScriptedModel(
        turns=[ScriptedTurn(text="Here is the summary: actual body")]
    )
    msgs = [
        _msg(Role.USER, "q1"),
        _msg(Role.ASSISTANT, "a1"),
        _msg(Role.USER, "q2"),
        _msg(Role.ASSISTANT, "a2"),
    ]
    new_msgs, summary = await maybe_auto_compact(
        msgs,
        summariser=summariser,
        at_tokens=10,
        current_token_count=1000,
        keep_recent_turns=1,
    )
    assert new_msgs is not None
    assert summary == "actual body"


async def test_compact_no_anchor_skips() -> None:
    """No user messages → no anchor to slice at → returns no-op."""
    summariser = ScriptedModel(turns=[ScriptedTurn(text="x")])
    msgs = [
        _msg(Role.SYSTEM, "head"),
        _msg(Role.ASSISTANT, "only assistant"),
    ]
    new_msgs, summary = await maybe_auto_compact(
        msgs,
        summariser=summariser,
        at_tokens=10,
        current_token_count=1000,
    )
    assert new_msgs is None
    assert summary == ""


async def test_compact_summariser_raise_is_noop() -> None:
    """Summariser exception → graceful no-op (never kill a turn)."""

    class _RaisingModel:
        name = "raiser"

        def stream(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            async def _gen():  # type: ignore[no-untyped-def]
                raise RuntimeError("API down")
                yield  # pragma: no cover
            return _gen()

    msgs = [
        _msg(Role.USER, "q1"),
        _msg(Role.ASSISTANT, "a1"),
        _msg(Role.USER, "q2"),
        _msg(Role.ASSISTANT, "a2"),
        _msg(Role.USER, "q3"),
        _msg(Role.ASSISTANT, "a3"),
    ]
    new_msgs, summary = await maybe_auto_compact(
        msgs,
        summariser=_RaisingModel(),  # type: ignore[arg-type]
        at_tokens=10,
        current_token_count=1000,
        keep_recent_turns=1,
    )
    assert new_msgs is None
    assert summary == ""


async def test_compact_empty_summary_is_noop() -> None:
    """Whitespace-only summary → no-op. Shipping an empty
    summary in place of real content would mislead the model
    into thinking it had context it doesn't."""
    summariser = ScriptedModel(turns=[ScriptedTurn(text="   \n  ")])
    msgs = [
        _msg(Role.USER, "q1"),
        _msg(Role.ASSISTANT, "a1"),
        _msg(Role.USER, "q2"),
        _msg(Role.ASSISTANT, "a2"),
    ]
    new_msgs, summary = await maybe_auto_compact(
        msgs,
        summariser=summariser,
        at_tokens=10,
        current_token_count=1000,
        keep_recent_turns=1,
    )
    assert new_msgs is None


# ---------------------------------------------------------------------------
# Agent kwarg validation
# ---------------------------------------------------------------------------


def test_agent_auto_compact_disabled_by_default() -> None:
    agent = Agent("you help", model="echo")
    assert agent._auto_compact_at_tokens is None
    assert agent._auto_compact_summariser is None


def test_agent_auto_compact_negative_threshold_rejected() -> None:
    with pytest.raises(
        ValueError, match="auto_compact_at_tokens must be > 0"
    ):
        Agent(
            "you help", model="echo", auto_compact_at_tokens=-1
        )


def test_agent_auto_compact_zero_threshold_rejected() -> None:
    """``0`` is rejected — the user means ``None`` (disable),
    expressing it as 0 is a typo trap that would otherwise mean
    'compact every turn even for tiny conversations'."""
    with pytest.raises(
        ValueError, match="auto_compact_at_tokens must be > 0"
    ):
        Agent(
            "you help", model="echo", auto_compact_at_tokens=0
        )


def test_agent_auto_compact_keep_recent_zero_rejected() -> None:
    with pytest.raises(
        ValueError, match="auto_compact_keep_recent_turns must be >= 1"
    ):
        Agent(
            "you help",
            model="echo",
            auto_compact_at_tokens=1000,
            auto_compact_keep_recent_turns=0,
        )


def test_agent_auto_compact_summariser_defaults_to_main_model() -> None:
    """When ``auto_compact_at_tokens=N`` is set but no separate
    summariser is given, the main model is reused. Single-kwarg
    opt-in for the common case."""
    agent = Agent(
        "you help",
        model="echo",
        auto_compact_at_tokens=5000,
    )
    assert agent._auto_compact_summariser is agent._model


def test_agent_auto_compact_explicit_summariser_wins() -> None:
    """When ``auto_compact_summariser=`` is provided, it's used
    instead of the main model. Useful for Opus-main + Haiku-
    summariser cost optimisation."""
    cheap = ScriptedModel(turns=[ScriptedTurn(text="x")])
    agent = Agent(
        "you help",
        model="echo",
        auto_compact_at_tokens=5000,
        auto_compact_summariser=cheap,
    )
    assert agent._auto_compact_summariser is cheap
