"""Cost-estimation tests.

Guards the regression we just fixed where every model adapter
emitted ``cost_usd=0.0`` because no pricing table existed. Now:

* Every model adapter routes through :func:`estimate_cost`.
* The table covers production model IDs as of May 2026.
* Date-suffixed snapshots (``gpt-4.1-mini-2026-05-13``) hit the
  base rate via longest-prefix match.
* Unknown models return 0.0 + emit a one-time warning.
"""

from __future__ import annotations

import warnings

import pytest

from loomflow.model._pricing import (
    _WARNED_UNKNOWN,
    PRICING_PER_MTOKEN,
    estimate_cost,
)


def test_known_model_exact_match() -> None:
    """gpt-4.1-mini at $0.40 in / $1.60 out per 1M:
    1000 in + 500 out = (1000*0.40 + 500*1.60) / 1_000_000
                      = (400 + 800) / 1_000_000 = $0.0012"""
    cost = estimate_cost("gpt-4.1-mini", 1000, 500)
    assert cost == pytest.approx(0.0012, abs=1e-9)


def test_snapshot_id_falls_back_to_base_via_longest_prefix() -> None:
    """OpenAI's date-suffixed snapshot IDs (which the API returns)
    should hit the base model's price via longest-prefix match
    so users don't have to add every snapshot date to the table."""
    cost_base = estimate_cost("gpt-4.1-mini", 10_000, 5_000)
    cost_snap = estimate_cost("gpt-4.1-mini-2026-05-13", 10_000, 5_000)
    assert cost_base == cost_snap
    assert cost_base > 0  # sanity


def test_longest_prefix_picks_more_specific_over_generic() -> None:
    """``claude-opus-4-7`` is more specific than ``claude-opus``;
    both are in the table, and the longer key should win."""
    # claude-opus-4-7 and claude-opus both happen to share the same
    # (15.00, 75.00) pricing — verify by tweaking one in-place so the
    # two keys disagree, then confirm the longer key wins.
    saved = PRICING_PER_MTOKEN["claude-opus-4-7"]
    try:
        PRICING_PER_MTOKEN["claude-opus-4-7"] = (99.0, 99.0)
        # 1000 in + 1000 out at $99/$99 per 1M tokens:
        #   (1000 * 99 + 1000 * 99) / 1_000_000 = 198_000 / 1e6 = $0.198
        cost = estimate_cost("claude-opus-4-7-20251022", 1_000, 1_000)
        assert cost == pytest.approx(0.198, abs=1e-9)
    finally:
        PRICING_PER_MTOKEN["claude-opus-4-7"] = saved


def test_zero_tokens_returns_zero() -> None:
    assert estimate_cost("gpt-4.1-mini", 0, 0) == 0.0


def test_empty_model_returns_zero() -> None:
    assert estimate_cost("", 100, 100) == 0.0


def test_unknown_model_returns_zero_and_warns_once() -> None:
    """Unknown models should not crash — return 0.0 + emit a
    single warning per model name (no flood of warnings)."""
    _WARNED_UNKNOWN.discard("totally-fake-model-xyz")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert estimate_cost("totally-fake-model-xyz", 1000, 500) == 0.0
        # Second call: still zero, but NO new warning (one-time guard).
        assert estimate_cost("totally-fake-model-xyz", 2000, 1000) == 0.0
    msgs = [str(w.message) for w in caught]
    matched = [m for m in msgs if "totally-fake-model-xyz" in m]
    assert len(matched) == 1, f"expected exactly 1 warning, got {len(matched)}: {msgs}"


# ---------------------------------------------------------------------------
# Adapter integration — usage objects must carry non-zero cost when
# tokens are non-zero and the model is recognised.
# ---------------------------------------------------------------------------


def test_openai_adapter_attaches_cost_to_usage() -> None:
    """The OpenAI adapter's ``complete()`` path must compute cost
    from the (input_tokens, output_tokens) pair returned by the
    API. Regression fixture for the original bug: $0.0000 even
    when tokens were non-zero."""
    from loomflow.model.openai import OpenAIModel

    class _FakeUsage:
        prompt_tokens = 1000
        completion_tokens = 500

    class _FakeMessage:
        content = "hi"
        tool_calls = None

    class _FakeChoice:
        message = _FakeMessage()
        finish_reason = "stop"

    class _FakeResponse:
        usage = _FakeUsage()
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **_kwargs: object) -> _FakeResponse:
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    # Construct the adapter with a fake API key + inject the fake client.
    model = OpenAIModel("gpt-4.1-mini", api_key="sk-fake")
    model._client = _FakeClient()  # type: ignore[attr-defined]  # noqa: SLF001

    import asyncio

    text, _calls, usage, _finish = asyncio.run(
        model.complete(messages=[])
    )
    assert text == "hi"
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 500
    # gpt-4.1-mini: $0.40 in + $1.60 out per 1M = $0.0012
    assert usage.cost_usd == pytest.approx(0.0012, abs=1e-9)


def test_table_covers_currently_used_models() -> None:
    """Smoke test: the models our resolvers + examples reference
    by default must be priced. Catches accidental table drift
    after model renames."""
    required = [
        "gpt-4.1-mini",  # examples / Discord bot default
        "gpt-4o",
        "gpt-4o-mini",
        "claude-opus-4-7",  # README mentions this
        "claude-haiku-4-5",
    ]
    missing = [m for m in required if m not in PRICING_PER_MTOKEN]
    assert not missing, f"pricing table missing core models: {missing}"
