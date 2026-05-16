"""Prompt caching tests — resolver, pricing math, adapter behaviour.

Guards the end-to-end caching wiring:

* :func:`_resolve_prompt_caching` normalises bool / dict / None
  into :class:`PromptCacheConfig`.
* :func:`estimate_cost` applies the per-provider cache-read discount
  (OpenAI 0.5x, Anthropic 0.1x) and the cache-write premium
  (Anthropic 1.25x for 5m, 2x for 1h; OpenAI no write fee).
* :class:`AnthropicModel` injects ``cache_control`` on the last
  system block + last tool when enabled, and parses
  ``cache_read_input_tokens`` / ``cache_creation_input_tokens``
  from the response.
* :class:`OpenAIModel` parses ``prompt_tokens_details.cached_tokens``
  and normalises to loomflow's separate-bucket Usage convention.
* :class:`Agent`'s ``prompt_caching=`` kwarg flows through.
* :class:`RunResult` carries the cache fields.
"""

from __future__ import annotations

import asyncio

import pytest

from loomflow import Agent, ConfigError
from loomflow.agent.api import _resolve_prompt_caching
from loomflow.core.types import PromptCacheConfig
from loomflow.model._pricing import estimate_cost
from loomflow.model.anthropic import (
    AnthropicModel,
    _apply_anthropic_cache_control,
    _cache_control_for,
)
from loomflow.model.openai import OpenAIModel

# ---------------------------------------------------------------------------
# Resolver — bool / dict / None → PromptCacheConfig
# ---------------------------------------------------------------------------


def test_resolver_none_disables() -> None:
    cfg = _resolve_prompt_caching(None)
    assert cfg.enabled is False


def test_resolver_false_disables() -> None:
    cfg = _resolve_prompt_caching(False)
    assert cfg.enabled is False


def test_resolver_true_enables_with_5m_default() -> None:
    cfg = _resolve_prompt_caching(True)
    assert cfg.enabled is True
    assert cfg.ttl == "5m"
    assert cfg.cache_key is None


def test_resolver_dict_with_ttl_and_cache_key() -> None:
    cfg = _resolve_prompt_caching(
        {"enabled": True, "ttl": "1h", "cache_key": "channel_42"}
    )
    assert cfg.enabled is True
    assert cfg.ttl == "1h"
    assert cfg.cache_key == "channel_42"


def test_resolver_dict_defaults_enabled_true() -> None:
    """Empty dict means ``{"enabled": True}`` — the user already
    expressed intent to configure something."""
    cfg = _resolve_prompt_caching({})
    assert cfg.enabled is True
    assert cfg.ttl == "5m"


def test_resolver_rejects_unknown_ttl() -> None:
    with pytest.raises(ConfigError, match="ttl"):
        _resolve_prompt_caching({"ttl": "30m"})


def test_resolver_rejects_non_string_cache_key() -> None:
    with pytest.raises(ConfigError, match="cache_key"):
        _resolve_prompt_caching({"cache_key": 42})


def test_resolver_rejects_unknown_type() -> None:
    with pytest.raises(ConfigError, match="unrecognised"):
        _resolve_prompt_caching("yes please")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pricing — cache-aware math, per-provider rates
# ---------------------------------------------------------------------------


def test_anthropic_cache_read_costs_one_tenth() -> None:
    """Claude Opus 4.7: $15/MTok input. 1000 cached tokens should
    cost (1000 * 15 * 0.1) / 1_000_000 = $0.0015 — 10x cheaper
    than the $0.015 those tokens would cost uncached."""
    cached_cost = estimate_cost(
        "claude-opus-4-7", 0, 0, cached_input_tokens=1000
    )
    uncached_cost = estimate_cost("claude-opus-4-7", 1000, 0)
    assert cached_cost == pytest.approx(0.0015, abs=1e-9)
    assert uncached_cost == pytest.approx(0.015, abs=1e-9)
    # Discount ratio: 10x.
    assert uncached_cost / cached_cost == pytest.approx(10.0, rel=1e-6)


def test_openai_cache_read_costs_half() -> None:
    """gpt-4.1-mini: $0.40/MTok input. 1000 cached tokens cost
    (1000 * 0.40 * 0.5) / 1e6 = $0.0002 — 2x cheaper."""
    cached_cost = estimate_cost(
        "gpt-4.1-mini", 0, 0, cached_input_tokens=1000
    )
    uncached_cost = estimate_cost("gpt-4.1-mini", 1000, 0)
    assert cached_cost == pytest.approx(0.0002, abs=1e-9)
    assert uncached_cost == pytest.approx(0.0004, abs=1e-9)


def test_anthropic_cache_write_5m_costs_1_25x() -> None:
    """5-minute TTL cache writes are billed at 1.25x the base input
    rate. Opus 4.7 at $15/MTok × 1000 tokens × 1.25 = $0.01875."""
    cost = estimate_cost(
        "claude-opus-4-7", 0, 0,
        cache_write_tokens=1000, cache_ttl="5m",
    )
    assert cost == pytest.approx(0.01875, abs=1e-9)


def test_anthropic_cache_write_1h_costs_2x() -> None:
    """1-hour TTL doubles the write rate. $15 × 1000 × 2 = $0.030."""
    cost = estimate_cost(
        "claude-opus-4-7", 0, 0,
        cache_write_tokens=1000, cache_ttl="1h",
    )
    assert cost == pytest.approx(0.030, abs=1e-9)


def test_openai_cache_write_costs_nothing() -> None:
    """OpenAI doesn't bill cache writes — they're free, regardless
    of the cache_write_tokens count."""
    cost = estimate_cost(
        "gpt-4.1-mini", 0, 0, cache_write_tokens=10_000
    )
    assert cost == 0.0


def test_full_breakdown_anthropic_cached_run() -> None:
    """Realistic Anthropic call: 100 uncached input, 5000 cached,
    500 output. Opus 4.7 ($15 in / $75 out). Expected:
    100 × $15/M = $0.0015
    5000 × $15/M × 0.1 = $0.0075
    500 × $75/M = $0.0375
    total = $0.0465
    """
    cost = estimate_cost(
        "claude-opus-4-7",
        input_tokens=100,
        output_tokens=500,
        cached_input_tokens=5000,
    )
    assert cost == pytest.approx(0.0465, abs=1e-9)


# ---------------------------------------------------------------------------
# Anthropic adapter — cache_control injection
# ---------------------------------------------------------------------------


def test_cache_control_for_disabled_returns_none() -> None:
    cfg = PromptCacheConfig(enabled=False)
    assert _cache_control_for(cfg) is None


def test_cache_control_for_5m_default() -> None:
    cfg = PromptCacheConfig(enabled=True, ttl="5m")
    block = _cache_control_for(cfg)
    assert block == {"type": "ephemeral"}


def test_cache_control_for_1h_adds_ttl() -> None:
    cfg = PromptCacheConfig(enabled=True, ttl="1h")
    block = _cache_control_for(cfg)
    assert block == {"type": "ephemeral", "ttl": "1h"}


def test_apply_cache_control_converts_single_system_part_to_block() -> None:
    """Single-part case (back-compat): one content block with one
    ``cache_control`` marker — same shape every pre-0.10.13 client
    saw, just constructed via the new list-based helper signature."""
    kwargs: dict[str, object] = {"system": "you are helpful"}
    _apply_anthropic_cache_control(
        kwargs, ["you are helpful"], [], {"type": "ephemeral"}
    )
    assert kwargs["system"] == [
        {
            "type": "text",
            "text": "you are helpful",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_apply_cache_control_marks_memory_block_independently() -> None:
    """The 0.10.13 unlock: when the architecture emits
    [instructions, memory] as two system parts, the helper renders
    them as two content blocks BOTH carrying ``cache_control`` — so
    cached reads hit independently of any per-turn-volatile recall
    block placed later."""
    kwargs: dict[str, object] = {}
    _apply_anthropic_cache_control(
        kwargs,
        ["instructions block", "memory block"],
        [],
        {"type": "ephemeral"},
    )
    blocks = kwargs["system"]
    assert blocks == [
        {
            "type": "text",
            "text": "instructions block",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": "memory block",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_apply_cache_control_marks_three_system_parts() -> None:
    """Three-part case — instructions / memory / recall. All three
    get ``cache_control`` (3 of Anthropic's 4 breakpoints), leaving
    the 4th for the tool array."""
    kwargs: dict[str, object] = {}
    _apply_anthropic_cache_control(
        kwargs,
        ["instructions", "memory", "recall"],
        [],
        {"type": "ephemeral"},
    )
    blocks = kwargs["system"]
    # All three blocks should carry the marker.
    for b in blocks:
        assert b["cache_control"] == {"type": "ephemeral"}


def test_apply_cache_control_caps_system_markers_at_three() -> None:
    """If an architecture ever emits 4+ system messages, only the
    LAST 3 carry ``cache_control`` — leaving room for the tool
    marker without busting Anthropic's 4-breakpoint hard cap."""
    kwargs: dict[str, object] = {}
    _apply_anthropic_cache_control(
        kwargs,
        ["a", "b", "c", "d"],
        [],
        {"type": "ephemeral"},
    )
    blocks = kwargs["system"]
    # First block: no marker. Last three: marked.
    assert "cache_control" not in blocks[0]
    assert all("cache_control" in b for b in blocks[1:])


def test_apply_cache_control_off_is_noop() -> None:
    """``cache_ctrl=None`` (caching disabled) leaves ``kwargs``
    untouched — the caller's pre-set ``system`` string survives."""
    kwargs: dict[str, object] = {"system": "untouched"}
    _apply_anthropic_cache_control(
        kwargs, ["untouched"], [], None
    )
    assert kwargs["system"] == "untouched"


def test_apply_cache_control_annotates_last_tool() -> None:
    """The last tool gets ``cache_control`` so Anthropic caches the
    full tool definitions array up to and including it."""
    tools = [
        {"name": "foo", "description": "...", "input_schema": {}},
        {"name": "bar", "description": "...", "input_schema": {}},
    ]
    kwargs: dict[str, object] = {"tools": tools}
    _apply_anthropic_cache_control(
        kwargs, [], tools, {"type": "ephemeral", "ttl": "1h"}
    )
    # Last tool now has cache_control; first does not.
    assert "cache_control" not in tools[0]
    assert tools[1]["cache_control"] == {
        "type": "ephemeral", "ttl": "1h",
    }


def test_apply_cache_control_no_op_when_ctrl_is_none() -> None:
    """When caching is disabled (ctrl is None), the helper must NOT
    rewrite anything — the kwargs the SDK receives stay byte-stable
    so cache hits from un-cached requests in the same prefix still
    work."""
    kwargs: dict[str, object] = {"system": "x", "tools": [{"name": "t"}]}
    _apply_anthropic_cache_control(
        kwargs, ["x"], [{"name": "t"}], None
    )
    assert kwargs["system"] == "x"  # untouched
    assert kwargs["tools"] == [{"name": "t"}]


# ---------------------------------------------------------------------------
# Anthropic adapter — usage parsing (uses a fake client)
# ---------------------------------------------------------------------------


def test_anthropic_complete_parses_cache_fields() -> None:
    """When Anthropic returns ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens`` in its usage block, the adapter
    must surface both on the Usage object AND compute cost using
    the discount + write premium."""

    class _FakeUsage:
        input_tokens = 50
        output_tokens = 200
        cache_read_input_tokens = 1000
        cache_creation_input_tokens = 500

    class _FakeBlock:
        type = "text"
        text = "hello"

    class _FakeResponse:
        usage = _FakeUsage()
        content = [_FakeBlock()]
        stop_reason = "end_turn"

    class _FakeMessages:
        async def create(self, **_kwargs: object) -> _FakeResponse:
            return _FakeResponse()

    class _FakeClient:
        messages = _FakeMessages()

    model = AnthropicModel("claude-opus-4-7", api_key="sk-fake")
    model._client = _FakeClient()  # type: ignore[attr-defined]  # noqa: SLF001

    _text, _calls, usage, _finish = asyncio.run(
        model.complete(
            messages=[],
            prompt_caching=PromptCacheConfig(enabled=True, ttl="5m"),
        )
    )
    assert usage.input_tokens == 50
    assert usage.cached_input_tokens == 1000
    assert usage.cache_write_tokens == 500
    assert usage.output_tokens == 200
    # Cost: 50 × $15/M + 1000 × $15/M × 0.1 + 500 × $15/M × 1.25 + 200 × $75/M
    #     = $0.00075 + $0.0015 + $0.009375 + $0.015 = $0.026625
    assert usage.cost_usd == pytest.approx(0.026625, abs=1e-9)


# ---------------------------------------------------------------------------
# OpenAI adapter — cached_tokens normalisation
# ---------------------------------------------------------------------------


def test_openai_complete_normalises_subset_to_separate_buckets() -> None:
    """OpenAI returns ``prompt_tokens = total`` with ``cached_tokens``
    as a subset. The adapter must split into:
        input_tokens = total - cached (cache miss portion)
        cached_input_tokens = cached
    so downstream code doesn't double-count.
    """

    class _FakeDetails:
        cached_tokens = 1500

    class _FakeUsage:
        prompt_tokens = 2000  # total; 1500 of which are cached
        completion_tokens = 300
        prompt_tokens_details = _FakeDetails()

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

    model = OpenAIModel("gpt-4.1-mini", api_key="sk-fake")
    model._client = _FakeClient()  # type: ignore[attr-defined]  # noqa: SLF001

    _text, _calls, usage, _finish = asyncio.run(model.complete(messages=[]))
    assert usage.input_tokens == 500   # 2000 - 1500
    assert usage.cached_input_tokens == 1500
    assert usage.cache_write_tokens == 0  # OpenAI doesn't surface
    assert usage.output_tokens == 300
    # Cost: 500 × $0.40/M + 1500 × $0.40/M × 0.5 + 300 × $1.60/M
    #     = $0.0002 + $0.0003 + $0.00048 = $0.00098
    assert usage.cost_usd == pytest.approx(0.00098, abs=1e-9)


def test_openai_complete_with_cache_key_forwards_to_api() -> None:
    """When ``PromptCacheConfig.cache_key`` is set, the adapter
    forwards it as ``prompt_cache_key`` to OpenAI's API for better
    routing on shared-prefix requests."""

    captured: dict[str, object] = {}

    class _FakeMessage:
        content = "x"
        tool_calls = None

    class _FakeChoice:
        message = _FakeMessage()
        finish_reason = "stop"

    class _FakeResponse:
        usage = None
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs: object) -> _FakeResponse:
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    model = OpenAIModel("gpt-4.1-mini", api_key="sk-fake")
    model._client = _FakeClient()  # type: ignore[attr-defined]  # noqa: SLF001

    asyncio.run(model.complete(
        messages=[],
        prompt_caching=PromptCacheConfig(
            enabled=True, cache_key="user_42"
        ),
    ))
    assert captured.get("prompt_cache_key") == "user_42"


# ---------------------------------------------------------------------------
# Agent integration — kwarg flows through to RunResult fields
# ---------------------------------------------------------------------------


def test_agent_accepts_prompt_caching_bool() -> None:
    agent = Agent("x", model="echo", prompt_caching=True)
    assert agent._prompt_caching.enabled is True  # noqa: SLF001
    assert agent._prompt_caching.ttl == "5m"  # noqa: SLF001


def test_agent_accepts_prompt_caching_dict() -> None:
    agent = Agent(
        "x",
        model="echo",
        prompt_caching={"enabled": True, "ttl": "1h", "cache_key": "k"},
    )
    assert agent._prompt_caching.enabled is True  # noqa: SLF001
    assert agent._prompt_caching.ttl == "1h"  # noqa: SLF001
    assert agent._prompt_caching.cache_key == "k"  # noqa: SLF001


def test_agent_default_prompt_caching_disabled() -> None:
    agent = Agent("x", model="echo")
    assert agent._prompt_caching.enabled is False  # noqa: SLF001


@pytest.mark.anyio
async def test_run_result_carries_cache_fields() -> None:
    """The RunResult surface mirrors Usage — users see
    ``result.cached_tokens_in`` + ``result.cache_write_tokens``
    after a run that hit cache."""
    from loomflow import ScriptedModel, ScriptedTurn
    from loomflow.core.types import Usage

    model = ScriptedModel(
        [
            ScriptedTurn(
                text="done",
                usage=Usage(
                    input_tokens=10,
                    cached_input_tokens=200,
                    cache_write_tokens=50,
                    output_tokens=20,
                    cost_usd=0.123,
                ),
            )
        ]
    )
    agent = Agent("x", model=model, prompt_caching=True)
    result = await agent.run("hi")
    assert result.tokens_in == 10
    assert result.cached_tokens_in == 200
    assert result.cache_write_tokens == 50
    assert result.tokens_out == 20
    assert result.cost_usd == pytest.approx(0.123, abs=1e-9)
