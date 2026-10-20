"""Model adapters against the real provider SDKs (no network, no keys).

Each test drives the installed SDK through a mock HTTP transport, so a
change in the SDK's own signatures breaks the test instead of every
user's first call. Fakes that accept ``**kwargs`` can't catch that.
"""

from __future__ import annotations

import importlib.metadata
import json
from typing import Any

import pytest

from loomflow import Agent, Message

pytestmark = pytest.mark.anyio

anthropic = pytest.importorskip("anthropic")


def _anthropic_client(sent: list[dict[str, Any]], reply: str = "hi") -> Any:
    """A real AsyncAnthropic whose HTTP layer records requests."""
    major = int(importlib.metadata.version("anthropic").split(".")[0])
    # SDK 1.x moved to httpx2 and rejects plain httpx clients.
    http = pytest.importorskip("httpx2") if major >= 1 else pytest.importorskip("httpx")

    def handler(request: Any) -> Any:
        sent.append(json.loads(request.content))
        return http.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": reply}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )

    return anthropic.AsyncAnthropic(
        api_key="sk-test",
        http_client=http.AsyncClient(transport=http.MockTransport(handler)),
        max_retries=0,
    )


async def test_agent_runs_on_the_installed_anthropic_sdk() -> None:
    # anthropic SDK 1.x removed `temperature` from messages.create; an
    # adapter that always sent it raised TypeError before any request.
    from loomflow.model.anthropic import AnthropicModel

    sent: list[dict[str, Any]] = []
    agent = Agent("You are helpful.", model=AnthropicModel("claude-opus-4-7", client=_anthropic_client(sent)))
    result = await agent.run("hello")
    assert result.output == "hi"
    assert len(sent) == 1
    assert "temperature" not in sent[0]


async def test_explicit_temperature_still_reaches_the_api() -> None:
    from loomflow.model.anthropic import AnthropicModel

    sent: list[dict[str, Any]] = []
    model = AnthropicModel("claude-haiku-4-5", client=_anthropic_client(sent))
    await model.complete([Message(role="user", content="x")], temperature=0.2)
    assert sent[-1]["temperature"] == 0.2


async def test_sdk_type_errors_are_not_swallowed() -> None:
    # A TypeError from a real create() used to be read as "fake client"
    # and retried via stream(), hiding the actual failure.
    from loomflow.model.anthropic import AnthropicModel

    class Messages:
        async def create(self, *, model: str, messages: Any, max_tokens: int) -> Any:
            raise TypeError("boom from the SDK")

        def stream(self, **kwargs: Any) -> Any:  # pragma: no cover - must not run
            raise AssertionError("stream() must not be used as a fallback")

    class Client:
        messages = Messages()

    model = AnthropicModel("claude-opus-4-7", client=Client())
    with pytest.raises(TypeError, match="boom from the SDK"):
        await model.complete([Message(role="user", content="x")])


# ---------------------------------------------------------------------------
# Pricing — official list prices, October 2026
# ---------------------------------------------------------------------------

_K = 1_000


@pytest.mark.parametrize(
    ("model", "inp", "out", "cached"),
    [
        # platform.claude.com/docs/en/about-claude/pricing
        ("claude-opus-5-5", 4.00, 20.00, 0.20),
        ("claude-sonnet-5-5", 2.00, 10.00, 0.10),
        ("claude-sonnet-5", 2.00, 10.00, 0.20),
        ("claude-haiku-5-5", 0.10, 0.50, 0.01),
        ("claude-fable-5-1", 10.00, 50.00, 0.25),
        ("claude-fable-5", 10.00, 50.00, 1.00),
        ("claude-opus-4-8", 5.00, 25.00, 0.50),
        ("claude-opus-4-20250514", 15.00, 75.00, 1.50),
        # developers.openai.com/api/docs/pricing
        ("gpt-6-sol", 2.00, 10.00, 0.20),
        ("gpt-6.1-sol", 2.00, 10.00, 0.10),
        ("gpt-6-astra", 10.00, 50.00, 1.00),
        ("gpt-5.6-terra", 2.00, 12.00, 0.20),
        ("gpt-5.2", 1.75, 14.00, 0.175),
        ("gpt-5-pro", 15.00, 120.00, 15.00),
        ("o1-pro", 150.00, 600.00, 150.00),
        ("o4-mini", 1.10, 4.40, 0.275),
        # api-docs.deepseek.com (peak-hour rate)
        ("deepseek-flash", 0.30, 1.20, 0.006),
        ("deepseek-v4-pro", 1.32, 3.96, 0.044),
    ],
)
def test_short_prompt_prices_match_the_official_tables(
    model: str, inp: float, out: float, cached: float
) -> None:
    from loomflow.model._pricing import estimate_cost

    # 10K tokens: below every long-context threshold.
    assert estimate_cost(model, 10 * _K, 0) == pytest.approx(inp / 100)
    assert estimate_cost(model, 0, 10 * _K) == pytest.approx(out / 100)
    assert estimate_cost(model, 0, 0, cached_input_tokens=10 * _K) == pytest.approx(cached / 100)


def test_long_context_tiers_bill_the_whole_request() -> None:
    from loomflow.model._pricing import estimate_cost

    # Haiku 5.5: $0.50 / $2.50 for prompts over 100K tokens.
    assert estimate_cost("claude-haiku-5-5", 150 * _K, _K) == pytest.approx(0.075 + 0.0025)
    assert estimate_cost("claude-haiku-5-5", 90 * _K, _K) == pytest.approx(0.009 + 0.0005)
    # Cached prompt tokens count towards the threshold.
    assert estimate_cost(
        "claude-haiku-5-5", 10 * _K, 0, cached_input_tokens=95 * _K
    ) == pytest.approx(0.005 + 0.00475)
    # GPT-5.4 doubles input above 272K; its mini variant has no tier.
    assert estimate_cost("gpt-5.4", 300 * _K, 0) == pytest.approx(1.50)
    assert estimate_cost("gpt-5.4-mini", 300 * _K, 0) == pytest.approx(0.225)


def test_cache_writes_bill_on_anthropic_and_new_openai_models_only() -> None:
    from loomflow.model._pricing import estimate_cost

    assert estimate_cost("claude-opus-5-5", 0, 0, cache_write_tokens=10 * _K) == pytest.approx(0.05)
    assert estimate_cost(
        "claude-opus-5-5", 0, 0, cache_write_tokens=10 * _K, cache_ttl="1h"
    ) == pytest.approx(0.08)
    assert estimate_cost("gpt-6-sol", 0, 0, cache_write_tokens=10 * _K) == pytest.approx(0.025)
    assert estimate_cost("gpt-5.2", 0, 0, cache_write_tokens=10 * _K) == 0.0


# ---------------------------------------------------------------------------
# Retries — classification, Retry-After, jitter, no stacked SDK retries
# ---------------------------------------------------------------------------


class _Headers(dict):  # type: ignore[type-arg]
    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key.lower(), default)


class _Resp:
    def __init__(self, **headers: str) -> None:
        self.headers = _Headers({k.lower().replace("_", "-"): v for k, v in headers.items()})


def _anthropic_status_error(status: int, body: dict[str, Any], **headers: str) -> Any:
    httpx = pytest.importorskip("httpx2") if int(
        importlib.metadata.version("anthropic").split(".")[0]
    ) >= 1 else pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, headers=headers, request=request)
    return anthropic.APIStatusError("boom", response=response, body=body)


def test_mid_stream_overloaded_error_is_transient() -> None:
    from loomflow.core.errors import RateLimitError, TransientModelError
    from loomflow.governance.retry import classify_model_error

    # The SDK raises an error event inside a stream with the stream's 200.
    overloaded = _anthropic_status_error(
        200, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    )
    assert isinstance(classify_model_error(overloaded), TransientModelError)
    limited = _anthropic_status_error(
        200, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
    )
    assert isinstance(classify_model_error(limited), RateLimitError)


@pytest.mark.parametrize("status", [408, 409])
def test_request_timeout_and_conflict_are_retried(status: int) -> None:
    from loomflow.core.errors import TransientModelError
    from loomflow.governance.retry import classify_model_error

    exc = _anthropic_status_error(status, {"type": "error", "error": {"type": "x"}})
    assert isinstance(classify_model_error(exc), TransientModelError)


def test_retry_after_headers() -> None:
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    from loomflow.governance.retry import _extract_retry_after

    class E(Exception):
        def __init__(self, **headers: str) -> None:
            self.response = _Resp(**headers)

    assert _extract_retry_after(E(retry_after_ms="1500")) == pytest.approx(1.5)
    assert _extract_retry_after(E(retry_after="2")) == 2.0
    assert _extract_retry_after(E(retry_after="inf")) is None
    assert _extract_retry_after(E(retry_after="-3")) is None
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    assert 25 < (_extract_retry_after(E(retry_after=future)) or 0) <= 30


async def test_a_retry_after_beyond_the_cap_stops_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    import anyio

    from loomflow.core.errors import RateLimitError
    from loomflow.governance.retry import RetryPolicy
    from loomflow.model.retrying import RetryingModel

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(anyio, "sleep", fake_sleep)
    calls = 0

    class Limited:
        name = "x"

        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            raise RateLimitError("wait a day", retry_after=86_400.0)

    with pytest.raises(RateLimitError):
        await RetryingModel(Limited(), RetryPolicy(max_attempts=3)).complete([])
    assert calls == 1 and sleeps == []


def test_full_jitter_is_the_default() -> None:
    import random

    from loomflow.governance.retry import RetryPolicy, compute_backoff

    policy = RetryPolicy()
    rng = random.Random(0)
    delays = [compute_backoff(policy, 3, rng=rng) for _ in range(200)]
    assert all(0.0 <= d <= 4.0 for d in delays)  # 1s * 2**2, uniform from 0
    assert min(delays) < 1.0 < max(delays)


def test_sdk_clients_loomflow_builds_do_not_retry_themselves(monkeypatch: pytest.MonkeyPatch) -> None:
    from loomflow.model.anthropic import AnthropicModel

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert AnthropicModel("claude-opus-5-5")._client.max_retries == 0
    openai = pytest.importorskip("openai")
    del openai
    from loomflow.model.openai import OpenAIModel

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert OpenAIModel("gpt-6-sol")._client.max_retries == 0



# ---------------------------------------------------------------------------
# Anthropic structured output: native JSON outputs, tools stay callable
# ---------------------------------------------------------------------------


class _Receipt(__import__("pydantic").BaseModel):
    amount: float
    currency: str
    note: str = __import__("pydantic").Field(default="", max_length=20)


async def test_output_schema_uses_output_config_format_and_keeps_tools() -> None:
    from loomflow import tool
    from loomflow.model.anthropic import AnthropicModel

    @tool
    async def lookup(q: str) -> str:
        """Look something up."""
        return q

    sent: list[dict[str, Any]] = []
    model = AnthropicModel(
        "claude-sonnet-5-5",
        client=_anthropic_client(sent, '{"amount": 4.5, "currency": "EUR"}'),
    )
    agent = Agent("x", model=model, tools=[lookup])
    result = await agent.run("receipt?", output_schema=_Receipt)
    assert result.parsed == _Receipt(amount=4.5, currency="EUR")
    body = sent[-1]
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    # Unsupported constraints move into the description (the agent
    # still validates them); real tools are offered, nothing forced.
    note = fmt["schema"]["properties"]["note"]
    assert "maxLength" not in note and "maxLength" in note["description"]
    assert [t["name"] for t in body["tools"]] == ["lookup"]
    assert "tool_choice" not in body


async def test_legacy_models_never_force_the_output_tool_over_real_tools() -> None:
    from loomflow.core.types import ToolDef
    from loomflow.model.anthropic import AnthropicModel

    sent: list[dict[str, Any]] = []
    model = AnthropicModel("claude-opus-4-1", client=_anthropic_client(sent))
    real = ToolDef(name="lookup", description="d", input_schema={"type": "object"})
    await model.complete([Message(role="user", content="x")], tools=[real], output_schema=_Receipt)
    assert sent[-1]["tool_choice"] == {"type": "any"}  # some tool, not necessarily __output__
    await model.complete(
        [Message(role="user", content="x")], tools=[real], output_schema=_Receipt, effort="high"
    )
    # A forced tool_choice can't be combined with thinking.
    assert "tool_choice" not in sent[-1]
    assert sent[-1]["thinking"]["type"] == "enabled"


async def test_effort_reaches_the_api_on_current_models() -> None:
    from loomflow.model.anthropic import AnthropicModel

    sent: list[dict[str, Any]] = []
    model = AnthropicModel("claude-opus-5-5", client=_anthropic_client(sent))
    await model.complete([Message(role="user", content="x")], effort="xhigh")
    assert sent[-1]["output_config"] == {"effort": "xhigh"}
    assert sent[-1]["thinking"] == {"type": "adaptive"}
    await model.complete([Message(role="user", content="x")], effort="minimal")
    assert sent[-1]["output_config"] == {"effort": "low"}


def test_bare_o_series_ids_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    from loomflow.agent.api import _resolve_model
    from loomflow.model.openai import OpenAIModel

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for spec in ("o1", "o3", "gpt-6-sol"):
        assert isinstance(_resolve_model(spec), OpenAIModel)


def test_gpt_6_tool_limits_in_chat_completions() -> None:
    from loomflow.core.errors import PermanentModelError
    from loomflow.model.openai import _apply_chat_tool_limits

    with pytest.raises(PermanentModelError, match="Responses API"):
        _apply_chat_tool_limits("gpt-6.1-sol", {"tools": [{}]})
    kwargs = {"tools": [{}], "reasoning_effort": "high"}
    with pytest.warns(UserWarning, match="reasoning_effort='none'"):
        _apply_chat_tool_limits("gpt-6-sol", kwargs)
    assert kwargs["reasoning_effort"] == "none"
    untouched = {"reasoning_effort": "high"}
    _apply_chat_tool_limits("gpt-6-sol", untouched)  # no tools: effort kept
    assert untouched["reasoning_effort"] == "high"


def test_context_windows_follow_the_official_tables() -> None:
    from loomflow.agent.auto_compact import context_window_for

    for name in (
        "claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5", "claude-haiku-5-5",
        "claude-opus-4-8", "claude-sonnet-5",
    ):
        assert context_window_for(name) == 1_000_000, name
    assert context_window_for("claude-opus-4-5") == 200_000
    assert context_window_for("claude-haiku-4-5") == 200_000
    assert context_window_for("gpt-6-sol") == 1_050_000
    assert context_window_for("deepseek-flash") == 1_000_000


async def test_token_estimates_count_tool_calls_images_and_use_o200k() -> None:
    pytest.importorskip("tiktoken")
    from loomflow.core.types import Image, ToolCall
    from loomflow.model.count_tokens import (
        IMAGE_TOKEN_ESTIMATE,
        _encoding_name,
        count_tokens,
    )

    class Named:
        name = "gpt-6-sol"

    assert _encoding_name("gpt-6-sol") == "o200k_base"
    assert _encoding_name("openai/gpt-4o-mini") == "o200k_base"
    assert _encoding_name("claude-opus-5-5") == "cl100k_base"
    plain = [Message(role="assistant", content="ok")]
    with_call = [
        Message(
            role="assistant",
            content="ok",
            tool_calls=(ToolCall(id="c", tool="search", args={"q": "a long query " * 20}),),
        )
    ]
    with_image = [
        Message(role="user", content="ok", images=(Image(data="aGk=", media_type="image/png"),))
    ]
    base = await count_tokens(Named(), plain)
    assert await count_tokens(Named(), with_call) > base + 20
    assert await count_tokens(Named(), with_image) >= base + IMAGE_TOKEN_ESTIMATE
