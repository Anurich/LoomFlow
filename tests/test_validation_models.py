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


def _anthropic_client(sent: list[dict[str, Any]]) -> Any:
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
                "content": [{"type": "text", "text": "hi"}],
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
