"""LiteLLMModel adapter tests using fake clients (no litellm needed).

Because :class:`LiteLLMModel` inherits from :class:`OpenAIModel` and
LiteLLM produces OpenAI-shaped chunks, we verify:

* The adapter accepts an injected ``client=`` (the same shape
  ``OpenAIModel`` accepts) without trying to import ``litellm``.
* Streaming roundtrips OpenAI-shaped chunks correctly (smoke test —
  full chunk normalisation is covered in ``test_openai.py``).
* The string resolver dispatches LiteLLM prefixes
  (``mistral-``, ``command-``, ``bedrock/``, ...) to ``LiteLLMModel``.
* The ``litellm/`` opt-in prefix routes through LiteLLM and strips
  the prefix before forwarding.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any

import pytest

from jeevesagent import Agent
from jeevesagent.core.errors import ConfigError
from jeevesagent.core.types import Message, ModelChunk, Role
from jeevesagent.model.litellm import LiteLLMModel

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fake OpenAI-shaped client (LiteLLM produces these chunks too)
# ---------------------------------------------------------------------------


class _FakeOAIStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> _FakeOAIStream:
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration as e:
            raise StopAsyncIteration from e


class _FakeCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.captured_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeOAIStream:
        self.captured_kwargs = kwargs
        return _FakeOAIStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks: list[Any]) -> None:
        self.completions = _FakeCompletions(chunks)


class _FakeClient:
    def __init__(self, chunks: list[Any]) -> None:
        self.chat = _FakeChat(chunks)


def _text(content: str) -> Any:
    return NS(
        usage=None,
        choices=[
            NS(
                delta=NS(content=content, tool_calls=None),
                finish_reason=None,
            )
        ],
    )


def _finish(reason: str = "stop") -> Any:
    return NS(
        usage=None,
        choices=[
            NS(
                delta=NS(content=None, tool_calls=None),
                finish_reason=reason,
            )
        ],
    )


def _usage(prompt: int, completion: int) -> Any:
    return NS(
        usage=NS(prompt_tokens=prompt, completion_tokens=completion),
        choices=[],
    )


async def _collect(stream: AsyncIterator[ModelChunk]) -> list[ModelChunk]:
    return [c async for c in stream]


# ---------------------------------------------------------------------------
# Construction with injected client
# ---------------------------------------------------------------------------


async def test_litellm_with_injected_client_skips_real_sdk_import() -> None:
    """Passing ``client=...`` bypasses the lazy ``import litellm``,
    so the test runs cleanly even when litellm isn't installed."""
    chunks = [_text("hi"), _finish("stop"), _usage(3, 1)]
    model = LiteLLMModel(
        "mistral-large", client=_FakeClient(chunks)
    )
    out = await _collect(
        model.stream([Message(role=Role.USER, content="ping")])
    )
    assert [c.text for c in out if c.kind == "text"] == ["hi"]
    (finish,) = [c for c in out if c.kind == "finish"]
    assert finish.usage is not None
    assert finish.usage.input_tokens == 3
    assert finish.usage.output_tokens == 1


async def test_litellm_model_name_passes_through() -> None:
    fake = _FakeClient([_finish("stop"), _usage(0, 0)])
    model = LiteLLMModel("mistral-large", client=fake)
    assert model.name == "mistral-large"


async def test_litellm_with_agent_runs_end_to_end() -> None:
    chunks = [_text("forty-two"), _finish("stop"), _usage(5, 3)]
    model = LiteLLMModel("mistral-large", client=_FakeClient(chunks))
    agent = Agent("hi", model=model)
    result = await agent.run("answer?")
    assert result.output == "forty-two"
    assert result.tokens_in == 5
    assert result.tokens_out == 3


# ---------------------------------------------------------------------------
# String resolver dispatches LiteLLM prefixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "mistral-large",
        "command-r-plus",
        "bedrock/anthropic.claude-3-sonnet",
        "vertex_ai/gemini-pro",
        "together_ai/meta-llama/Llama-3-70b",
        "ollama/llama3",
        "gemini/gemini-1.5-pro",
        "groq/llama-3.1-70b",
        "replicate/meta/llama-2-70b-chat",
        "azure/gpt-4o",
    ],
)
def test_string_resolver_dispatches_litellm_prefixes(
    monkeypatch: pytest.MonkeyPatch, spec: str
) -> None:
    """Each known LiteLLM prefix should produce a ``LiteLLMModel``
    instance from the string resolver. Dummy env vars keep the
    underlying SDK construction quiet — but the lazy ``import litellm``
    inside ``LiteLLMModel.__init__`` runs, so this test silently
    skips the litellm import path by using ``client=`` injection
    is not possible here — instead we just patch litellm in.
    """
    # Provide a stub ``litellm`` module so the lazy import succeeds
    # without the real package installed.
    import sys
    import types

    fake_litellm = types.ModuleType("litellm")

    async def _fake_acompletion(**kwargs: Any) -> Any:
        return _FakeOAIStream([_finish("stop"), _usage(0, 0)])

    fake_litellm.acompletion = _fake_acompletion  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    agent = Agent("hi", model=spec)
    assert type(agent._model).__name__ == "LiteLLMModel"
    # ``litellm/`` prefix gets stripped before forwarding.
    expected = spec[len("litellm/"):] if spec.startswith("litellm/") else spec
    assert agent._model.name == expected


def test_litellm_explicit_prefix_strips_before_forwarding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``litellm/<inner>`` is the explicit opt-in: forces LiteLLM even
    for specs the direct paths would otherwise take."""
    import sys
    import types

    fake_litellm = types.ModuleType("litellm")

    async def _fake_acompletion(**kwargs: Any) -> Any:
        return _FakeOAIStream([_finish("stop"), _usage(0, 0)])

    fake_litellm.acompletion = _fake_acompletion  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    # Without the prefix, "claude-3-haiku" would route to AnthropicModel.
    # The explicit ``litellm/`` opts into LiteLLM instead.
    agent = Agent("hi", model="litellm/claude-3-haiku")
    assert type(agent._model).__name__ == "LiteLLMModel"
    assert agent._model.name == "claude-3-haiku"  # prefix stripped


def test_unknown_model_string_raises_config_error_not_value_error() -> None:
    """0.1.x raised ``ValueError``; 0.2.0 harmonises to ``ConfigError``."""
    with pytest.raises(ConfigError) as excinfo:
        Agent("hi", model="totally-unknown-spec")
    msg = str(excinfo.value)
    assert "unknown model spec" in msg
    assert "claude-" in msg
    assert "gpt-" in msg
    assert "litellm/" in msg
