"""LiteLLM-backed model adapter — one adapter, every provider.

`LiteLLM <https://github.com/BerriAI/litellm>`_ normalises 100+
provider APIs to OpenAI's chat-completion shape, including:

* Anthropic (``claude-*``) — though :class:`AnthropicModel` is a
  faster direct path
* OpenAI (``gpt-*``) — same; :class:`OpenAIModel` is the direct path
* Cohere (``command-r``, ``command-r-plus``)
* Mistral (``mistral-large``, ``mistral-small``, ...)
* AWS Bedrock (``bedrock/anthropic.claude-3-...``)
* Google Vertex AI (``vertex_ai/gemini-pro``)
* Together AI (``together_ai/...``)
* Groq, Replicate, Ollama, …

Because LiteLLM produces OpenAI-shaped streaming chunks, this adapter
can subclass :class:`OpenAIModel` and reuse its entire chunk
aggregation / tool-call delta accumulation logic. The only
difference: where :class:`OpenAIModel` calls
``self._client.chat.completions.create``, this one routes through
``litellm.acompletion``.

Usage::

    from jeevesagent import Agent
    from jeevesagent.model.litellm import LiteLLMModel

    agent = Agent(
        "...",
        model=LiteLLMModel("mistral-large", api_key="..."),
    )

The string-based resolver in :mod:`jeevesagent.agent.api` recognises
several common LiteLLM prefixes (``mistral-``, ``command-``,
``bedrock/``, ``vertex_ai/``, ``together_ai/``, ``ollama/``,
``gemini/``) so passing the bare model spec works too.
"""

from __future__ import annotations

from typing import Any

from .openai import OpenAIModel


class _LiteLLMCompletions:
    """OpenAI-shaped ``completions`` surface backed by ``litellm.acompletion``."""

    def __init__(self, defaults: dict[str, Any]) -> None:
        self._defaults = defaults

    async def create(self, **kwargs: Any) -> Any:
        from litellm import acompletion  # type: ignore[import-not-found, import-untyped]

        merged = {**self._defaults, **kwargs}
        return await acompletion(**merged)


class _LiteLLMChat:
    def __init__(self, completions: _LiteLLMCompletions) -> None:
        self.completions = completions


class _LiteLLMClient:
    """A duck-typed ``openai.AsyncOpenAI`` that routes through LiteLLM.

    Exposes ``client.chat.completions.create`` so :class:`OpenAIModel`'s
    inherited :meth:`stream` works without modification.
    """

    def __init__(self, **defaults: Any) -> None:
        self.chat = _LiteLLMChat(_LiteLLMCompletions(defaults))


class LiteLLMModel(OpenAIModel):
    """Talks to any LiteLLM-supported provider.

    Inherits chunk normalisation, tool-call delta aggregation, and
    message-conversion from :class:`OpenAIModel` because LiteLLM
    produces OpenAI-shaped outputs.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        **litellm_kwargs: Any,
    ) -> None:
        if client is None:
            try:
                import litellm  # type: ignore[import-not-found, import-untyped]  # noqa: F401
            except ImportError as exc:  # pragma: no cover — depends on user env
                raise ImportError(
                    "LiteLLM is not installed. "
                    "Install with: pip install 'jeevesagent[litellm]'"
                ) from exc

            defaults: dict[str, Any] = dict(litellm_kwargs)
            if api_key is not None:
                defaults["api_key"] = api_key
            client = _LiteLLMClient(**defaults)

        # Skip OpenAIModel's openai-SDK import path by passing the
        # client we just built. ``api_key`` is included as a kwarg
        # purely to satisfy OpenAIModel's signature; LiteLLM itself
        # picks up provider-specific keys from environment variables
        # or the ``api_key=`` we shoved into the defaults above.
        super().__init__(model, client=client, api_key=api_key)
