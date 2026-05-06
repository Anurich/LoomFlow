"""07_litellm — Talking to non-Anthropic, non-OpenAI providers.

What it shows:
* The string resolver dispatches LiteLLM prefixes to ``LiteLLMModel``,
  which wraps the ``litellm`` SDK to talk to ~100 providers through
  one adapter (Cohere, Mistral, Bedrock, Vertex, Together, Ollama,
  Gemini, Groq, ...).
* Falls back to ``EchoModel`` when no provider key is set, so the
  example always runs.

To make a real call against Mistral::

    pip install 'jeevesagent[litellm]'
    export MISTRAL_API_KEY=...
    python examples/07_litellm.py

Other prefixes that route through LiteLLM:

* ``command-r-plus``      (Cohere; needs ``COHERE_API_KEY``)
* ``bedrock/anthropic.claude-3-sonnet-20240229-v1:0``
                          (AWS Bedrock; AWS creds via env)
* ``vertex_ai/gemini-pro`` (Google; needs Vertex AI auth)
* ``together_ai/mistralai/Mistral-7B-Instruct-v0.3``
                          (Together; ``TOGETHER_API_KEY``)
* ``ollama/llama3``       (local Ollama at http://localhost:11434)
* ``gemini/gemini-1.5-pro`` (Google AI Studio; ``GEMINI_API_KEY``)
* ``groq/llama-3.1-70b-versatile`` (Groq; ``GROQ_API_KEY``)
* ``replicate/meta/llama-2-70b-chat``

Use ``litellm/<spec>`` to force the LiteLLM path even for specs the
direct adapters would otherwise grab (e.g. ``litellm/claude-3-haiku``).

Run:
    python examples/07_litellm.py
"""

from __future__ import annotations

import asyncio
import os


def _pick_model() -> str:
    """Pick a model based on which provider key is in the env."""
    if os.environ.get("MISTRAL_API_KEY"):
        return "mistral-large-latest"
    if os.environ.get("COHERE_API_KEY"):
        return "command-r-plus"
    if os.environ.get("GROQ_API_KEY"):
        return "groq/llama-3.1-70b-versatile"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini/gemini-1.5-pro"
    print(
        "(no MISTRAL_API_KEY / COHERE_API_KEY / GROQ_API_KEY / "
        "GEMINI_API_KEY; falling back to echo.)"
    )
    return "echo"


async def main() -> None:
    from jeevesagent import Agent

    model = _pick_model()
    agent = Agent("You are a haiku poet.", model=model)
    result = await agent.run("Write a haiku about open-source.")
    print(f"--- {model} ---")
    print(result.output)
    print(
        f"\nUsed {result.total_tokens} tokens "
        f"(in={result.tokens_in} / out={result.tokens_out}), "
        f"${result.cost_usd:.4f}, "
        f"in {result.duration.total_seconds():.2f}s"
    )


if __name__ == "__main__":
    asyncio.run(main())
