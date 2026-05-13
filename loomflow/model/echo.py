"""A trivial model that echoes the last user message back, in chunks.

Useful for proving the loop end-to-end without API keys or network. It
emits one ``text`` chunk per word followed by a single ``finish`` chunk
with a synthetic usage record.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anyio

from ..core.types import Message, ModelChunk, Role, ToolCall, ToolDef, Usage


class EchoModel:
    """Echo-style model for tests and demos."""

    name: str = "echo"

    def __init__(
        self,
        *,
        prefix: str = "Echo: ",
        chunk_delay_s: float = 0.0,
        cost_per_token: float = 0.0,
    ) -> None:
        self._prefix = prefix
        self._chunk_delay = chunk_delay_s
        self._cost_per_token = cost_per_token

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        output_schema: Any | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        prompt_caching: Any = None,
    ) -> tuple[str, list[ToolCall], Usage, str]:
        """Single-shot echo. Returns the echoed user prompt as one
        string with synthetic usage. No per-token chunking — used by
        the non-streaming hot path (``agent.run()``).

        ``output_schema`` is accepted for protocol compatibility but
        ignored — Echo is a zero-key dev fake; structured-output
        constraints don't apply.
        """
        last_user = next(
            (m for m in reversed(messages) if m.role == Role.USER),
            None,
        )
        text = f"{self._prefix}{last_user.content if last_user else ''}"
        input_tokens = sum(len(m.content.split()) for m in messages)
        output_tokens = max(1, len(text.split()))
        cost = (input_tokens + output_tokens) * self._cost_per_token
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        return text, [], usage, "stop"

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        output_schema: Any | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        prompt_caching: Any = None,
    ) -> AsyncIterator[ModelChunk]:
        last_user = next(
            (m for m in reversed(messages) if m.role == Role.USER),
            None,
        )
        text = f"{self._prefix}{last_user.content if last_user else ''}"

        words = text.split(" ") or [text]
        emitted = 0
        for i, word in enumerate(words):
            piece = word if i == 0 else " " + word
            if self._chunk_delay > 0:
                await anyio.sleep(self._chunk_delay)
            yield ModelChunk(kind="text", text=piece)
            emitted += 1

        # Cheap-and-cheerful usage estimate: 1 token per whitespace-separated
        # word in the input, same in the output.
        input_tokens = sum(len(m.content.split()) for m in messages)
        output_tokens = emitted
        cost = (input_tokens + output_tokens) * self._cost_per_token

        yield ModelChunk(
            kind="finish",
            finish_reason="stop",
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            ),
        )
