"""A test-only model that replays a fixed sequence of turns.

Each :class:`ScriptedTurn` describes one model response: optional text
followed by zero or more tool calls, terminated by a finish chunk. The
model advances through the script with each call to :meth:`stream`,
which lets a single agent ``run()`` exercise multi-turn flows end-to-end.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..core.types import Message, ModelChunk, ToolCall, ToolDef, Usage


@dataclass
class ScriptedTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class ScriptedModel:
    """Model that emits canned responses, one per call to :meth:`stream`."""

    name: str = "scripted"

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        self._turns = list(turns)
        self._idx = 0

    @property
    def remaining(self) -> int:
        return max(0, len(self._turns) - self._idx)

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
        """Single-shot replay of the next scripted turn.

        Mirrors :meth:`stream` but returns the turn's text +
        tool_calls + usage in one tuple. Used by the non-streaming
        hot path (``agent.run()``); ``agent.stream()`` keeps using
        :meth:`stream` for per-chunk replay.

        ``output_schema`` is accepted for protocol compatibility
        but ignored — Scripted is a deterministic test fake; turns
        are pre-baked, not constrained at decode time.
        """
        if self._idx >= len(self._turns):
            return "", [], Usage(), "stop"
        turn = self._turns[self._idx]
        self._idx += 1
        finish_reason = "tool_use" if turn.tool_calls else "stop"
        return (
            turn.text,
            list(turn.tool_calls),
            turn.usage,
            finish_reason,
        )

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
        if self._idx >= len(self._turns):
            yield ModelChunk(kind="text", text="")
            yield ModelChunk(
                kind="finish",
                finish_reason="stop",
                usage=Usage(),
            )
            return

        turn = self._turns[self._idx]
        self._idx += 1

        if turn.text:
            for i, word in enumerate(turn.text.split(" ")):
                piece = word if i == 0 else " " + word
                yield ModelChunk(kind="text", text=piece)

        for tc in turn.tool_calls:
            yield ModelChunk(kind="tool_call", tool_call=tc)

        finish_reason = "tool_use" if turn.tool_calls else "stop"
        yield ModelChunk(
            kind="finish",
            finish_reason=finish_reason,
            usage=turn.usage,
        )
