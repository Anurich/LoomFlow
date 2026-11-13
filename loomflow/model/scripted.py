"""A test-only model that replays a fixed sequence of turns.

Each :class:`ScriptedTurn` describes one model response: optional text
followed by zero or more tool calls, terminated by a finish chunk. The
model advances through the script with each call to :meth:`stream`,
which lets a single agent ``run()`` exercise multi-turn flows end-to-end.

Every call is recorded on :attr:`ScriptedModel.calls`, so a test can
assert on the prompt and tools the agent actually sent::

    model = ScriptedModel([ScriptedTurn(text="4")])
    agent = Agent("You are a calculator.", model=model)
    await agent.run("2 + 2?")
    assert model.calls[0].messages[-1].content == "2 + 2?"

Asking for more turns than were scripted raises
:class:`ScriptExhaustedError` (pass ``on_exhausted="empty"`` for the
old behaviour of answering with empty text).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from ..core.errors import LoomError
from ..core.types import Message, ModelChunk, ToolCall, ToolDef, Usage


class ScriptExhaustedError(LoomError):
    """A :class:`ScriptedModel` was called more times than it has turns."""


@dataclass
class ScriptedTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class ScriptedCall:
    """What a :class:`ScriptedModel` was sent on one call.

    ``messages`` and ``tools`` are snapshots taken at call time (the
    agent keeps appending to its own message list afterwards).
    """

    messages: list[Message]
    tools: list[ToolDef] | None = None
    output_schema: Any | None = None
    effort: str | None = None


class ScriptedModel:
    """Model that emits canned responses, one per model call.

    ``turns`` are replayed in order. Each call is recorded on
    :attr:`calls` (messages + tools the agent sent), so tests can
    assert on prompts.

    ``on_exhausted`` decides what happens when the agent asks for a
    turn past the end of the script:

    * ``"raise"`` (default) — raise :class:`ScriptExhaustedError`
      naming how many turns were scripted. An agent that makes more
      model calls than the test expected is almost always a bug in
      the test or the agent, and an empty answer hides it.
    * ``"empty"`` — answer with empty text and ``finish_reason="stop"``
      (the old behaviour).
    """

    name: str = "scripted"

    def __init__(
        self,
        turns: list[ScriptedTurn],
        *,
        on_exhausted: Literal["raise", "empty"] = "raise",
    ) -> None:
        if on_exhausted not in ("raise", "empty"):
            raise ValueError(
                f"on_exhausted must be 'raise' or 'empty', not {on_exhausted!r}"
            )
        self._turns = list(turns)
        self._idx = 0
        self._on_exhausted = on_exhausted
        # ``calls`` is the public view. The model appends to its own
        # reference so a subclass that keeps a ``self.calls`` counter of
        # its own (a common pattern before this existed) keeps working.
        self._calls: list[ScriptedCall] = []
        self.calls: list[ScriptedCall] = self._calls

    @property
    def remaining(self) -> int:
        """How many scripted turns have not been played yet."""
        return max(0, len(self._turns) - self._idx)

    def _next_turn(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        output_schema: Any | None,
        effort: str | None,
    ) -> ScriptedTurn | None:
        """Record the call and return the next turn (``None`` when the
        script is exhausted and ``on_exhausted="empty"``)."""
        self._calls.append(
            ScriptedCall(
                messages=list(messages),
                tools=list(tools) if tools is not None else None,
                output_schema=output_schema,
                effort=effort,
            )
        )
        if self._idx >= len(self._turns):
            if self._on_exhausted == "empty":
                return None
            n = len(self._turns)
            raise ScriptExhaustedError(
                f"ScriptedModel ran out of turns: {n} turn(s) were scripted "
                f"but the agent made model call #{len(self._calls)}. Add a "
                "ScriptedTurn for it (inspect `model.calls[-1].messages` to "
                "see what was asked), or pass on_exhausted='empty' to answer "
                "with empty text instead."
            )
        turn = self._turns[self._idx]
        self._idx += 1
        return turn

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
        turn = self._next_turn(messages, tools, output_schema, effort)
        if turn is None:
            return "", [], Usage(), "stop"
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
        turn = self._next_turn(messages, tools, output_schema, effort)
        if turn is None:
            yield ModelChunk(kind="text", text="")
            yield ModelChunk(
                kind="finish",
                finish_reason="stop",
                usage=Usage(),
            )
            return

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
