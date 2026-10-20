"""Output-stage guardrails for streamed text.

The ``output`` stage checks ``session.output`` when a run finishes, but
a streaming caller has already been sent every text chunk by then —
so a redacting :class:`~loomflow.guardrails.PIIGuard` cleaned the final
output while the raw text had gone out chunk by chunk.

:class:`GuardedStream` wraps a run's event sink. It holds streamed text
back, runs the output-stage guards over the text so far each time
``chunk_chars`` more has arrived (or an eighth of the text so far, if
that's more, which keeps the total work linear in the length), and
releases the guarded text except for the last ``holdback_chars`` — a
match still being written (half an email address) is redacted before
any of it is sent. Any other event (a tool call, the end of the model
turn, the run finishing) releases the rest first, so event order is
kept. When a guard blocks, the rest
of that text is withheld; the run's own output-stage check then
blocks the run as usual.

Text released before a guard blocks has already been sent — as with
NeMo Guardrails' chunked output rails, the hold-back window bounds
what can slip through rather than buffering whole answers.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from ..core.types import Event, EventKind
from .base import Guardrail, apply_guardrails

if TYPE_CHECKING:
    from ..core.context import RunContext

__all__ = ["GuardedStream"]


class GuardedStream:
    """An event sink that passes streamed text through the output-stage
    guards before forwarding it to ``emit``."""

    def __init__(
        self,
        guards: Sequence[Guardrail],
        emit: Callable[[Event], Awaitable[None]],
        *,
        context: RunContext | None = None,
        chunk_chars: int = 200,
        holdback_chars: int = 128,
    ) -> None:
        self._guards = tuple(g for g in guards if "output" in g.stages)
        self._emit = emit
        self._context = context
        self._chunk_chars = chunk_chars
        self._holdback_chars = holdback_chars
        self._parts: list[str] = []
        self._unchecked = 0  # raw chars since the last guard run
        self._total = 0  # raw chars held this segment
        self._released = 0  # guarded chars already sent
        self._blocked = False
        self._last: Event | None = None  # template for released chunks

    async def __call__(self, event: Event) -> None:
        if event.kind == EventKind.MODEL_CHUNK:
            chunk = event.payload.get("chunk") or {}
            if chunk.get("kind") == "text":
                text = chunk.get("text") or ""
                self._parts.append(text)
                self._unchecked += len(text)
                self._total += len(text)
                self._last = event
                if self._unchecked >= max(self._chunk_chars, self._total // 8):
                    await self._release(final=False)
                return
        await self.flush()
        await self._emit(event)

    async def flush(self) -> None:
        """Release all held text (it has been guarded) and start over."""
        if self._parts:
            await self._release(final=True)
        self._parts.clear()
        self._unchecked = 0
        self._total = 0
        self._released = 0
        self._blocked = False
        self._last = None

    async def _release(self, *, final: bool) -> None:
        self._unchecked = 0
        if self._blocked or self._last is None:
            return
        outcome = await apply_guardrails(
            self._guards, "".join(self._parts), stage="output", context=self._context
        )
        if outcome.blocked:
            self._blocked = True
            return
        text = outcome.text
        end = len(text) if final else len(text) - self._holdback_chars
        if end <= self._released:
            return
        piece = text[self._released : end]
        self._released = end
        template = self._last
        await self._emit(
            template.model_copy(
                update={
                    "payload": {
                        **template.payload,
                        "chunk": {**template.payload["chunk"], "text": piece},
                    }
                }
            )
        )
