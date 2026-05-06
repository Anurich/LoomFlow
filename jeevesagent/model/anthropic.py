"""Adapter for Anthropic's Claude models via the official ``anthropic`` SDK.

Streams via ``messages.stream``; normalises Anthropic's event types into
our :class:`ModelChunk` shape:

* ``text_delta`` -> ``ModelChunk(kind="text", text=...)``
* ``input_json_delta`` accumulates partial tool-use JSON; on
  ``content_block_stop`` we emit ``ModelChunk(kind="tool_call", ...)``
* ``message_delta`` carries the final ``stop_reason`` and output token
  count; ``message_start`` carries the input token count
* a single trailing ``ModelChunk(kind="finish", ...)`` is emitted when
  the stream ends, regardless of whether tools were called

The SDK is imported lazily inside ``__init__`` so users can
``from jeevesagent.model import AnthropicModel`` without the
``anthropic`` extra installed; the import only fires when the
constructor needs to build a default client.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..core.types import Message, ModelChunk, Role, ToolCall, ToolDef, Usage

DEFAULT_MAX_TOKENS = 4096


@dataclass
class _PartialTool:
    id: str = ""
    name: str = ""
    args_json: str = ""


class AnthropicModel:
    """Talks to Claude via :class:`anthropic.AsyncAnthropic`."""

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        *,
        client: Any = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.name = model
        self._max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover — depends on user env
                raise ImportError(
                    "Anthropic SDK not installed. "
                    "Install with: pip install 'jeevesagent[anthropic]'"
                ) from exc
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = AsyncAnthropic(api_key=key)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ModelChunk]:
        system, anth_messages = _to_anthropic_messages(messages)
        anth_tools = [_to_anthropic_tool(t) for t in (tools or [])]

        kwargs: dict[str, Any] = {
            "model": self.name,
            "messages": anth_messages,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if anth_tools:
            kwargs["tools"] = anth_tools

        partials: dict[int, _PartialTool] = {}
        agg_input = 0
        agg_output = 0
        finish_reason: str | None = None

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", None)

                if etype == "message_start":
                    msg = getattr(event, "message", None)
                    usage = getattr(msg, "usage", None) if msg is not None else None
                    if usage is not None:
                        agg_input += getattr(usage, "input_tokens", 0) or 0
                        agg_output += getattr(usage, "output_tokens", 0) or 0

                elif etype == "content_block_start":
                    block = event.content_block
                    if getattr(block, "type", None) == "tool_use":
                        partials[event.index] = _PartialTool(
                            id=getattr(block, "id", "") or "",
                            name=getattr(block, "name", "") or "",
                        )

                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta":
                        text = getattr(delta, "text", "") or ""
                        if text:
                            yield ModelChunk(kind="text", text=text)
                    elif dtype == "input_json_delta":
                        partial = partials.get(event.index)
                        if partial is not None:
                            partial.args_json += (
                                getattr(delta, "partial_json", "") or ""
                            )

                elif etype == "content_block_stop":
                    partial = partials.pop(event.index, None)
                    if partial is not None:
                        try:
                            args = (
                                json.loads(partial.args_json)
                                if partial.args_json
                                else {}
                            )
                        except json.JSONDecodeError:
                            args = {}
                        yield ModelChunk(
                            kind="tool_call",
                            tool_call=ToolCall(
                                id=partial.id,
                                tool=partial.name,
                                args=args,
                            ),
                        )

                elif etype == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        agg_output += getattr(usage, "output_tokens", 0) or 0
                    delta = getattr(event, "delta", None)
                    if delta is not None:
                        sr = getattr(delta, "stop_reason", None)
                        if sr:
                            finish_reason = sr

        yield ModelChunk(
            kind="finish",
            finish_reason=finish_reason or "end_turn",
            usage=Usage(input_tokens=agg_input, output_tokens=agg_output),
        )


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _to_anthropic_messages(
    messages: list[Message],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert our messages to ``(system_text, [anthropic_message, ...])``.

    Anthropic requires ``system`` as a top-level field and structures
    tool calls as ``tool_use`` content blocks on the assistant turn,
    with ``tool_result`` blocks returned in the next user turn.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    for m in messages:
        if m.role == Role.SYSTEM:
            system_parts.append(m.content)
            continue

        # Flush queued tool_results into a user turn before emitting any
        # non-tool message that follows.
        if m.role != Role.TOOL and pending_results:
            out.append({"role": "user", "content": pending_results})
            pending_results = []

        if m.role == Role.USER:
            out.append({"role": "user", "content": m.content})

        elif m.role == Role.ASSISTANT:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.tool,
                        "input": tc.args,
                    }
                )
            out.append(
                {"role": "assistant", "content": blocks if blocks else m.content}
            )

        elif m.role == Role.TOOL:
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content,
                }
            )

    if pending_results:
        out.append({"role": "user", "content": pending_results})

    return "\n\n".join(system_parts), out


def _to_anthropic_tool(t: ToolDef) -> dict[str, Any]:
    return {
        "name": t.name,
        "description": t.description,
        "input_schema": t.input_schema or {"type": "object", "properties": {}},
    }
