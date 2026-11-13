"""Shared parsing for the tool-call arguments a model emits.

Providers hand tool-call arguments over as a JSON *string* (OpenAI
``function.arguments``, Anthropic streamed ``input_json_delta``). That
string is model output: it can be cut off mid-object by the token
limit, or be valid JSON that is not an object (``null``, a list, a
bare string). Neither may be turned into ``{}`` and executed — a tool
would then run with its *defaults* (``wipe(path="/")``). The adapters
call :func:`parse_tool_args` and put the error on
:attr:`ToolCall.args_error`; the tool dispatcher refuses to run such a
call and returns the error to the model instead.

:func:`collect_stream` drains a ``ModelChunk`` stream into the
``complete()`` return tuple, for wrappers around stream-only models.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..core.types import ModelChunk, ToolCall, Usage

# How much of the unusable argument text is echoed back to the model.
_RAW_PREVIEW_CHARS = 200


def parse_tool_args(raw: str | None) -> tuple[dict[str, Any], str | None]:
    """``(args, error)`` for a raw tool-call argument string.

    An empty string is a call with no arguments (``{}``, no error).
    On a parse failure or a non-object value, ``args`` is ``{}`` and
    ``error`` says what was wrong (for :attr:`ToolCall.args_error`).
    """
    if not raw:
        return {}, None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return {}, (
            f"the arguments for this tool call were not valid JSON ({exc}); "
            f"received: {_preview(raw)}"
        )
    if not isinstance(value, dict):
        kind = "null" if value is None else type(value).__name__
        return {}, (
            "the arguments for this tool call must be a JSON object "
            f"({{\"param\": value, ...}}), not {kind}; received: {_preview(raw)}"
        )
    return value, None


def _preview(raw: str) -> str:
    if len(raw) <= _RAW_PREVIEW_CHARS:
        return repr(raw)
    return repr(raw[:_RAW_PREVIEW_CHARS]) + f"... ({len(raw)} chars)"


async def collect_stream(
    chunks: AsyncIterator[ModelChunk],
) -> tuple[str, list[ToolCall], Usage, str]:
    """Drain a ``ModelChunk`` stream into the ``complete()`` tuple
    ``(text, tool_calls, usage, finish_reason)``."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    usage = Usage()
    finish_reason = "stop"
    async for chunk in chunks:
        if chunk.kind == "text" and chunk.text is not None:
            text_parts.append(chunk.text)
        elif chunk.kind == "tool_call" and chunk.tool_call is not None:
            tool_calls.append(chunk.tool_call)
        elif chunk.kind == "finish":
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
    return "".join(text_parts), tool_calls, usage, finish_reason
