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
``from loomflow.model import AnthropicModel`` without the
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
        secrets: Any | None = None,
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
                    "Install with: pip install 'loomflow[anthropic]'"
                ) from exc
            # Resolution order: api_key= → secrets.lookup_sync →
            # os.environ. Same precedence as OpenAIModel — see the
            # comment there for rationale.
            resolved_key = api_key
            if resolved_key is None and secrets is not None:
                resolved_key = secrets.lookup_sync("ANTHROPIC_API_KEY")
            if resolved_key is None:
                resolved_key = os.environ.get("ANTHROPIC_API_KEY")
            self._client = AsyncAnthropic(api_key=resolved_key)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        output_schema: Any | None = None,
    ) -> tuple[str, list[ToolCall], Usage, str]:
        """Single-shot non-streaming completion.

        Calls ``client.messages.create(...)`` (no ``stream=True``,
        no ``stream`` context manager) — Anthropic returns the full
        ``Message`` in one HTTP response. We walk its ``content``
        blocks once to assemble ``(text, tool_calls, usage,
        stop_reason)``. Used by the non-streaming hot path
        (``agent.run()``); ``agent.stream()`` keeps using
        :meth:`stream`.

        ``output_schema`` (when set) is implemented via the
        forced-tool-call pattern Anthropic recommends for structured
        output: a synthetic ``__output__`` tool is appended to the
        tool list with the schema as its ``input_schema``, and
        ``tool_choice`` forces the model to invoke it. The model's
        constrained tool-args are extracted and returned as the
        message ``text`` (a JSON string the agent loop can parse).

        Falls back to consuming :meth:`stream` if the underlying
        client raises (test fakes that only support streaming, or
        transports that don't honour single-shot creation).
        """
        system, anth_messages = _to_anthropic_messages(messages)
        anth_tools = [_to_anthropic_tool(t) for t in (tools or [])]

        # Structured output: synthesize a forced tool call.
        synthetic_tool_name = ""
        if output_schema is not None:
            synthetic = _schema_as_tool(output_schema)
            if synthetic is not None:
                anth_tools.append(synthetic)
                synthetic_tool_name = synthetic["name"]

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
        if synthetic_tool_name:
            # Force the model to invoke the structured-output tool.
            # The model can still chain real-tool calls before it,
            # but its terminal response MUST be the synthetic one.
            kwargs["tool_choice"] = {
                "type": "tool",
                "name": synthetic_tool_name,
            }

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception:  # noqa: BLE001 — fallback for fake / non-conforming clients
            return await _consume_anthropic_stream(
                self.stream(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    output_schema=output_schema,
                )
            )

        # If the SDK actually returned a stream object instead of a
        # Message (some test fakes), drain the stream path.
        if hasattr(response, "__aiter__") and not hasattr(response, "content"):
            return await _consume_anthropic_stream(
                self.stream(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    output_schema=output_schema,
                )
            )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in getattr(response, "content", None) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_name = getattr(block, "name", "") or ""
                args_raw = getattr(block, "input", None)
                args: dict[str, Any] = (
                    dict(args_raw) if isinstance(args_raw, dict) else {}
                )
                # Synthetic structured-output tool: the args ARE the
                # schema-validated output. Surface as text so the
                # agent loop's parser sees a JSON string and the
                # validate-with-retry path succeeds first try.
                if synthetic_tool_name and tool_name == synthetic_tool_name:
                    text_parts.append(json.dumps(args))
                    continue
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", "") or "",
                        tool=tool_name,
                        args=args,
                    )
                )

        u = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
        )
        stop_reason = (
            getattr(response, "stop_reason", None) or "end_turn"
        )
        return "".join(text_parts), tool_calls, usage, str(stop_reason)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        output_schema: Any | None = None,
    ) -> AsyncIterator[ModelChunk]:
        system, anth_messages = _to_anthropic_messages(messages)
        anth_tools = [_to_anthropic_tool(t) for t in (tools or [])]
        synthetic_tool_name = ""
        if output_schema is not None:
            synthetic = _schema_as_tool(output_schema)
            if synthetic is not None:
                anth_tools.append(synthetic)
                synthetic_tool_name = synthetic["name"]

        kwargs: dict[str, Any] = {
            "model": self.name,
            "messages": anth_messages,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if synthetic_tool_name:
            kwargs["tool_choice"] = {
                "type": "tool",
                "name": synthetic_tool_name,
            }
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


async def _consume_anthropic_stream(
    chunks: AsyncIterator[ModelChunk],
) -> tuple[str, list[ToolCall], Usage, str]:
    """Drain a ``ModelChunk`` stream into the same return tuple as
    :meth:`AnthropicModel.complete`. Used when the non-streaming
    transport path is unavailable (test fakes / niche SDKs)."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    usage = Usage()
    finish_reason = "end_turn"
    async for chunk in chunks:
        if chunk.kind == "text" and chunk.text is not None:
            text_parts.append(chunk.text)
        elif (
            chunk.kind == "tool_call" and chunk.tool_call is not None
        ):
            tool_calls.append(chunk.tool_call)
        elif chunk.kind == "finish":
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
    return "".join(text_parts), tool_calls, usage, finish_reason


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


def _schema_as_tool(output_schema: Any | None) -> dict[str, Any] | None:
    """Translate a Pydantic ``BaseModel`` into a synthetic Anthropic
    tool whose ``input_schema`` IS the requested output schema.

    Combined with ``tool_choice={"type": "tool", "name": ...}`` on
    the request, this forces the model to emit a single tool_use
    block whose ``input`` is a JSON object matching the schema —
    Anthropic's idiomatic structured-output pattern. The agent loop
    parses that JSON via the existing validate-with-retry path,
    which now almost never has to retry.

    Returns ``None`` when the supplied object isn't a Pydantic
    model (defensive — the protocol types this loosely as ``Any``).
    """
    if output_schema is None:
        return None
    # Tagged unions are handled by prompt-augmentation + validate-
    # with-retry in the agent loop. Native forced-tool-call pattern
    # only fits a single concrete schema.
    if not (isinstance(output_schema, type) and hasattr(output_schema, "model_json_schema")):
        return None
    schema_method = getattr(output_schema, "model_json_schema", None)
    if not callable(schema_method):
        return None
    return {
        "name": "__output__",
        "description": (
            "Emit the final response. The provided arguments must be "
            "a JSON object matching the schema; this is your only "
            "way to return a result for this turn."
        ),
        "input_schema": schema_method(),
    }
