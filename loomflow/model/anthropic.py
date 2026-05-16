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
from ._pricing import estimate_cost


def _cache_control_for(prompt_caching: Any) -> dict[str, Any] | None:
    """Build the ``cache_control`` block for Anthropic's API when
    caching is enabled, otherwise return ``None``.

    Anthropic accepts ``{"type": "ephemeral"}`` (5-min TTL, default)
    or ``{"type": "ephemeral", "ttl": "1h"}`` (1-hour TTL at 2x
    write premium). The dict goes onto the LAST cacheable content
    block in ``system`` / ``tools`` / ``messages`` — Anthropic
    caches everything up to and including that breakpoint.
    """
    if prompt_caching is None or not getattr(prompt_caching, "enabled", False):
        return None
    ttl = getattr(prompt_caching, "ttl", "5m")
    out: dict[str, Any] = {"type": "ephemeral"}
    if ttl == "1h":
        out["ttl"] = "1h"
    return out


def _apply_anthropic_cache_control(
    kwargs: dict[str, Any],
    system_parts: list[str],
    anth_tools: list[dict[str, Any]],
    cache_ctrl: dict[str, Any] | None,
) -> None:
    """Inject Anthropic cache breakpoints when caching is enabled.

    Anthropic accepts **up to four** ``cache_control`` markers per
    request, each creating an independent cache entry (the longest
    matching prefix wins on read). We use them like this:

    1. The LAST tool definition — caches the entire tool array.
    2. The LAST *system content block* — caches the full system
       prompt (catches turns where recall is empty / unchanged).
    3. Earlier system blocks, working backward — caches stable
       prefixes (instructions, memory blocks) independently of any
       per-turn-volatile recall block that sits later.

    Architectures emit system content as a list of strings (one per
    semantic chunk — instructions, memory, recall). We render each
    as its own content block and place ``cache_control`` on the
    LAST N (N = min(3, num_blocks)) so:

    * Single-block messages still get one marker (back-compat).
    * Two-block messages (instructions + memory OR instructions +
      recall) get two markers — both blocks independently cached.
    * Three-block messages (instructions + memory + recall) use all
      three system breakpoints, leaving exactly one for tools.

    ``cache_control`` placed beyond block 3 would put us at 5 total
    markers and Anthropic would reject the request; we cap at 3
    system + 1 tools = the 4 the API supports.

    Mutates ``kwargs`` in place: rewrites ``kwargs["system"]`` as a
    content-block list when caching is on, and annotates
    ``anth_tools[-1]`` with cache_control.
    """
    if cache_ctrl is None:
        return
    if system_parts:
        # Build content blocks for each system part; mark the LAST
        # N with cache_control. Anthropic supports 4 breakpoints
        # total; we reserve 1 for tools (below) and use the other
        # 3 for system. With <3 parts, only that many markers fire.
        max_system_markers = 3
        n_parts = len(system_parts)
        n_marked = min(max_system_markers, n_parts)
        first_marked = n_parts - n_marked

        blocks: list[dict[str, Any]] = []
        for i, part in enumerate(system_parts):
            block: dict[str, Any] = {"type": "text", "text": part}
            if i >= first_marked:
                block["cache_control"] = cache_ctrl
            blocks.append(block)
        kwargs["system"] = blocks
    if anth_tools:
        # Annotate the LAST tool with cache_control; caches every
        # tool definition up to and including this one. This is the
        # 4th breakpoint, leaving room for up to 3 system markers
        # above.
        anth_tools[-1] = {**anth_tools[-1], "cache_control": cache_ctrl}
        kwargs["tools"] = anth_tools

DEFAULT_MAX_TOKENS = 4096


@dataclass
class _PartialTool:
    id: str = ""
    name: str = ""
    args_json: str = ""


class AnthropicModel:
    """Talks to Claude via :class:`anthropic.AsyncAnthropic`."""

    # See ``OpenAIModel.supports_native_structured_output``. Anthropic
    # has no first-party ``response_format``; we translate
    # ``output_schema`` into a forced tool call (``tool_choice``
    # pointing at a synthetic ``__output__`` tool whose ``input_schema``
    # IS the requested schema). The tool is required, so the model
    # MUST emit a JSON object matching the schema. Equivalent
    # constraint, no schema text needed in the prompt.
    supports_native_structured_output: bool = True

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

    async def count_tokens(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
    ) -> int:
        """Native token count via Anthropic's
        ``messages.count_tokens`` beta endpoint — exact, no
        round-trip cost beyond a single API call.

        Mirrors the same ``messages`` / ``tools`` shaping the
        ``complete`` and ``stream`` paths use, so the count is
        what the actual completion would be billed for.
        :func:`loomflow.model.count_tokens.count_tokens` discovers
        this method via ``hasattr`` and prefers it over the
        tiktoken / char-based fallbacks.
        """
        system_parts, anth_messages = _to_anthropic_messages(messages)
        anth_tools = [_to_anthropic_tool(t) for t in (tools or [])]
        kwargs: dict[str, Any] = {
            "model": self.name,
            "messages": anth_messages,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if anth_tools:
            kwargs["tools"] = anth_tools
        # The ``count_tokens`` endpoint is on the ``messages``
        # namespace and returns ``{"input_tokens": N}``. Older
        # SDKs route it through ``messages.beta.count_tokens``;
        # we try the modern path first and fall back.
        try:
            resp = await self._client.messages.count_tokens(**kwargs)
        except (AttributeError, TypeError):
            resp = await self._client.beta.messages.count_tokens(
                **kwargs
            )
        return int(getattr(resp, "input_tokens", 0))

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
        system_parts, anth_messages = _to_anthropic_messages(messages)
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
        if system_parts:
            # Default: join into one string (caching-off shape).
            # ``_apply_anthropic_cache_control`` below rewrites this
            # into a content-block list when caching is enabled.
            kwargs["system"] = "\n\n".join(system_parts)
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
        # Reasoning-effort translation. Adapter picks the right
        # regime (Opus 4.7 adaptive-only / 4.6 adaptive+effort /
        # legacy budget_tokens) based on the model name; drops the
        # kwarg with a warning on models that don't support it.
        from ._effort import anthropic_kwargs

        kwargs.update(
            anthropic_kwargs(effort, self.name, strict=strict_effort)
        )

        # Prompt cache injection (no-op when caching is disabled).
        # Must run AFTER ``kwargs["tools"]`` / ``kwargs["system"]``
        # are set, since the helper rewrites them with cache_control
        # markers when enabled.
        cache_ctrl = _cache_control_for(prompt_caching)
        _apply_anthropic_cache_control(
            kwargs, system_parts, anth_tools, cache_ctrl
        )

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
                    effort=effort,
                    strict_effort=strict_effort,
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
        in_tok = getattr(u, "input_tokens", 0) or 0
        out_tok = getattr(u, "output_tokens", 0) or 0
        # Anthropic uses ``separate buckets`` semantics:
        # ``input_tokens`` = tokens AFTER the last cache breakpoint
        # (charged at full rate); ``cache_read_input_tokens`` =
        # cache hits (0.1x); ``cache_creation_input_tokens`` =
        # tokens written into cache on this call (1.25x for 5m,
        # 2x for 1h). Total prompt processed =
        # in_tok + cache_read + cache_write. We forward this shape
        # directly through Usage.
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        ttl = getattr(prompt_caching, "ttl", "5m") if prompt_caching else "5m"
        usage = Usage(
            input_tokens=in_tok,
            cached_input_tokens=cache_read,
            cache_write_tokens=cache_write,
            output_tokens=out_tok,
            cost_usd=estimate_cost(
                self.name,
                in_tok,
                out_tok,
                cached_input_tokens=cache_read,
                cache_write_tokens=cache_write,
                cache_ttl=ttl,
            ),
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
        effort: str | None = None,
        strict_effort: bool = False,
        prompt_caching: Any = None,
    ) -> AsyncIterator[ModelChunk]:
        system_parts, anth_messages = _to_anthropic_messages(messages)
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
        if system_parts:
            # Default: caching-off shape (single joined string). The
            # cache-control helper rewrites this to a multi-block list
            # when caching is enabled.
            kwargs["system"] = "\n\n".join(system_parts)
        if synthetic_tool_name:
            kwargs["tool_choice"] = {
                "type": "tool",
                "name": synthetic_tool_name,
            }
        if anth_tools:
            kwargs["tools"] = anth_tools
        from ._effort import anthropic_kwargs

        kwargs.update(
            anthropic_kwargs(effort, self.name, strict=strict_effort)
        )

        # Prompt cache injection (no-op when caching is disabled).
        cache_ctrl = _cache_control_for(prompt_caching)
        _apply_anthropic_cache_control(
            kwargs, system_parts, anth_tools, cache_ctrl
        )

        partials: dict[int, _PartialTool] = {}
        agg_input = 0
        agg_cache_read = 0
        agg_cache_write = 0
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
                        # Cache stats are emitted on ``message_start``
                        # in the streaming event sequence (the API
                        # decides cache hit/miss before producing
                        # tokens). Pull them once here.
                        agg_cache_read += (
                            getattr(usage, "cache_read_input_tokens", 0) or 0
                        )
                        agg_cache_write += (
                            getattr(usage, "cache_creation_input_tokens", 0)
                            or 0
                        )

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

        ttl = (
            getattr(prompt_caching, "ttl", "5m")
            if prompt_caching else "5m"
        )
        yield ModelChunk(
            kind="finish",
            finish_reason=finish_reason or "end_turn",
            usage=Usage(
                input_tokens=agg_input,
                cached_input_tokens=agg_cache_read,
                cache_write_tokens=agg_cache_write,
                output_tokens=agg_output,
                cost_usd=estimate_cost(
                    self.name,
                    agg_input,
                    agg_output,
                    cached_input_tokens=agg_cache_read,
                    cache_write_tokens=agg_cache_write,
                    cache_ttl=ttl,
                ),
            ),
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
) -> tuple[list[str], list[dict[str, Any]]]:
    """Convert our messages to ``(system_parts, [anthropic_message, ...])``.

    Anthropic requires ``system`` as a top-level field and structures
    tool calls as ``tool_use`` content blocks on the assistant turn,
    with ``tool_result`` blocks returned in the next user turn.

    Returns ``system_parts`` as a list (one string per Role.SYSTEM
    message) so the cache-control helper can emit one content block
    per semantic chunk (instructions / memory / recall) with its own
    cache_control marker. Callers that don't enable caching join
    with ``\\n\\n`` to recover the old single-string shape.
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

    return system_parts, out


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
