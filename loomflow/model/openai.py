"""Adapter for OpenAI chat completions via the official ``openai`` SDK.

Streams via ``chat.completions.create(stream=True)``. OpenAI streams
text in ``delta.content`` and tool calls in ``delta.tool_calls`` arrays
where each entry carries an ``index``; the same index across chunks
refers to the same tool call (so we accumulate by index). The final
chunk with ``stream_options={"include_usage": True}`` carries token
counts.

The SDK is imported lazily inside ``__init__`` so users without the
``openai`` extra installed can still ``from loomflow.model import
OpenAIModel`` — the import only fires when constructing without
passing a ``client``.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..core.ids import new_id
from ..core.types import Message, ModelChunk, Role, ToolCall, ToolDef, Usage
from ._pricing import estimate_cost


@dataclass
class _OAIPartial:
    id: str = ""
    name: str = ""
    args_json: str = ""


class OpenAIModel:
    """Talks to OpenAI via :class:`openai.AsyncOpenAI`."""

    # Tells the agent loop this adapter translates ``output_schema``
    # into a provider-native call (``response_format=json_schema``,
    # ``strict=True``). When True, the loop skips the redundant
    # in-prompt JSON Schema directive — the API-level constraint is
    # enough on its own. Saves ~2k input tokens per structured-output
    # call without changing reliability (the validate-with-retry
    # path still adds the schema to the retry message if validation
    # ever fails). Off-by-default so unknown / custom adapters keep
    # the prompt-augmentation safety net.
    supports_native_structured_output: bool = True

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        client: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        secrets: Any | None = None,
    ) -> None:
        self.name = model
        if client is not None:
            self._client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover — depends on user env
                raise ImportError(
                    "OpenAI SDK not installed. "
                    "Install with: pip install 'loomflow[openai]'"
                ) from exc
            # Resolution order for the API key:
            #   1. Explicit ``api_key=`` argument
            #   2. ``secrets.lookup_sync("OPENAI_API_KEY")`` if a
            #      Secrets backend is wired (vault / dict-based)
            #   3. ``os.environ["OPENAI_API_KEY"]`` as the bare
            #      fallback for unconfigured callers
            resolved_key = api_key
            if resolved_key is None and secrets is not None:
                resolved_key = secrets.lookup_sync("OPENAI_API_KEY")
            if resolved_key is None:
                resolved_key = os.environ.get("OPENAI_API_KEY")
            self._client = AsyncOpenAI(
                api_key=resolved_key,
                base_url=base_url,
            )

    def _effort_kwargs(
        self, effort: str | None, strict_effort: bool
    ) -> dict[str, Any]:
        """Translate the reasoning-effort dial into provider kwargs.

        Default is the OpenAI prefix-matched mapping in
        :mod:`loomflow.model._effort`. Subclasses that route through a
        different transport (e.g. :class:`LiteLLMModel` — same wire
        shape, many providers) override this to do their own
        translation, since the ``self.name`` check below only knows
        about real OpenAI models.
        """
        from ._effort import openai_kwargs
        return openai_kwargs(effort, self.name, strict=strict_effort)

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
        """Single-shot completion (no per-chunk yields).

        Tries the OpenAI non-streaming endpoint
        (``stream=False``) first. If that fails — e.g. when a test
        fake client only supports streaming, or a transport doesn't
        honor ``stream=False`` — falls back to consuming
        :meth:`stream` internally and accumulating the result. The
        fallback still saves the per-chunk yield + Event
        construction overhead on the architecture side because
        ReAct calls ``complete`` with a single ``await``.

        ``output_schema`` (when set) is translated to OpenAI's
        ``response_format=json_schema`` with ``strict=True``. The
        model is then constrained at decode time to emit valid JSON
        matching the schema, eliminating the need for the agent
        loop's validation-retry round-trip on the happy path.
        """
        oai_messages = _to_openai_messages(messages)
        oai_tools = [_to_openai_tool(t) for t in (tools or [])]

        kwargs: dict[str, Any] = {
            "model": self.name,
            "messages": oai_messages,
            "stream": False,
        }
        if temperature != 1.0:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if oai_tools:
            kwargs["tools"] = oai_tools
        rf = _build_response_format(output_schema)
        if rf is not None:
            kwargs["response_format"] = rf
        # Reasoning-effort translation. Adapter knows which OpenAI
        # models accept ``reasoning_effort`` and drops the kwarg on
        # everything else (warn-once or strict-raise depending on
        # the caller). See ``loomflow/model/_effort.py``.
        kwargs.update(self._effort_kwargs(effort, strict_effort))

        # OpenAI prompt caching is **automatic** — no request-side
        # change required. The optional ``prompt_cache_key`` improves
        # cache-hit routing when many requests share a long prefix
        # (e.g. per-user system prompt). Forward it when the caller
        # supplied one via ``PromptCacheConfig(cache_key=...)``.
        if prompt_caching is not None and getattr(
            prompt_caching, "cache_key", None
        ):
            kwargs["prompt_cache_key"] = prompt_caching.cache_key

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception:  # noqa: BLE001 — surface the fallback transparently
            return await _consume_stream(
                self.stream(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    output_schema=output_schema,
                    effort=effort,
                    strict_effort=strict_effort,
                    prompt_caching=prompt_caching,
                )
            )

        choices = getattr(response, "choices", None) or []
        if not choices:
            # Some clients (test fakes; transports that don't honour
            # ``stream=False``) return an iterator instead of a
            # single response object. Fall back to consuming the
            # adapter's own ``stream`` — it knows how to parse the
            # chunks into ModelChunks.
            if hasattr(response, "__aiter__"):
                return await _consume_stream(
                    self.stream(
                        messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                )
            return ("", [], Usage(), "stop")
        choice = choices[0]
        message = getattr(choice, "message", None)

        text = (getattr(message, "content", None) or "") if message else ""

        tool_calls: list[ToolCall] = []
        raw_tool_calls = (
            getattr(message, "tool_calls", None) if message else None
        )
        for tc in raw_tool_calls or []:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            args_json = getattr(fn, "arguments", "") if fn else ""
            try:
                args = json.loads(args_json) if args_json else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", None) or new_id("tc"),
                    tool=name,
                    args=args,
                )
            )

        u = getattr(response, "usage", None)
        total_in = getattr(u, "prompt_tokens", 0) or 0
        out_tok = getattr(u, "completion_tokens", 0) or 0
        # OpenAI uses ``subset`` semantics: ``prompt_tokens`` is the
        # TOTAL prompt size (cache hits + misses), with
        # ``prompt_tokens_details.cached_tokens`` as the cached
        # portion. Normalise to loomflow's ``separate buckets``
        # convention so downstream Usage math doesn't double-count.
        details = getattr(u, "prompt_tokens_details", None)
        cache_read = getattr(details, "cached_tokens", 0) or 0
        in_tok = max(0, total_in - cache_read)  # uncached portion
        usage = Usage(
            input_tokens=in_tok,
            cached_input_tokens=cache_read,
            # OpenAI doesn't surface cache_write_tokens — writes are
            # free and not billed separately. Always 0 for OpenAI.
            output_tokens=out_tok,
            cost_usd=estimate_cost(
                self.name,
                in_tok,
                out_tok,
                cached_input_tokens=cache_read,
            ),
        )
        finish_reason = getattr(choice, "finish_reason", None) or "stop"
        return text, tool_calls, usage, str(finish_reason)

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
        oai_messages = _to_openai_messages(messages)
        oai_tools = [_to_openai_tool(t) for t in (tools or [])]

        kwargs: dict[str, Any] = {
            "model": self.name,
            "messages": oai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature != 1.0:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if oai_tools:
            kwargs["tools"] = oai_tools
        rf = _build_response_format(output_schema)
        if rf is not None:
            kwargs["response_format"] = rf
        kwargs.update(self._effort_kwargs(effort, strict_effort))
        # See the ``complete()`` path for the rationale — caching is
        # automatic on OpenAI; only the optional routing hint flows
        # through here.
        if prompt_caching is not None and getattr(
            prompt_caching, "cache_key", None
        ):
            kwargs["prompt_cache_key"] = prompt_caching.cache_key

        partials: dict[int, _OAIPartial] = {}
        usage = Usage()
        finish_reason: str | None = None

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                total_in = getattr(chunk_usage, "prompt_tokens", 0) or 0
                out_tok = getattr(chunk_usage, "completion_tokens", 0) or 0
                # Same cache-aware normalisation as ``complete()``:
                # cached_tokens is a subset of prompt_tokens; surface
                # them as separate buckets.
                details = getattr(chunk_usage, "prompt_tokens_details", None)
                cache_read = getattr(details, "cached_tokens", 0) or 0
                in_tok = max(0, total_in - cache_read)
                usage = Usage(
                    input_tokens=in_tok,
                    cached_input_tokens=cache_read,
                    output_tokens=out_tok,
                    cost_usd=estimate_cost(
                        self.name,
                        in_tok,
                        out_tok,
                        cached_input_tokens=cache_read,
                    ),
                )

            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if content:
                yield ModelChunk(kind="text", text=content)

            tool_call_deltas = getattr(delta, "tool_calls", None)
            if tool_call_deltas:
                for tc in tool_call_deltas:
                    idx = getattr(tc, "index", 0) or 0
                    p = partials.setdefault(idx, _OAIPartial())
                    tc_id = getattr(tc, "id", None)
                    if tc_id:
                        p.id = tc_id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        fn_name = getattr(fn, "name", None)
                        if fn_name:
                            p.name = fn_name
                        fn_args = getattr(fn, "arguments", None)
                        if fn_args:
                            p.args_json += fn_args

            fr = getattr(choice, "finish_reason", None)
            if fr:
                finish_reason = fr

        # OpenAI emits each tool call across many deltas keyed by index;
        # only after the stream ends do we have a complete picture.
        for idx in sorted(partials.keys()):
            p = partials[idx]
            try:
                args = json.loads(p.args_json) if p.args_json else {}
            except json.JSONDecodeError:
                args = {}
            yield ModelChunk(
                kind="tool_call",
                tool_call=ToolCall(
                    id=p.id or new_id("tc"),
                    tool=p.name,
                    args=args,
                ),
            )

        yield ModelChunk(
            kind="finish",
            finish_reason=finish_reason or "stop",
            usage=usage,
        )


# ---------------------------------------------------------------------------
# Stream-consumption fallback for complete()
# ---------------------------------------------------------------------------


async def _consume_stream(
    chunks: AsyncIterator[ModelChunk],
) -> tuple[str, list[ToolCall], Usage, str]:
    """Drain a ``ModelChunk`` stream into the same return tuple as
    :meth:`OpenAIModel.complete`. Used when the non-streaming
    transport isn't available."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    usage = Usage()
    finish_reason = "stop"
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


async def _replay(it: AsyncIterator[Any]) -> AsyncIterator[ModelChunk]:
    """Last-resort: yield from an unknown async iterator that we
    couldn't classify as a Response. Only used when a fake test
    client returns an iterator from a ``stream=False`` call."""
    async for item in it:
        if isinstance(item, ModelChunk):
            yield item


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert our messages to OpenAI's role/tool_call_id wire format."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            out.append({"role": "system", "content": m.content})
        elif m.role == Role.USER:
            out.append({"role": "user", "content": m.content})
        elif m.role == Role.ASSISTANT:
            msg: dict[str, Any] = {"role": "assistant"}
            if m.content:
                msg["content"] = m.content
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool,
                            "arguments": json.dumps(tc.args),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif m.role == Role.TOOL:
            out.append(
                {
                    "role": "tool",
                    "content": m.content,
                    "tool_call_id": m.tool_call_id or "",
                }
            )
    return out


def _to_openai_tool(t: ToolDef) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema or {"type": "object", "properties": {}},
        },
    }


def _build_response_format(output_schema: Any | None) -> dict[str, Any] | None:
    """Translate a Pydantic ``BaseModel`` subclass into OpenAI's
    ``response_format=json_schema`` payload.

    Returns ``None`` when the caller didn't request structured
    output, or when the supplied object isn't a Pydantic model
    (defensive — the protocol types this loosely as ``Any``).

    The ``strict=True`` flag tells OpenAI to enforce the schema at
    decode time. The model is constrained to emit valid JSON
    matching the schema, which means the agent loop's
    validation-with-retry path becomes a fallback that almost
    never fires. Supported on ``gpt-4o``, ``gpt-4o-mini``,
    ``gpt-4.1`` and newer; older models silently fall back to
    prompt-augmentation behaviour at the OpenAI side.

    Pydantic's emitted schemas are *not* compatible with OpenAI's
    strict mode out of the box: strict mode requires every object
    to carry ``additionalProperties: false`` AND lists every
    property in ``required``. We normalise the schema in place
    before sending so callers don't need to hand-roll a strict-
    compatible Pydantic model.
    """
    if output_schema is None:
        return None
    # Tagged unions (``A | B``) — OpenAI strict mode requires the
    # root schema to be a single ``type: object``, so we can't use
    # native response_format here. The agent loop's prompt-
    # augmentation + validate-with-retry path handles unions
    # correctly; returning ``None`` opts into that fallback.
    if not (isinstance(output_schema, type) and hasattr(output_schema, "model_json_schema")):
        return None
    schema_method = getattr(output_schema, "model_json_schema", None)
    if not callable(schema_method):
        return None
    name = getattr(output_schema, "__name__", "Output")
    schema = _normalise_schema_for_openai_strict(schema_method())
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": True,
        },
    }


def _normalise_schema_for_openai_strict(schema: Any) -> Any:
    """Mutate a JSON Schema in place so it satisfies OpenAI's
    ``response_format=json_schema, strict=True`` constraints.

    Two transformations are applied to every nested ``"type":
    "object"`` node (top-level + ``$defs`` + nested properties):

    1. ``additionalProperties`` is set to ``False`` if not already
       present. OpenAI strict mode rejects schemas that don't
       explicitly forbid extra fields.
    2. The ``required`` array is set to every key in
       ``properties``. Strict mode rejects schemas with optional
       properties; Pydantic ``Optional[T]`` fields are emitted as
       ``anyOf: [T, null]`` which strict mode DOES accept when
       listed as required, so this is the right move.

    Returns the same object (mutated). Non-dict inputs return
    unchanged.
    """
    if isinstance(schema, dict):
        # Walk $defs first so the referenced sub-schemas are
        # normalised before strict-mode validation reads them.
        defs = schema.get("$defs")
        if isinstance(defs, dict):
            for d in defs.values():
                _normalise_schema_for_openai_strict(d)
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            props = schema.get("properties")
            if isinstance(props, dict):
                schema["required"] = list(props.keys())
                for v in props.values():
                    _normalise_schema_for_openai_strict(v)
        # Recurse into other constructs (anyOf / oneOf / allOf /
        # arrays / etc.) so nested objects deeper in the tree are
        # also normalised.
        for key, v in list(schema.items()):
            if key in ("$defs", "properties"):
                continue
            if isinstance(v, (dict, list)):
                _normalise_schema_for_openai_strict(v)
    elif isinstance(schema, list):
        for item in schema:
            _normalise_schema_for_openai_strict(item)
    return schema
