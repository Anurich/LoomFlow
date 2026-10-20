"""OpenTelemetry GenAI semantic-convention helpers (``gen_ai.*``).

Pure functions — no OTel SDK dependency — so span emit sites stay
one-liners. Built against the *Development*-status draft of the OTel
GenAI semantic conventions:

* model-call ("chat") spans: ``gen_ai.operation.name="chat"`` (Required),
  ``gen_ai.provider.name`` (Required), ``gen_ai.request.model``
  (Recommended), plus post-hoc ``gen_ai.usage.input_tokens`` /
  ``gen_ai.usage.output_tokens`` / ``gen_ai.response.finish_reasons``
  once the call has completed.
* tool-execution spans: ``gen_ai.operation.name="execute_tool"``,
  ``gen_ai.tool.name``, ``gen_ai.tool.call.id``.

Emit sites keep their ``loom.*`` span names, which loomflow's own
sinks (in-memory, console, JSONL) record as-is. The OpenTelemetry
exporter gives semconv spans their spec names and kinds instead —
``chat {model}`` (kind ``CLIENT``), ``execute_tool {tool}``,
``invoke_agent`` — carrying the ``loom.*`` name as the
``loom.span.name`` attribute (see :func:`otel_span_identity`).

Metrics follow the client-inference conventions: the
``gen_ai.client.inference.*`` token counters and histograms, the
inference and tool duration histograms (seconds, with the spec's
bucket boundaries as an advisory — opentelemetry-sdk releases that
ignore advisories need a View for them), described by
:data:`METRIC_INSTRUMENTS`. The ``loom.*`` metrics are still emitted
alongside them.

Post-hoc attributes (token usage is only known after the model call
returns) are applied through :func:`set_span_attributes`, which
duck-types against the yielded :class:`~loomflow.core.types.Span`
value object: a ``set_attribute`` method wins if present, otherwise
the mutable ``attributes`` dict is updated in place. Telemetry sinks
read the yielded span's attributes at close, so late additions are
captured without any protocol change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: ``gen_ai.operation.name`` for model inference calls.
OPERATION_CHAT = "chat"
#: ``gen_ai.operation.name`` for tool executions requested by the model.
OPERATION_EXECUTE_TOOL = "execute_tool"
#: ``gen_ai.operation.name`` for a whole agent run.
OPERATION_INVOKE_AGENT = "invoke_agent"
#: Fallback ``gen_ai.provider.name`` when the provider can't be derived.
PROVIDER_OTHER = "_OTHER"

# Well-known ``gen_ai.provider.name`` values keyed by the lowercase
# provider token found in a LiteLLM-style ``provider/model`` id or an
# adapter's explicit ``provider`` attribute.
_PROVIDER_TOKENS: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "azure": "azure.ai.openai",
    "azure_ai": "azure.ai.inference",
    "bedrock": "aws.bedrock",
    "gemini": "gcp.gemini",
    "google": "gcp.gemini",
    "vertex_ai": "gcp.vertex_ai",
    "mistral": "mistral_ai",
    "mistralai": "mistral_ai",
    "cohere": "cohere",
    "deepseek": "deepseek",
    "groq": "groq",
    "perplexity": "perplexity",
    "xai": "x_ai",
}

# Bare-model-id prefix heuristics ("claude-sonnet-4-5" → anthropic).
_MODEL_ID_PREFIXES: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("chatgpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini", "gcp.gemini"),
    ("mistral", "mistral_ai"),
    ("command", "cohere"),
    ("deepseek", "deepseek"),
    ("grok", "x_ai"),
    ("sonar", "perplexity"),
)


def provider_name(model: Any) -> str:
    """Best-effort ``gen_ai.provider.name`` for a model adapter.

    Resolution order:

    1. an explicit ``provider`` attribute on the adapter (duck-typed);
    2. the adapter's class name (``AnthropicModel`` → ``anthropic``);
    3. the model id: a LiteLLM-style ``provider/model`` prefix, then
       well-known bare-id prefixes (``claude-*`` → ``anthropic``);
    4. :data:`PROVIDER_OTHER`.
    """
    explicit = getattr(model, "provider", None)
    if isinstance(explicit, str) and explicit:
        return _PROVIDER_TOKENS.get(explicit.lower(), explicit.lower())

    type_name = type(model).__name__.lower()
    if "anthropic" in type_name:
        return "anthropic"
    # LiteLLMModel subclasses OpenAIModel, so an "openai" *type-name*
    # match is safe — routing info for LiteLLM lives in the model id,
    # which never contains "openai" in the class name.
    if "openai" in type_name:
        return "openai"

    model_id = str(getattr(model, "name", "") or "").lower()
    if "/" in model_id:
        token = model_id.split("/", 1)[0]
        if token in _PROVIDER_TOKENS:
            return _PROVIDER_TOKENS[token]
    for prefix, provider in _MODEL_ID_PREFIXES:
        if model_id.startswith(prefix):
            return provider
    return PROVIDER_OTHER


def chat_span_name(model_id: str) -> str:
    """Spec span name for a model call: ``chat {model_id}``."""
    return f"chat {model_id}" if model_id else "chat"


def tool_span_name(tool_name: str) -> str:
    """Spec span name for a tool execution: ``execute_tool {name}``."""
    return f"execute_tool {tool_name}" if tool_name else "execute_tool"


def chat_attrs(model: Any) -> dict[str, Any]:
    """``gen_ai.*`` attributes known *before* a model call starts."""
    request_model = str(getattr(model, "name", "") or "") or None
    return {
        "gen_ai.operation.name": OPERATION_CHAT,
        "gen_ai.provider.name": provider_name(model),
        "gen_ai.request.model": request_model,
    }


def usage_attrs(
    input_tokens: int,
    output_tokens: int,
    finish_reasons: str | Sequence[str] | None = None,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> dict[str, Any]:
    """``gen_ai.*`` attributes known *after* a model call completes.

    ``input_tokens`` is loomflow's full-rate bucket; per the spec
    ``gen_ai.usage.input_tokens`` counts *all* input tokens, so the
    cache reads and writes are added in (and reported on their own
    when non-zero).
    """
    attrs: dict[str, Any] = {
        "gen_ai.usage.input_tokens": int(input_tokens + cache_read_tokens + cache_write_tokens),
        "gen_ai.usage.output_tokens": int(output_tokens),
    }
    if cache_read_tokens:
        attrs["gen_ai.usage.cache_read.input_tokens"] = int(cache_read_tokens)
    if cache_write_tokens:
        attrs["gen_ai.usage.cache_write.input_tokens"] = int(cache_write_tokens)
    if isinstance(finish_reasons, str):
        finish_reasons = (finish_reasons,)
    if finish_reasons:
        attrs["gen_ai.response.finish_reasons"] = tuple(
            str(r) for r in finish_reasons
        )
    return attrs


def tool_attrs(tool_name: str, *, call_id: str | None = None) -> dict[str, Any]:
    """``gen_ai.*`` attributes for a tool-execution span."""
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": OPERATION_EXECUTE_TOOL,
        "gen_ai.tool.name": tool_name,
    }
    if call_id:
        attrs["gen_ai.tool.call.id"] = call_id
    return attrs


def otel_span_identity(name: str, attrs: Mapping[str, Any]) -> tuple[str, str]:
    """The spec ``(span name, span kind)`` for a span loomflow calls
    ``name``: ``("chat {model}", "CLIENT")`` for model calls,
    ``("execute_tool {tool}", "INTERNAL")`` for tools,
    ``("invoke_agent", "INTERNAL")`` for runs, else ``(name,
    "INTERNAL")``."""
    operation = attrs.get("gen_ai.operation.name")
    if operation == OPERATION_CHAT:
        return chat_span_name(str(attrs.get("gen_ai.request.model") or "")), "CLIENT"
    if operation == OPERATION_EXECUTE_TOOL:
        return tool_span_name(str(attrs.get("gen_ai.tool.name") or "")), "INTERNAL"
    if operation == OPERATION_INVOKE_AGENT:
        agent = attrs.get("gen_ai.agent.name")
        return (f"invoke_agent {agent}" if agent else "invoke_agent"), "INTERNAL"
    return name, "INTERNAL"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_DURATION_BUCKETS = (
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
)
_TOKEN_BUCKETS = (
    1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864,
)

#: Semconv instruments: ``name -> (kind, unit, bucket boundaries)``.
METRIC_INSTRUMENTS: dict[str, tuple[str, str, tuple[float, ...] | None]] = {
    "gen_ai.client.inference.duration": ("histogram", "s", _DURATION_BUCKETS),
    "gen_ai.client.inference.usage.input_tokens": ("counter", "{token}", None),
    "gen_ai.client.inference.usage.output_tokens": ("counter", "{token}", None),
    "gen_ai.client.inference.usage.cache_read.input_tokens": ("counter", "{token}", None),
    "gen_ai.client.inference.usage.cache_write.input_tokens": ("counter", "{token}", None),
    "gen_ai.client.inference.operation.input_tokens": ("histogram", "{token}", _TOKEN_BUCKETS),
    "gen_ai.client.inference.operation.output_tokens": ("histogram", "{token}", _TOKEN_BUCKETS),
    "gen_ai.execute_tool.duration": ("histogram", "s", _DURATION_BUCKETS),
}


def instrument_for(name: str) -> tuple[str, str, tuple[float, ...] | None]:
    """``(kind, unit, buckets)`` for a metric name: the semconv table,
    else by suffix (``_ms`` / ``_seconds`` / ``_bytes`` histograms
    with that unit; ``loom.tokens.*`` token counters; other counters)."""
    known = METRIC_INSTRUMENTS.get(name)
    if known is not None:
        return known
    if name.endswith("_ms"):
        return "histogram", "ms", None
    if name.endswith("_seconds"):
        return "histogram", "s", None
    if name.endswith("_bytes"):
        return "histogram", "By", None
    if name.startswith("loom.tokens."):
        return "counter", "{token}", None
    return "counter", "", None


def inference_metrics(
    model: Any,
    usage: Any,
    duration_s: float,
    *,
    error_type: str | None = None,
) -> list[tuple[str, float, dict[str, Any]]]:
    """The client-inference metrics for one model call, as
    ``(name, value, attributes)`` to pass to ``emit_metric``."""
    base: dict[str, Any] = {
        "gen_ai.operation.name": OPERATION_CHAT,
        "gen_ai.provider.name": provider_name(model),
        "gen_ai.request.model": str(getattr(model, "name", "") or "") or None,
    }
    timed = {**base, "error.type": error_type} if error_type else base
    tokens = {**base, "gen_ai.token.modality": "text"}
    cache_read = int(getattr(usage, "cached_input_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_write_tokens", 0) or 0)
    total_in = int(getattr(usage, "input_tokens", 0) or 0) + cache_read + cache_write
    out = int(getattr(usage, "output_tokens", 0) or 0)
    metrics: list[tuple[str, float, dict[str, Any]]] = [
        ("gen_ai.client.inference.duration", duration_s, timed),
        ("gen_ai.client.inference.usage.input_tokens", total_in, tokens),
        ("gen_ai.client.inference.usage.output_tokens", out, tokens),
        ("gen_ai.client.inference.operation.input_tokens", total_in, base),
        ("gen_ai.client.inference.operation.output_tokens", out, base),
    ]
    if cache_read:
        metrics.append(("gen_ai.client.inference.usage.cache_read.input_tokens", cache_read, tokens))
    if cache_write:
        metrics.append(
            ("gen_ai.client.inference.usage.cache_write.input_tokens", cache_write, tokens)
        )
    return metrics


def set_span_attributes(span: Any, attrs: Mapping[str, Any]) -> None:
    """Set attributes on a live span handle, post-hoc.

    Duck-typed so any Telemetry implementation works unchanged:
    ``span.set_attribute(key, value)`` is used when available,
    otherwise the span's mutable ``attributes`` dict is updated in
    place (the shipped sinks read the yielded span's attributes at
    close, so late additions land in the captured record). ``None``
    spans (fast-telemetry null context) and ``None`` values are
    ignored.
    """
    if span is None:
        return
    setter = getattr(span, "set_attribute", None)
    if callable(setter):
        for key, value in attrs.items():
            if value is not None:
                setter(key, value)
        return
    target = getattr(span, "attributes", None)
    if isinstance(target, dict):
        target.update(
            {k: v for k, v in attrs.items() if v is not None}
        )


__all__ = [
    "METRIC_INSTRUMENTS",
    "OPERATION_CHAT",
    "OPERATION_EXECUTE_TOOL",
    "OPERATION_INVOKE_AGENT",
    "PROVIDER_OTHER",
    "chat_attrs",
    "chat_span_name",
    "inference_metrics",
    "instrument_for",
    "otel_span_identity",
    "provider_name",
    "set_span_attributes",
    "tool_attrs",
    "tool_span_name",
    "usage_attrs",
]
