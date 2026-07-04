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

The framework's legacy ``loom.*`` span names and attributes are kept
untouched; the ``gen_ai.*`` attributes ride on the *same* spans
(purely additive — zero back-compat break). The spec's preferred span
names (``chat {model}`` / ``execute_tool {tool}``) are provided by
:func:`chat_span_name` / :func:`tool_span_name` for sinks or a future
naming switch, but existing tests pin the ``loom.*`` names so the
semconv identity lives in attributes today.

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
) -> dict[str, Any]:
    """``gen_ai.*`` attributes known *after* a model call completes."""
    attrs: dict[str, Any] = {
        "gen_ai.usage.input_tokens": int(input_tokens),
        "gen_ai.usage.output_tokens": int(output_tokens),
    }
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
    "OPERATION_CHAT",
    "OPERATION_EXECUTE_TOOL",
    "PROVIDER_OTHER",
    "chat_attrs",
    "chat_span_name",
    "provider_name",
    "set_span_attributes",
    "tool_attrs",
    "tool_span_name",
    "usage_attrs",
]
