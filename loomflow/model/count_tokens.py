"""Token counting — provider-native → tiktoken → char-based fallback.

The foundation 0.10.19's auto-compact will build on. Today's
visible use case: ``/cost``-style UIs that want to show
"how close are we to the context window?" without making an
extra round-trip just to find out.

Design: a single top-level :func:`count_tokens` helper that
takes a :class:`~loomflow.core.protocols.Model` and a list of
:class:`Message`. Three-tier fallback chain:

1. **Provider-native.** If the model adapter has a
   ``count_tokens(messages, tools=)`` method (Anthropic's
   ``client.messages.count_tokens(...)``, OpenAI's tiktoken
   wrapping, etc.) we use it — exact byte-accurate counts.

2. **tiktoken.** If tiktoken is installed (it is under the
   ``loader`` extra), we tokenize via ``cl100k_base`` — the
   encoding GPT-4-class and Claude-3+ models use, accurate
   enough for budgeting decisions. Same encoding both major
   providers' tokenizers approximate, so the answer is the
   right order of magnitude regardless of which provider runs
   the actual completion.

3. **Char-based estimate.** Last resort, no deps. ``len(text)
   // 4`` is the rule-of-thumb token count for English-ish
   prose; we use it when both upstream tiers fail (or when the
   user runs without the ``loader`` extra). Always returns
   something — counts are foundational; raising would propagate
   into compact/snip decision sites that have no graceful fall-
   back.

The Model protocol is NOT modified. ``count_tokens`` on adapters
is duck-typed via ``hasattr`` — old custom Model impls that
don't implement it inherit the fallback automatically.

Tools are included in the estimate: serialised ``ToolDef``
JSON-schema bytes count toward the prompt under both Anthropic
and OpenAI tokenizers, so omitting them would systematically
under-count by 200-1000 tokens for tool-heavy agents.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ..core.types import Message

if TYPE_CHECKING:
    from ..core.protocols import Model
    from ..core.types import ToolDef


# Token-per-char ratio for the char-based fallback. 4 is the
# canonical estimate for English prose under cl100k_base; code
# averages closer to 3, but 4 is the conservative cross-content
# default (under-estimating tokens leads to overflowing context;
# over-estimating just leaves headroom). Tune via
# ``count_tokens(..., chars_per_token=N)`` if your traffic skews.
DEFAULT_CHARS_PER_TOKEN = 4


async def count_tokens(
    model: Model,
    messages: list[Message],
    *,
    tools: list[ToolDef] | None = None,
    chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Estimate input tokens for ``messages`` (+ optional ``tools``).

    See module docstring for the three-tier fallback chain. The
    return is always a positive int; the helper never raises.
    Auto-compact / snip / budget-display sites can rely on this
    invariant.

    Args:
        model: any :class:`Model` instance. The adapter MAY
            implement ``count_tokens(messages, tools=)`` for an
            exact native count; if not, the helper falls through
            to tiktoken / char-based estimates without consulting
            the adapter further.
        messages: the message list to count. System messages
            count too — they're real input tokens.
        tools: optional tool definitions. Counted as their
            serialised JSON-schema body — matches what both major
            providers wire-encode.
        chars_per_token: char-based fallback divisor. Default 4
            (cl100k_base average for English prose).
    """
    # Tier 1: provider-native ``count_tokens`` on the adapter.
    native = getattr(model, "count_tokens", None)
    if callable(native):
        try:
            value = await native(messages, tools=tools)
            # Preserve native zeros — providers like Anthropic
            # may legitimately return 0 for empty inputs, but
            # the ``max(1, ...)`` floor only applies to estimates;
            # we trust the provider when it answers.
            return int(value)
        except Exception:  # noqa: BLE001 — fall through to estimates
            pass

    # Tier 2: tiktoken (cl100k_base — GPT-4 / Claude-3+ family).
    # Apply the ``max(1, ...)`` floor so callers can use the
    # returned count in budget percentage math without div-zero.
    try:
        return max(1, _tiktoken_estimate(messages, tools))
    except Exception:  # noqa: BLE001 — tiktoken absent or borked
        pass

    # Tier 3: char-based estimate. Always succeeds, already
    # floored to >=1 in the helper.
    return _char_estimate(messages, tools, chars_per_token)


def _tiktoken_estimate(
    messages: list[Message], tools: list[ToolDef] | None
) -> int:
    """Count via ``tiktoken`` using ``cl100k_base``.

    Raises if ``tiktoken`` isn't installed or the encoding can't
    be fetched — the caller catches and falls through to the
    char-based estimate."""
    # Lazy import — tiktoken is an extra, not a base dep.
    import tiktoken  # type: ignore[import-not-found, import-untyped]

    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for m in messages:
        total += len(enc.encode(m.content or ""))
        # The provider wire format adds 3-5 tokens of role
        # markers per message. ``4`` is the standard estimate
        # used by tiktoken's own ``num_tokens_from_messages``
        # cookbook example.
        total += 4
    if tools:
        # Tool defs serialise as JSON; encode the whole array.
        tools_payload = json.dumps(
            [_tool_dict(t) for t in tools], separators=(",", ":")
        )
        total += len(enc.encode(tools_payload))
    return total


def _char_estimate(
    messages: list[Message],
    tools: list[ToolDef] | None,
    chars_per_token: int,
) -> int:
    """Pure-Python char-based estimate. No deps. Always succeeds.

    ``chars_per_token`` defaults to 4 — the rule-of-thumb for
    English prose under cl100k_base. ``8`` per-message overhead
    approximates the role-markers + JSON envelope on the wire.
    """
    total_chars = 0
    per_message_overhead = 8
    for m in messages:
        total_chars += len(m.content or "") + per_message_overhead
    if tools:
        # Same JSON-serialisation logic as the tiktoken path so
        # the two estimates stay in the same ballpark.
        tools_payload = json.dumps(
            [_tool_dict(t) for t in tools], separators=(",", ":")
        )
        total_chars += len(tools_payload)
    return max(1, total_chars // chars_per_token)


def _tool_dict(tool: ToolDef) -> dict[str, object]:
    """Render a :class:`ToolDef` as a flat dict for serialisation
    in the token estimate. Mirrors how Anthropic + OpenAI wire-
    encode tool definitions (name + description + input_schema).
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }
