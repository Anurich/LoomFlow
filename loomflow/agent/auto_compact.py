"""Auto-compact — summarise old turns when conversation grows large.

The third tier of context-budget defence, paired with snip (0.10.16,
cheap pure-slicing) and tool_result_summarizer (0.10.14, per-result
compression). Auto-compact is the heavy-hitter: when ``session.messages``
crosses a token threshold, it fires an LLM call to summarise the older
half of the conversation into a single ``system`` message, keeping the
last N turn groups verbatim.

Where it fires
--------------

Before every model call of a ReAct turn (``Dependencies.compact_context``)
— so a run whose own tool results pile up is compacted while it is
running, and a session whose rehydrated history has grown past the
threshold is compacted before its first call — and after an
architecture pass when a checkpoint is written or a stop-hook pass
follows. Tokens are counted with the token-counting helper (only once
the transcript is long enough that it could be over the threshold, so
short conversations cost nothing). Past ``auto_compact_at_tokens`` the
older messages are replaced with a single summary system message; the
most recent N user-anchored turn groups stay verbatim.
:class:`CachingSummariser` remembers summaries by the transcript they
cover, so the same rehydrated history isn't summarised again on every
run.

What survives the compact
-------------------------

* The leading ``Role.SYSTEM`` head — always (identity / instructions).
* A NEW ``Role.SYSTEM`` message containing the summary of dropped
  content — prefixed with ``[auto-compacted summary]`` so the model
  knows what it's reading.
* The last N user-anchored turn groups — verbatim, so the model has
  recent concrete context (the most recent tool result is usually
  load-bearing for the next decision).

Where the summary lives
-----------------------

In-conversation as a new system message — NOT written to long-term
memory. Auto-compact is per-run; the framework doesn't assume a
``session_summary`` working-block convention exists (loom-code's
``compact.py`` writes to ``session_summary`` via memory.update_block
for its own purposes, but that's an app-level choice, not a
framework one).

Failure semantics
-----------------

If the summariser raises or returns an empty string, auto-compact is
a no-op for this iteration — the conversation continues uncompacted.
Same principle as :mod:`loomflow.tools.result_summarizer`: framework-
level token optimisations must NEVER kill a turn.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from ..core.types import Message, ModelChunk, Role

if TYPE_CHECKING:
    from ..core.protocols import Model


# Default token windows for known model families. Conservative — the
# real windows are larger, but leaving headroom for the current
# turn's prompt + response + tool I/O is the point. Substring match
# against the model name; first hit wins. Anything unmatched falls
# back to ``_DEFAULT_CONTEXT_WINDOW``.
#
# Order matters: the lookup returns the FIRST substring hit, so the
# more-specific 1M keys must precede the 200k catch-alls (e.g.
# "claude-sonnet-4-6" before "claude-sonnet", "gpt-5.4" before
# "gpt-5").
_KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    # Claude (platform.claude.com context-windows): every model from
    # Opus / Sonnet 4.6 on, Haiku 5.5, Fable and Mythos has 1M by
    # default; Opus 4.5 and earlier, Sonnet 4.5 / 4 and Haiku 4.5 200k.
    "claude-fable": 1_000_000,
    "claude-mythos": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-5": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-sonnet": 200_000,
    "claude-haiku-5": 1_000_000,
    "claude-haiku": 200_000,
    "claude-3-5": 200_000,
    # OpenAI (developers.openai.com model pages): GPT-6.x, GPT-5.6,
    # 5.5 and 5.4 = 1.05M; GPT-5.3 / 5.x / codex = 400k.
    "gpt-6": 1_050_000,
    "gpt-5.6": 1_050_000,
    "gpt-5.5": 1_050_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.3": 400_000,
    "gpt-5": 400_000,
    "gpt-4.1": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_385,
    "o3": 200_000,
    "o4": 200_000,
    "o1": 200_000,
    # DeepSeek V4 (api-docs.deepseek.com) = 1M; V3.x chat/reasoner 128k.
    "deepseek-flash": 1_000_000,
    "deepseek-v4": 1_000_000,
    "deepseek": 128_000,
    # Gemini
    "gemini-1.5": 1_000_000,
    "gemini-2": 1_000_000,
}

# Fallback for any model not in the table. 8k is the lowest-common-
# denominator across local Ollama / small open models — the user can
# always override via ``auto_compact_at_tokens=`` explicitly.
_DEFAULT_CONTEXT_WINDOW = 8_192

# Model names we've already warned about falling back to the default
# window — once per process per name, so a hot loop calling
# ``context_window_for`` doesn't spam stderr.
_warned_unknown_models: set[str] = set()

# Default trigger fraction of the window. 0.8 leaves headroom for
# the current turn's prompt + tool I/O + response without bumping
# the actual limit.
DEFAULT_AUTO_COMPACT_PCT = 0.8


def context_window_for(model_name: str) -> int:
    """Best-effort context-window lookup by model-name substring.

    Returns the largest matching known window; falls back to
    :data:`_DEFAULT_CONTEXT_WINDOW` (8192) for unrecognised models,
    warning once per model name so the squeeze is never silent.
    Substring match is intentional — model names drift (suffixes
    like ``-20250115`` or ``-preview``) and we want the lookup to
    survive that drift.
    """
    lowered = model_name.lower()
    for hint, window in _KNOWN_CONTEXT_WINDOWS.items():
        if hint in lowered:
            return window
    if model_name not in _warned_unknown_models:
        _warned_unknown_models.add(model_name)
        warnings.warn(
            f"context_window_for: unknown model {model_name!r} — "
            f"falling back to the conservative default of "
            f"{_DEFAULT_CONTEXT_WINDOW} tokens. If the real window is "
            f"larger, compaction thresholds derived from this value "
            f"will fire far too early; pass an explicit "
            f"auto_compact_at_tokens= instead.",
            stacklevel=2,
        )
    return _DEFAULT_CONTEXT_WINDOW


# The summariser instruction. Tight: preserves facts the model
# needs to continue working, drops mechanics + repetition.
_SUMMARY_TEMPLATE = """\
You are condensing the opening portion of a coding-agent
conversation. Your summary REPLACES the dropped turns in the
agent's working context — the agent will continue from your
summary plus its most recent turns.

Preserve:
- Concrete decisions made (files written, commits, choices)
- Facts established (function/class names, file paths, error
  messages, version constraints)
- Open questions or blockers the agent was working on
- User preferences or constraints stated

Drop:
- Per-turn mechanical detail (which tool was called when)
- Already-resolved sub-tasks (don't list "found file X, then
  read it"; just list what was learned from file X)
- Repeated information
- Greetings / acknowledgements

Output ONLY the summary as plain text — no preamble, no
markdown headers. Aim for roughly 20% of the original length.

Conversation to summarise:
{transcript}
"""


def _render_transcript(messages: list[Message]) -> str:
    """Render a list of messages as a readable transcript for the
    summariser. Role markers + content only; we strip tool-call
    JSON because the model needs facts, not mechanics."""
    lines: list[str] = []
    for m in messages:
        role = m.role.value.upper()
        body = (m.content or "").strip()
        if not body:
            continue
        lines.append(f"[{role}] {body}")
    return "\n\n".join(lines)


def _split_at_user_anchor(
    messages: list[Message], keep_last_n_turns: int
) -> tuple[list[Message], list[Message]]:
    """Return ``(to_summarise, to_keep)`` split at a user-message
    boundary.

    Mirror of :func:`loomflow.agent.snip.snip_messages` logic — we
    cut at user-message indices so the kept tail starts with a
    user message (no orphan tool_result before its tool_call).
    Leading system head is left out of both halves so the caller
    can re-attach it.
    """
    user_indices = [
        i for i, m in enumerate(messages) if m.role == Role.USER
    ]
    if len(user_indices) <= keep_last_n_turns:
        # Not enough turns to compact yet.
        return [], list(messages)
    cut_at = user_indices[-keep_last_n_turns]
    return list(messages[:cut_at]), list(messages[cut_at:])


def _split_within_turn(
    messages: list[Message], keep_last_n_exchanges: int
) -> tuple[list[Message], list[Message], list[Message]]:
    """``(lead, to_summarise, to_keep)`` for a body whose last turn is
    one long tool loop (too few user messages to cut at): ``lead`` is
    everything up to and including the turn's user message, ``to_keep``
    the last ``keep_last_n_exchanges`` assistant exchanges (an
    assistant message and the tool results after it — never split),
    and ``to_summarise`` the exchanges in between."""
    users = [i for i, m in enumerate(messages) if m.role == Role.USER]
    start = users[-1] + 1 if users else 0
    exchanges = [i for i in range(start, len(messages)) if messages[i].role == Role.ASSISTANT]
    if len(exchanges) <= keep_last_n_exchanges:
        return list(messages), [], []
    cut = exchanges[-keep_last_n_exchanges]
    return list(messages[:start]), list(messages[start:cut]), list(messages[cut:])


async def maybe_auto_compact(
    messages: list[Message],
    *,
    summariser: Model,
    at_tokens: int,
    current_token_count: int,
    keep_recent_turns: int = 4,
) -> tuple[list[Message] | None, str]:
    """If ``current_token_count`` exceeds ``at_tokens``, compact
    ``messages`` and return the new list + summary text.

    Returns ``(new_messages, summary)`` on a successful compact;
    returns ``(None, "")`` when no compact happened (below
    threshold, no anchor to slice at, summariser failed, or
    summary came back empty).

    The new message list has shape:

    * Leading ``Role.SYSTEM`` head from the input (preserved).
    * One new ``Role.SYSTEM`` message: ``"[auto-compacted
      summary] " + summary``.
    * The last ``keep_recent_turns`` user-anchored turn groups
      from the input, verbatim — or, when the context grew inside
      one turn's tool loop (too few user messages to cut at), that
      turn's user message, then the summary of its older exchanges,
      then its last ``keep_recent_turns`` exchanges.
    """
    if current_token_count <= at_tokens:
        return None, ""
    if not messages:
        return None, ""

    # Separate leading system head from the rest. The head is
    # part of the agent's stable identity — never summarised,
    # never dropped.
    head: list[Message] = []
    body_start = 0
    for i, m in enumerate(messages):
        if m.role == Role.SYSTEM:
            head.append(m)
            body_start = i + 1
        else:
            break
    body = messages[body_start:]

    older, recent = _split_at_user_anchor(body, keep_recent_turns)
    lead: list[Message] = []
    if not older:
        # Too few turns to cut at: a run whose context grew inside one
        # long tool loop. Summarise that loop's older exchanges.
        lead, older, recent = _split_within_turn(body, keep_recent_turns)
        if not older:
            return None, ""

    transcript = _render_transcript(older)
    if not transcript:
        return None, ""

    prompt = _SUMMARY_TEMPLATE.format(transcript=transcript)
    summariser_msgs = [Message(role=Role.USER, content=prompt)]

    # Use the same stream-and-aggregate shape as
    # :mod:`loomflow.tools.result_summarizer` — the Model protocol
    # only mandates ``stream``; we don't depend on ``complete``
    # being implemented.
    parts: list[str] = []
    try:
        # No temperature: current Claude models reject non-default
        # values with a 400, which silently disabled compaction.
        async for chunk in summariser.stream(
            summariser_msgs,
            tools=None,
            max_tokens=2048,
        ):
            text = getattr(chunk, "text", None)
            if text:
                parts.append(text)
    except Exception as exc:  # noqa: BLE001 — never kill a turn
        # Graceful no-op, but NOT silent: a summariser that fails on
        # every iteration effectively disables compaction, and the
        # conversation will keep growing until it blows the context
        # window. Warn so the squeeze is observable.
        warnings.warn(
            f"auto-compact summariser "
            f"{getattr(summariser, 'name', type(summariser).__name__)!r} "
            f"failed ({type(exc).__name__}: {exc}); continuing with the "
            f"conversation uncompacted.",
            stacklevel=2,
        )
        return None, ""

    summary = "".join(parts).strip()
    # Strip any obvious "Here is the summary:" preamble the
    # summariser might add despite the instruction.
    summary = re.sub(
        r"^(here(?:'s| is)? (?:the )?(?:summary|condensed version):?\s*)",
        "",
        summary,
        flags=re.IGNORECASE,
    )
    if not summary:
        return None, ""

    summary_msg = Message(
        role=Role.SYSTEM,
        content=f"[auto-compacted summary of {len(older)} dropped "
        f"messages]\n{summary}",
    )
    new_messages = head + lead + [summary_msg] + recent
    return new_messages, summary


class CachingSummariser:
    """A summariser model that remembers its last ``max_entries``
    summaries by the exact messages it was given."""

    def __init__(self, inner: Model, *, max_entries: int = 64) -> None:
        self._inner = inner
        self._max = max_entries
        self._cache: OrderedDict[str, str] = OrderedDict()

    @property
    def name(self) -> str:
        return str(getattr(self._inner, "name", "summariser"))

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._inner, attr)

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelChunk]:
        key = hashlib.sha256(
            "\x00".join(f"{m.role}\x01{m.content}" for m in messages).encode("utf-8", "replace")
        ).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            yield ModelChunk(kind="text", text=cached)
            yield ModelChunk(kind="finish", finish_reason="stop")
            return
        parts: list[str] = []
        async for chunk in self._inner.stream(messages, **kwargs):
            if chunk.kind == "text" and chunk.text:
                parts.append(chunk.text)
            yield chunk
        text = "".join(parts)
        if text.strip():
            self._cache[key] = text
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
