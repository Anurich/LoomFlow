"""Tool-result summarization — Claude Code's ``tool_use_summary`` ported.

The problem this solves
-----------------------

Long tool results (large ``read_file``, multi-page ``grep``, verbose
``bash`` output) stay in conversation history forever. Across 5-10
turns the bulk of the prompt is OLD tool results the agent has
already extracted what it needed from. They are pure cost — same
tokens shipped + cached + billed every subsequent turn.

What this module does
---------------------

After a tool returns content, the ReAct loop hands the content to
:func:`summarize_tool_result` (when the agent was constructed with
``tool_result_summarizer=<model>``). For results longer than
``threshold`` characters the helper calls the summarizer model with
a fixed prompt and returns a short summary. The ReAct loop uses the
summary as the message content — the original is never shipped on
the next turn.

For results UNDER the threshold the helper is a no-op (returns the
original). Failures (model raises, returns empty, returns garbage)
all fall back to the original — the principle is "summarization is
best-effort; never let it kill a turn."

Design choices
--------------

* **Always-replace, never keep both.** No "summary plus original"
  storage — the savings come from removing the original from the
  conversation. The agent can re-read the file or re-run the
  command if it really needs the verbatim text later.

* **Summarize in-turn, not lazily.** The summary lands in the same
  message slot the original would have, so the next model call
  sees the summary in place. No multi-turn delayed-replace dance
  (which would invalidate prompt cache prefixes on every turn).

* **Single-call summarization, no chain-of-thought.** We're paying
  for tokens to save tokens; a one-shot haiku call is the only
  shape that nets out positive.

* **Fixed prompt — no per-tool customisation.** Tools have widely
  varying output shapes (line-oriented grep vs. block-oriented
  bash vs. table-shaped diff) but a single "preserve the facts
  the agent needs later" instruction generalises adequately.
  Tool-specific summarisers would be a future axis if the metrics
  show value.

Provider scope
--------------

Any :class:`Model` works — the summarizer is invoked via the same
``model.complete()`` protocol the agent itself uses. In practice
you want a small fast model (Claude Haiku, GPT-4.1-nano) so the
summarisation cost stays well below the savings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.protocols import Model

# Default char threshold below which we don't bother summarising —
# the API round-trip would cost more than we'd save. 500 chars is
# roughly 100 tokens; anything smaller is dwarfed by the summariser
# call itself.
DEFAULT_SUMMARY_THRESHOLD = 500

# The instruction sent to the summariser model. Kept tight: tells
# the model what to preserve, what to drop, and what shape to
# return. NOT a prompt-engineering puzzle — short and direct works
# best on small models.
_SUMMARY_TEMPLATE = """\
You are condensing a tool result for a coding agent. The agent has
already SEEN this output once in its current turn. Your summary
will replace the verbatim output in the agent's conversation
history for FUTURE turns.

Preserve:
- File paths, identifiers, function/class names mentioned
- Counts, line numbers, version strings, error codes
- Concrete findings (e.g. "function X is defined in file Y line Z",
  "tests A and B failed", "config flag C is set to D")

Drop:
- Decorative output (banner lines, separator dashes)
- Repeated headers or column labels in tabular output
- Stack traces beyond the top frame (keep the top frame)
- Boilerplate ("Loading...", "Done.", progress bars)

Output ONLY the summary as plain text — no preamble, no "Here is
the summary:" introduction, no closing pleasantries. Aim for
roughly 10% of the original length.

Tool: {tool_name}
Original output (verbatim, {char_count} chars):
{content}
"""


async def summarize_tool_result(
    content: str,
    *,
    tool_name: str,
    summarizer: Model,
    threshold: int = DEFAULT_SUMMARY_THRESHOLD,
) -> str:
    """Return a short summary of ``content``, or ``content`` itself.

    Returns the original content unchanged when:

    * ``len(content) <= threshold`` — too small to be worth a round-
      trip; the summariser would cost more than it saves.
    * The summariser raises any exception — graceful degradation;
      tool-result summarisation must never break the agent's turn.
    * The summariser returns an empty string or whitespace-only
      response — fall through to original to avoid shipping a
      worthless empty message.

    On the success path returns the summariser's text output
    verbatim. The caller is responsible for wrapping it in a
    :class:`Message` and emitting any telemetry events.

    Args:
        content: the tool's output text.
        tool_name: passed into the summariser prompt so the model
            can pick conventions appropriate to the tool (e.g.
            keep grep line numbers, drop bash separator dashes).
        summarizer: any :class:`Model` instance — typically a small
            fast model like Haiku. The full ``model.complete()``
            protocol is used; ``Dependencies.model`` need not be
            passed (the summarizer is independent of the main
            agent's model).
        threshold: char count below which the helper is a no-op.

    Returns:
        Either the summary text (success path) or ``content`` itself
        (under threshold / summariser failed / summariser returned
        empty).
    """
    if len(content) <= threshold:
        return content

    from ..core.types import Message, Role

    prompt = _SUMMARY_TEMPLATE.format(
        tool_name=tool_name,
        char_count=len(content),
        content=content,
    )
    messages = [
        Message(role=Role.USER, content=prompt),
    ]
    # Aggregate via ``stream()`` (the only method the :class:`Model`
    # protocol requires — ``complete()`` is an adapter-specific
    # convenience). Low temperature for deterministic compression.
    # ``max_tokens=512`` caps the summary at ~2KB so it stays well
    # under the threshold to be re-summarisation-eligible itself.
    parts: list[str] = []
    try:
        async for chunk in summarizer.stream(
            messages,
            tools=None,
            temperature=0.0,
            max_tokens=512,
        ):
            text = getattr(chunk, "text", None)
            if text:
                parts.append(text)
    except Exception:  # noqa: BLE001 — fall back to original
        return content

    summary = "".join(parts).strip()
    if not summary:
        return content
    return summary
