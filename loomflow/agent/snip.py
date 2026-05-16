"""Snip — drop older conversation turns above a sliding window.

The pattern: keep the last N user-anchored turn groups in
``session.messages``, drop the rest. Runs BEFORE each
architecture invocation in :meth:`Agent._loop`, so the model
sees a bounded conversation regardless of how long the REPL
session has been open.

Versus :mod:`loomflow.tools.result_summarizer` (0.10.14):

* **Summariser** compresses individual tool results in place —
  same number of messages, smaller content per message.
  Requires an LLM call per oversized result.
* **Snip** drops whole turn groups (user + assistant + tool
  results) without ever calling an LLM. Cheaper. Loses more
  information.

Versus the future auto-compact (0.10.19):

* **Snip** is the always-on cheap defence — pure list slicing,
  no API call.
* **Auto-compact** is the heavy-hitter that fires at, say, 80%
  of context window and summarises everything outside the
  protected tail.

In practice you want **both**: snip keeps day-to-day history
bounded; auto-compact catches the edge cases where snip alone
isn't enough (very long single turns, dense tool output) before
the model rejects the request.

Why "user-anchored turn groups" instead of just "last N messages"
---------------------------------------------------------------

Cutting in the middle would orphan tool-result messages — the
model API requires every ``tool_result`` to follow the
``tool_call`` that triggered it. Same problem in reverse for
the architecture: if we drop an assistant message that emitted
tool_calls, we'd leave dangling tool_results in the next user
turn that reference nothing. Snipping at user-message
boundaries preserves the invariant that each kept fragment is
self-contained.

Leading ``Role.SYSTEM`` messages (architectures occasionally
inject one or two as preamble — though most rebuild system
content per turn) are preserved unconditionally; they're the
identity / instructions, not conversation history.
"""

from __future__ import annotations

from ..core.types import Message, Role


def snip_messages(
    messages: list[Message], keep_last_n_turns: int
) -> tuple[list[Message], int]:
    """Trim ``messages`` to the last ``keep_last_n_turns`` user-
    anchored turn groups, preserving leading system messages.

    A "turn group" is one ``Role.USER`` message plus everything
    after it until the next ``Role.USER`` (or end-of-list). So
    ``[user1, assistant1+tool_call, tool_result1, user2,
    assistant2]`` is two turn groups; keep_last_n_turns=1 returns
    ``[user2, assistant2]`` (plus any leading system head).

    Returns ``(snipped_messages, num_dropped)``. ``num_dropped``
    is the count of messages removed (zero when no snip happened
    — fewer turns existed than the window, or the window is 0
    which disables the helper entirely).

    Args:
        messages: the conversation history to snip. Usually
            ``session.messages`` from an :class:`AgentSession`.
        keep_last_n_turns: how many user-anchored turn groups to
            keep. ``0`` or negative disables snipping (returns
            ``messages`` unchanged with ``num_dropped == 0``).

    Edge cases handled:

    * ``messages`` is empty / shorter than the window → no-op.
    * No user messages exist (system-only or assistant-only
      history) → no-op; the helper has no anchor to slice at.
    * Leading consecutive ``Role.SYSTEM`` messages are stripped
      off the head and re-attached after slicing — they survive
      every snip.
    """
    if keep_last_n_turns <= 0:
        return messages, 0
    if not messages:
        return messages, 0

    # Separate leading system head from the rest. The head
    # survives every snip; only the body gets sliced.
    head: list[Message] = []
    body_start = 0
    for i, m in enumerate(messages):
        if m.role == Role.SYSTEM:
            head.append(m)
            body_start = i + 1
        else:
            break
    body = messages[body_start:]

    # Find indices of user messages in the body. These are the
    # anchor points we can safely slice at.
    user_indices = [
        i for i, m in enumerate(body) if m.role == Role.USER
    ]
    if len(user_indices) <= keep_last_n_turns:
        # Already at or under the window — nothing to drop.
        return messages, 0

    # Slice starting from the Nth-from-end user message.
    cut_at = user_indices[-keep_last_n_turns]
    kept_body = body[cut_at:]
    dropped = len(body) - len(kept_body)
    return head + kept_body, dropped
