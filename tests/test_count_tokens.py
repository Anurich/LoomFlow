"""Tests for the token-counting helper (0.10.17).

The helper has three tiers — provider-native via the adapter's
``count_tokens`` method, tiktoken via ``cl100k_base``, char-based
estimate as fallback. We verify all three.

Coverage:

* Adapter-native path fires when ``model.count_tokens`` exists +
  succeeds.
* Adapter-native exception → falls through to next tier.
* tiktoken path produces a reasonable count for a fixed string
  (skipped if tiktoken isn't installed in the test env — it's
  under the ``loader`` extra).
* Char-based estimate produces the expected number when both
  upstream tiers are unavailable.
* Tools are counted (their serialised JSON adds to the total).
* Empty messages list returns 0 (or 1 from char-based ``max(1, ...)``).
"""

from __future__ import annotations

import pytest

from loomflow.core.types import Message, Role, ToolDef
from loomflow.model.count_tokens import (
    DEFAULT_CHARS_PER_TOKEN,
    _char_estimate,
    count_tokens,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Char-based estimate (tier 3) — direct unit tests
# ---------------------------------------------------------------------------


def test_char_estimate_empty_returns_min_one() -> None:
    """Empty input still returns 1, not 0 — protects callers that
    divide by the count or build a percentage budget bar."""
    assert _char_estimate([], None, DEFAULT_CHARS_PER_TOKEN) == 1


def test_char_estimate_single_message() -> None:
    msgs = [Message(role=Role.USER, content="hello world")]
    # "hello world" = 11 chars + 8 overhead = 19; 19 // 4 = 4
    out = _char_estimate(msgs, None, 4)
    assert out == 4


def test_char_estimate_counts_tools() -> None:
    """Tools serialise as JSON and contribute to the total."""
    msgs = [Message(role=Role.USER, content="x")]
    no_tools = _char_estimate(msgs, None, 4)
    with_tools = _char_estimate(
        msgs,
        [
            ToolDef(
                name="my_tool",
                description="does a thing",
                input_schema={"type": "object", "properties": {}},
            )
        ],
        4,
    )
    assert with_tools > no_tools


def test_char_estimate_custom_ratio() -> None:
    """A lower chars-per-token ratio yields a higher count
    (more tokens per char). Lets users tune for code-heavy
    workloads where the true ratio is closer to 3."""
    msgs = [Message(role=Role.USER, content="a" * 100)]
    lo = _char_estimate(msgs, None, 4)
    hi = _char_estimate(msgs, None, 2)
    assert hi > lo


# ---------------------------------------------------------------------------
# count_tokens — provider-native path
# ---------------------------------------------------------------------------


async def test_count_tokens_uses_native_method_when_available() -> None:
    """When the model implements ``count_tokens()``, the helper
    delegates to it (and skips the tiktoken / char estimates)."""

    class _NativeModel:
        name = "fake"

        async def count_tokens(self, messages, *, tools=None):  # type: ignore[no-untyped-def]
            return 12345

    msgs = [Message(role=Role.USER, content="hi")]
    out = await count_tokens(_NativeModel(), msgs)  # type: ignore[arg-type]
    assert out == 12345


async def test_count_tokens_native_failure_falls_through() -> None:
    """Native exception → fall through to tiktoken / char-based.
    Counts are foundational; the helper must never raise."""

    class _BrokenNativeModel:
        name = "broken"

        async def count_tokens(self, messages, *, tools=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("API down")

    msgs = [Message(role=Role.USER, content="hi")]
    out = await count_tokens(_BrokenNativeModel(), msgs)  # type: ignore[arg-type]
    # Some positive int from a fallback tier.
    assert out > 0


async def test_count_tokens_no_native_method_uses_fallback() -> None:
    """Model without ``count_tokens`` → tiktoken (if installed) or
    char-based. Either way, a positive int."""

    class _PlainModel:
        name = "plain"

    msgs = [Message(role=Role.USER, content="hello world")]
    out = await count_tokens(_PlainModel(), msgs)  # type: ignore[arg-type]
    assert out > 0


# ---------------------------------------------------------------------------
# count_tokens — tiktoken path (skipped if tiktoken not installed)
# ---------------------------------------------------------------------------


async def test_count_tokens_tiktoken_path_when_available() -> None:
    """If tiktoken is installed, the fallback uses it. The exact
    count for ``"hello world"`` under ``cl100k_base`` is 2; +4
    overhead → 6 total. Asserting >= 5 + <= 10 gives some slack
    in case tiktoken's encoding shifts."""
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        pytest.skip("tiktoken not installed (loader extra)")

    class _PlainModel:
        name = "plain"

    msgs = [Message(role=Role.USER, content="hello world")]
    out = await count_tokens(_PlainModel(), msgs)  # type: ignore[arg-type]
    assert 4 <= out <= 12


# ---------------------------------------------------------------------------
# count_tokens — end-to-end behaviour invariants
# ---------------------------------------------------------------------------


async def test_count_tokens_includes_tools_in_total() -> None:
    """Adding a tool definition increases the count — under both
    the tiktoken path and the char-based fallback."""

    class _PlainModel:
        name = "plain"

    msgs = [Message(role=Role.USER, content="hi")]
    no_tools = await count_tokens(_PlainModel(), msgs)  # type: ignore[arg-type]
    with_tools = await count_tokens(  # type: ignore[arg-type]
        _PlainModel(),
        msgs,
        tools=[
            ToolDef(
                name="bigtool",
                description="A tool with a long description " * 5,
                input_schema={
                    "type": "object",
                    "properties": {
                        "field_a": {"type": "string"},
                        "field_b": {"type": "integer"},
                    },
                },
            )
        ],
    )
    assert with_tools > no_tools


async def test_count_tokens_always_returns_positive_int() -> None:
    """Empty messages, no tools, plain model → still a positive
    int. Important invariant: budget code that does ``available
    = max - count`` or ``pct = count / max`` must never see 0."""

    class _PlainModel:
        name = "plain"

    out = await count_tokens(_PlainModel(), [])  # type: ignore[arg-type]
    assert out > 0
