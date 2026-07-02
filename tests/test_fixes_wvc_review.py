"""Regression tests for the wvc review-fix batch.

Covers:

1. Anthropic thinking-replay cache is LRU: entries replayed by a
   live conversation survive far past the cache bound while cold
   entries from finished conversations are evicted.
2. Thinking-block replay is gated on thinking being enabled for the
   CURRENT request — ``count_tokens`` / ``complete`` strip thinking
   blocks when the request runs without thinking, and inject them
   when it runs with thinking, so counts match real requests.
3. ``MultiTelemetry`` with two capture sinks records IDENTICAL
   trace/span/parent ids in both sinks — the module contextvars are
   managed once by the composer, not stomped per-sink.
4. ``FilesystemSandbox`` recurses into list / dict arguments, so
   ``{"paths": ["/etc/passwd"]}`` and nested dicts are denied.
5. ``FileAuditLog`` no longer scans the existing file in
   ``__init__`` — seq recovery is deferred to the first append.
6. Seatbelt profile generation rejects roots containing quotes,
   backslashes, or control characters instead of interpolating
   them into the profile.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as _NS
from typing import Any

import pytest

from loomflow.core.types import Message, Role, ToolCall
from loomflow.model.anthropic import (
    _THINKING_CACHE_MAX,
    AnthropicModel,
    _to_anthropic_messages,
)
from loomflow.observability.tracing import (
    InMemoryTelemetry,
    MultiTelemetry,
)
from loomflow.security import FilesystemSandbox
from loomflow.security.audit import FileAuditLog
from loomflow.security.sandbox.os_sandbox import (
    _seatbelt_profile,
    _validate_seatbelt_root,
)
from loomflow.tools import InProcessToolHost, tool

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# 1. Thinking-replay cache — LRU keeps hot entries resident
# ---------------------------------------------------------------------------


def _blocks(tag: str) -> list[dict[str, Any]]:
    return [{"type": "thinking", "thinking": tag, "signature": "sig"}]


def test_thinking_cache_lru_keeps_hot_replayed_entry() -> None:
    """A conversation's turn-1 thinking block is replayed on EVERY
    subsequent request (it stays in the message history). Simulate a
    long-lived conversation whose requests keep touching the hot
    entry while cold entries from other turns/conversations churn
    past the cache bound: the hot entry must survive."""
    model = AnthropicModel("claude-sonnet-4-5", client=object())
    hot_call = ToolCall(id="tu_hot", tool="t", args={})
    model._remember_thinking(_blocks("hot"), [hot_call])  # noqa: SLF001

    history = [
        Message(role=Role.USER, content="q"),
        Message(role=Role.ASSISTANT, content="", tool_calls=(hot_call,)),
        Message(role=Role.TOOL, content="ok", tool_call_id="tu_hot"),
    ]

    churn = _THINKING_CACHE_MAX + 300
    for i in range(churn):
        # Each request rebuilds the whole history — replaying (and
        # thereby LRU-touching) the hot entry...
        _to_anthropic_messages(
            history, thinking_map=model._thinking  # noqa: SLF001
        )
        # ...then the response caches a fresh (cold) entry.
        model._remember_thinking(  # noqa: SLF001
            _blocks(f"cold{i}"),
            [ToolCall(id=f"tu_cold{i}", tool="t", args={})],
        )

    thinking = model._thinking  # noqa: SLF001
    assert "tu_hot" in thinking, (
        "hot (still-replayed) entry was evicted — eviction is not LRU"
    )
    assert thinking["tu_hot"] == _blocks("hot")
    assert len(thinking) <= _THINKING_CACHE_MAX
    # The oldest cold entries (never touched again) were evicted.
    assert "tu_cold0" not in thinking


def test_thinking_cache_reinsert_moves_entry_to_end() -> None:
    model = AnthropicModel("claude-sonnet-4-5", client=object())
    model._remember_thinking(  # noqa: SLF001
        _blocks("a"), [ToolCall(id="a", tool="t", args={})]
    )
    model._remember_thinking(  # noqa: SLF001
        _blocks("b"), [ToolCall(id="b", tool="t", args={})]
    )
    model._remember_thinking(  # noqa: SLF001
        _blocks("a2"), [ToolCall(id="a", tool="t", args={})]
    )
    assert list(model._thinking) == ["b", "a"]  # noqa: SLF001
    assert model._thinking["a"] == _blocks("a2")  # noqa: SLF001


# ---------------------------------------------------------------------------
# 2. Thinking replay gated on the request's thinking config
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.create_calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self._responses.pop(0)

    async def count_tokens(self, **kwargs: Any) -> Any:
        self.count_calls.append(kwargs)
        return _NS(input_tokens=42)


class _FakeClient:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.messages = _FakeMessages(responses or [])


def _tool_use_history(model: AnthropicModel) -> list[Message]:
    """Prime the model's replay cache and return a matching history."""
    call = ToolCall(id="tu_1", tool="get", args={})
    model._remember_thinking(_blocks("cached"), [call])  # noqa: SLF001
    return [
        Message(role=Role.USER, content="q"),
        Message(role=Role.ASSISTANT, content="", tool_calls=(call,)),
        Message(role=Role.TOOL, content="ok", tool_call_id="tu_1"),
    ]


def _has_thinking_block(anth_messages: list[dict[str, Any]]) -> bool:
    for m in anth_messages:
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "thinking"
            for b in content
        ):
            return True
    return False


async def test_count_tokens_strips_thinking_when_thinking_off() -> None:
    client = _FakeClient()
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    history = _tool_use_history(model)

    n = await model.count_tokens(history)
    assert n == 42
    sent = client.messages.count_calls[0]["messages"]
    assert not _has_thinking_block(sent)


async def test_count_tokens_replays_thinking_when_thinking_on() -> None:
    client = _FakeClient()
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    history = _tool_use_history(model)

    await model.count_tokens(history, effort="high")
    sent = client.messages.count_calls[0]["messages"]
    assert _has_thinking_block(sent)
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert assistant["content"][0]["type"] == "thinking"


async def test_complete_strips_thinking_replay_when_thinking_off() -> None:
    """A request made WITHOUT thinking must not contain thinking
    blocks (the API rejects them when thinking is disabled) — and
    it must match what count_tokens counted for the same shape."""
    client = _FakeClient(
        [
            _NS(
                content=[_NS(type="text", text="done")],
                usage=_NS(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )
        ]
    )
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    history = _tool_use_history(model)

    text, _calls, _usage, _stop = await model.complete(history)
    assert text == "done"
    sent = client.messages.create_calls[0]["messages"]
    assert not _has_thinking_block(sent)
    assert "thinking" not in client.messages.create_calls[0]

    # Parity: count_tokens with the same (no-effort) config builds
    # the identical message shape.
    await model.count_tokens(history)
    assert client.messages.count_calls[0]["messages"] == sent


async def test_complete_replays_thinking_when_thinking_on() -> None:
    client = _FakeClient(
        [
            _NS(
                content=[_NS(type="text", text="done")],
                usage=_NS(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )
        ]
    )
    model = AnthropicModel("claude-sonnet-4-5", client=client)
    history = _tool_use_history(model)

    await model.complete(history, effort="high")
    sent = client.messages.create_calls[0]["messages"]
    assert _has_thinking_block(sent)


# ---------------------------------------------------------------------------
# 3. MultiTelemetry — consistent parent linkage across capture sinks
# ---------------------------------------------------------------------------


async def test_multi_telemetry_two_inmemory_sinks_agree_on_ids() -> None:
    sink_a = InMemoryTelemetry()
    sink_b = InMemoryTelemetry()
    tel = MultiTelemetry([sink_a, sink_b])

    async with tel.trace("outer") as outer_span:
        async with tel.trace("inner"):
            pass

    for sink in (sink_a, sink_b):
        spans = {s.name: s for s in sink.spans()}
        assert set(spans) == {"outer", "inner"}
        assert spans["outer"].parent_span_id is None
        # The nested span's parent must be the OUTER span — in both
        # sinks (previously the second sink's contextvar set stomped
        # the first's, corrupting linkage).
        assert spans["inner"].parent_span_id == spans["outer"].span_id
        assert spans["inner"].trace_id == spans["outer"].trace_id

    ids_a = {s.name: (s.trace_id, s.span_id) for s in sink_a.spans()}
    ids_b = {s.name: (s.trace_id, s.span_id) for s in sink_b.spans()}
    assert ids_a == ids_b, "sinks recorded different span identities"
    # The yielded Span carries the shared identity.
    assert outer_span.span_id == ids_a["outer"][1]
    assert outer_span.trace_id == ids_a["outer"][0]
    # The internal seam never leaks into recorded attributes.
    for sink in (sink_a, sink_b):
        for s in sink.spans():
            assert "_loomflow_span_ctx" not in s.attributes


async def test_multi_telemetry_deep_nesting_consistent_chain() -> None:
    sink_a = InMemoryTelemetry()
    sink_b = InMemoryTelemetry()
    tel = MultiTelemetry([sink_a, sink_b])

    async with tel.trace("run"):
        async with tel.trace("turn"):
            async with tel.trace("tool"):
                pass

    for sink in (sink_a, sink_b):
        spans = {s.name: s for s in sink.spans()}
        assert spans["turn"].parent_span_id == spans["run"].span_id
        assert spans["tool"].parent_span_id == spans["turn"].span_id


async def test_standalone_inmemory_sink_still_links_parents() -> None:
    """The seam is opt-in: a bare sink outside MultiTelemetry keeps
    managing the contextvars itself."""
    sink = InMemoryTelemetry()
    async with sink.trace("outer"):
        async with sink.trace("inner"):
            pass
    spans = {s.name: s for s in sink.spans()}
    assert spans["outer"].parent_span_id is None
    assert spans["inner"].parent_span_id == spans["outer"].span_id


# ---------------------------------------------------------------------------
# 4. FilesystemSandbox — container args are validated
# ---------------------------------------------------------------------------


@tool
async def batch_read(paths: list[str]) -> str:
    """Read several files."""
    return ",".join(paths)


@tool
async def configured(options: dict[str, Any]) -> str:
    """Run with options."""
    return str(options)


async def test_filesystem_sandbox_rejects_list_path_escape(
    tmp_path: Path,
) -> None:
    sandbox = FilesystemSandbox(
        InProcessToolHost([batch_read]), roots=[tmp_path]
    )
    result = await sandbox.call(
        "batch_read", {"paths": ["/etc/passwd"]}, call_id="c1"
    )
    assert not result.ok
    assert result.denied
    assert "/etc/passwd" in (result.reason or "")


async def test_filesystem_sandbox_rejects_separatorless_list_leaf(
    tmp_path: Path,
) -> None:
    """``paths`` is a path-like name — its leaves are validated even
    without a separator in the value."""
    sandbox = FilesystemSandbox(
        InProcessToolHost([batch_read]), roots=[tmp_path]
    )
    # Relative name resolves against cwd, which isn't under tmp_path.
    result = await sandbox.call(
        "batch_read", {"paths": ["passwd"]}, call_id="c2"
    )
    assert not result.ok and result.denied


async def test_filesystem_sandbox_rejects_nested_dict_escape(
    tmp_path: Path,
) -> None:
    sandbox = FilesystemSandbox(
        InProcessToolHost([configured]), roots=[tmp_path]
    )
    result = await sandbox.call(
        "configured",
        {"options": {"path": "/etc/passwd"}},
        call_id="c3",
    )
    assert not result.ok and result.denied


async def test_filesystem_sandbox_allows_container_paths_inside_root(
    tmp_path: Path,
) -> None:
    inside = tmp_path / "a.txt"
    sandbox = FilesystemSandbox(
        InProcessToolHost([batch_read]), roots=[tmp_path]
    )
    result = await sandbox.call(
        "batch_read", {"paths": [str(inside)]}, call_id="c4"
    )
    assert result.ok


async def test_filesystem_sandbox_ignores_non_path_containers(
    tmp_path: Path,
) -> None:
    sandbox = FilesystemSandbox(
        InProcessToolHost([configured]), roots=[tmp_path]
    )
    result = await sandbox.call(
        "configured",
        {"options": {"labels": ["alpha", "beta"], "count": 3}},
        call_id="c5",
    )
    assert result.ok


async def test_filesystem_sandbox_explicit_args_cover_list_leaves(
    tmp_path: Path,
) -> None:
    """Explicit path_args mode: leaves of a declared container arg
    inherit its name and are contained."""
    sandbox = FilesystemSandbox(
        InProcessToolHost([batch_read]),
        roots=[tmp_path],
        path_args=["paths"],
    )
    bad = await sandbox.call(
        "batch_read", {"paths": ["/etc/passwd"]}, call_id="c6"
    )
    assert not bad.ok and bad.denied
    ok = await sandbox.call(
        "batch_read", {"paths": [str(tmp_path / "x")]}, call_id="c7"
    )
    assert ok.ok


# ---------------------------------------------------------------------------
# 5. FileAuditLog — seq recovery deferred out of __init__
# ---------------------------------------------------------------------------


async def test_file_audit_log_init_does_not_scan_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audit.jsonl"
    seed = FileAuditLog(path, secret="x")
    await seed.append(session_id="s", actor="a", action="x", payload={})
    await seed.append(session_id="s", actor="a", action="x", payload={})

    scans: list[int] = []
    original = FileAuditLog._scan_max_seq

    def spy(self: FileAuditLog) -> int:
        scans.append(1)
        return original(self)

    monkeypatch.setattr(FileAuditLog, "_scan_max_seq", spy)

    log = FileAuditLog(path, secret="x")
    assert scans == [], "__init__ read the audit file eagerly"

    entry = await log.append(
        session_id="s", actor="a", action="y", payload={}
    )
    assert scans == [1]
    assert entry.seq == 3  # recovered lazily, still monotonic

    await log.append(session_id="s", actor="a", action="z", payload={})
    assert scans == [1], "seq was re-scanned after the first append"


async def test_file_audit_log_lazy_recovery_missing_file(
    tmp_path: Path,
) -> None:
    log = FileAuditLog(tmp_path / "fresh.jsonl", secret="x")
    entry = await log.append(
        session_id="s", actor="a", action="x", payload={}
    )
    assert entry.seq == 1


# ---------------------------------------------------------------------------
# 6. Seatbelt profile — root validation
# ---------------------------------------------------------------------------


def test_seatbelt_profile_rejects_quote_in_root() -> None:
    with pytest.raises(ValueError, match="cannot be safely embedded"):
        _seatbelt_profile((Path('/tmp/evil"dir'),), allow_network=False)


def test_seatbelt_profile_rejects_newline_in_root() -> None:
    with pytest.raises(ValueError, match="cannot be safely embedded"):
        _seatbelt_profile(
            (Path("/tmp/evil\n(allow network*)"),), allow_network=False
        )


def test_seatbelt_profile_rejects_backslash_and_control_chars() -> None:
    with pytest.raises(ValueError):
        _validate_seatbelt_root(Path("/tmp/back\\slash"))
    with pytest.raises(ValueError):
        _validate_seatbelt_root(Path("/tmp/ctrl\x01char"))


def test_seatbelt_profile_accepts_clean_root(tmp_path: Path) -> None:
    profile = _seatbelt_profile((tmp_path.resolve(),), allow_network=False)
    assert f'(subpath "{tmp_path.resolve()}")' in profile
    assert "(deny default)" in profile
