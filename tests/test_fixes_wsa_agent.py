"""Regression tests for the reviewed agent/api fixes (WSA batch).

Covers:

1. Ambient contextvars (living-plan / workspace / citations) are
   reset on EVERY exit path from ``Agent._loop`` — including failed
   runs (``OutputValidationError``) and wall-clock timeouts
   (``RunTimeout`` via ``fail_after`` cancellation). Before the fix,
   the resets only ran on the success path, leaking one agent's
   ambient state into the caller's task.
2. Auto-compact fires for default agents (no stop hooks): the
   compaction check runs after every architecture pass instead of
   only between Ralph-loop iterations. Also: token-count failure now
   emits an ``auto_compact.failed`` architecture event instead of
   silently disabling compaction, and a summariser failure warns.
3. ``max_stop_hook_iterations=0`` genuinely disables the Ralph loop
   (documented semantics) — no re-invocation, no hook polling, and
   NO ``interrupted`` / ``stop_hook_iterations_exhausted`` stamp.
4. ``Agent.stream()`` setup failures emit an ERROR event carrying
   the run's real session id instead of ``""``.
6. ``BoundedDict`` TTL eviction is amortised: per-key operations no
   longer trigger a full O(n) sweep each time, while expired keys
   remain unobservable through every per-key read in the meantime.

(Numbering matches the review findings.)
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest
from pydantic import BaseModel

from loomflow import (
    Agent,
    EchoModel,
    OutputValidationError,
    RunTimeout,
    ScriptedModel,
    ScriptedTurn,
    StopHookResult,
    Tuning,
)
from loomflow.agent.auto_compact import maybe_auto_compact
from loomflow.architecture.base import AgentSession
from loomflow.core._eviction import BoundedDict
from loomflow.core.context import _ambient_living_plan_var
from loomflow.core.types import (
    Event,
    EventKind,
    Message,
    ModelChunk,
    Role,
    Usage,
)

pytestmark = pytest.mark.anyio


class _Out(BaseModel):
    value: int


# ---------------------------------------------------------------------------
# 1. Contextvar cleanup on failed runs
# ---------------------------------------------------------------------------


async def test_contextvar_reset_after_output_validation_error() -> None:
    """A run that dies with OutputValidationError must still reset
    the ambient living-plan contextvar it installed at run start."""
    a = Agent(
        "hi",
        model=EchoModel(),
        living_plan={"auto_stop_hook": False},
    )
    assert _ambient_living_plan_var.get() is None
    with pytest.raises(OutputValidationError):
        # Echo replies with plain text — never valid JSON for _Out.
        await a.run(
            "not json",
            output_schema=_Out,
            output_validation_retries=0,
        )
    assert _ambient_living_plan_var.get() is None


class _SlowModel:
    """Model that hangs long enough for ``Agent(timeout=)`` to fire."""

    name = "slow"

    async def complete(
        self, messages: list[Message], **kwargs: Any
    ) -> tuple[str, list[Any], Usage, str]:
        await anyio.sleep(30)
        return "late", [], Usage(), "stop"

    async def stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[ModelChunk]:
        await anyio.sleep(30)
        yield ModelChunk(kind="finish", finish_reason="stop", usage=Usage())


async def test_contextvar_reset_after_run_timeout() -> None:
    """RunTimeout cancels the run via ``fail_after`` — the ambient
    contextvars must still be reset during the cancellation unwind."""
    a = Agent(
        "hi",
        model=_SlowModel(),
        timeout=0.1,
        living_plan={"auto_stop_hook": False},
    )
    with pytest.raises(RunTimeout):
        await a.run("hang")
    assert _ambient_living_plan_var.get() is None


async def test_contextvar_reset_after_successful_run() -> None:
    """Sanity: the success path resets too (pre-fix behaviour kept)."""
    a = Agent(
        "hi",
        model=EchoModel(),
        living_plan={"auto_stop_hook": False},
    )
    r = await a.run("hello")
    assert r.output
    assert _ambient_living_plan_var.get() is None


# ---------------------------------------------------------------------------
# 2. Auto-compact fires for default (no-stop-hook) agents
# ---------------------------------------------------------------------------


async def test_auto_compact_checked_without_stop_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A default agent (no stop hooks → fast_stop_hooks=True) with
    ``auto_compact_at_tokens=`` configured must run the compaction
    check after its architecture pass. Before the fix the check
    lived inside the ``if not fast_stop_hooks:`` branch and never
    ran for default agents."""
    calls: list[str] = []
    orig = Agent._maybe_compact

    async def spy(self: Agent, session: AgentSession, emit: Any) -> None:
        calls.append(session.id)
        await orig(self, session, emit)

    monkeypatch.setattr(Agent, "_maybe_compact", spy)
    a = Agent("hi", model=EchoModel(), auto_compact_at_tokens=1_000_000)
    r = await a.run("hello")
    assert r.turns == 1
    assert len(calls) == 1


async def test_maybe_compact_compacts_and_emits_event() -> None:
    """Over-threshold session → older turns replaced with a summary
    system message and an ``auto_compacted`` event is emitted."""
    summariser = ScriptedModel(turns=[ScriptedTurn(text="SUMMARY")])
    a = Agent(
        "hi",
        model=EchoModel(),
        auto_compact_at_tokens=10,
        tuning=Tuning(
            auto_compact_summariser=summariser,
            auto_compact_keep_recent_turns=1,
        ),
    )
    session = AgentSession(id="s-compact", instructions="hi")
    session.messages = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="q1 " * 60),
        Message(role=Role.ASSISTANT, content="a1 " * 60),
        Message(role=Role.USER, content="q2"),
        Message(role=Role.ASSISTANT, content="a2"),
    ]
    events: list[Event] = []

    async def emit(e: Event) -> None:
        events.append(e)

    await a._maybe_compact(session, emit)

    names = [e.payload.get("name") for e in events]
    assert "auto_compacted" in names
    assert any(
        "[auto-compacted summary" in (m.content or "")
        for m in session.messages
    )
    # Recent tail kept verbatim.
    assert session.messages[-1].content == "a2"


async def test_maybe_compact_count_failure_emits_failed_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When token counting blows up, compaction is skipped for the
    pass but the squeeze is observable via ``auto_compact.failed``
    (pre-fix: silently mapped to ``current = 0``)."""
    # ``loomflow.model.__init__`` re-exports the function under the
    # same name, shadowing the submodule attribute — resolve the real
    # module so the monkeypatch reaches api.py's local import.
    import importlib

    ct = importlib.import_module("loomflow.model.count_tokens")

    async def boom(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("counter down")

    monkeypatch.setattr(ct, "count_tokens", boom)
    a = Agent("hi", model=EchoModel(), auto_compact_at_tokens=10)
    session = AgentSession(id="s-fail", instructions="hi")
    session.messages = [Message(role=Role.USER, content="q")]
    events: list[Event] = []

    async def emit(e: Event) -> None:
        events.append(e)

    await a._maybe_compact(session, emit)

    failed = [
        e for e in events if e.payload.get("name") == "auto_compact.failed"
    ]
    assert len(failed) == 1
    assert failed[0].session_id == "s-fail"


async def test_summariser_failure_warns() -> None:
    """A raising summariser is still a graceful no-op — but now it
    warns so a permanently-broken summariser can't silently disable
    compaction forever."""

    class _RaisingModel:
        name = "raiser"

        def stream(self, *args: Any, **kwargs: Any) -> Any:
            async def _gen() -> AsyncIterator[ModelChunk]:
                raise RuntimeError("API down")
                yield  # pragma: no cover

            return _gen()

    msgs = [
        Message(role=Role.USER, content="q1"),
        Message(role=Role.ASSISTANT, content="a1"),
        Message(role=Role.USER, content="q2"),
        Message(role=Role.ASSISTANT, content="a2"),
    ]
    with pytest.warns(UserWarning, match="auto-compact summariser"):
        new_msgs, summary = await maybe_auto_compact(
            msgs,
            summariser=_RaisingModel(),  # type: ignore[arg-type]
            at_tokens=10,
            current_token_count=1000,
            keep_recent_turns=1,
        )
    assert new_msgs is None
    assert summary == ""


# ---------------------------------------------------------------------------
# 3. max_stop_hook_iterations=0 disables the loop entirely
# ---------------------------------------------------------------------------


async def test_zero_cap_disables_loop_without_interruption() -> None:
    """Cap of 0 → one architecture pass, hooks never polled, and the
    run is NOT stamped interrupted/exhausted (pre-fix: the
    ``while ... else`` fired immediately and marked every run
    interrupted)."""
    polled: list[int] = []

    class _WouldContinue:
        name = "would"

        async def __call__(
            self, session: Any, deps: Any, *, iteration: int
        ) -> StopHookResult:
            polled.append(iteration)
            return StopHookResult(inject_message="go", reason="r")

    a = Agent(
        "hi",
        model=EchoModel(),
        tuning=Tuning(
            stop_hooks=[_WouldContinue()],
            max_stop_hook_iterations=0,
        ),
    )
    r = await a.run("once")
    assert r.turns == 1
    assert r.interrupted is False
    assert r.interruption_reason is None
    # The loop is disabled entirely — hooks are never even polled.
    assert polled == []


async def test_positive_cap_exhaustion_still_marks_interrupted() -> None:
    """The exhaustion stamp still fires for a genuinely-exhausted
    positive cap (unchanged semantics)."""

    class _AlwaysGo:
        name = "always_go"

        async def __call__(
            self, session: Any, deps: Any, *, iteration: int
        ) -> StopHookResult:
            return StopHookResult(inject_message="go", reason="loop")

    sm = ScriptedModel(turns=[ScriptedTurn(text="t") for _ in range(5)])
    a = Agent(
        "hi",
        model=sm,
        tuning=Tuning(
            stop_hooks=[_AlwaysGo()],
            max_stop_hook_iterations=2,
        ),
    )
    r = await a.run("start")
    assert r.turns == 3  # initial pass + 2 continuations
    assert r.interrupted is True
    assert r.interruption_reason == "stop_hook_iterations_exhausted"


# ---------------------------------------------------------------------------
# 4. stream() error events carry the real session id
# ---------------------------------------------------------------------------


async def test_stream_error_event_carries_session_id() -> None:
    """A failing streamed run emits an ERROR event attributable to
    the run's session id (pre-fix: ``Event.error("", exc)``)."""
    a = Agent("hi", model=EchoModel())
    events: list[Event] = []
    with contextlib.suppress(Exception):
        async for e in a.stream(
            "not json",
            session_id="sess-known",
            output_schema=_Out,
            output_validation_retries=0,
        ):
            events.append(e)
    errors = [e for e in events if e.kind == EventKind.ERROR]
    assert errors
    assert errors[-1].session_id == "sess-known"


async def test_stream_error_event_has_generated_session_id() -> None:
    """Without an explicit session_id, the error event carries the
    framework-generated one — never the empty string."""
    a = Agent("hi", model=EchoModel())
    events: list[Event] = []
    with contextlib.suppress(Exception):
        async for e in a.stream(
            "not json",
            output_schema=_Out,
            output_validation_retries=0,
        ):
            events.append(e)
    errors = [e for e in events if e.kind == EventKind.ERROR]
    assert errors
    assert errors[-1].session_id != ""
    # Same id as the run's other events (attributable to one run).
    started = [e for e in events if e.kind == EventKind.STARTED]
    assert started and started[0].session_id == errors[-1].session_id


# ---------------------------------------------------------------------------
# 6. BoundedDict — amortised TTL sweep, correctness preserved
# ---------------------------------------------------------------------------


def test_bounded_dict_expired_keys_unobservable_via_per_key_reads() -> None:
    """Even between amortised full sweeps, per-key reads must never
    observe an expired entry."""
    d: BoundedDict[str, int] = BoundedDict(ttl_seconds=0.05)
    d["a"] = 1
    d["b"] = 2
    d["c"] = 3
    d["d"] = 4
    time.sleep(0.08)
    assert d.get("a") is None
    assert d.get("b", 99) == 99
    assert "c" not in d
    with pytest.raises(KeyError):
        _ = d["d"]
    assert len(d) == 0


def test_bounded_dict_setdefault_replaces_expired_entry() -> None:
    d: BoundedDict[str, int] = BoundedDict(ttl_seconds=0.05)
    d["k"] = 1
    time.sleep(0.08)
    assert d.setdefault("k", 99) == 99


def test_bounded_dict_sweep_is_amortised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-key ops must NOT trigger a full O(n) sweep each time —
    only once per _SWEEP_EVERY (128) operations."""
    sweeps = {"n": 0}
    orig = BoundedDict._sweep_expired

    def spy(self: BoundedDict[Any, Any]) -> int:
        sweeps["n"] += 1
        return orig(self)

    monkeypatch.setattr(BoundedDict, "_sweep_expired", spy)
    d: BoundedDict[str, int] = BoundedDict(ttl_seconds=3600.0)
    d["k"] = 1  # op 1
    for _ in range(100):  # ops 2..101 — under the 128 threshold
        d.get("k")
    assert sweeps["n"] == 0
    for _ in range(100):  # crosses 128 exactly once, then resets
        d.get("k")
    assert sweeps["n"] == 1


def test_bounded_dict_no_ttl_never_sweeps_on_per_key_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a TTL there is nothing to sweep — per-key ops stay
    sweep-free no matter how many run."""
    sweeps = {"n": 0}
    orig = BoundedDict._sweep_expired

    def spy(self: BoundedDict[Any, Any]) -> int:
        sweeps["n"] += 1
        return orig(self)

    monkeypatch.setattr(BoundedDict, "_sweep_expired", spy)
    d: BoundedDict[str, int] = BoundedDict()
    for i in range(300):
        d[str(i)] = i
        d.get(str(i))
    assert sweeps["n"] == 0


def test_bounded_dict_view_ops_still_sweep() -> None:
    """The O(n) view operations always sweep, so ``len``/``items``
    never report expired entries."""
    d: BoundedDict[str, int] = BoundedDict(ttl_seconds=0.05)
    d["a"] = 1
    d["b"] = 2
    time.sleep(0.08)
    assert len(d) == 0
    assert d.items() == []
    assert d.keys() == []
    assert d.values() == []
    assert list(iter(d)) == []
