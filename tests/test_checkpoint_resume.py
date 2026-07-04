"""G4b — agent-loop checkpoint writing + real ``Agent.resume``.

The contract under test:

* Checkpoint writing is OPT-IN via ``Tuning(checkpoint=True)`` — the
  default agent writes nothing even on a checkpoint-capable runtime.
* A checkpoint lands after every architecture pass (first pass + each
  stop-hook / Ralph iteration), snapshotting messages / turns /
  cumulative usage.
* A crashed run resumes from the latest checkpoint: messages, turns
  and usage are restored, memory-rehydration seeding is SKIPPED (the
  transcript IS the state), the new prompt is appended as a USER turn,
  and prior turns are never re-executed or re-billed.
* ``resume(prompt=None)`` injects the internal continuation nudge.
* ``Agent.list_checkpoints`` delegates to the runtime (empty list on
  runtimes without checkpoint support).
* Resuming from an OLDER checkpoint id forks: fresh session_id, the
  original session untouched.
* Runtimes without checkpoint support (or sessions with no
  checkpoints) fall back to the legacy resume — ``run()`` with memory
  rehydration / journal replay.
* A failing ``put_checkpoint`` warns and never kills the run.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

from loomflow import Agent, InMemoryMemory, Tuning
from loomflow.agent.api import _RESUME_CONTINUATION_PROMPT
from loomflow.agent.stop_hooks import StopHookResult
from loomflow.core.types import Message, Role, Usage
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.runtime import (
    Checkpoint,
    InMemoryJournalStore,
    InProcRuntime,
    JournaledRuntime,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingModel(ScriptedModel):
    """ScriptedModel that records the message list of every call."""

    name = "recording"

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.seen: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], **kwargs: Any
    ) -> tuple[str, list[Any], Usage, str]:
        self.seen.append(list(messages))
        return await super().complete(messages, **kwargs)


class CrashAfterModel(ScriptedModel):
    """Replays its script, then RAISES instead of returning empties —
    simulates the process dying on the (N+1)th model call."""

    name = "crash_after"

    async def complete(
        self, messages: list[Message], **kwargs: Any
    ) -> tuple[str, list[Any], Usage, str]:
        if self.remaining == 0:
            raise RuntimeError("simulated crash")
        return await super().complete(messages, **kwargs)


class SpyMemory(InMemoryMemory):
    """InMemoryMemory that counts seed-time recall calls so a test can
    assert the resume path never re-seeds."""

    def __init__(self) -> None:
        super().__init__()
        self.working_calls = 0
        self.recall_calls = 0

    async def working(self, *args: Any, **kwargs: Any) -> Any:
        self.working_calls += 1
        return await super().working(*args, **kwargs)

    async def recall(self, *args: Any, **kwargs: Any) -> Any:
        self.recall_calls += 1
        return await super().recall(*args, **kwargs)


class ContinueTimes:
    """Stop hook that forces N extra architecture passes."""

    name = "continue_times"

    def __init__(self, times: int) -> None:
        self.times = times
        self.fired = 0

    async def __call__(
        self, session: Any, deps: Any, *, iteration: int
    ) -> StopHookResult | None:
        if self.fired >= self.times:
            return None
        self.fired += 1
        return StopHookResult(inject_message="keep going", reason="test")


class MinimalRuntime:
    """A protocol-satisfying runtime with NO checkpoint / signal
    extensions — exercises every hasattr-gated fallback."""

    name = "minimal"

    async def step(
        self,
        name: str,
        fn: Callable[..., Awaitable[Any]],
        *args: Any,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return await fn(*args, **kwargs)

    def stream_step(
        self,
        name: str,
        fn: Callable[..., AsyncIterator[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        return fn(*args, **kwargs)

    @asynccontextmanager
    async def session(self, session_id: str) -> AsyncIterator[Any]:
        class _S:
            id = session_id

        yield _S()


def _tuned_agent(
    model: ScriptedModel,
    runtime: Any,
    *,
    checkpoint: bool = True,
    memory: Any | None = None,
    stop_hooks: list[Any] | None = None,
) -> Agent:
    return Agent(
        "hi",
        model=model,
        memory=memory if memory is not None else InMemoryMemory(),
        runtime=runtime,
        tuning=Tuning(checkpoint=checkpoint, stop_hooks=stop_hooks),
    )


# ---------------------------------------------------------------------------
# Checkpoint writing — opt-in only
# ---------------------------------------------------------------------------


async def test_checkpoint_written_after_pass_when_opted_in() -> None:
    runtime = InProcRuntime()
    agent = _tuned_agent(
        ScriptedModel([ScriptedTurn(text="done", usage=Usage(
            input_tokens=10, output_tokens=5, cost_usd=0.01,
        ))]),
        runtime,
    )
    result = await agent.run("go", session_id="ckpt-sess")
    metas = await agent.list_checkpoints("ckpt-sess")
    assert len(metas) == 1
    assert metas[0].turn == result.turns == 1
    cp = await runtime.get_checkpoint("ckpt-sess", metas[0].checkpoint_id)
    assert cp is not None
    assert cp.cumulative_usage.input_tokens == 10
    assert cp.cumulative_usage.output_tokens == 5
    # The snapshot carries the full transcript: system + user + reply.
    roles = [m.role for m in cp.messages]
    assert roles[0] == Role.SYSTEM
    assert Role.USER in roles
    assert cp.messages[-1].role == Role.ASSISTANT


async def test_no_checkpoint_without_opt_in() -> None:
    runtime = InProcRuntime()
    agent = _tuned_agent(
        ScriptedModel([ScriptedTurn(text="done")]),
        runtime,
        checkpoint=False,
    )
    await agent.run("go", session_id="no-ckpt")
    assert await agent.list_checkpoints("no-ckpt") == []


async def test_checkpoint_after_each_ralph_iteration() -> None:
    """One checkpoint per architecture pass: first pass + each
    stop-hook iteration."""
    runtime = JournaledRuntime(InMemoryJournalStore())
    agent = _tuned_agent(
        ScriptedModel([
            ScriptedTurn(text="pass one"),
            ScriptedTurn(text="pass two"),
            ScriptedTurn(text="pass three"),
        ]),
        runtime,
        stop_hooks=[ContinueTimes(2)],
    )
    await agent.run("go", session_id="ralph")
    metas = await agent.list_checkpoints("ralph")
    assert len(metas) == 3
    # Newest first, one turn added per pass.
    assert [m.turn for m in metas] == [3, 2, 1]


# ---------------------------------------------------------------------------
# Crash → resume
# ---------------------------------------------------------------------------


async def test_kill_after_turn_then_resume_restores_state() -> None:
    runtime = InProcRuntime()
    sid = "crash-sess"

    # Pass 1 succeeds (checkpoint written); the stop hook forces a
    # second pass whose model call raises — the "kill after turn 1".
    crash_model = CrashAfterModel([
        ScriptedTurn(
            text="partial progress",
            usage=Usage(input_tokens=100, output_tokens=40, cost_usd=0.02),
        ),
    ])
    agent_a = _tuned_agent(
        crash_model, runtime, stop_hooks=[ContinueTimes(1)]
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        await agent_a.run("do the task", session_id=sid)

    metas = await agent_a.list_checkpoints(sid)
    assert len(metas) == 1 and metas[0].turn == 1
    cp = await runtime.get_checkpoint(sid, metas[0].checkpoint_id)
    assert cp is not None

    # Resume with a fresh agent sharing the runtime (InProc = same
    # process). SpyMemory proves seeding is skipped; RecordingModel
    # proves the model saw the restored history + new prompt ONLY.
    spy = SpyMemory()
    resume_model = RecordingModel([
        ScriptedTurn(
            text="finished",
            usage=Usage(input_tokens=50, output_tokens=20, cost_usd=0.01),
        ),
    ])
    agent_b = _tuned_agent(resume_model, runtime, memory=spy)
    result = await agent_b.resume("pick it up", session_id=sid)

    assert result.output == "finished"
    assert result.session_id == sid
    # Turns continue from the checkpoint (1 restored + 1 new).
    assert result.turns == 2
    # Usage rolls up restored + new — prior turns are accounted, not
    # re-billed (no second model execution for turn 1).
    assert result.tokens_in == 150
    assert result.tokens_out == 60
    assert result.cost_usd == pytest.approx(0.03)

    # The model's input was EXACTLY the checkpoint transcript plus the
    # resume prompt — no re-seeded system/memory blocks, no
    # rehydration duplicates.
    assert len(resume_model.seen) == 1
    seen = resume_model.seen[0]
    assert len(seen) == len(cp.messages) + 1
    assert [
        (m.role, m.content) for m in seen[: len(cp.messages)]
    ] == [(m.role, m.content) for m in cp.messages]
    assert seen[-1].role == Role.USER
    assert seen[-1].content == "pick it up"
    # Seeding (memory recall) never ran on the resume path.
    assert spy.working_calls == 0
    assert spy.recall_calls == 0


async def test_resume_without_prompt_uses_continuation_nudge() -> None:
    runtime = InProcRuntime()
    sid = "nudge-sess"
    agent = _tuned_agent(
        ScriptedModel([ScriptedTurn(text="checkpointed")]), runtime
    )
    await agent.run("start", session_id=sid)

    model_b = RecordingModel([ScriptedTurn(text="continued")])
    agent_b = _tuned_agent(model_b, runtime)
    result = await agent_b.resume(session_id=sid)
    assert result.output == "continued"
    assert model_b.seen[0][-1].role == Role.USER
    assert model_b.seen[0][-1].content == _RESUME_CONTINUATION_PROMPT


# ---------------------------------------------------------------------------
# Time travel — fork from an older checkpoint
# ---------------------------------------------------------------------------


async def test_resume_from_older_checkpoint_forks() -> None:
    runtime = JournaledRuntime(InMemoryJournalStore())
    sid = "fork-sess"
    agent = _tuned_agent(
        ScriptedModel([
            ScriptedTurn(text="pass one"),
            ScriptedTurn(text="pass two"),
        ]),
        runtime,
        stop_hooks=[ContinueTimes(1)],
    )
    await agent.run("go", session_id=sid)
    metas = await agent.list_checkpoints(sid)
    assert len(metas) == 2
    older = metas[-1]  # newest-first ⇒ last is the turn-1 checkpoint
    assert older.turn == 1

    model_b = RecordingModel([ScriptedTurn(text="alternate path")])
    agent_b = _tuned_agent(model_b, runtime)
    result = await agent_b.resume(
        "try something else",
        session_id=sid,
        from_checkpoint=older.checkpoint_id,
    )
    # Fork: a fresh session id, derived — not the original.
    assert result.session_id != sid
    assert result.session_id.startswith("sess")
    assert result.output == "alternate path"
    # Fork continues from the OLDER snapshot (turn 1), not the latest.
    assert result.turns == 2
    # The original session's checkpoint history is untouched.
    assert [
        m.checkpoint_id for m in await agent.list_checkpoints(sid)
    ] == [m.checkpoint_id for m in metas]
    # The forked run wrote its own checkpoint under the fork id.
    fork_metas = await agent_b.list_checkpoints(result.session_id)
    assert len(fork_metas) == 1


async def test_resume_by_latest_checkpoint_id_does_not_fork() -> None:
    runtime = InProcRuntime()
    sid = "latest-id-sess"
    agent = _tuned_agent(
        ScriptedModel([ScriptedTurn(text="one")]), runtime
    )
    await agent.run("go", session_id=sid)
    [meta] = await agent.list_checkpoints(sid)

    agent_b = _tuned_agent(
        ScriptedModel([ScriptedTurn(text="two")]), runtime
    )
    result = await agent_b.resume(
        "more", session_id=sid, from_checkpoint=meta.checkpoint_id
    )
    assert result.session_id == sid


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


async def test_unsupported_runtime_falls_back_to_legacy_resume() -> None:
    """A runtime without checkpoint methods degrades to run()."""
    agent = Agent(
        "hi",
        model=ScriptedModel([ScriptedTurn(text="legacy path")]),
        memory=InMemoryMemory(),
        runtime=MinimalRuntime(),
        tuning=Tuning(checkpoint=True),  # opt-in is harmless here
    )
    result = await agent.resume("go", session_id="legacy-sess")
    assert result.output == "legacy path"
    assert result.session_id == "legacy-sess"
    assert await agent.list_checkpoints("legacy-sess") == []


async def test_no_checkpoint_found_falls_back_to_legacy_resume() -> None:
    """Checkpoint-capable runtime, but nothing recorded for the
    session (e.g. prior runs predate ``Tuning(checkpoint=True)``)."""
    runtime = InProcRuntime()
    model = RecordingModel([ScriptedTurn(text="rehydrated")])
    agent = _tuned_agent(model, runtime)
    result = await agent.resume("go", session_id="never-checkpointed")
    assert result.output == "rehydrated"
    # Legacy path DOES seed: the first message is the system prompt
    # built fresh for this run.
    assert model.seen[0][0].role == Role.SYSTEM


async def test_checkpoint_write_failure_warns_but_run_succeeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenCheckpointRuntime(InProcRuntime):
        async def put_checkpoint(self, cp: Checkpoint) -> None:
            raise OSError("disk full")

    agent = _tuned_agent(
        ScriptedModel([ScriptedTurn(text="survived")]),
        BrokenCheckpointRuntime(),
    )
    with caplog.at_level(logging.WARNING, logger="loomflow.agent"):
        result = await agent.run("go", session_id="broken-ckpt")
    assert result.output == "survived"
    assert any(
        "checkpoint write failed" in r.message and "disk full" in r.message
        for r in caplog.records
    )


async def test_checkpoint_lookup_failure_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenLookupRuntime(InProcRuntime):
        async def get_latest_checkpoint(
            self, session_id: str
        ) -> Checkpoint | None:
            raise OSError("store offline")

    agent = _tuned_agent(
        ScriptedModel([ScriptedTurn(text="fell back")]),
        BrokenLookupRuntime(),
        checkpoint=False,
    )
    with caplog.at_level(logging.WARNING, logger="loomflow.agent"):
        result = await agent.resume("go", session_id="broken-lookup")
    assert result.output == "fell back"
    assert any(
        "checkpoint lookup failed" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# JSON round-trip sanity on the loop-written checkpoint
# ---------------------------------------------------------------------------


async def test_loop_checkpoint_survives_json_round_trip() -> None:
    runtime = InProcRuntime()
    agent = _tuned_agent(
        ScriptedModel([ScriptedTurn(text="snapshot me")]), runtime
    )
    await agent.run("go", session_id="json-sess")
    [meta] = await agent.list_checkpoints("json-sess")
    cp = await runtime.get_checkpoint("json-sess", meta.checkpoint_id)
    assert cp is not None
    restored = Checkpoint.model_validate_json(cp.model_dump_json())
    assert restored == cp
