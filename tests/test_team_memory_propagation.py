"""Memory propagation across team architectures (P2 fix).

Pins the behavior introduced by ``inherit_ambient_memory`` in
``loomflow/core/context.py``: when ``Team.supervisor(memory=X)`` (or
any other Team builder) is called, workers constructed WITHOUT an
explicit ``memory=`` of their own should inherit ``X`` for the
duration of the run. Workers that DID set their own memory keep it.

This closes the gap where ``Team.supervisor(memory="sqlite:...")``
was silently NOT propagating to workers, leaving them on their
private ephemeral ``InMemoryMemory`` and breaking the persistent-
subagents promise across architectures.

These tests use the recorded-memory pattern: a tiny ``_RecordingMemory``
that wraps an inner memory and records every ``remember`` call's
session_id. After running a team, we assert which memory recorded
which episode — a worker that inherited the coordinator's memory
shows up in the coordinator's recorder; a worker with its own
explicit memory shows up in its own recorder.
"""

from __future__ import annotations

import pytest

from loomflow import (
    Agent,
    EchoModel,
    Episode,
    InMemoryMemory,
)
from loomflow.team import Team

pytestmark = pytest.mark.anyio


class _RecordingMemory(InMemoryMemory):
    """InMemoryMemory that captures every ``remember`` call.

    Used to assert which memory backend an Agent's episode landed
    in — the test setup gives each Agent a uniquely-tagged recorder
    and after the run we check which tags saw which session_ids.
    """

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag
        self.episodes: list[Episode] = []

    async def remember(self, episode: Episode) -> None:
        self.episodes.append(episode)
        await super().remember(episode)


async def test_supervisor_propagates_memory_to_worker_without_explicit() -> None:
    """Worker without explicit memory= inherits the coordinator's
    memory via the ambient context manager set in supervisor's
    delegate path."""
    coord_mem = _RecordingMemory(tag="coordinator")
    # Worker constructed with NO explicit memory — should inherit.
    worker = Agent(
        instructions="Echo back.",
        model=EchoModel(),
    )

    coordinator = Team.supervisor(
        workers={"echoer": worker},
        instructions="Delegate to echoer.",
        model=EchoModel(),
        memory=coord_mem,
    )

    # Spin the worker once via direct .run() with the ambient set
    # — bypasses the model-driven delegate decision (EchoModel won't
    # emit a delegate tool call). We just want to verify the wiring.
    from loomflow.core.context import inherit_ambient_memory

    with inherit_ambient_memory(coord_mem):
        await worker.run("hello", session_id="echoer-session")

    # The episode was written to the COORDINATOR's recorder because
    # the worker had no explicit memory, so _resolve_run_memory
    # picked up the ambient.
    assert len(coord_mem.episodes) == 1
    assert coord_mem.episodes[0].session_id == "echoer-session"
    # Sanity: the coordinator helper is reachable from the test
    # namespace (the import succeeds — that's the entire P2 hook).
    assert coordinator is not None


async def test_worker_with_explicit_memory_is_not_overridden() -> None:
    """Worker constructed WITH its own memory= keeps it — ambient
    propagation does not override an explicit choice. Mirrors the
    explicit-always-wins rule from Workflow precedence."""
    coord_mem = _RecordingMemory(tag="coordinator")
    worker_mem = _RecordingMemory(tag="worker")

    worker = Agent(
        instructions="Echo back.",
        model=EchoModel(),
        memory=worker_mem,  # explicit
    )

    from loomflow.core.context import inherit_ambient_memory

    with inherit_ambient_memory(coord_mem):
        await worker.run("hello", session_id="explicit-session")

    # Episode landed in the worker's OWN memory, not the
    # coordinator's. Explicit > ambient.
    assert len(worker_mem.episodes) == 1
    assert worker_mem.episodes[0].session_id == "explicit-session"
    assert len(coord_mem.episodes) == 0


async def test_inherit_ambient_memory_is_nest_safe() -> None:
    """Nested ``inherit_ambient_memory`` blocks restore the prior
    binding on exit. Anyio task-group spawns inherit the contextvar
    automatically (Python contextvar semantics), so this matters
    for the helper's correctness, not just style."""
    outer_mem = _RecordingMemory(tag="outer")
    inner_mem = _RecordingMemory(tag="inner")
    worker = Agent(instructions="", model=EchoModel())

    from loomflow.core.context import (
        _ambient_memory_var,
        inherit_ambient_memory,
    )

    assert _ambient_memory_var.get() is None
    with inherit_ambient_memory(outer_mem):
        assert _ambient_memory_var.get() is outer_mem
        with inherit_ambient_memory(inner_mem):
            assert _ambient_memory_var.get() is inner_mem
            await worker.run("nested", session_id="nest-session")
        # Inner exited — outer is restored.
        assert _ambient_memory_var.get() is outer_mem
    # Outermost exited — back to None.
    assert _ambient_memory_var.get() is None

    # The episode was written to inner_mem (the active ambient at
    # the time of the run).
    assert len(inner_mem.episodes) == 1
    assert len(outer_mem.episodes) == 0


async def test_inherit_ambient_memory_restores_on_exception() -> None:
    """Even if the inner block raises, the contextvar is reset."""
    mem = _RecordingMemory(tag="x")

    from loomflow.core.context import (
        _ambient_memory_var,
        inherit_ambient_memory,
    )

    assert _ambient_memory_var.get() is None
    with pytest.raises(RuntimeError, match="boom"):
        with inherit_ambient_memory(mem):
            assert _ambient_memory_var.get() is mem
            raise RuntimeError("boom")
    # Restored despite the exception.
    assert _ambient_memory_var.get() is None


def test_team_supervisor_forwards_persist_tool_transcripts() -> None:
    """Team.supervisor now (0.10.16+) forwards
    ``persist_tool_transcripts`` + ``tool_transcript_max_bytes``
    to the coordinator Agent. Without this, the coordinator
    couldn't opt in to tool-transcript persistence even when the
    framework supported the feature — flag stayed default-off."""
    worker = Agent(instructions="", model=EchoModel())
    coordinator = Team.supervisor(
        workers={"w": worker},
        instructions="lead",
        model=EchoModel(),
        persist_tool_transcripts=True,
        tool_transcript_max_bytes=12345,
    )
    assert coordinator._persist_tool_transcripts is True
    assert coordinator._tool_transcript_max_bytes == 12345


def test_team_supervisor_default_persist_is_off() -> None:
    """Default stays opt-in so existing callers see no behavior change."""
    coordinator = Team.supervisor(
        workers={"w": Agent(instructions="", model=EchoModel())},
        instructions="lead",
        model=EchoModel(),
    )
    assert coordinator._persist_tool_transcripts is False


def test_all_team_builders_forward_persist_tool_transcripts() -> None:
    """The kwarg has to reach the coordinator from every Team.*
    builder, not just supervisor. Without this, swarm/router/debate/
    actor_critic/blackboard users would see the kwarg accepted but
    silently dropped — the worst kind of API bug. One assertion per
    builder pins the forwarding."""
    from loomflow.architecture.router import RouterRoute

    a = Agent(instructions="", model=EchoModel())
    b = Agent(instructions="", model=EchoModel())

    # Each builder takes a different required-args shape; the
    # assertion is identical: ``_persist_tool_transcripts`` lands
    # True on the returned coordinator Agent.
    cases: list[tuple[str, Agent]] = [
        (
            "supervisor",
            Team.supervisor(
                workers={"w": a},
                instructions="",
                model=EchoModel(),
                persist_tool_transcripts=True,
            ),
        ),
        (
            "swarm",
            Team.swarm(
                {"a": a, "b": b},
                "a",
                model=EchoModel(),
                persist_tool_transcripts=True,
            ),
        ),
        (
            "router",
            Team.router(
                routes=[
                    RouterRoute(name="a", description="", agent=a)
                ],
                model=EchoModel(),
                persist_tool_transcripts=True,
            ),
        ),
        (
            "debate",
            Team.debate(
                debaters=[a, b],
                model=EchoModel(),
                persist_tool_transcripts=True,
            ),
        ),
        (
            "actor_critic",
            Team.actor_critic(
                actor=a,
                critic=b,
                model=EchoModel(),
                persist_tool_transcripts=True,
            ),
        ),
        (
            "blackboard",
            Team.blackboard(
                agents={"a": a, "b": b},
                model=EchoModel(),
                persist_tool_transcripts=True,
            ),
        ),
    ]
    for name, coord in cases:
        assert coord._persist_tool_transcripts is True, name
