"""Reflexion architecture tests.

Covers:

* Protocol satisfaction + resolver string.
* Constructor validation (max_attempts, threshold ranges).
* Score parsing (the helper that extracts a 0-1 from evaluator
  output): "score: X" pattern, fallback float-in-text, parse-failure
  defaults to 0.0.
* Threshold-met early exit (score >= threshold on attempt 1 → no
  reflection, no extra attempts).
* Reflect → persist → retry cycle (lesson is appended to memory
  block).
* max_attempts enforcement (never converges, gives up).
* End-to-end via the resolver string.
* Lessons block accumulates across attempts within one run.
* Lessons block is visible to the base architecture's seed_context
  (via memory.working()).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from jeevesagent import (
    Agent,
    Architecture,
    InMemoryMemory,
    ReAct,
    Reflexion,
    ScriptedModel,
    ScriptedTurn,
)
from jeevesagent.architecture import AgentSession, Dependencies
from jeevesagent.architecture.reflexion import _parse_score
from jeevesagent.architecture.resolver import resolve_architecture
from jeevesagent.core.types import Event

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_reflexion_satisfies_architecture_protocol() -> None:
    assert isinstance(Reflexion(), Architecture)


def test_reflexion_name() -> None:
    assert Reflexion().name == "reflexion"


def test_reflexion_declares_no_workers() -> None:
    """Reflexion uses one model in three roles (generator via base,
    evaluator, reflector) — no separate Agents declared."""
    assert Reflexion().declared_workers() == {}


def test_resolver_handles_reflexion_string() -> None:
    arch = resolve_architecture("reflexion")
    assert isinstance(arch, Reflexion)


def test_reflexion_rejects_max_attempts_lt_1() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        Reflexion(max_attempts=0)


def test_reflexion_rejects_threshold_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="threshold"):
        Reflexion(threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        Reflexion(threshold=-0.1)


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


def test_parse_score_picks_up_score_colon_pattern() -> None:
    assert _parse_score("score: 0.85\nReasoning: ...") == 0.85


def test_parse_score_handles_score_equals() -> None:
    assert _parse_score("score=0.4") == 0.4


def test_parse_score_falls_back_to_first_float_in_text() -> None:
    assert _parse_score("the answer scored 0.6 overall") == 0.6


def test_parse_score_clamps_to_0_1_range() -> None:
    """The model might emit scores out of range — clamp."""
    assert _parse_score("score: 1.5") == 1.0
    # Negative numbers don't match our regex (no leading minus), so
    # they fall through to 0.0.
    assert _parse_score("score: bogus") == 0.0


def test_parse_score_returns_zero_on_parse_failure() -> None:
    assert _parse_score("no number anywhere here") == 0.0


# ---------------------------------------------------------------------------
# Test fixtures: a no-op base that sets a fixed output
# ---------------------------------------------------------------------------


class _PrebakedBase:
    """Round-0 architecture that sets ``session.output`` to one of a
    sequence of values. Used so we can exercise Reflexion's loop
    without paying for a full ReAct invocation."""

    name = "_prebaked"

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self._idx = 0

    def declared_workers(self) -> dict[str, Agent]:
        return {}

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        if self._idx < len(self._outputs):
            session.output = self._outputs[self._idx]
            self._idx += 1
        else:
            session.output = "[exhausted]"
        session.turns += 1
        if False:  # keeps function an async generator
            yield Event.budget_warning(  # pragma: no cover
                session.id, None,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Threshold-met on first attempt
# ---------------------------------------------------------------------------


async def test_reflexion_terminates_when_threshold_met_on_first_attempt() -> None:
    """Evaluator returns 0.95 → above threshold 0.8 → no reflector,
    no second attempt. session.output is the first attempt's."""
    base = _PrebakedBase(outputs=["good answer"])
    # Only one model call: the evaluator. No reflector.
    model = ScriptedModel([ScriptedTurn(text="score: 0.95\nLooks good.")])
    agent = Agent(
        "test",
        model=model,
        architecture=Reflexion(base=base, threshold=0.8, max_attempts=3),
    )
    result = await agent.run("task")
    assert result.output == "good answer"
    # 1 (base) + 1 (evaluator) = 2
    assert result.turns == 2


# ---------------------------------------------------------------------------
# One reflect cycle, then converge
# ---------------------------------------------------------------------------


async def test_reflexion_reflects_and_retries_until_threshold() -> None:
    """Attempt 1: score 0.4 → reflect → lesson persisted. Attempt 2:
    score 0.9 → terminate. Final output is attempt-2's."""
    base = _PrebakedBase(outputs=["v1 (bad)", "v2 (good)"])
    model = ScriptedModel(
        [
            ScriptedTurn(text="score: 0.4\nMissing key info."),
            ScriptedTurn(text="Be more specific about dates."),
            ScriptedTurn(text="score: 0.9\nLooks great."),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        architecture=Reflexion(base=base, threshold=0.8, max_attempts=3),
    )
    result = await agent.run("task")
    assert result.output == "v2 (good)"
    # 1 (base R1) + 1 (eval R1) + 1 (reflect R1) + 1 (base R2) + 1 (eval R2) = 5
    assert result.turns == 5


async def test_reflexion_persists_lesson_to_memory_block() -> None:
    """After a failed attempt, the lesson lands in
    memory.working() under the configured block name."""
    base = _PrebakedBase(outputs=["v1", "v2"])
    model = ScriptedModel(
        [
            ScriptedTurn(text="score: 0.3"),
            ScriptedTurn(text="Use ISO dates always."),
            ScriptedTurn(text="score: 0.9"),
        ]
    )
    memory = InMemoryMemory()
    agent = Agent(
        "test",
        model=model,
        memory=memory,
        architecture=Reflexion(
            base=base,
            threshold=0.8,
            max_attempts=3,
            lessons_block_name="my_lessons",
        ),
    )
    await agent.run("task")
    blocks = await memory.working()
    by_name = {b.name: b for b in blocks}
    assert "my_lessons" in by_name
    assert "Use ISO dates always" in by_name["my_lessons"].content


# ---------------------------------------------------------------------------
# max_attempts enforcement
# ---------------------------------------------------------------------------


async def test_reflexion_gives_up_at_max_attempts() -> None:
    """Evaluator never returns >= threshold → reflexion runs for
    max_attempts and returns the latest attempt's output."""
    base = _PrebakedBase(outputs=["v1", "v2"])
    # Two attempts both fail; max_attempts=2 means after attempt 2's
    # evaluation, we hit the cap and return.
    model = ScriptedModel(
        [
            ScriptedTurn(text="score: 0.3"),
            ScriptedTurn(text="lesson 1"),
            ScriptedTurn(text="score: 0.4"),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        architecture=Reflexion(base=base, threshold=0.8, max_attempts=2),
    )
    result = await agent.run("task")
    assert result.output == "v2"
    # 1 (base R1) + 1 (eval R1) + 1 (reflect R1) + 1 (base R2) + 1 (eval R2) = 5
    assert result.turns == 5


async def test_reflexion_emits_max_attempts_event() -> None:
    base = _PrebakedBase(outputs=["v1"])
    model = ScriptedModel(
        [
            ScriptedTurn(text="score: 0.1"),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        architecture=Reflexion(base=base, threshold=0.8, max_attempts=1),
    )
    events = [e async for e in agent.stream("task")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "reflexion.max_attempts_reached" in arch_names


# ---------------------------------------------------------------------------
# Memory-block visibility to base architecture
# ---------------------------------------------------------------------------


class _CaptureSeedBase:
    """A base that captures the seed context blocks it sees, so we
    can assert that prior-attempt lessons are visible on later
    attempts."""

    name = "_capture"

    def __init__(self) -> None:
        self.seen_blocks_per_attempt: list[list[tuple[str, str]]] = []

    def declared_workers(self) -> dict[str, Agent]:
        return {}

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        blocks = await deps.memory.working()
        self.seen_blocks_per_attempt.append(
            [(b.name, b.content) for b in blocks]
        )
        session.output = f"seen {len(blocks)} blocks"
        session.turns += 1
        if False:  # async generator
            yield Event.budget_warning(  # pragma: no cover
                session.id, None,  # type: ignore[arg-type]
            )


async def test_reflexion_lesson_visible_to_base_on_next_attempt() -> None:
    """Attempt 1: base sees no blocks. Reflexion persists a lesson.
    Attempt 2: base sees the lesson in memory.working()."""
    base = _CaptureSeedBase()
    model = ScriptedModel(
        [
            # Attempt 1: low score
            ScriptedTurn(text="score: 0.2"),
            # Reflect R1
            ScriptedTurn(text="critical lesson about X"),
            # Attempt 2: high score (terminate)
            ScriptedTurn(text="score: 0.9"),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        memory=InMemoryMemory(),
        architecture=Reflexion(
            base=base, threshold=0.8, max_attempts=3
        ),
    )
    await agent.run("task")
    # Attempt 1 saw no lesson block.
    assert all(
        n != "reflexion_lessons"
        for n, _ in base.seen_blocks_per_attempt[0]
    )
    # Attempt 2 saw the lesson block with the persisted content.
    found = False
    for name, content in base.seen_blocks_per_attempt[1]:
        if name == "reflexion_lessons" and "critical lesson about X" in content:
            found = True
    assert found


# ---------------------------------------------------------------------------
# Composition with ReAct as the actual base
# ---------------------------------------------------------------------------


async def test_reflexion_with_react_base_end_to_end() -> None:
    """Smoke test: Reflexion(base=ReAct()) runs a full loop. ReAct's
    text response becomes the attempt; evaluator scores it; if
    threshold met, terminate. Just verifies plumbing works."""
    model = ScriptedModel(
        [
            ScriptedTurn(text="here is my answer"),
            ScriptedTurn(text="score: 0.95"),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        architecture=Reflexion(base=ReAct(), threshold=0.8),
    )
    result = await agent.run("question")
    assert "here is my answer" in result.output
    # ReAct turn (1) + eval (1) = 2
    assert result.turns == 2


# ---------------------------------------------------------------------------
# Resolver string with default base (ReAct)
# ---------------------------------------------------------------------------


async def test_reflexion_via_resolver_string_uses_react_default() -> None:
    """``architecture="reflexion"`` constructs Reflexion() which
    defaults base=ReAct()."""
    model = ScriptedModel(
        [
            ScriptedTurn(text="my output"),
            ScriptedTurn(text="score: 0.9"),
        ]
    )
    agent = Agent("test", model=model, architecture="reflexion")
    result = await agent.run("hi")
    assert "my output" in result.output


# ---------------------------------------------------------------------------
# Selective lesson recall via VectorStore
# ---------------------------------------------------------------------------


async def test_reflexion_persists_lessons_to_vector_store() -> None:
    """When a ``lesson_store`` is configured, lessons go to the
    vector store on each failed attempt — not the memory block."""
    from jeevesagent import HashEmbedder, InMemoryVectorStore

    store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=64))
    model = ScriptedModel(
        [
            ScriptedTurn(text="bad answer 1"),
            ScriptedTurn(text="score: 0.2"),
            ScriptedTurn(text="lesson: try harder"),
            ScriptedTurn(text="bad answer 2"),
            ScriptedTurn(text="score: 0.3"),
            ScriptedTurn(text="lesson: try even harder"),
            ScriptedTurn(text="finally good"),
            ScriptedTurn(text="score: 0.95"),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        architecture=Reflexion(
            base=ReAct(),
            threshold=0.8,
            max_attempts=3,
            lesson_store=store,
        ),
    )
    result = await agent.run("a hard question")
    assert "finally good" in result.output
    assert await store.count() == 2


async def test_reflexion_recalls_top_k_from_store() -> None:
    """A populated lesson_store rewrites the working memory block
    on each attempt with at most top_k_lessons retrieved bullets."""
    from jeevesagent import HashEmbedder, InMemoryVectorStore
    from jeevesagent.loader.base import Chunk

    store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=64))
    await store.add(
        [
            Chunk(
                content=f"lesson #{i} about topic {i}", metadata={}
            )
            for i in range(5)
        ]
    )
    model = ScriptedModel(
        [
            ScriptedTurn(text="ok answer"),
            ScriptedTurn(text="score: 0.95"),
        ]
    )
    memory = InMemoryMemory()
    agent = Agent(
        "test",
        model=model,
        memory=memory,
        architecture=Reflexion(
            base=ReAct(),
            threshold=0.9,
            lesson_store=store,
            top_k_lessons=2,
        ),
    )
    await agent.run("topic 3 query")
    blocks = await memory.working()
    block = next(
        (b for b in blocks if b.name == "reflexion_lessons"), None
    )
    assert block is not None
    bullet_count = block.content.count("- ")
    assert bullet_count <= 2


def test_reflexion_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k_lessons"):
        Reflexion(top_k_lessons=0)
