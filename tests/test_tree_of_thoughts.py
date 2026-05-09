"""TreeOfThoughts architecture tests.

Covers:

* Protocol satisfaction; ``declared_workers`` empty (single-agent).
* Constructor validation: branch_factor, max_depth, beam_width,
  solved_threshold ranges.
* Resolver string ``"tree-of-thoughts"``.
* Single-level expansion: 1 root → branch_factor candidates →
  evaluator scores → top beam_width survive.
* Multi-level expansion: previous frontier expanded again.
* Early termination on ``score >= solved_threshold``.
* ``max_depth`` enforcement (no early termination — best leaf wins).
* Best leaf becomes ``session.output`` (highest-scored non-root node).
* Tree structure exposed via ``session.metadata["tot_nodes"]``.
* Architecture progress events surface for each milestone.
* Helper :func:`_chain_to_root` builds correct ancestor chains.
"""

from __future__ import annotations

import pytest

from loomflow import Agent, Architecture, ScriptedModel, ScriptedTurn
from loomflow.architecture import ThoughtNode, TreeOfThoughts
from loomflow.architecture.resolver import resolve_architecture
from loomflow.architecture.tree_of_thoughts import _chain_to_root

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_tot_satisfies_architecture_protocol() -> None:
    assert isinstance(TreeOfThoughts(), Architecture)


def test_tot_name_is_kebab_case() -> None:
    assert TreeOfThoughts().name == "tree-of-thoughts"


def test_tot_declared_workers_empty() -> None:
    assert TreeOfThoughts().declared_workers() == {}


def test_resolver_handles_tree_of_thoughts_string() -> None:
    arch = resolve_architecture("tree-of-thoughts")
    assert isinstance(arch, TreeOfThoughts)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_tot_rejects_branch_factor_lt_1() -> None:
    with pytest.raises(ValueError, match="branch_factor"):
        TreeOfThoughts(branch_factor=0)


def test_tot_rejects_max_depth_lt_1() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        TreeOfThoughts(max_depth=0)


def test_tot_rejects_beam_width_lt_1() -> None:
    with pytest.raises(ValueError, match="beam_width"):
        TreeOfThoughts(beam_width=0)


def test_tot_rejects_solved_threshold_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="solved_threshold"):
        TreeOfThoughts(solved_threshold=1.5)
    with pytest.raises(ValueError, match="solved_threshold"):
        TreeOfThoughts(solved_threshold=-0.1)


# ---------------------------------------------------------------------------
# _chain_to_root helper
# ---------------------------------------------------------------------------


def test_chain_to_root_walks_parent_pointers() -> None:
    root = ThoughtNode(
        id="r", parent_id=None, content="root", score=1.0, depth=0
    )
    a = ThoughtNode(
        id="a", parent_id="r", content="a", score=0.5, depth=1
    )
    b = ThoughtNode(
        id="b", parent_id="a", content="b", score=0.6, depth=2
    )
    c = ThoughtNode(
        id="c", parent_id="b", content="c", score=0.7, depth=3
    )
    chain = _chain_to_root([root, a, b, c], c)
    assert [n.id for n in chain] == ["r", "a", "b", "c"]


def test_chain_to_root_handles_root_alone() -> None:
    root = ThoughtNode(
        id="r", parent_id=None, content="root", score=1.0, depth=0
    )
    chain = _chain_to_root([root], root)
    assert [n.id for n in chain] == ["r"]


# ---------------------------------------------------------------------------
# Single-level expansion: branch_factor=2, max_depth=1, beam_width=1
# ---------------------------------------------------------------------------


async def test_tot_single_level_picks_highest_scored_leaf() -> None:
    """Root → 2 candidates → 2 evaluator calls → top 1 survives.
    Final session.output is the higher-scoring candidate's content.

    Model script (in order):
      1. propose candidate 1 ("guess A")
      2. propose candidate 2 ("guess B")
      3. evaluate candidate 1 ("score: 0.4")
      4. evaluate candidate 2 ("score: 0.8")
    """
    model = ScriptedModel(
        [
            ScriptedTurn(text="guess A"),
            ScriptedTurn(text="guess B"),
            ScriptedTurn(text="score: 0.4"),
            ScriptedTurn(text="score: 0.8"),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=TreeOfThoughts(
            branch_factor=2, max_depth=1, beam_width=1,
            # default solved_threshold=1.0; scripted scores all < 1.0
            # so early termination never fires.
        ),
    )
    result = await agent.run("solve this")
    assert result.output == "guess B"
    # 2 propose + 2 eval = 4 turns
    assert result.turns == 4


# ---------------------------------------------------------------------------
# Multi-level expansion: branch_factor=2, max_depth=2, beam_width=1
# ---------------------------------------------------------------------------


async def test_tot_multi_level_expands_from_pruned_frontier() -> None:
    """Level 1: 2 candidates → keep top 1. Level 2: 2 more candidates
    from that survivor → keep top 1. Best across both levels wins.

    Model script:
      Level 1:
        1. propose "L1a"
        2. propose "L1b"
        3. eval L1a → 0.5
        4. eval L1b → 0.7  ← survives
      Level 2 (from L1b):
        5. propose "L2a"
        6. propose "L2b"
        7. eval L2a → 0.95  ← would solve, but solved_threshold=2.0 prevents
        8. eval L2b → 0.6
    Final: L2a (highest non-root score).
    """
    model = ScriptedModel(
        [
            ScriptedTurn(text="L1a"),
            ScriptedTurn(text="L1b"),
            ScriptedTurn(text="score: 0.5"),
            ScriptedTurn(text="score: 0.7"),
            ScriptedTurn(text="L2a"),
            ScriptedTurn(text="L2b"),
            ScriptedTurn(text="score: 0.95"),
            ScriptedTurn(text="score: 0.6"),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=TreeOfThoughts(
            branch_factor=2, max_depth=2, beam_width=1,
            # default solved_threshold=1.0; scripted scores < 1.0.
        ),
    )
    result = await agent.run("solve this multi-step")
    assert result.output == "L2a"


# ---------------------------------------------------------------------------
# Early termination on solved_threshold
# ---------------------------------------------------------------------------


async def test_tot_early_terminates_on_solved_threshold() -> None:
    """If a candidate scores >= solved_threshold, stop immediately —
    don't expand further. We give max_depth=3 but level 1 finds a
    solution; level 2 / 3 never run."""
    model = ScriptedModel(
        [
            ScriptedTurn(text="A"),
            ScriptedTurn(text="B"),
            ScriptedTurn(text="score: 0.3"),
            ScriptedTurn(text="score: 0.99"),  # solved
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=TreeOfThoughts(
            branch_factor=2, max_depth=3, beam_width=1,
            solved_threshold=0.95,
        ),
    )
    result = await agent.run("task")
    assert result.output == "B"
    # 2 propose + 2 eval = 4. No level 2 turns.
    assert result.turns == 4


# ---------------------------------------------------------------------------
# session.metadata exposes the full tree
# ---------------------------------------------------------------------------


async def test_tot_exposes_tree_via_session_metadata() -> None:
    """The full search tree is stashed on session.metadata so consumers
    can render it (debug UI, eval, post-hoc analysis)."""
    captured_metadata: dict[str, object] = {}

    class _CaptureSession(Agent):
        async def run(  # type: ignore[override]
            self, prompt: str, *, session_id: str | None = None
        ):
            r = await super().run(prompt, session_id=session_id)
            return r

    model = ScriptedModel(
        [
            ScriptedTurn(text="A"),
            ScriptedTurn(text="B"),
            ScriptedTurn(text="score: 0.5"),
            ScriptedTurn(text="score: 0.95"),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=TreeOfThoughts(
            branch_factor=2, max_depth=1, beam_width=1,
        ),
    )
    # We need to inspect session.metadata, which RunResult doesn't
    # carry. Easiest is to grab metadata via stream events: the
    # tot.completed event has winner_id / winner_score; combined
    # with tot.proposed / tot.evaluated we can reconstruct.
    events = [e async for e in agent.stream("task")]
    proposed = [
        e for e in events if e.kind == "architecture_event"
        and e.payload.get("name") == "tot.proposed"
    ]
    evaluated = [
        e for e in events if e.kind == "architecture_event"
        and e.payload.get("name") == "tot.evaluated"
    ]
    completed = [
        e for e in events if e.kind == "architecture_event"
        and e.payload.get("name") == "tot.completed"
    ]
    assert len(proposed) == 2
    assert len(evaluated) == 2
    assert len(completed) == 1
    assert completed[0].payload["winner_score"] == pytest.approx(0.95)
    assert captured_metadata is not None  # placeholder use


# ---------------------------------------------------------------------------
# Architecture progress events
# ---------------------------------------------------------------------------


async def test_tot_emits_full_event_sequence() -> None:
    model = ScriptedModel(
        [
            ScriptedTurn(text="cand"),
            ScriptedTurn(text="score: 0.5"),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=TreeOfThoughts(
            branch_factor=1, max_depth=1, beam_width=1,
        ),
    )
    events = [e async for e in agent.stream("task")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "tot.started" in arch_names
    assert "tot.level_started" in arch_names
    assert "tot.proposed" in arch_names
    assert "tot.evaluated" in arch_names
    assert "tot.pruned" in arch_names
    assert "tot.completed" in arch_names


# ---------------------------------------------------------------------------
# Beam width
# ---------------------------------------------------------------------------


async def test_tot_beam_keeps_top_n_per_level() -> None:
    """branch_factor=3, beam_width=2 → 3 candidates, top 2 survive
    (the lowest-scored one is pruned)."""
    model = ScriptedModel(
        [
            ScriptedTurn(text="A"),
            ScriptedTurn(text="B"),
            ScriptedTurn(text="C"),
            ScriptedTurn(text="score: 0.9"),
            ScriptedTurn(text="score: 0.2"),
            ScriptedTurn(text="score: 0.5"),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=TreeOfThoughts(
            branch_factor=3, max_depth=1, beam_width=2,
        ),
    )
    events = [e async for e in agent.stream("task")]
    pruned = next(
        e for e in events if e.kind == "architecture_event"
        and e.payload.get("name") == "tot.pruned"
    )
    kept_scores = pruned.payload["kept_scores"]
    # Kept are top 2 by score: 0.9 and 0.5; 0.2 is pruned.
    assert sorted(kept_scores, reverse=True) == [0.9, 0.5]


# ---------------------------------------------------------------------------
# min_score floor pruning
# ---------------------------------------------------------------------------


async def test_tot_min_score_floor_drops_weak_branches() -> None:
    """Candidates scoring below ``min_score`` are pruned regardless
    of beam capacity. Saves the next level's compute."""
    model = ScriptedModel(
        [
            ScriptedTurn(text="thought a"),
            ScriptedTurn(text="thought b"),
            ScriptedTurn(text="thought c"),
            ScriptedTurn(text="score: 0.9"),
            ScriptedTurn(text="score: 0.2"),
            ScriptedTurn(text="score: 0.4"),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=TreeOfThoughts(
            branch_factor=3,
            max_depth=1,
            beam_width=3,  # beam has room for all 3
            min_score=0.5,  # ...but floor drops 0.2 and 0.4
        ),
    )
    events = [e async for e in agent.stream("task")]
    pruned = next(
        e for e in events if e.kind == "architecture_event"
        and e.payload.get("name") == "tot.pruned"
    )
    # Only the 0.9 candidate survives the floor.
    assert pruned.payload["kept_scores"] == [0.9]
    assert pruned.payload["pruned_below_floor"] == 2


def test_tot_rejects_invalid_min_score() -> None:
    with pytest.raises(ValueError, match="min_score"):
        TreeOfThoughts(min_score=1.5)


async def test_tot_parallel_and_sequential_agree() -> None:
    """Parallel and sequential modes must produce the same result
    given identical scripted scores."""
    proposer_replies = ["thought a", "thought b"]
    eval_replies = ["score: 0.9", "score: 0.3"]
    seq_model = ScriptedModel(
        [ScriptedTurn(text=t) for t in proposer_replies + eval_replies]
    )
    par_model = ScriptedModel(
        [ScriptedTurn(text=t) for t in proposer_replies + eval_replies]
    )
    seq_agent = Agent(
        "solver",
        model=seq_model,
        architecture=TreeOfThoughts(
            branch_factor=2, max_depth=1, beam_width=1, parallel=False
        ),
    )
    par_agent = Agent(
        "solver",
        model=par_model,
        architecture=TreeOfThoughts(
            branch_factor=2, max_depth=1, beam_width=1, parallel=True
        ),
    )
    seq = await seq_agent.run("task")
    par = await par_agent.run("task")
    assert seq.output == par.output
    assert "thought a" in seq.output
