"""Tree of Thoughts: branching exploration with per-node evaluation.

Yao et al. 2023 — `Tree of Thoughts: Deliberate Problem Solving with
Large Language Models <https://arxiv.org/abs/2305.10601>`_. Useful
for combinatorial reasoning, multi-step planning, math (Game of 24),
puzzle solving — anywhere a single straight-shot ReAct trajectory
would commit too early.

Pattern (BFS beam search)
-------------------------

1. **Root** is the problem statement.
2. **For each level up to ``max_depth``:**

   a. **Expand:** for every frontier node, the proposer generates
      ``branch_factor`` candidate "thoughts" (next steps toward a
      solution).
   b. **Evaluate:** the evaluator scores each candidate 0-1 (how
      promising is this branch?).
   c. **Prune:** keep only the top ``beam_width`` scored
      candidates as the next frontier.
   d. **Early exit:** if any candidate scores ``>= solved_threshold``,
      we stop early and use that branch.

3. **Best leaf wins.** The highest-scoring leaf across the whole
   tree is the final answer (its content goes to ``session.output``).

This is the "BFS-with-beam" variant — DFS with backtracking is a
follow-up. For a structured combinatorial task, BFS-beam covers most
of what users need.

Cost
----
``branch_factor × beam_width × max_depth × 2`` model calls (one
proposer + one evaluator per candidate). With defaults
``(3, 2, 3)`` that's 36 calls. Reserve ToT for problems where the
search structure earns the cost — math/planning tasks where ReAct
visibly meanders.

Strengths
---------
* **Explicit search tree.** Every candidate, score, and decision is
  observable through ``architecture_event`` events.
* **Composable.** Wrap inside :class:`Reflexion` to learn which
  evaluation patterns predict real success.
* **Replay-correct.** Each proposer / evaluator call is a named
  ``runtime.step``, so journaled runtimes replay deterministically.

Weaknesses
----------
* **Expensive.** 30-50× a single ReAct turn for typical settings.
* **Evaluator-quality bound.** A weak evaluator picks weak branches
  and the search wastes budget on dead ends.
* **Domain-specific.** Branch-and-evaluate makes sense for
  combinatorial problems; for open-ended writing tasks, use
  Self-Refine or Actor-Critic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import anyio
from pydantic import BaseModel

from ..core.ids import new_id
from ..core.types import Event, Message, Role, Usage
from .base import AgentSession, Dependencies
from .helpers import add_usage, parse_score, text_only_model_call

if TYPE_CHECKING:
    from ..agent.api import Agent


DEFAULT_PROPOSER_PROMPT = """\
You are exploring possible reasoning paths to solve a problem.

Given the problem and any prior steps, propose ONE next step (a
"thought") toward a solution. A thought can be a sub-step,
intermediate calculation, sub-decision, or partial answer.

Output only the thought itself — concise, one paragraph at most.
Do not number it; do not preface with "Thought:".
"""


DEFAULT_EVALUATOR_PROMPT = """\
You evaluate a candidate reasoning step. Given the original problem
and the proposed thought, score how promising this thought is for
arriving at the correct solution.

Output exactly one line:
score: <number between 0 and 1>

Then optionally one line of brief justification. The first line
must match the score format exactly so it can be parsed.

- 1.0 = this thought is correct and final / will obviously lead to a
  correct answer
- 0.7-0.9 = strong direction, likely correct
- 0.4-0.6 = plausible but uncertain
- 0.0-0.3 = wrong direction or contradicts the problem
"""


class ThoughtNode(BaseModel):
    """One node in the Tree-of-Thoughts search tree.

    Children are stored implicitly (each node has a ``parent_id``).
    The full tree is reconstructable from the node list ToT keeps in
    its session metadata.
    """

    id: str
    parent_id: str | None
    content: str
    score: float = 0.0
    depth: int


class TreeOfThoughts:
    """Branch + evaluate + prune. BFS beam search over thoughts."""

    name = "tree-of-thoughts"

    def __init__(
        self,
        *,
        branch_factor: int = 3,
        max_depth: int = 3,
        beam_width: int = 2,
        solved_threshold: float = 1.0,
        min_score: float = 0.0,
        parallel: bool = True,
        proposer_prompt: str | None = None,
        evaluator_prompt: str | None = None,
    ) -> None:
        if branch_factor < 1:
            raise ValueError("branch_factor must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if not 0.0 <= solved_threshold <= 1.0:
            raise ValueError(
                "solved_threshold must be in [0.0, 1.0]"
            )
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be in [0.0, 1.0]")
        self._branch_factor = branch_factor
        self._max_depth = max_depth
        self._beam_width = beam_width
        self._solved_threshold = solved_threshold
        # Floor below which a candidate is dropped REGARDLESS of beam
        # capacity. Lets bad branches die quickly instead of riding
        # along just because the beam has room. 0.0 = legacy behavior
        # (no floor).
        self._min_score = min_score
        # Run proposer + evaluator calls within a level concurrently
        # via anyio.create_task_group. Pure speedup — branch_factor *
        # beam_width independent calls are now wall-clock parallel
        # instead of sequential. Disable for deterministic test
        # ordering or when your model provider has tight rate limits.
        self._parallel = parallel
        self._proposer_prompt = (
            proposer_prompt or DEFAULT_PROPOSER_PROMPT
        )
        self._evaluator_prompt = (
            evaluator_prompt or DEFAULT_EVALUATOR_PROMPT
        )

    def declared_workers(self) -> dict[str, Agent]:
        return {}

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # Root represents the problem itself; depth 0; no model call.
        root = ThoughtNode(
            id=new_id("thot"),
            parent_id=None,
            content=prompt,
            score=1.0,  # root is "perfect" by definition
            depth=0,
        )
        all_nodes: list[ThoughtNode] = [root]
        frontier: list[ThoughtNode] = [root]

        yield Event.architecture_event(
            session.id,
            "tot.started",
            branch_factor=self._branch_factor,
            beam_width=self._beam_width,
            max_depth=self._max_depth,
        )

        for depth in range(1, self._max_depth + 1):
            status = await deps.budget.allows_step()
            if status.blocked:
                session.interrupted = True
                session.interruption_reason = (
                    f"budget:{status.reason}"
                )
                yield Event.budget_exceeded(session.id, status)
                break
            if status.warn:
                yield Event.budget_warning(session.id, status)

            yield Event.architecture_event(
                session.id,
                "tot.level_started",
                depth=depth,
                frontier_size=len(frontier),
            )

            # === Expand: generate branch_factor candidates per node ===
            #
            # Parallel mode runs every (parent × k) proposer call
            # concurrently — independent calls, big wall-clock win
            # at moderate fan-out. Sequential mode preserves
            # deterministic ordering for tests / strict rate limits.
            candidates: list[ThoughtNode] = []
            propose_jobs = [
                (parent, k)
                for parent in frontier
                for k in range(self._branch_factor)
            ]
            propose_results: list[tuple[str, Usage] | None] = [
                None
            ] * len(propose_jobs)

            # B023 false positive: the task_group below joins on
            # all spawned tasks before the for-loop advances, so the
            # captured ``depth`` / ``propose_results`` are stable
            # for the closure's entire lifetime.
            async def _propose_one(  # noqa: B023
                idx: int, parent: ThoughtNode, k: int
            ) -> None:
                chain = _chain_to_root(all_nodes, parent)
                msgs = _proposer_messages(
                    self._proposer_prompt, prompt, chain
                )
                text, usage = await text_only_model_call(
                    deps,
                    f"tot_propose_d{depth}_p{parent.id}_k{k}",  # noqa: B023
                    msgs,
                )
                propose_results[idx] = (text, usage)  # noqa: B023

            if self._parallel:
                async with anyio.create_task_group() as tg:
                    for idx, (parent, k) in enumerate(propose_jobs):
                        tg.start_soon(_propose_one, idx, parent, k)
            else:
                for idx, (parent, k) in enumerate(propose_jobs):
                    await _propose_one(idx, parent, k)

            for (parent, _k), pr in zip(
                propose_jobs, propose_results, strict=True
            ):
                assert pr is not None
                text, usage = pr
                await deps.budget.consume(
                    tokens_in=usage.input_tokens,
                    tokens_out=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                )
                session.cumulative_usage = add_usage(
                    session.cumulative_usage, usage
                )
                session.turns += 1
                candidate = ThoughtNode(
                    id=new_id("thot"),
                    parent_id=parent.id,
                    content=text.strip(),
                    depth=depth,
                )
                candidates.append(candidate)
                all_nodes.append(candidate)
                yield Event.architecture_event(
                    session.id,
                    "tot.proposed",
                    depth=depth,
                    node_id=candidate.id,
                    parent_id=parent.id,
                    content=candidate.content[:200],
                )

            # === Evaluate every candidate (parallel where possible) ===
            eval_results: list[tuple[float, Usage] | None] = [
                None
            ] * len(candidates)

            # Same B023-safe pattern as the proposer task group above.
            async def _eval_one(idx: int, cand: ThoughtNode) -> None:  # noqa: B023
                chain = _chain_to_root(all_nodes, cand)
                msgs = _evaluator_messages(
                    self._evaluator_prompt, prompt, chain, cand
                )
                text, usage = await text_only_model_call(
                    deps, f"tot_eval_d{depth}_n{cand.id}", msgs  # noqa: B023
                )
                eval_results[idx] = (parse_score(text), usage)  # noqa: B023

            if self._parallel:
                async with anyio.create_task_group() as tg:
                    for idx, cand in enumerate(candidates):
                        tg.start_soon(_eval_one, idx, cand)
            else:
                for idx, cand in enumerate(candidates):
                    await _eval_one(idx, cand)

            for cand, er in zip(
                candidates, eval_results, strict=True
            ):
                assert er is not None
                score, usage = er
                await deps.budget.consume(
                    tokens_in=usage.input_tokens,
                    tokens_out=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                )
                session.cumulative_usage = add_usage(
                    session.cumulative_usage, usage
                )
                session.turns += 1
                cand.score = score
                yield Event.architecture_event(
                    session.id,
                    "tot.evaluated",
                    depth=depth,
                    node_id=cand.id,
                    score=cand.score,
                )

            # === Prune: top beam_width by score, AND drop anything
            # below the min_score floor. The floor lets a clearly-
            # losing branch die immediately even if the beam has
            # room — saves the next level's compute. ===
            candidates.sort(key=lambda n: n.score, reverse=True)
            survivors = [
                c for c in candidates if c.score >= self._min_score
            ]
            frontier = survivors[: self._beam_width]
            n_pruned_floor = len(candidates) - len(survivors)
            yield Event.architecture_event(
                session.id,
                "tot.pruned",
                depth=depth,
                kept=[n.id for n in frontier],
                kept_scores=[n.score for n in frontier],
                pruned_below_floor=n_pruned_floor,
            )

            # === Early exit if any candidate is "solved" ===
            if frontier and frontier[0].score >= self._solved_threshold:
                yield Event.architecture_event(
                    session.id,
                    "tot.solved",
                    depth=depth,
                    node_id=frontier[0].id,
                    score=frontier[0].score,
                )
                break

            if not frontier:
                # Beam went empty (shouldn't happen unless candidates
                # was empty too). Bail with whatever's best so far.
                yield Event.architecture_event(
                    session.id,
                    "tot.empty_beam",
                    depth=depth,
                )
                break

        # Pick the best non-root node we've seen across the whole tree.
        non_root = [n for n in all_nodes if n.parent_id is not None]
        if not non_root:
            session.output = ""
            yield Event.architecture_event(
                session.id,
                "tot.no_thoughts",
                total_nodes=len(all_nodes),
            )
            return
        best = max(non_root, key=lambda n: n.score)
        session.output = best.content
        # Stash the full tree on session.metadata so consumers can
        # render the search tree post-hoc.
        session.metadata["tot_nodes"] = [
            n.model_dump() for n in all_nodes
        ]
        session.metadata["tot_winner_id"] = best.id

        yield Event.architecture_event(
            session.id,
            "tot.completed",
            winner_id=best.id,
            winner_score=best.score,
            total_nodes=len(all_nodes),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_to_root(
    all_nodes: list[ThoughtNode], leaf: ThoughtNode
) -> list[ThoughtNode]:
    """Reconstruct the chain from root to ``leaf`` by parent pointers.

    Returns root first, leaf last.
    """
    by_id = {n.id: n for n in all_nodes}
    chain: list[ThoughtNode] = []
    cursor: ThoughtNode | None = leaf
    while cursor is not None:
        chain.append(cursor)
        if cursor.parent_id is None:
            break
        cursor = by_id.get(cursor.parent_id)
    return list(reversed(chain))


def _proposer_messages(
    system_prompt: str, problem: str, chain: list[ThoughtNode]
) -> list[Message]:
    """Build messages for a proposer call.

    The chain from root to current frontier node provides the
    "prior steps so far"; the proposer extends with one new thought.
    """
    # Drop the root (which holds the original prompt) since we send
    # the prompt explicitly.
    prior_steps = [n for n in chain if n.parent_id is not None]
    prior_text = (
        "\n".join(
            f"Step {i + 1}: {n.content}"
            for i, n in enumerate(prior_steps)
        )
        if prior_steps
        else "(no prior steps yet — propose the first one)"
    )
    return [
        Message(role=Role.SYSTEM, content=system_prompt),
        Message(
            role=Role.USER,
            content=(
                f"Problem:\n{problem}\n\n"
                f"Prior steps:\n{prior_text}\n\n"
                f"Propose ONE next step."
            ),
        ),
    ]


def _evaluator_messages(
    system_prompt: str,
    problem: str,
    chain: list[ThoughtNode],
    candidate: ThoughtNode,
) -> list[Message]:
    """Build messages for an evaluator call.

    The chain shows prior steps; the candidate is the new step
    being evaluated.
    """
    # Chain includes the candidate at the end; the prior chain is
    # everything before the candidate.
    prior = [
        n
        for n in chain
        if n.parent_id is not None and n.id != candidate.id
    ]
    prior_text = (
        "\n".join(
            f"Step {i + 1}: {n.content}"
            for i, n in enumerate(prior)
        )
        if prior
        else "(none)"
    )
    return [
        Message(role=Role.SYSTEM, content=system_prompt),
        Message(
            role=Role.USER,
            content=(
                f"Problem:\n{problem}\n\n"
                f"Prior steps:\n{prior_text}\n\n"
                f"Candidate next step:\n{candidate.content}\n\n"
                f"Score this candidate."
            ),
        ),
    ]
