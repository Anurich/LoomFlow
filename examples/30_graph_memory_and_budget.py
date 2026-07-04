"""30_graph_memory_and_budget.py — multi-hop graph memory + token-budgeted injection.

Two memory upgrades that shipped in v0.11:

**Graph memory** — facts are bi-temporal SPO triples; ``recall_graph``
builds a directed entity graph from them and BFS-traverses it, so
*multi-hop* questions flat recall can't answer become answerable::

    alice —works_at→ acme —operates_in→ tokyo

The traversal is **point-in-time correct**: when a fact is superseded
(alice changes jobs), a current query walks the NEW edge while a
``valid_at=<past>`` query still walks the edge that was true then —
the same bi-temporal semantics the fact store already enforces.

**Token-budgeted injection** — memory injection used to be item-count
based (5 facts / 3 episodes). With ``Tuning(memory_token_budget=...)``
the seed's recalled-memory block is filled greedily by
``relevance x recency-decay`` score under a hard token budget, with
the first non-fitting item truncated and pinned working blocks never
dropped::

    Agent(..., tuning=Tuning(
        memory_token_budget=800,
        memory_decay_half_life_days=30,
    ))

This example runs OFFLINE (no API key) against the in-memory fact
store and ``EchoModel``.

Run with::

    python examples/30_graph_memory_and_budget.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import anyio

from loomflow import Agent, EchoModel, Episode, Fact, InMemoryMemory, Tuning
from loomflow.memory import recall_graph
from loomflow.memory.facts import InMemoryFactStore


async def graph_demo() -> None:
    print("=" * 60)
    print("1) Graph memory — 2-hop, point-in-time correct")
    print("=" * 60)

    store = InMemoryFactStore()
    jan = datetime(2026, 1, 1, tzinfo=UTC)
    jul = datetime(2026, 7, 1, tzinfo=UTC)

    # Alice's world in January…
    await store.append(
        Fact(user_id="alice", subject="alice", predicate="works_at",
             object="acme", valid_from=jan)
    )
    await store.append(
        Fact(user_id="alice", subject="acme", predicate="operates_in",
             object="tokyo", valid_from=jan)
    )
    # …in July she changes jobs. Appending the same (subject, predicate)
    # SUPERSEDES the old fact (closes its valid_until) — bi-temporal.
    await store.append(
        Fact(user_id="alice", subject="alice", predicate="works_at",
             object="initech", valid_from=jul)
    )
    await store.append(
        Fact(user_id="alice", subject="initech", predicate="operates_in",
             object="austin", valid_from=jul)
    )
    # Bob's facts live in a different partition — never in alice's graph.
    await store.append(
        Fact(user_id="bob", subject="bob", predicate="works_at",
             object="globex", valid_from=jan)
    )

    query = "where does alice's employer operate?"

    print(f'  query: "{query}"')
    print("  — current (valid_at=None):")
    for path in await recall_graph(store, query, user_id="alice", hops=2):
        print(f"      {path.render()}")

    march = datetime(2026, 3, 15, tzinfo=UTC)
    print(f"  — as of {march.date()} (valid_at=<past>):")
    for path in await recall_graph(
        store, query, user_id="alice", hops=2, valid_at=march
    ):
        print(f"      {path.render()}")


class _RecordingEcho:
    """Wrap EchoModel's stream so we can inspect the seeded prompt."""

    name = "recording-echo"

    def __init__(self) -> None:
        self._inner = EchoModel()
        self.seen_messages: list[Any] = []

    def stream(self, messages: Any, **kwargs: Any) -> Any:
        self.seen_messages = list(messages)
        return self._inner.stream(messages, **kwargs)


def _memory_block_chars(messages: list[Any]) -> int:
    return sum(
        len(m.content or "")
        for m in messages
        if "memory" in (m.content or "").lower()
        or "episode" in (m.content or "").lower()
    )


async def budget_demo() -> None:
    print()
    print("=" * 60)
    print("2) Token-budgeted memory injection")
    print("=" * 60)

    async def seeded_memory() -> InMemoryMemory:
        memory = InMemoryMemory()
        for i in range(3):
            await memory.remember(
                Episode(
                    session_id=f"s{i}",
                    user_id="alice",
                    input=f"deployment question #{i}",
                    output="deployment runbook step " * 400,  # oversized
                )
            )
        return memory

    # Unbudgeted: every recalled episode lands in the seed verbatim.
    plain_model = _RecordingEcho()
    plain = Agent("You are helpful.", model=plain_model,
                  memory=await seeded_memory())
    await plain.run("how do we deploy?", user_id="alice")

    # Budgeted: the recalled block is ranked, decayed, and hard-capped.
    tight_model = _RecordingEcho()
    tight = Agent(
        "You are helpful.",
        model=tight_model,
        memory=await seeded_memory(),
        tuning=Tuning(memory_token_budget=400,
                      memory_decay_half_life_days=30),
    )
    await tight.run("how do we deploy?", user_id="alice")

    print(f"  unbudgeted memory block : {_memory_block_chars(plain_model.seen_messages):>6} chars")
    print(f"  budgeted (400 tokens)   : {_memory_block_chars(tight_model.seen_messages):>6} chars")
    print("  Working blocks are pinned (never dropped); episodes/facts are")
    print("  ranked by relevance x recency-decay and greedily fitted.")


async def main() -> None:
    await graph_demo()
    await budget_demo()


if __name__ == "__main__":
    anyio.run(main)
