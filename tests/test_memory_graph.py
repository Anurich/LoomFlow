"""Temporal fact graph tests (G11): build, traversal, bi-temporal edges.

Covers: graph construction + entity normalization/aliases, neighbor
direction + temporal filtering, 2-hop resolution that flat recall
misses, supersession as edge invalidation (point-in-time replay),
path ranking, seed extraction, ``recall_graph`` end-to-end against
InMemoryFactStore and SqliteFactStore, per-tenant isolation, explicit
``merge()`` aliasing, and limit caps.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path as FsPath

import pytest

from loomflow.core.types import Fact
from loomflow.memory import (
    Edge,
    FactGraph,
    InMemoryFactStore,
    SqliteFactStore,
    recall_graph,
)

pytestmark = pytest.mark.anyio

_T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _fact(
    subject: str,
    predicate: str,
    object_: str,
    *,
    user_id: str | None = None,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    recorded_at: datetime | None = None,
) -> Fact:
    return Fact(
        user_id=user_id,
        subject=subject,
        predicate=predicate,
        object=object_,
        valid_from=valid_from or _T0,
        valid_until=valid_until,
        recorded_at=recorded_at or (valid_from or _T0),
    )


def _two_hop_facts(user_id: str | None = None) -> list[Fact]:
    return [
        _fact("alice", "works_at", "acme", user_id=user_id, valid_from=_T0),
        _fact(
            "acme",
            "operates_in",
            "tokyo",
            user_id=user_id,
            valid_from=_T0 + timedelta(days=1),
        ),
    ]


# ---------------------------------------------------------------------------
# Construction, normalization, aliases
# ---------------------------------------------------------------------------


async def test_from_facts_normalizes_entities_and_keeps_aliases() -> None:
    graph = FactGraph.from_facts(
        [
            _fact("  Alice  Smith ", "works_at", "ACME"),
            _fact("alice smith", "lives_in", "Tokyo"),
        ]
    )
    # Both spellings collapse to one node with both raws as aliases.
    assert "alice smith" in graph.entities
    assert graph.aliases["alice smith"] == {"  Alice  Smith ", "alice smith"}
    assert graph.aliases["acme"] == {"ACME"}
    out = graph.neighbors("Alice   SMITH", direction="out")
    assert {e.object for e in out} == {"acme", "tokyo"}


async def test_add_fact_skips_empty_entities() -> None:
    graph = FactGraph()
    assert graph.add_fact(_fact("   ", "works_at", "acme")) is None
    assert graph.add_fact(_fact("alice", "works_at", "acme")) is not None
    assert graph.entities == {"alice", "acme"}


# ---------------------------------------------------------------------------
# Neighbors: direction + temporal filtering
# ---------------------------------------------------------------------------


async def test_neighbors_direction() -> None:
    graph = FactGraph.from_facts(_two_hop_facts())
    out = graph.neighbors("acme", direction="out")
    assert [e.object for e in out] == ["tokyo"]
    in_ = graph.neighbors("acme", direction="in")
    assert [e.subject for e in in_] == ["alice"]
    both = graph.neighbors("acme", direction="both")
    assert {(e.subject, e.object) for e in both} == {("acme", "tokyo"), ("alice", "acme")}
    with pytest.raises(ValueError):
        graph.neighbors("acme", direction="sideways")


async def test_neighbors_temporal_filtering() -> None:
    closed = _fact(
        "alice",
        "works_at",
        "acme",
        valid_from=_T0,
        valid_until=_T0 + timedelta(days=10),  # closed in the past
    )
    open_ = _fact("alice", "works_at", "initech", valid_from=_T0 + timedelta(days=10))
    graph = FactGraph.from_facts([closed, open_])

    # valid_at=None → currently valid only: the closed edge is excluded.
    now_edges = graph.neighbors("alice", direction="out")
    assert [e.object for e in now_edges] == ["initech"]

    # Point-in-time inside the closed window → the old edge is back.
    then = _T0 + timedelta(days=5)
    then_edges = graph.neighbors("alice", direction="out", valid_at=then)
    assert [e.object for e in then_edges] == ["acme"]

    # Before valid_from → nothing.
    assert graph.neighbors("alice", valid_at=_T0 - timedelta(days=1)) == []

    # A future valid_until still counts as currently valid.
    future = _fact(
        "alice",
        "visits",
        "osaka",
        valid_from=_T0,
        valid_until=datetime.now(UTC) + timedelta(days=365),
    )
    graph.add_fact(future)
    assert any(e.object == "osaka" for e in graph.neighbors("alice", direction="out"))


# ---------------------------------------------------------------------------
# Multi-hop traversal
# ---------------------------------------------------------------------------


async def test_two_hop_resolution() -> None:
    """The G11 acceptance case: 'where does alice's employer operate?'"""
    graph = FactGraph.from_facts(_two_hop_facts())
    paths = graph.traverse(["alice"], hops=2)
    rendered = [p.render() for p in paths]
    assert "alice —works_at→ acme —operates_in→ tokyo" in rendered
    # A 1-hop traversal (≈ flat recall) never reaches tokyo.
    one_hop = graph.traverse(["alice"], hops=1)
    assert all("tokyo" not in p.render() for p in one_hop)


async def test_traverse_backward_direction_renders_reversed() -> None:
    graph = FactGraph.from_facts(_two_hop_facts())
    paths = graph.traverse(["tokyo"], hops=2)
    rendered = [p.render() for p in paths]
    assert "tokyo ←operates_in— acme ←works_at— alice" in rendered


async def test_path_nodes_and_facts() -> None:
    graph = FactGraph.from_facts(_two_hop_facts())
    paths = graph.traverse(["alice"], hops=2)
    full = [p for p in paths if len(p.edges) == 2][0]
    assert full.nodes == ("alice", "acme", "tokyo")
    assert [f.predicate for f in full.facts] == ["works_at", "operates_in"]


async def test_traverse_supersession_point_in_time() -> None:
    """Supersession closes the old edge for 'now' but keeps it for
    point-in-time queries inside its historical window."""
    store = InMemoryFactStore()
    t_old = _T0
    t_new = _T0 + timedelta(days=30)
    await store.append(_fact("alice", "works_at", "acme", valid_from=t_old, recorded_at=t_old))
    await store.append(_fact("alice", "works_at", "initech", valid_from=t_new, recorded_at=t_new))
    await store.append(_fact("acme", "operates_in", "tokyo", valid_from=t_old, recorded_at=t_old))
    await store.append(
        _fact("initech", "operates_in", "austin", valid_from=t_old, recorded_at=t_old)
    )
    facts = await store.query(user_id=None, limit=100)
    graph = FactGraph.from_facts(facts)

    # Current traversal goes through initech, not acme.
    now_renders = [p.render() for p in graph.traverse(["alice"], hops=2)]
    assert any("initech —operates_in→ austin" in r for r in now_renders)
    assert all("works_at→ acme" not in r for r in now_renders)

    # Replay inside the old window: acme is back, initech doesn't exist yet.
    then = t_old + timedelta(days=5)
    then_renders = [p.render() for p in graph.traverse(["alice"], hops=2, valid_at=then)]
    assert any("acme —operates_in→ tokyo" in r for r in then_renders)
    assert all("initech" not in r for r in then_renders)


async def test_path_ranking_shorter_then_newer() -> None:
    graph = FactGraph.from_facts(
        [
            _fact("alice", "knows", "bob", recorded_at=_T0),
            _fact("alice", "works_at", "acme", recorded_at=_T0 + timedelta(days=2)),
            _fact("acme", "operates_in", "tokyo", recorded_at=_T0 + timedelta(days=1)),
        ]
    )
    paths = graph.traverse(["alice"], hops=2)
    lengths = [len(p.edges) for p in paths]
    assert lengths == sorted(lengths), "shorter paths must rank first"
    # Among the 1-hop paths, the newer fact (works_at) outranks the older.
    one_hop = [p for p in paths if len(p.edges) == 1]
    assert one_hop[0].edges[0].predicate == "works_at"
    assert one_hop[1].edges[0].predicate == "knows"


async def test_traverse_limit_caps_results() -> None:
    facts = [_fact("hub", "links_to", f"node{i}") for i in range(20)]
    graph = FactGraph.from_facts(facts)
    assert len(graph.traverse(["hub"], hops=2, limit=5)) == 5
    assert len(graph.traverse(["hub"], hops=2, limit=50)) == 20
    assert graph.traverse(["hub"], hops=0) == []


async def test_traverse_unknown_seed_is_empty() -> None:
    graph = FactGraph.from_facts(_two_hop_facts())
    assert graph.traverse(["zorp"], hops=2) == []
    assert graph.traverse([], hops=2) == []


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


async def test_seed_entities_token_overlap() -> None:
    graph = FactGraph.from_facts(
        [
            _fact("Alice Smith", "works_at", "acme"),
            _fact("bob", "lives_in", "tokyo"),
        ]
    )
    assert graph.seed_entities("where does Alice's employer operate?") == ["alice smith"]
    # Tokens shorter than 3 chars never match ("bo" ≠ bob).
    assert graph.seed_entities("bo") == []
    assert graph.seed_entities("") == []
    # Multiple seeds, sorted.
    assert graph.seed_entities("alice and bob") == ["alice smith", "bob"]


# ---------------------------------------------------------------------------
# Explicit aliasing (merge)
# ---------------------------------------------------------------------------


async def test_merge_unifies_entities() -> None:
    graph = FactGraph.from_facts(
        [
            _fact("bob", "likes", "sushi"),
            _fact("Robert", "works_at", "acme"),
            _fact("acme", "operates_in", "tokyo"),
        ]
    )
    # Before the merge, bob's traversal never reaches acme.
    assert all("acme" not in p.render() for p in graph.traverse(["bob"], hops=2))

    canonical = graph.merge("bob", "Robert")
    assert canonical == "bob"
    merged_out = graph.neighbors("robert", direction="out")
    assert {e.object for e in merged_out} == {"sushi", "acme"}
    renders = [p.render() for p in graph.traverse(["bob"], hops=2)]
    assert any("acme —operates_in→ tokyo" in r for r in renders)
    # Idempotent and transitive-safe.
    assert graph.merge("bob", "robert") == "bob"
    with pytest.raises(ValueError):
        graph.merge("bob", "   ")


# ---------------------------------------------------------------------------
# recall_graph end-to-end (protocol-only, both backends)
# ---------------------------------------------------------------------------


async def _seed_store(store: InMemoryFactStore | SqliteFactStore) -> None:
    for fact in _two_hop_facts(user_id="alice-tenant"):
        await store.append(fact)
    # Another tenant's facts — must never leak into alice's graph.
    await store.append(_fact("alice", "works_at", "globex", user_id="bob-tenant"))
    await store.append(_fact("globex", "operates_in", "berlin", user_id="bob-tenant"))


async def test_recall_graph_inmemory_end_to_end() -> None:
    store = InMemoryFactStore()
    await _seed_store(store)
    paths = await recall_graph(
        store, "where does alice's employer operate?", user_id="alice-tenant"
    )
    renders = [p.render() for p in paths]
    assert "alice —works_at→ acme —operates_in→ tokyo" in renders
    assert all("globex" not in r and "berlin" not in r for r in renders)


async def test_recall_graph_sqlite_end_to_end(tmp_path: FsPath) -> None:
    store = SqliteFactStore(tmp_path / "facts.db")
    await _seed_store(store)
    try:
        paths = await recall_graph(
            store, "where does alice's employer operate?", user_id="alice-tenant"
        )
        renders = [p.render() for p in paths]
        assert "alice —works_at→ acme —operates_in→ tokyo" in renders
        assert all("globex" not in r for r in renders)
        # Per-tenant isolation cuts both ways.
        bob_paths = await recall_graph(store, "alice employer", user_id="bob-tenant")
        bob_renders = [p.render() for p in bob_paths]
        assert any("globex" in r for r in bob_renders)
        assert all("acme" not in r and "tokyo" not in r for r in bob_renders)
    finally:
        await store.aclose()


async def test_recall_graph_no_seed_match_returns_empty() -> None:
    store = InMemoryFactStore()
    await store.append(_fact("alice", "works_at", "acme"))
    assert await recall_graph(store, "zzz qqq") == []


async def test_recall_graph_limit_and_valid_at() -> None:
    store = InMemoryFactStore()
    t_old, t_new = _T0, _T0 + timedelta(days=30)
    await store.append(_fact("alice", "works_at", "acme", valid_from=t_old, recorded_at=t_old))
    await store.append(_fact("alice", "works_at", "initech", valid_from=t_new, recorded_at=t_new))
    await store.append(_fact("alice", "likes", "sushi", valid_from=t_old, recorded_at=t_old))

    # limit caps the ranked result set.
    assert len(await recall_graph(store, "alice", limit=1)) == 1

    # Current recall: initech, never acme.
    now_renders = [p.render() for p in await recall_graph(store, "alice")]
    assert any("initech" in r for r in now_renders)
    assert all("acme" not in r for r in now_renders)

    # Point-in-time recall inside the superseded window: acme, never initech.
    then_renders = [
        p.render() for p in await recall_graph(store, "alice", valid_at=t_old + timedelta(days=1))
    ]
    assert any("acme" in r for r in then_renders)
    assert all("initech" not in r for r in then_renders)


# ---------------------------------------------------------------------------
# Edge / Path dataclass surface
# ---------------------------------------------------------------------------


async def test_edge_and_path_render() -> None:
    fact = _fact("alice", "works_at", "acme")
    graph = FactGraph.from_facts([fact])
    edge = graph.neighbors("alice", direction="out")[0]
    assert isinstance(edge, Edge)
    assert edge.render() == "alice —works_at→ acme"
    assert edge.fact is fact
    from loomflow.memory import Path

    path = Path(edges=(edge,))
    assert path.render() == "alice —works_at→ acme"
    assert path.nodes == ("alice", "acme")
    with pytest.raises(ValueError):
        Path(edges=(edge,), directions=(True, False))
