"""Temporal fact graph — multi-hop traversal over bi-temporal SPO facts.

This is the in-process half of the Zep/Graphiti-style temporal knowledge
graph: :class:`~loomflow.core.types.Fact` triples become directed edges
between normalized entity nodes, and :meth:`FactGraph.traverse` answers
multi-hop questions that flat ``recall_text`` misses ("where does
Alice's employer operate?" — alice —works_at→ acme —operates_in→ tokyo).

Design notes (v1):

* **Store-agnostic.** The graph is built in process from facts fetched
  via the stable :class:`~loomflow.memory.facts.FactStore` protocol —
  no new tables, no extra dependencies, works against every backend.
  :func:`recall_graph` is the one-call convenience wrapper.
* **Entity linking is exact-normalized only.** Nodes are keyed by
  ``casefold + strip + collapse-whitespace`` of the raw subject/object
  text; every raw spelling is kept in :attr:`FactGraph.aliases`.
  Fuzzy matching (prefix/suffix, embeddings) is deliberately *not*
  attempted — silent wrong merges are worse than missed ones. When you
  know two spellings are the same entity, say so explicitly with
  :meth:`FactGraph.merge`.
* **Temporal correctness is the differentiator.** Every edge carries
  its fact's bi-temporal window. ``valid_at=None`` means "currently
  valid" (``valid_until`` unset or in the future); a concrete
  ``valid_at`` replays the graph as of that world-time instant, so a
  superseded fact is excluded from current traversals but *included*
  for point-in-time queries inside its window. This mirrors the
  ``valid_at`` semantics of :meth:`FactStore.query` — supersession is
  edge invalidation, never edge deletion.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..core.types import Fact
from .facts import FactStore, _is_valid_at

__all__ = ["Edge", "FactGraph", "Path", "recall_graph"]

# Safety valve for path enumeration on dense graphs: traverse() stops
# expanding after this many discovered paths (pre-ranking), regardless
# of ``limit``. At the documented v1 scale bound (``fetch_limit=2000``
# facts) this is never the binding constraint in practice.
_MAX_PATHS = 10_000


def _normalize_entity(raw: str) -> str:
    """Casefold + strip + collapse internal whitespace."""
    return " ".join(raw.casefold().split())


def _edge_valid(fact: Fact, valid_at: datetime | None) -> bool:
    """Is the fact's world-time window open at ``valid_at``?

    ``valid_at=None`` means *currently valid*: ``valid_until`` unset,
    or set to a future instant. A concrete ``valid_at`` uses the same
    half-open ``[valid_from, valid_until)`` window as ``FactStore.query``.
    """
    if valid_at is not None:
        return _is_valid_at(fact, valid_at)
    until = fact.valid_until
    if until is None:
        return True
    # Match naive/aware-ness of the stored timestamp (stores use
    # UTC-aware datetimes; naive values are treated as UTC).
    now = datetime.now(UTC)
    if until.tzinfo is None:
        now = now.replace(tzinfo=None)
    return until > now


def _query_tokens(text: str) -> set[str]:
    """Casefolded alphanumeric tokens of length >= 3."""
    out: set[str] = set()
    buf: list[str] = []
    for ch in text.casefold():
        if ch.isalnum():
            buf.append(ch)
        else:
            if len(buf) >= 3:
                out.add("".join(buf))
            buf = []
    if len(buf) >= 3:
        out.add("".join(buf))
    return out


@dataclass(frozen=True)
class Edge:
    """A directed edge: normalized subject —predicate→ normalized object.

    ``subject`` / ``object`` are the *normalized* node names as built;
    the backing :class:`Fact` retains the raw spellings. Edges merged
    via :meth:`FactGraph.merge` keep their original node names — merge
    affects traversal identity, not provenance.
    """

    subject: str
    predicate: str
    object: str
    fact: Fact

    def render(self) -> str:
        return f"{self.subject} —{self.predicate}→ {self.object}"


@dataclass(frozen=True)
class Path:
    """An ordered chain of edges discovered by :meth:`FactGraph.traverse`.

    ``directions[i]`` is ``True`` when ``edges[i]`` was traversed in its
    natural subject→object orientation, ``False`` for a backward hop.
    """

    edges: tuple[Edge, ...]
    directions: tuple[bool, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.directions and self.edges:
            # Default: every edge traversed forward.
            object.__setattr__(self, "directions", (True,) * len(self.edges))
        if len(self.directions) != len(self.edges):
            raise ValueError("directions must have one entry per edge")

    @property
    def facts(self) -> list[Fact]:
        return [e.fact for e in self.edges]

    @property
    def nodes(self) -> tuple[str, ...]:
        """Node names in traversal order (``len(edges) + 1`` entries)."""
        if not self.edges:
            return ()
        first, fwd0 = self.edges[0], self.directions[0]
        out = [first.subject if fwd0 else first.object]
        for edge, fwd in zip(self.edges, self.directions, strict=True):
            out.append(edge.object if fwd else edge.subject)
        return tuple(out)

    def render(self) -> str:
        """Human-readable chain: ``alice —works_at→ acme —operates_in→ tokyo``.

        Backward hops render with a reversed arrow so the direction of
        the underlying claim is never misstated:
        ``tokyo ←operates_in— acme``.
        """
        if not self.edges:
            return ""
        first, fwd0 = self.edges[0], self.directions[0]
        parts = [first.subject if fwd0 else first.object]
        for edge, fwd in zip(self.edges, self.directions, strict=True):
            if fwd:
                parts.append(f"—{edge.predicate}→ {edge.object}")
            else:
                parts.append(f"←{edge.predicate}— {edge.subject}")
        return " ".join(parts)


class FactGraph:
    """Directed multigraph over normalized entities, built from facts.

    Node identity is exact-normalized entity text plus any explicit
    :meth:`merge` calls. All raw spellings feeding a node are kept in
    :attr:`aliases`.
    """

    def __init__(self) -> None:
        # Adjacency keyed by build-time normalized node name.
        self._out: dict[str, list[Edge]] = {}
        self._in: dict[str, list[Edge]] = {}
        # normalized name -> raw spellings seen in facts.
        self._aliases: dict[str, set[str]] = {}
        # Explicit-alias union: normalized name -> canonical name, and
        # canonical name -> all member names (including itself).
        self._canon: dict[str, str] = {}
        self._groups: dict[str, set[str]] = {}

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_facts(cls, facts: Iterable[Fact]) -> FactGraph:
        graph = cls()
        for fact in facts:
            graph.add_fact(fact)
        return graph

    def add_fact(self, fact: Fact) -> Edge | None:
        """Add one fact as a directed edge. Returns the edge, or ``None``
        when subject or object normalizes to the empty string."""
        subject = _normalize_entity(fact.subject)
        object_ = _normalize_entity(fact.object)
        if not subject or not object_:
            return None
        self._aliases.setdefault(subject, set()).add(fact.subject)
        self._aliases.setdefault(object_, set()).add(fact.object)
        edge = Edge(subject=subject, predicate=fact.predicate, object=object_, fact=fact)
        self._out.setdefault(subject, []).append(edge)
        self._in.setdefault(object_, []).append(edge)
        return edge

    # ---- entity identity ---------------------------------------------------

    @property
    def aliases(self) -> dict[str, set[str]]:
        """Normalized entity name -> set of raw spellings seen."""
        return {name: set(raws) for name, raws in self._aliases.items()}

    @property
    def entities(self) -> set[str]:
        """All node names, resolved through explicit merges."""
        return {self._resolve(name) for name in self._aliases}

    def merge(self, a: str, b: str) -> str:
        """Declare that ``a`` and ``b`` are the same entity.

        This is the *explicit* aliasing API — v1 entity linking is
        exact-normalized-match only, because fuzzy rules silently merge
        distinct entities. Returns the canonical name (``a``'s side
        wins). Both arguments are normalized first; neither needs to
        exist in the graph yet.
        """
        canon_a = self._resolve(_normalize_entity(a))
        canon_b = self._resolve(_normalize_entity(b))
        if not canon_a or not canon_b:
            raise ValueError("merge() requires non-empty entity names")
        if canon_a == canon_b:
            return canon_a
        group_a = self._groups.setdefault(canon_a, {canon_a})
        group_b = self._groups.pop(canon_b, {canon_b})
        for member in group_b:
            self._canon[member] = canon_a
        group_a.update(group_b)
        return canon_a

    def _resolve(self, normalized: str) -> str:
        """Follow explicit merges to the canonical node name."""
        return self._canon.get(normalized, normalized)

    def _members(self, canonical: str) -> set[str]:
        """All build-time node names identified with ``canonical``."""
        return self._groups.get(canonical, {canonical})

    # ---- queries -----------------------------------------------------------

    def neighbors(
        self,
        entity: str,
        *,
        direction: str = "both",
        valid_at: datetime | None = None,
    ) -> list[Edge]:
        """Edges incident to ``entity``, filtered by temporal validity.

        ``direction``: ``"out"`` (entity is subject), ``"in"`` (entity
        is object), or ``"both"``. ``valid_at=None`` keeps only
        currently-valid edges; a concrete instant replays that moment.
        """
        if direction not in ("out", "in", "both"):
            raise ValueError(f"direction must be 'out', 'in', or 'both', got {direction!r}")
        canonical = self._resolve(_normalize_entity(entity))
        edges: list[Edge] = []
        seen: set[int] = set()
        for member in self._members(canonical):
            candidates: list[Edge] = []
            if direction in ("out", "both"):
                candidates.extend(self._out.get(member, ()))
            if direction in ("in", "both"):
                candidates.extend(self._in.get(member, ()))
            for edge in candidates:
                if id(edge) in seen:  # self-loops appear in both indexes
                    continue
                seen.add(id(edge))
                if _edge_valid(edge.fact, valid_at):
                    edges.append(edge)
        edges.sort(key=lambda e: e.fact.recorded_at, reverse=True)
        return edges

    def seed_entities(self, query: str) -> list[str]:
        """Entities whose raw/alias text shares a token (len >= 3,
        casefolded) with ``query``. Returns canonical names, sorted."""
        tokens = _query_tokens(query)
        if not tokens:
            return []
        seeds: set[str] = set()
        for name, raws in self._aliases.items():
            entity_tokens = _query_tokens(name)
            for raw in raws:
                entity_tokens |= _query_tokens(raw)
            if entity_tokens & tokens:
                seeds.add(self._resolve(name))
        return sorted(seeds)

    def traverse(
        self,
        seeds: list[str],
        *,
        hops: int = 2,
        valid_at: datetime | None = None,
        limit: int = 50,
    ) -> list[Path]:
        """BFS out to ``hops`` edges from each seed, both directions.

        Returns up to ``limit`` simple paths (no node revisited within
        a path), ranked shortest-first, ties broken by the recency of
        the newest fact on the path. Edges invalid at ``valid_at`` are
        never traversed (``None`` = currently valid).
        """
        if hops < 1:
            return []
        paths: list[Path] = []
        seen_paths: set[tuple[tuple[str, bool], ...]] = set()
        seed_nodes: list[str] = []
        for seed in seeds:
            canonical = self._resolve(_normalize_entity(seed))
            if canonical and canonical not in seed_nodes:
                seed_nodes.append(canonical)

        for start in seed_nodes:
            # (node, edges-so-far, directions-so-far, visited-canonical)
            queue: deque[tuple[str, tuple[Edge, ...], tuple[bool, ...], frozenset[str]]]
            queue = deque([(start, (), (), frozenset({start}))])
            while queue:
                node, edges, dirs, visited = queue.popleft()
                if len(edges) >= hops:
                    continue
                for edge, forward, nxt in self._expand(node, valid_at):
                    if nxt in visited:
                        continue
                    new_edges = (*edges, edge)
                    new_dirs = (*dirs, forward)
                    key = tuple((e.fact.id, d) for e, d in zip(new_edges, new_dirs, strict=True))
                    if key not in seen_paths:
                        seen_paths.add(key)
                        paths.append(Path(edges=new_edges, directions=new_dirs))
                        if len(paths) >= _MAX_PATHS:
                            return _rank(paths)[:limit]
                    queue.append((nxt, new_edges, new_dirs, visited | {nxt}))
        return _rank(paths)[:limit]

    def _expand(
        self,
        canonical: str,
        valid_at: datetime | None,
    ) -> list[tuple[Edge, bool, str]]:
        """Valid outgoing steps from a canonical node, both directions.

        Yields ``(edge, forward, next_canonical)`` tuples.
        """
        steps: list[tuple[Edge, bool, str]] = []
        for member in self._members(canonical):
            for edge in self._out.get(member, ()):
                if _edge_valid(edge.fact, valid_at):
                    steps.append((edge, True, self._resolve(edge.object)))
            for edge in self._in.get(member, ()):
                if _edge_valid(edge.fact, valid_at):
                    steps.append((edge, False, self._resolve(edge.subject)))
        return steps


def _rank(paths: list[Path]) -> list[Path]:
    """Shorter paths first; ties broken by newest fact on the path
    (most recently recorded wins)."""

    def key(path: Path) -> tuple[int, float]:
        newest = max(f.recorded_at.timestamp() for f in path.facts)
        return (len(path.edges), -newest)

    return sorted(paths, key=key)


async def recall_graph(
    fact_store: FactStore,
    query: str,
    *,
    user_id: str | None = None,
    hops: int = 2,
    valid_at: datetime | None = None,
    limit: int = 10,
    fetch_limit: int = 2000,
) -> list[Path]:
    """Multi-hop recall over any :class:`FactStore`.

    Fetches up to ``fetch_limit`` of the user's facts (the documented
    v1 scale bound — the graph is built in process per call), seeds
    from ``query`` token overlap, and BFS-traverses ``hops`` edges.
    ``valid_at=None`` traverses only currently-valid facts; a concrete
    instant replays the graph as of that world time — superseded facts
    are included inside their historical window.

    Works against every backend via the protocol ``query()`` surface::

        paths = await recall_graph(store, "where does alice's employer operate?",
                                   user_id="alice")
        for path in paths:
            print(path.render())   # alice —works_at→ acme —operates_in→ tokyo
    """
    facts = await fact_store.query(user_id=user_id, limit=fetch_limit)
    graph = FactGraph.from_facts(facts)
    seeds = graph.seed_entities(query)
    if not seeds:
        return []
    return graph.traverse(seeds, hops=hops, valid_at=valid_at, limit=limit)
