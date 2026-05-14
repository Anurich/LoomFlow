"""19_workspace_lifecycle.py — the workspace v0.10 lifecycle.

Example 16 shows the workspace as a multi-agent COORDINATION
layer. This one shows the v0.10 LIFECYCLE + SELF-IMPROVEMENT
surface — the features that make the notebook a substrate an
agent gets smarter on, run over run:

1. **Namespacing** — sub-buckets within one workspace.
2. **Versioning** — every ``update_note`` snapshots history;
   ``list_versions`` / ``read_version`` walk it back.
3. **Archive** — soft-hide stale notes (``archive_note``);
   excluded from listings, still readable by slug.
4. **Questions** — ``ask_question`` / ``answer_question`` /
   ``list_open_questions`` (opt-in via ``questions=True`` on
   ``make_workspace_tools``).
5. **Semantic search** — optional ``embedder=`` on the backend;
   ``search_notes(mode="semantic"|"hybrid")``.
6. **Citation tracking + outcome attribution** — ``read_note``
   logs citations into a per-run set; ``attribute_outcome``
   updates ``cited_count`` / ``success_count`` / ``last_cited_at``.
7. **Relevance-aware search** — ``search_notes(boost_relevance=
   True)`` ranks frequently-used-on-success notes higher.
8. **Retention** — ``prune()`` GCs stale, low-value notes;
   citation-aware so it keeps what's been *used*.

Fully offline — uses :class:`InMemoryWorkspace` and a tiny
deterministic stub embedder, so no API key is needed.

Run with::

    python examples/19_workspace_lifecycle.py
"""

from __future__ import annotations

import asyncio
import math

from loomflow import InMemoryWorkspace, PruneResult
from loomflow.core.context import RunContext, set_run_context
from loomflow.workspace.tools import make_workspace_tools


class _StubEmbedder:
    """Deterministic char-bucket embedder — zero deps, zero keys.

    Real deployments pass an ``OpenAIEmbedder`` / ``VoyageEmbedder``
    etc. This stub just makes 'apple' and 'apples' land near each
    other so the semantic-search demo is reproducible offline.
    """

    name = "stub-embedder-v1"
    dimensions = 26

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * 26
        for ch in text.lower():
            if ch.isalpha():
                vec[ord(ch) - ord("a")] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


async def demo_namespacing() -> None:
    print("=" * 60)
    print("1. NAMESPACING — sub-buckets within one workspace")
    print("=" * 60)
    ws = InMemoryWorkspace()
    await ws.write_note(
        author="agent", title="API rate limits", body="...",
        user_id="u", namespace="backend",
    )
    await ws.write_note(
        author="agent", title="Button hover states", body="...",
        user_id="u", namespace="frontend",
    )
    # Default list sees ALL namespaces (namespace is metadata,
    # not a partition — teammates' adjacent work stays visible).
    everything = await ws.list_notes(user_id="u")
    backend_only = await ws.list_notes(
        user_id="u", namespace="backend"
    )
    print(f"  all namespaces:      {len(everything)} notes")
    print(f"  namespace=backend:   {len(backend_only)} notes")
    print()


async def demo_versioning() -> None:
    print("=" * 60)
    print("2. VERSIONING — every update_note snapshots history")
    print("=" * 60)
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="Design doc", body="draft 1",
        user_id="u",
    )
    await ws.update_note(
        author="agent", slug=n.slug, body="draft 2", user_id="u",
    )
    await ws.update_note(
        author="agent", slug=n.slug, body="final", user_id="u",
    )
    versions = await ws.list_versions(
        n.slug, author="agent", user_id="u"
    )
    live = await ws.read_note(n.slug, user_id="u")
    print(f"  live body:  {live.body!r}" if live else "  (gone)")
    print(f"  history:    {len(versions)} prior revision(s)")
    for v in versions:
        snap = await ws.read_version(
            n.slug, v.version, author="agent", user_id="u"
        )
        print(f"    v{v.version}: {snap.body!r}" if snap else "")
    print()


async def demo_archive() -> None:
    print("=" * 60)
    print("3. ARCHIVE — soft-hide stale notes")
    print("=" * 60)
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="Old approach", body="...",
        user_id="u",
    )
    await ws.archive_note(author="agent", slug=n.slug, user_id="u")
    visible = await ws.list_notes(user_id="u")
    with_archived = await ws.list_notes(
        user_id="u", include_archived=True
    )
    still_readable = await ws.read_note(n.slug, user_id="u")
    print(f"  list_notes():                 {len(visible)} (hidden)")
    print(f"  list_notes(include_archived):  {len(with_archived)}")
    print(
        f"  read_note(slug):               "
        f"{'still works' if still_readable else 'gone'}"
    )
    print()


async def demo_questions() -> None:
    print("=" * 60)
    print("4. QUESTIONS — ask / answer / list_open (opt-in tools)")
    print("=" * 60)
    ws = InMemoryWorkspace()
    # The three question tools only appear when questions=True.
    tools = make_workspace_tools(
        ws, author="alice", questions=True
    )
    tool_names = {t.name for t in tools}
    print(f"  tools wired: {sorted(tool_names)}")
    # Alice asks; Bob answers (cross-author safe).
    async with set_run_context(RunContext(user_id="u")):
        q = await ws.write_note(
            author="alice", title="Which DB?", body="postgres or sqlite?",
            user_id="u", kind="question", answered=False,
        )
        answer = await ws.write_note(
            author="bob", title="Answer: Which DB?",
            body="postgres — we need concurrent writes",
            user_id="u", kind="finding", parent_slug=q.slug,
        )
        # Bob marks Alice's question answered via the cross-author
        # carve-out — he can flip the flag without owning the note.
        await ws.update_note(
            author="bob", slug=q.slug, body=q.body, user_id="u",
            mark_answered=answer.slug,
        )
        resolved = await ws.read_note(q.slug, user_id="u")
    print(
        f"  question answered: {resolved.answered}"
        if resolved else "  (?)"
    )
    print(
        f"  answered_by:       {resolved.answered_by}"
        if resolved else ""
    )
    print()


async def demo_semantic_search() -> None:
    print("=" * 60)
    print("5. SEMANTIC SEARCH — optional embedder on the backend")
    print("=" * 60)
    ws = InMemoryWorkspace(embedder=_StubEmbedder())
    await ws.write_note(
        author="agent", title="apple orchard guide",
        body="how to grow apple trees", user_id="u",
    )
    await ws.write_note(
        author="agent", title="zebra migration patterns",
        body="savanna grazing routes", user_id="u",
    )
    hits = await ws.search_notes(
        "apple", user_id="u", mode="semantic"
    )
    print(f"  semantic search 'apple' → top hit: {hits[0].summary.title!r}")
    print()


async def demo_self_improvement() -> None:
    print("=" * 60)
    print("6+7. CITATION TRACKING + RELEVANCE-AWARE SEARCH")
    print("=" * 60)
    ws = InMemoryWorkspace()
    popular = await ws.write_note(
        author="agent", title="The good fix", body="useful thing",
        user_id="u",
    )
    await ws.write_note(
        author="agent", title="The other note", body="useful thing",
        user_id="u",
    )
    # Simulate 5 successful runs that each read `popular`.
    from loomflow.core.context import _ambient_citations_var
    for _ in range(5):
        cites: set[str] = set()
        token = _ambient_citations_var.set(cites)
        try:
            async with set_run_context(RunContext(user_id="u")):
                await ws.read_note(popular.slug, user_id="u")
                await ws.attribute_outcome(success=True, user_id="u")
        finally:
            _ambient_citations_var.reset(token)
    after = await ws.read_note(popular.slug, user_id="u")
    print(
        f"  '{popular.title}' — cited_count={after.cited_count}, "
        f"success_count={after.success_count}"
        if after else ""
    )
    # Without boost, both notes are equal-tier text matches.
    # With boost, the cited-on-success one ranks first.
    boosted = await ws.search_notes(
        "useful", user_id="u", boost_relevance=True
    )
    print(
        f"  boost_relevance=True → top hit: "
        f"{boosted[0].summary.title!r} (it's the proven one)"
    )
    print()


async def demo_prune() -> None:
    print("=" * 60)
    print("8. PRUNE — citation-aware retention / GC")
    print("=" * 60)
    ws = InMemoryWorkspace()
    # Three notes: one cited, one a protected 'decision', one
    # plain + uncited (the GC target).
    cited = await ws.write_note(
        author="agent", title="referenced note", body="...",
        user_id="u",
    )
    await ws.write_note(
        author="agent", title="a sticky decision", body="...",
        user_id="u", kind="decision",
    )
    await ws.write_note(
        author="agent", title="forgotten scratch note", body="...",
        user_id="u",
    )
    # Cite the first note once so it survives.
    from loomflow.core.context import _ambient_citations_var
    cites: set[str] = set()
    token = _ambient_citations_var.set(cites)
    try:
        async with set_run_context(RunContext(user_id="u")):
            await ws.read_note(cited.slug, user_id="u")
            await ws.attribute_outcome(success=True, user_id="u")
    finally:
        _ambient_citations_var.reset(token)
    # Prune: no age filter, keep cited (min_cited_count=1) +
    # keep decisions. Only the uncited scratch note should go.
    result: PruneResult = await ws.prune(
        older_than=None,
        min_cited_count=1,
        keep_kinds=["decision"],
        user_id="u",
    )
    print(
        f"  prune → deleted={result.notes_deleted}, "
        f"kept={result.notes_kept}"
    )
    survivors = await ws.list_notes(user_id="u")
    print(f"  survivors: {[s.title for s in survivors]}")
    print(
        "  (cited note + decision kept; uncited scratch note GC'd "
        "— retention keeps what's been USED, not just what's "
        "recent)"
    )
    print()


async def main() -> None:
    await demo_namespacing()
    await demo_versioning()
    await demo_archive()
    await demo_questions()
    await demo_semantic_search()
    await demo_self_improvement()
    await demo_prune()
    print("All v0.10 workspace lifecycle features demonstrated.")


if __name__ == "__main__":
    asyncio.run(main())
