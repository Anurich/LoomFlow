"""Example 23 — RAG ingest: one-liner document → vector store → search.

The LangChain-parity ergonomic. To put a file into a searchable index
you need exactly three things, and ``index_document`` collapses the
plumbing between them:

  1. an embedder (turns text into vectors),
  2. a vector store (built ONCE, with the embedder),
  3. the file.

  store = ChromaVectorStore(embedder=emb, persist_directory="./db")
  await index_document("research.pdf", store)   # load + chunk + add
  results = await store.search("what is X?")

The store is built once with its embedder; ``index_document`` and
``store.add`` reuse that embedder — you never pass it twice. (The
``from_texts`` / ``from_chunks`` classmethods DO take ``embedder=``
because they are factories that *build* a fresh store — don't call
them on a store you already made; that builds a throwaway and drops
the write. Use ``index_document`` / ``store.add`` to grow an existing
index.)

This file uses the zero-key ``HashEmbedder`` + ``InMemoryVectorStore``
so it runs out of the box with no API key and no extras. Swap in
``OpenAIEmbedder()`` + ``ChromaVectorStore(..., persist_directory=)``
for a real, persistent index — the rest of the code is identical.

Run::

    python examples/23_index_document.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from loomflow.memory import HashEmbedder
from loomflow.vectorstore import InMemoryVectorStore, index_document


async def main() -> None:
    # A throwaway markdown doc to ingest (a real run points at your
    # own .pdf / .docx / .md — the loader auto-detects the format).
    tmp = Path(tempfile.mkdtemp()) / "handbook.md"
    tmp.write_text(
        "# Onboarding\n\n"
        "New engineers get a laptop on day one and complete the "
        "security training within the first week.\n\n"
        "# Expenses\n\n"
        "Reimbursements are filed in the portal and approved by your "
        "manager. Travel over $500 needs VP sign-off.\n\n"
        "# Time off\n\n"
        "Vacation is unlimited but must be logged two weeks ahead for "
        "anything longer than three days.\n",
        encoding="utf-8",
    )

    # Build the store ONCE with its embedder.
    store = InMemoryVectorStore(embedder=HashEmbedder())

    # One call: load the file, chunk it, embed + add. No manual
    # loader → chunker → add threading, no embedder repeat.
    ids = await index_document(str(tmp), store)
    print(f"indexed {len(ids)} chunks from {tmp.name}\n")

    # Grow the index over time — index_document ADDS to the same
    # store (this is the difference from the from_* factories).
    # (Re-indexing the same file here just to show the count grow.)
    await index_document(str(tmp), store)
    print(f"total chunks after a second ingest: {await store.count()}\n")

    # Search. SearchResult.chunk.content is the passage; .score ranks.
    # NOTE: HashEmbedder has NO semantic meaning (it's a deterministic
    # test stand-in), so the ranking below is effectively random —
    # that's the embedder, not the store. Swap in OpenAIEmbedder() and
    # the right chunk rises to the top.
    for query in ("how do I get reimbursed?", "what about vacation?"):
        print(f"Q: {query}")
        results = await store.search(query, k=2)
        for r in results:
            snippet = r.chunk.content.replace("\n", " ")[:90]
            print(f"  {r.score:.3f}  {snippet}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
