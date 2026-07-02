"""One-liner document ingest — load + chunk + add, in one call.

LangChain's whole pitch is ``Chroma.from_documents(docs, embedding)``.
loomflow made you thread ``load()`` → ``chunker.split()`` →
``store.add()`` by hand. :func:`index_document` collapses that to:

    from loomflow.vectorstore import index_document, ChromaVectorStore

    store = ChromaVectorStore(embedder=..., persist_directory="./db")
    ids = await index_document("research.pdf", store)

Unlike the ``from_texts`` / ``from_chunks`` *factories* (which build a
NEW store each call — a footgun if used in a loop), this ADDS to the
store you pass, which is the right primitive for growing an index over
time (call it once per document).

A free function over the :class:`~loomflow.vectorstore.VectorStore`
Protocol rather than a per-backend method: it works against all four
stores unchanged, and the loader import — gated behind the optional
``loomflow[loader]`` extra — stays lazy (inside the call), so importing
this module never drags in the loader.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import anyio.to_thread

if TYPE_CHECKING:
    from ..loader.chunking import Chunker
    from .base import VectorStore


async def index_document(
    path: str | Path,
    store: VectorStore,
    *,
    chunker: Chunker | None = None,
) -> list[str]:
    """Load ``path``, chunk it, and add the chunks to ``store``.

    Returns the new chunk ids. ``chunker`` defaults to
    :class:`~loomflow.loader.RecursiveChunker` (the format-agnostic
    production default); pass ``MarkdownChunker`` / ``TokenChunker``
    etc. to override. Raises :class:`ImportError` with an install
    hint when the loader extra isn't present.
    """
    try:
        from ..loader.chunking import RecursiveChunker
        from ..loader.dispatch import load
    except ImportError as exc:  # pragma: no cover - extra not installed
        raise ImportError(
            "index_document needs the loader. "
            "Install with: pip install 'loomflow[loader]'."
        ) from exc

    # ``load`` (PDF / DOCX parsing) and ``chunker.split`` are
    # CPU-bound sync work — offload to a worker thread so the event
    # loop stays responsive during large-document ingest.
    doc = await anyio.to_thread.run_sync(load, path)
    used = chunker if chunker is not None else RecursiveChunker()
    source = str(doc.metadata.get("source", path))
    chunks = await anyio.to_thread.run_sync(
        partial(used.split, doc.content, source=source)
    )
    return await store.add(chunks)
