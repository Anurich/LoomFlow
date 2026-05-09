"""Document loaders + chunking.

Reads ``.pdf``, ``.docx``, ``.xlsx``, ``.csv``, ``.tsv``, ``.md``,
``.txt``, and ``.html`` files into a normalized :class:`Document`
whose ``content`` is markdown text. From there, four chunking
strategies break the document into LLM-friendly pieces:

* :class:`RecursiveChunker` — the production workhorse (LangChain-
  compatible behaviour)
* :class:`MarkdownChunker` — splits on ``#`` heading boundaries;
  preserves the header trail in chunk metadata. Best for the
  markdown produced by the PDF / DOCX / Excel loaders.
* :class:`SentenceChunker` — sentence-boundary chunks for QA-style
  RAG.
* :class:`TokenChunker` — chunk by token count via ``tiktoken``
  (lazy import).

One-liner usage::

    from loomflow.loader import load, chunk

    doc = load("research.pdf")              # auto-detect format
    chunks = chunk(doc.content)             # default: RecursiveChunker

Or pick the loader and chunker explicitly::

    from loomflow.loader import load_pdf, MarkdownChunker

    doc = load_pdf("research.pdf")
    chunker = MarkdownChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.split(doc.content)

Optional dependencies
---------------------

The PDF / DOCX / Excel / HTML loaders are gated behind extras so
the framework's base install stays lean::

    pip install 'loomflow[loader]'              # all four
    pip install 'loomflow[loader-pdf]'          # just pypdf
    pip install 'loomflow[loader-docx]'         # just python-docx
    pip install 'loomflow[loader-excel]'        # just openpyxl
    pip install 'loomflow[loader-html]'         # just beautifulsoup4

Each loader raises a helpful :class:`ImportError` if its dependency
is missing.
"""

from .base import Chunk, Document
from .chunking import (
    MarkdownChunker,
    RecursiveChunker,
    SentenceChunker,
    TokenChunker,
    chunk,
)
from .csv import load_csv, load_tsv
from .dispatch import load
from .docx import load_docx
from .excel import load_excel
from .html import load_html
from .pdf import load_pdf
from .text import load_markdown, load_text

__all__ = [
    "Chunk",
    "Document",
    "MarkdownChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "TokenChunker",
    "chunk",
    "load",
    "load_csv",
    "load_docx",
    "load_excel",
    "load_html",
    "load_markdown",
    "load_pdf",
    "load_text",
    "load_tsv",
]
