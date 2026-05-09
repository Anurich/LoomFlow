"""Tiny BM25 implementation for hybrid lexical+vector search.

Standalone (no extra deps), tokenizer is a simple ``\\w+`` regex
lower-cased. Good enough for the hybrid-search use case where BM25
catches exact terms (model names, error codes, person names) that
embedding similarity smears together.

Used internally by :class:`InMemoryVectorStore.search_hybrid` —
not part of the public surface (yet).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 index over a parallel-list corpus.

    The index is rebuilt on every :meth:`remove` because BM25 stats
    (document frequency, average doc length) are corpus-global —
    incremental decrement would work but adds complexity for a
    rare operation in our usage pattern.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tokens: list[list[str]] = []
        self._freqs: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._n_docs = 0
        self._avg_dl = 0.0

    def add(self, texts: list[str]) -> None:
        for text in texts:
            tokens = _tokenize(text)
            freqs = Counter(tokens)
            self._tokens.append(tokens)
            self._freqs.append(freqs)
            for term in freqs:
                self._df[term] += 1
        self._recompute_stats()

    def remove_by_indices(self, indices: set[int]) -> None:
        if not indices:
            return
        self._tokens = [
            t for i, t in enumerate(self._tokens) if i not in indices
        ]
        self._freqs = [
            f for i, f in enumerate(self._freqs) if i not in indices
        ]
        self._df = Counter()
        for f in self._freqs:
            for term in f:
                self._df[term] += 1
        self._recompute_stats()

    def _recompute_stats(self) -> None:
        self._n_docs = len(self._tokens)
        total_len = sum(len(t) for t in self._tokens)
        self._avg_dl = total_len / self._n_docs if self._n_docs else 0.0

    def score(self, query: str, doc_idx: int) -> float:
        if doc_idx < 0 or doc_idx >= self._n_docs:
            return 0.0
        q_tokens = _tokenize(query)
        if not q_tokens or self._avg_dl == 0:
            return 0.0
        doc_freqs = self._freqs[doc_idx]
        dl = len(self._tokens[doc_idx])
        score = 0.0
        for term in q_tokens:
            f = doc_freqs.get(term, 0)
            if f == 0:
                continue
            df = self._df.get(term, 0)
            idf = math.log(
                (self._n_docs - df + 0.5) / (df + 0.5) + 1
            )
            num = f * (self.k1 + 1)
            denom = f + self.k1 * (
                1 - self.b + self.b * (dl / self._avg_dl)
            )
            score += idf * num / denom if denom else 0.0
        return score

    def search(
        self, query: str, k: int
    ) -> list[tuple[int, float]]:
        """Top-``k`` ``(doc_idx, score)`` pairs ranked by BM25 score."""
        if self._n_docs == 0:
            return []
        scored = [(i, self.score(query, i)) for i in range(self._n_docs)]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    *,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Combine multiple rankings via Reciprocal Rank Fusion.

    Each ranking is ``[(doc_idx, score), ...]`` already sorted
    best-first. RRF scores by rank position only (ignores raw
    scores), which is robust when the rankings come from different
    scoring systems (cosine vs BM25). The constant ``k=60`` is the
    convention from Cormack et al. — it dampens the weight of
    top-1 just enough that doc #2 in ranking A can outrank doc #1
    in ranking B if it appears in both.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
