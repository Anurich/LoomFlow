"""Shared hybrid-recall helpers for memory backends.

Two pieces:

* :func:`default_recall_scored` — a no-op fallback that wraps each
  :class:`Episode` from a backend's :meth:`recall` in an
  :class:`EpisodeMatch` with a neutral ``score=1.0``. Used by
  backends that haven't implemented native hybrid scoring yet, so
  the ``Memory`` protocol stays coherent.

* :class:`HybridRanker` — a tiny BM25 + cosine + Reciprocal Rank
  Fusion (RRF) implementation that backends can use to compute
  scored matches over their episode rows. RRF is the field-standard
  fusion algorithm: it scores by rank position only (ignoring raw
  score magnitudes), which is robust when the two rankings come
  from different scoring systems (cosine ∈ [-1, 1] vs BM25 ∈
  [0, ∞)).

Lives here rather than in :mod:`loomflow.vectorstore` to keep the
memory module self-contained — vectorstore has its own copy for
RAG retrieval and doesn't share state with this one.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ._embedding_util import cosine

if TYPE_CHECKING:
    from ..core.types import Episode, EpisodeMatch

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Default fallback for backends without native hybrid recall
# ---------------------------------------------------------------------------


def default_recall_scored(
    episodes: Iterable[Episode],
    *,
    score: float = 1.0,
) -> list[EpisodeMatch]:
    """Wrap a list of :class:`Episode` rows as :class:`EpisodeMatch`
    with a neutral score.

    Backends that don't compute their own retrieval scores (e.g.
    pure-recency :class:`InMemoryMemory.recall`, or external stores
    where Loom doesn't see the score) call this in their
    ``recall_scored`` to satisfy the protocol without inventing
    scores they don't actually have. Returns one match per episode
    in input order.
    """
    from ..core.types import EpisodeMatch

    return [EpisodeMatch(episode=ep, score=score) for ep in episodes]


# ---------------------------------------------------------------------------
# Hybrid ranker — BM25 + cosine + RRF
# ---------------------------------------------------------------------------


# ``cosine`` is re-exported (see the import at the top of the module)
# from ``memory._embedding_util`` — the single shared implementation
# for all memory backends. Kept in ``__all__`` for back-compat with
# callers that imported it from here.


class _BM25:
    """Minimal Okapi BM25 over a parallel-list corpus.

    Stateless across queries; the corpus is rebuilt by the caller
    on each ``recall_scored`` call (since memory is small and the
    rebuild cost is dominated by the tokenizer regex, not by IDF
    computation). For a real production fact store with millions
    of episodes, swap this for a persistent inverted index — but
    in-process memory backends won't have that scale.
    """

    def __init__(
        self, texts: list[str], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self._tokens: list[list[str]] = [_tokenize(t) for t in texts]
        self._freqs: list[Counter[str]] = [
            Counter(toks) for toks in self._tokens
        ]
        self._df: Counter[str] = Counter()
        for f in self._freqs:
            for term in f:
                self._df[term] += 1
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

    def rank(self, query: str) -> list[tuple[int, float]]:
        """``[(doc_idx, bm25_score), ...]`` sorted best-first.

        Zero-score docs are filtered out so RRF doesn't pollute the
        ranking with completely-irrelevant rows."""
        scored = [(i, self.score(query, i)) for i in range(self._n_docs)]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    *,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Combine multiple per-component rankings via RRF.

    Each ``rankings`` entry is ``[(doc_idx, raw_score), ...]``
    already sorted best-first. The fused score is
    ``sum(1 / (k + rank + 1))`` across components a doc appears in.
    ``k=60`` is the original Cormack et al. recommendation —
    dampens the weight of top-1 just enough that doc #2 in ranking
    A can outrank doc #1 in ranking B when it also appears in B.

    Raw scores are intentionally ignored: the whole point of RRF is
    to be robust across mismatched scoring scales (cosine vs BM25).
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def hybrid_rank(
    *,
    bm25_ranking: list[tuple[int, float]],
    vector_ranking: list[tuple[int, float]],
    alpha: float,
    k: int = 60,
) -> list[tuple[int, float, float | None, float | None]]:
    """Run RRF over BM25 + vector rankings, returning
    ``[(doc_idx, fused_score, bm25_score | None, vector_score | None), ...]``.

    ``alpha`` ∈ ``[0, 1]`` controls the lexical-vs-vector weighting:
    ``0`` = pure BM25, ``1`` = pure vector cosine, ``0.5`` = balanced.
    Implemented by *replicating* the favoured ranking when ``alpha``
    is near an extreme — RRF treats each occurrence as another vote.
    Crude but matches the public hybrid behaviour in
    ``loomflow.vectorstore.InMemoryVectorStore`` so users get
    consistent ranking semantics across the two surfaces.

    Per-component raw scores ride along so the caller can populate
    :class:`EpisodeMatch` ``bm25_score`` / ``vector_score`` fields
    for downstream consumers (rerankers, A/B experiments).
    """
    alpha = max(0.0, min(1.0, alpha))
    bm25_weight = max(1, int(round((1 - alpha) * 4)))
    vector_weight = max(1, int(round(alpha * 4)))
    rankings: list[list[tuple[int, float]]] = []
    rankings.extend([bm25_ranking] * bm25_weight)
    rankings.extend([vector_ranking] * vector_weight)
    fused = reciprocal_rank_fusion(rankings, k=k)

    bm25_lookup = {idx: s for idx, s in bm25_ranking}
    vec_lookup = {idx: s for idx, s in vector_ranking}
    out: list[tuple[int, float, float | None, float | None]] = []
    for idx, score in fused:
        out.append(
            (
                idx,
                score,
                bm25_lookup.get(idx),
                vec_lookup.get(idx),
            )
        )
    return out


def hybrid_rank_episodes(
    candidates: list[Episode],
    *,
    query: str,
    query_embedding: list[float] | None,
    alpha: float,
    limit: int,
    embeddings: list[list[float]] | None = None,
) -> list[EpisodeMatch]:
    """Shared BM25 + cosine + RRF ranking tail for the backends'
    ``recall_scored`` implementations.

    Every backend fetches its own candidate pool (already partitioned
    by ``user_id`` and filtered by ``time_range``) and hands it here:

    * ``query_embedding`` — the embedded query, or ``None`` for
      backends without a vector arm (BM25-only ranking then).
    * ``embeddings`` — optional parallel list of candidate vectors;
      when ``None``, each candidate's own ``Episode.embedding`` is
      used (candidates without one simply don't score on the vector
      arm).

    Non-positive cosine similarities are dropped so RRF doesn't
    promote them. When the fused ranking is empty (no lexical AND no
    vector signal), falls back to recency ordering with ``0.0``
    scores so the caller always gets *something* useful. Component
    scores (``bm25_score`` / ``vector_score``) ride along on each
    :class:`EpisodeMatch`.
    """
    from ..core.types import EpisodeMatch

    if not candidates:
        return []

    # Vector arm — cosine over candidate embeddings; drop
    # non-positive sims so RRF doesn't promote them.
    vector_scores: list[tuple[int, float]] = []
    if query_embedding is not None:
        for i, ep in enumerate(candidates):
            emb = (
                embeddings[i]
                if embeddings is not None
                else ep.embedding
            )
            if emb is None:
                continue
            sim = cosine(query_embedding, emb)
            if sim > 0:
                vector_scores.append((i, sim))
        vector_scores.sort(key=lambda x: x[1], reverse=True)

    # BM25 arm — lexical ranking over the same candidate pool.
    texts = [f"{e.input}\n{e.output}" for e in candidates]
    bm25_ranking = _BM25(texts).rank(query)

    fused = hybrid_rank(
        bm25_ranking=bm25_ranking,
        vector_ranking=vector_scores,
        alpha=alpha,
    )
    if not fused:
        recent = sorted(
            candidates, key=lambda e: e.occurred_at, reverse=True
        )[:limit]
        return [EpisodeMatch(episode=e, score=0.0) for e in recent]
    return [
        EpisodeMatch(
            episode=candidates[idx],
            score=score,
            bm25_score=bm25_score,
            vector_score=vector_score,
        )
        for idx, score, bm25_score, vector_score in fused[:limit]
    ]


# Kept for symmetry with the public API of vectorstore._bm25; some
# downstream tests / extensions may want to reach into this module.
__all__ = [
    "default_recall_scored",
    "cosine",
    "_BM25",
    "reciprocal_rank_fusion",
    "hybrid_rank",
    "hybrid_rank_episodes",
]


# Quiet "imported but unused" complaints for the optional Any helper
_ = Any
