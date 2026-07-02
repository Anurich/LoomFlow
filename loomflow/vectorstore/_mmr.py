"""Maximal Marginal Relevance (MMR) reranking.

Given a query vector and a candidate pool of (vector, original
score) pairs, returns the top-k indices in selection order,
balancing relevance to the query against diversity from already-
selected candidates.

The framework's :meth:`VectorStore.search` exposes this via a
``diversity: float | None`` argument scaled 0..1 where 0 = pure
relevance, 1 = maximum diversity. We invert internally to the
classical ``lambda_mult`` (relevance weight) that the algorithm
expects: ``lambda_mult = 1 - diversity``.

This is a pure helper — no I/O, no async — so backends with their
own vector storage (Chroma, Postgres, FAISS) can fetch candidates
in whichever native way they prefer, then call this to rerank.
"""

from __future__ import annotations

import math
from typing import TypeVar

from ._util import cosine as _cosine

_T = TypeVar("_T")


def mmr_select(
    query_vec: list[float],
    candidate_vecs: list[list[float]],
    k: int,
    *,
    diversity: float = 0.5,
) -> list[int]:
    """Return ``k`` candidate indices selected by MMR.

    ``diversity`` is in [0, 1]. 0 = pure relevance (degenerates to
    plain top-k by query similarity), 1 = pure diversity (selects
    the most spread-out points regardless of query similarity).
    Values outside the range are clamped.

    If the pool has fewer than ``k`` candidates, all are returned
    in MMR order.
    """
    if not candidate_vecs:
        return []
    diversity = max(0.0, min(1.0, diversity))
    lambda_mult = 1.0 - diversity

    n = len(candidate_vecs)
    k = min(k, n)
    if k == 0:
        return []

    # Pre-compute similarity to query for every candidate.
    sim_q = [_cosine(query_vec, v) for v in candidate_vecs]

    # First pick: most similar to the query.
    selected: list[int] = []
    remaining = set(range(n))
    first = max(remaining, key=lambda i: sim_q[i])
    selected.append(first)
    remaining.remove(first)

    # Greedy MMR for the rest. For each remaining candidate, score =
    # lambda * sim_to_query - (1 - lambda) * max_sim_to_already_selected.
    while remaining and len(selected) < k:
        best_score = -math.inf
        best_idx = -1
        for i in remaining:
            max_sim_sel = max(
                _cosine(candidate_vecs[i], candidate_vecs[j])
                for j in selected
            )
            score = lambda_mult * sim_q[i] - (1.0 - lambda_mult) * max_sim_sel
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def rerank_tail(
    query_vec: list[float],
    candidates: list[_T],
    candidate_vecs: list[list[float]],
    k: int,
    diversity: float | None,
) -> list[_T]:
    """Shared per-backend search tail: plain top-``k`` when
    ``diversity`` is off (``None`` / ``<= 0``), MMR rerank otherwise.

    ``candidates`` and ``candidate_vecs`` are parallel lists already
    sorted best-first by the backend's native similarity ranking.
    """
    if diversity is None or diversity <= 0:
        return candidates[:k]
    chosen = mmr_select(query_vec, candidate_vecs, k, diversity=diversity)
    return [candidates[i] for i in chosen]
