"""Reciprocal Rank Fusion of dense + BM25.

RRF is a simple, parameter-light fusion that works well across diverse
score scales — the dense and BM25 raw scores are not directly comparable,
but their *ranks* are. Constant `k_rrf=60` is the value used in the
original Cormack et al. paper.
"""
from __future__ import annotations

from mypdfcv_ai.retrieval.base import Hit, Retriever


class HybridRetriever:
    strategy_name = "hybrid"

    def __init__(self, dense: Retriever, bm25: Retriever, k_rrf: int = 60) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._k_rrf = k_rrf

    def search(self, query: str, k: int = 8) -> list[Hit]:
        # Fetch a wider pool from each and fuse.
        dense_hits = self._dense.search(query, k=max(k * 3, 12))
        bm25_hits = self._bm25.search(query, k=max(k * 3, 12))

        scores: dict[str, float] = {}
        hit_lookup: dict[str, Hit] = {}
        for rank, h in enumerate(dense_hits):
            scores[h.fact_id] = scores.get(h.fact_id, 0.0) + 1.0 / (self._k_rrf + rank + 1)
            hit_lookup[h.fact_id] = h
        for rank, h in enumerate(bm25_hits):
            scores[h.fact_id] = scores.get(h.fact_id, 0.0) + 1.0 / (self._k_rrf + rank + 1)
            hit_lookup.setdefault(h.fact_id, h)

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        # Reproject the RRF score onto each hit so callers see a single score.
        return [
            Hit(
                fact_id=fid,
                content=hit_lookup[fid].content,
                score=score,
                source_type=hit_lookup[fid].source_type,
                metadata=hit_lookup[fid].metadata,
            )
            for fid, score in ranked
        ]
