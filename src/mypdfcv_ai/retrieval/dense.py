"""Dense retrieval over precomputed embeddings.

At demo scale (<1k facts/user) we hold the matrix in memory and use numpy
dot product. The repository pattern means this is the only file that
changes when we swap to pgvector — query becomes
`embedding <=> :q_vec ORDER BY 1 LIMIT k`.
"""
from __future__ import annotations

import json

import numpy as np

from mypdfcv_ai.db.models import CareerFact
from mypdfcv_ai.ingestion.embedder import embed_query
from mypdfcv_ai.retrieval.base import Hit


class DenseRetriever:
    strategy_name = "dense"

    def __init__(self, facts: list[CareerFact]) -> None:
        self._facts = facts
        self._matrix = _stack_embeddings(facts)

    def search(self, query: str, k: int = 8) -> list[Hit]:
        if not self._facts or self._matrix.size == 0:
            return []
        q = embed_query(query)
        # All vectors are L2-normalised at ingest time, so dot product == cosine sim.
        scores = self._matrix @ q
        top_idx = np.argsort(-scores)[:k]
        return [
            Hit(
                fact_id=self._facts[i].id,
                content=self._facts[i].content,
                score=float(scores[i]),
                source_type=self._facts[i].source_type,
                metadata=self._facts[i].fact_metadata or {},
            )
            for i in top_idx
        ]


def _stack_embeddings(facts: list[CareerFact]) -> np.ndarray:
    rows = []
    for f in facts:
        if not f.embedding_json:
            continue
        rows.append(json.loads(f.embedding_json))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)
