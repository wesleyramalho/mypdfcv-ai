"""BM25 retrieval via rank-bm25.

A keyword baseline that the dense retriever has to beat in the eval to
justify its existence. In practice we find dense wins on paraphrase queries
and BM25 wins on rare-term queries — which is exactly why we also ship a
hybrid retriever.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from mypdfcv_ai.db.models import CareerFact
from mypdfcv_ai.retrieval.base import Hit

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    strategy_name = "bm25"

    def __init__(self, facts: list[CareerFact]) -> None:
        self._facts = facts
        self._tokenized = [_tokenize(f.content) for f in facts]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, k: int = 8) -> list[Hit]:
        if not self._facts or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            Hit(
                fact_id=self._facts[i].id,
                content=self._facts[i].content,
                score=float(scores[i]),
                source_type=self._facts[i].source_type,
                metadata=self._facts[i].fact_metadata or {},
            )
            for i in ranked
            if scores[i] > 0
        ]
