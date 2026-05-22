"""Build a retriever for a user, given a strategy name."""
from __future__ import annotations

from sqlalchemy.orm import Session

from mypdfcv_ai.db.models import CareerFact
from mypdfcv_ai.db.repository import CareerFactsRepository
from mypdfcv_ai.retrieval.base import Retriever, Strategy
from mypdfcv_ai.retrieval.bm25 import BM25Retriever
from mypdfcv_ai.retrieval.dense import DenseRetriever
from mypdfcv_ai.retrieval.hybrid import HybridRetriever


def build_retriever(session: Session, user_id: str, strategy: Strategy = "hybrid") -> Retriever:
    facts = CareerFactsRepository(session).list_for_user(user_id)
    return build_retriever_from_facts(facts, strategy=strategy)


def build_retriever_from_facts(
    facts: list[CareerFact], strategy: Strategy = "hybrid"
) -> Retriever:
    """Same dispatcher as build_retriever, but takes facts in directly.

    Used by the stateless /v1/tailor-resume endpoint where the facts are
    constructed in memory from a request payload rather than fetched from
    the DB. Facts must already have their embeddings set for the dense /
    hybrid strategies.
    """
    if strategy == "dense":
        return DenseRetriever(facts)
    if strategy == "bm25":
        return BM25Retriever(facts)
    if strategy == "hybrid":
        return HybridRetriever(DenseRetriever(facts), BM25Retriever(facts))
    raise ValueError(f"Unknown strategy: {strategy}")
