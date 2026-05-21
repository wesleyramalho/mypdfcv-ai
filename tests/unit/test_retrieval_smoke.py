"""End-to-end smoke test of ingest + retrieve, no LLM involved.

This validates the foundation (DB → embeddings → retrievers) before we
layer the agent on top. Catches dimension mismatches, normalisation bugs,
and the tokenizer not biting BM25.
"""
from __future__ import annotations

import os

# Make sure tests use an isolated DB.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from mypdfcv_ai.db.models import CareerFact
from mypdfcv_ai.db.repository import CareerFactsRepository
from mypdfcv_ai.db.session import get_sessionmaker, init_db
from mypdfcv_ai.ingestion.embedder import embed_texts
from mypdfcv_ai.retrieval.factory import build_retriever


def _seed_facts(session, user_id: str) -> None:
    raw = [
        ("Led migration of monolith to microservices on AWS, cutting p99 latency by 40%.", "experience"),
        ("Built ETL pipelines in Python and Airflow, processing 50TB/day of clickstream data.", "experience"),
        ("Designed embedding-based retrieval over product catalog, lifting CTR by 12%.", "project"),
        ("BSc Computer Science, Federal University of Rio de Janeiro, 2018.", "education"),
        ("Spoke at PyCon Brazil 2023 on RAG evaluation methodology.", "freeform"),
        ("Open-sourced a Postgres extension for approximate nearest neighbour search.", "project"),
    ]
    vecs = embed_texts([t for t, _ in raw])
    facts = []
    for (content, source), vec in zip(raw, vecs, strict=True):
        f = CareerFact(user_id=user_id, source_type=source, content=content)
        f.set_embedding(vec.tolist())
        facts.append(f)
    CareerFactsRepository(session).add_many(facts)


def test_dense_finds_paraphrase_match():
    init_db()
    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        _seed_facts(session, user_id="u1")
        session.commit()

        retriever = build_retriever(session, "u1", strategy="dense")
        hits = retriever.search("vector search over product data", k=3)

    assert hits, "dense retriever returned no hits"
    # Embedding-based retrieval should fetch the catalog one, even though the
    # query uses different vocabulary.
    top_contents = [h.content for h in hits]
    assert any("product catalog" in c for c in top_contents), top_contents


def test_bm25_finds_exact_term_match():
    init_db()
    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        _seed_facts(session, user_id="u2")
        session.commit()

        retriever = build_retriever(session, "u2", strategy="bm25")
        hits = retriever.search("Airflow ETL", k=3)

    assert hits
    assert "Airflow" in hits[0].content


def test_hybrid_combines_both_signals():
    init_db()
    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        _seed_facts(session, user_id="u3")
        session.commit()

        retriever = build_retriever(session, "u3", strategy="hybrid")
        hits = retriever.search("PyCon presentation about evaluation", k=3)

    contents = [h.content for h in hits]
    assert any("PyCon" in c for c in contents), contents
