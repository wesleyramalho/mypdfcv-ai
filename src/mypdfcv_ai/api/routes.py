from __future__ import annotations

from fastapi import APIRouter, HTTPException

from mypdfcv_ai.agents.tailor_agent import run_tailor_agent
from mypdfcv_ai.api.schemas import (
    CitedFact,
    IngestRequest,
    IngestResponse,
    SearchHitOut,
    SearchRequest,
    SearchResponse,
    TailoredBulletOut,
    TailorRequest,
    TailorResponse,
)
from mypdfcv_ai.db.models import CareerFact, TailoringRun
from mypdfcv_ai.db.repository import CareerFactsRepository, TailoringRunsRepository
from mypdfcv_ai.db.session import get_sessionmaker
from mypdfcv_ai.ingestion.embedder import embed_texts
from mypdfcv_ai.retrieval.factory import build_retriever

router = APIRouter(prefix="/v1")


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/career/facts", response_model=IngestResponse)
def ingest_facts(req: IngestRequest) -> IngestResponse:
    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        repo = CareerFactsRepository(session)
        if req.replace:
            repo.delete_for_user(req.user_id)

        embeddings = embed_texts([f.content for f in req.facts])
        records = []
        for f, vec in zip(req.facts, embeddings, strict=True):
            fact = CareerFact(
                user_id=req.user_id,
                source_type=f.source_type,
                source_id=f.source_id,
                content=f.content,
                fact_metadata=f.metadata,
            )
            fact.set_embedding(vec.tolist())
            records.append(fact)
        repo.add_many(records)
        session.commit()

    return IngestResponse(user_id=req.user_id, inserted=len(req.facts))


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if req.strategy not in {"dense", "bm25", "hybrid"}:
        raise HTTPException(status_code=400, detail="Unknown strategy")
    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        retriever = build_retriever(session, req.user_id, strategy=req.strategy)  # type: ignore[arg-type]
        hits = retriever.search(req.query, k=req.k)
    return SearchResponse(
        hits=[
            SearchHitOut(fact_id=h.fact_id, content=h.content, score=h.score, source_type=h.source_type)
            for h in hits
        ]
    )


@router.post("/tailor", response_model=TailorResponse)
def tailor(req: TailorRequest) -> TailorResponse:
    if req.strategy not in {"dense", "bm25", "hybrid"}:
        raise HTTPException(status_code=400, detail="Unknown strategy")

    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        retriever = build_retriever(session, req.user_id, strategy=req.strategy)  # type: ignore[arg-type]
        result = run_tailor_agent(
            jd_text=req.jd_text,
            target_sections=req.target_sections,
            retriever=retriever,
            user_id=req.user_id,
        )
        run = TailoringRun(
            user_id=req.user_id,
            jd_text=req.jd_text,
            iterations=result.iterations,
            duration_ms=result.duration_ms,
            model_used=result.model_used,
        )
        run.set_bullets(
            [
                {
                    "section": b.section,
                    "text": b.text,
                    "confidence": b.confidence,
                    "citations": b.citations,
                }
                for b in result.bullets
            ]
        )
        TailoringRunsRepository(session).add(run)
        session.commit()

    return TailorResponse(
        bullets=[
            TailoredBulletOut(
                section=b.section,
                text=b.text,
                confidence=b.confidence,
                citations=[
                    CitedFact(
                        fact_id=c.fact_id,
                        content=c.content,
                        source_type=c.source_type,
                        score=c.score,
                    )
                    for c in b.cited_facts
                ],
            )
            for b in result.bullets
        ],
        iterations=result.iterations,
        duration_ms=result.duration_ms,
        model_used=result.model_used,
        finish_summary=result.finish_summary,
        notes=result.notes,
    )
