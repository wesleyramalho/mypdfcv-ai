"""Pydantic request/response schemas for the HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CareerFactIn(BaseModel):
    content: str
    source_type: str = "experience"
    source_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    user_id: str
    facts: list[CareerFactIn]
    replace: bool = True


class IngestResponse(BaseModel):
    user_id: str
    inserted: int


class TailorRequest(BaseModel):
    user_id: str
    jd_text: str
    target_sections: list[str] = Field(
        default_factory=lambda: ["summary", "experience"]
    )
    strategy: str = "hybrid"


class CitedFact(BaseModel):
    fact_id: str
    content: str
    source_type: str
    score: float


class TailoredBulletOut(BaseModel):
    section: str
    text: str
    confidence: float
    citations: list[CitedFact]


class TailorResponse(BaseModel):
    bullets: list[TailoredBulletOut]
    iterations: int
    duration_ms: int
    model_used: str
    finish_summary: str
    notes: list[str]


class SearchRequest(BaseModel):
    user_id: str
    query: str
    strategy: str = "hybrid"
    k: int = 6


class SearchHitOut(BaseModel):
    fact_id: str
    content: str
    score: float
    source_type: str


class SearchResponse(BaseModel):
    hits: list[SearchHitOut]
