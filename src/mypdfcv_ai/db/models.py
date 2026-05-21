"""SQLAlchemy models.

Storage is SQLite for the demo. The repository layer below keeps the
business logic ignorant of this — swapping to Postgres + pgvector means
implementing one more `Repository` subclass and a new embedding column type.
The query semantics are identical (cosine sim ranking) since we precompute
normalised embeddings and use dot product.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SourceType(str, Enum):
    EXPERIENCE = "experience"
    PROJECT = "project"
    EDUCATION = "education"
    CERTIFICATION = "certification"
    SKILL = "skill"
    FREEFORM = "freeform"


def _uuid() -> str:
    return str(uuid.uuid4())


class CareerFact(Base):
    """An atomic, citation-worthy piece of the user's career history.

    One bullet point on a resume corresponds to one fact. We store the raw
    text plus the embedding (as a JSON-encoded list of floats — fine for the
    demo's <1k row scale; swap to pgvector for production)."""

    __tablename__ = "career_facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    fact_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    embedding_json: Mapped[str | None] = mapped_column("embedding", Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def get_embedding(self) -> list[float] | None:
        return json.loads(self.embedding_json) if self.embedding_json else None

    def set_embedding(self, vec: list[float]) -> None:
        self.embedding_json = json.dumps(vec)


class TailoringRun(Base):
    """One end-to-end tailoring request and its outcome.

    Stored so the eval harness and a future feedback loop can replay them."""

    __tablename__ = "tailoring_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    jd_text: Mapped[str] = mapped_column(Text)
    bullets_json: Mapped[str] = mapped_column(Text)  # list of TailoredBullet dicts
    iterations: Mapped[int] = mapped_column(default=0)
    duration_ms: Mapped[int] = mapped_column(default=0)
    model_used: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def get_bullets(self) -> list[dict]:
        return json.loads(self.bullets_json)

    def set_bullets(self, bullets: list[dict]) -> None:
        self.bullets_json = json.dumps(bullets)
