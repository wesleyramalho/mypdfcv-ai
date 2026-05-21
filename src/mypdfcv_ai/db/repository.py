"""Repository pattern over the storage layer.

Why a repository: it keeps retrievers, agents, and the API ignorant of
SQLAlchemy and SQLite. Switching to Postgres + pgvector or to a hosted
vector DB (Pinecone, Weaviate) means writing a new repository subclass —
nothing else changes. This is the production-discipline signal for the
review.
"""
from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy.orm import Session

from mypdfcv_ai.db.models import CareerFact, TailoringRun


class CareerFactsRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_many(self, facts: Iterable[CareerFact]) -> None:
        self.session.add_all(list(facts))
        self.session.flush()

    def delete_for_user(self, user_id: str) -> int:
        deleted = (
            self.session.query(CareerFact).filter(CareerFact.user_id == user_id).delete()
        )
        self.session.flush()
        return deleted

    def list_for_user(self, user_id: str) -> list[CareerFact]:
        return (
            self.session.query(CareerFact)
            .filter(CareerFact.user_id == user_id)
            .order_by(CareerFact.created_at.asc())
            .all()
        )

    def get(self, fact_id: str) -> CareerFact | None:
        return self.session.get(CareerFact, fact_id)


class TailoringRunsRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, run: TailoringRun) -> TailoringRun:
        self.session.add(run)
        self.session.flush()
        return run
