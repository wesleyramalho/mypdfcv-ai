"""Stateless /v1/tailor-resume endpoint — contract + auth + DB isolation.

The agent loop itself is exercised in tests/unit/test_grounding.py and the
eval harness; here we stub `run_tailor_agent` so the test stays fast and
doesn't depend on OpenRouter being reachable. Strategy is forced to "bm25"
so the sentence-transformers model isn't loaded either.
"""
from __future__ import annotations

import os
import tempfile

# Force an isolated SQLite *before* any mypdfcv_ai import touches the engine.
# A temp file (not :memory:) so the lifespan's init_db() and the test's own
# session see the same database — pure :memory: is per-connection.
_TMP_DB = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
_TMP_DB.close()
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.name}"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mypdfcv_ai.agents.tailor_agent import TailoredBullet, TailoringResult  # noqa: E402
from mypdfcv_ai.config import get_settings  # noqa: E402
from mypdfcv_ai.db.session import get_sessionmaker  # noqa: E402
from mypdfcv_ai.main import app  # noqa: E402
from mypdfcv_ai.retrieval.base import Hit  # noqa: E402

SAMPLE_PAYLOAD = {
    "resume_data": {
        "fullName": "Ada Lovelace",
        "headline": "Senior AI Engineer",
        "summary": "Ten years building grounded RAG systems.",
        "contact": {
            "email": "",
            "phone": "",
            "location": "",
            "linkedin": "",
            "website": "",
        },
        "experience": [
            {
                "id": "exp-1",
                "company": "Acme Logistics",
                "title": "Senior Engineer",
                "location": "",
                "startDate": "2022-01",
                "endDate": None,
                "current": True,
                "description": "Cut p99 latency by 40% via FastAPI migration.",
            }
        ],
        "education": [],
        "skillGroups": [{"id": "sk-1", "category": "Languages", "skills": ["Python"]}],
        "projects": [],
    },
    "jd_text": "Looking for a senior Python engineer with latency optimisation experience.",
    "target_sections": ["summary", "experience.exp-1"],
    "strategy": "bm25",
}


@pytest.fixture(autouse=True)
def _reset_token():
    """Each test starts with no auth token (matches local-dev default)."""
    settings = get_settings()
    original = settings.tailor_api_token
    settings.tailor_api_token = ""
    yield
    settings.tailor_api_token = original


@pytest.fixture
def client():
    # `with` triggers FastAPI lifespan, which runs init_db() so the
    # `does_not_write_to_db` assertion has tables to query.
    with TestClient(app) as c:
        yield c


def _scripted_agent_result(fact_id: str) -> TailoringResult:
    """Mimic what run_tailor_agent would return for one grounded bullet."""
    return TailoringResult(
        bullets=[
            TailoredBullet(
                section="experience.exp-1",
                text="Cut p99 latency by 40% via FastAPI migration at Acme Logistics.",
                citations=[fact_id],
                cited_facts=[
                    Hit(
                        fact_id=fact_id,
                        content="At Acme Logistics as Senior Engineer: Cut p99 latency by 40% via FastAPI migration.",
                        score=0.91,
                        source_type="experience",
                        metadata={},
                    )
                ],
                confidence=0.82,
            )
        ],
        iterations=3,
        duration_ms=42,
        model_used="stub/test-model",
        finish_summary="ok",
        notes=[],
    )


def test_tailor_resume_returns_bullets_with_source_id(monkeypatch, client):
    captured: dict[str, object] = {}

    def fake_run_tailor_agent(*, jd_text, target_sections, retriever, user_id, client=None):  # noqa: ARG001
        captured["jd_text"] = jd_text
        captured["target_sections"] = target_sections
        captured["user_id"] = user_id
        # Pick the experience bullet from the in-memory retriever so the
        # response mapping has a known-good source_id to round-trip.
        hits = retriever.search("latency FastAPI migration", k=5)
        fact_id = next(
            (h.fact_id for h in hits if "FastAPI" in h.content),
            hits[0].fact_id if hits else "00000000-0000-0000-0000-000000000000",
        )
        return _scripted_agent_result(fact_id)

    monkeypatch.setattr(
        "mypdfcv_ai.api.routes.run_tailor_agent", fake_run_tailor_agent
    )

    resp = client.post("/v1/tailor-resume", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["iterations"] == 3
    assert body["model_used"] == "stub/test-model"
    assert len(body["bullets"]) == 1

    bullet = body["bullets"][0]
    assert bullet["section"] == "experience.exp-1"
    assert bullet["confidence"] == 0.82
    assert len(bullet["citations"]) == 1
    citation = bullet["citations"][0]
    # The point of this endpoint: citations carry a source_id the FE can
    # use to highlight the originating resume entry.
    assert citation["source_id"] is not None
    assert citation["source_id"].startswith("experience.exp-1")

    # The handler should have forwarded the request fields straight through.
    assert captured["target_sections"] == ["summary", "experience.exp-1"]
    assert captured["user_id"] == "__inline__"


def test_tailor_resume_rejects_unknown_strategy(client):
    bad = {**SAMPLE_PAYLOAD, "strategy": "bogus"}
    resp = client.post("/v1/tailor-resume", json=bad)
    assert resp.status_code == 400


def test_tailor_resume_rejects_empty_resume(client):
    empty = {
        "resume_data": {},
        "jd_text": "anything",
        "target_sections": ["summary"],
        "strategy": "bm25",
    }
    resp = client.post("/v1/tailor-resume", json=empty)
    assert resp.status_code == 400


def test_tailor_resume_enforces_shared_secret_when_set(monkeypatch, client):
    settings = get_settings()
    settings.tailor_api_token = "expected-secret"

    def fake_run_tailor_agent(*, retriever, **_):
        hits = retriever.search("x", k=1)
        fid = hits[0].fact_id if hits else "00000000-0000-0000-0000-000000000000"
        return _scripted_agent_result(fid)

    monkeypatch.setattr(
        "mypdfcv_ai.api.routes.run_tailor_agent", fake_run_tailor_agent
    )

    # Missing header → 401.
    resp_missing = client.post("/v1/tailor-resume", json=SAMPLE_PAYLOAD)
    assert resp_missing.status_code == 401

    # Wrong header → 401.
    resp_wrong = client.post(
        "/v1/tailor-resume",
        json=SAMPLE_PAYLOAD,
        headers={"X-Tailor-Token": "nope"},
    )
    assert resp_wrong.status_code == 401

    # Correct header → 200.
    resp_ok = client.post(
        "/v1/tailor-resume",
        json=SAMPLE_PAYLOAD,
        headers={"X-Tailor-Token": "expected-secret"},
    )
    assert resp_ok.status_code == 200, resp_ok.text


def test_tailor_resume_does_not_write_to_db(monkeypatch, client):
    def fake_run_tailor_agent(*, retriever, **_):
        hits = retriever.search("x", k=1)
        fid = hits[0].fact_id if hits else "00000000-0000-0000-0000-000000000000"
        return _scripted_agent_result(fid)

    monkeypatch.setattr(
        "mypdfcv_ai.api.routes.run_tailor_agent", fake_run_tailor_agent
    )

    resp = client.post("/v1/tailor-resume", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200, resp.text

    # Confirm no rows landed in either table.
    from mypdfcv_ai.db.models import CareerFact, TailoringRun

    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        assert session.query(CareerFact).count() == 0
        assert session.query(TailoringRun).count() == 0
