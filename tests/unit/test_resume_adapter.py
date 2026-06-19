"""Adapter: FE ResumeData → list of in-memory CareerFact records.

Anchors the contract used by /v1/tailor-resume so an FE change to the
resume shape gets caught here before it hits the agent loop.
"""
from __future__ import annotations

from mypdfcv_ai.adapters.resume_payload import resume_to_facts
from mypdfcv_ai.api.schemas import ResumeDataIn


def _sample_resume() -> ResumeDataIn:
    return ResumeDataIn.model_validate(
        {
            "fullName": "Ada Lovelace",
            "headline": "Senior AI Engineer",
            "summary": "Ten years building grounded RAG systems.",
            "contact": {
                "email": "ada@example.com",
                "phone": "",
                "location": "Rio",
                "linkedin": "",
                "website": "",
            },
            "experience": [
                {
                    "id": "exp-1",
                    "company": "Acme Logistics",
                    "title": "Senior Engineer",
                    "location": "Remote",
                    "startDate": "2022-01",
                    "endDate": None,
                    "current": True,
                    "description": (
                        "- Cut p99 latency by 40% via FastAPI migration.\n"
                        "- Built ETL pipelines processing 50TB/day.\n"
                    ),
                },
                {
                    "id": "exp-2",
                    "company": "Globex",
                    "title": "Engineer",
                    "location": "",
                    "startDate": "2019-01",
                    "endDate": "2021-12",
                    "current": False,
                    "description": "Shipped a recommendation system used by 2M users.",
                },
            ],
            "education": [
                {
                    "id": "edu-1",
                    "school": "UFRJ",
                    "degree": "BSc",
                    "field": "Computer Science",
                    "startDate": "2014",
                    "endDate": "2018",
                    "gpa": "9.1",
                    "highlights": "Thesis on approximate nearest neighbour search.",
                }
            ],
            "skillGroups": [
                {
                    "id": "sk-1",
                    "category": "Languages",
                    "skills": ["Python", "TypeScript"],
                },
                {"id": "sk-empty", "category": "Empty", "skills": []},
            ],
            "projects": [
                {
                    "id": "proj-1",
                    "name": "mypdfcv-ai",
                    "description": "Grounded resume tailor with RAG.",
                    "url": None,
                    "technologies": ["FastAPI", "BGE-small"],
                    "startDate": "2026-05",
                    "endDate": None,
                }
            ],
            "sections": {
                "summary": True,
                "experience": True,
                "education": True,
                "skills": True,
                "projects": True,
            },
            "sectionOrder": ["personal", "summary", "experience"],
        }
    )


def test_adapter_emits_one_fact_per_atomic_entry():
    facts = resume_to_facts(_sample_resume())

    by_source = {f.source_id: f for f in facts}

    # summary + headline
    assert "summary" in by_source
    assert "headline" in by_source
    assert "Ada Lovelace" in by_source["headline"].content
    assert "Senior AI Engineer" in by_source["headline"].content

    # experience header + one bullet per non-empty line
    assert "experience.exp-1" in by_source
    assert by_source["experience.exp-1"].source_type == "experience"
    assert "experience.exp-1.bullet.0" in by_source
    assert "experience.exp-1.bullet.1" in by_source
    assert "experience.exp-1.bullet.2" not in by_source  # only 2 non-empty lines
    # Bullet content should be prefixed with employer context so the
    # grounding regex sees the proper noun.
    assert "Acme Logistics" in by_source["experience.exp-1.bullet.0"].content
    assert "40%" in by_source["experience.exp-1.bullet.0"].content

    # single-line experience → header + 1 bullet
    assert "experience.exp-2" in by_source
    assert "experience.exp-2.bullet.0" in by_source

    # education / projects / skills
    assert "education.edu-1" in by_source
    assert "UFRJ" in by_source["education.edu-1"].content
    assert "project.proj-1" in by_source
    assert "FastAPI" in by_source["project.proj-1"].content
    assert "skills.sk-1" in by_source
    assert "Python" in by_source["skills.sk-1"].content

    # empty skills group is dropped
    assert "skills.sk-empty" not in by_source


def test_adapter_ids_are_unique_and_36_char_uuids():
    facts = resume_to_facts(_sample_resume())
    ids = [f.id for f in facts]
    assert len(ids) == len(set(ids)), "fact ids must be unique"
    for fid in ids:
        assert len(fid) == 36, f"fact id is not a 36-char UUID: {fid!r}"


def test_adapter_handles_empty_resume():
    facts = resume_to_facts(ResumeDataIn())
    assert facts == []
