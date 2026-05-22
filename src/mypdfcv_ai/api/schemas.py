"""Pydantic request/response schemas for the HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CareerFactIn(BaseModel):
    content: str
    source_type: str = "experience"
    source_id: str | None = None
    metadata: dict = Field(default_factory=dict)


# --- Resume payload mirrors the mypdfcv FE TypeScript ResumeData shape.
# Kept intentionally permissive (extra fields ignored, optionals tolerated)
# so the FE can evolve without breaking this contract.


class ContactInfoIn(BaseModel):
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    website: str = ""


class ExperienceEntryIn(BaseModel):
    id: str
    company: str = ""
    title: str = ""
    location: str = ""
    start_date: str = Field("", alias="startDate")
    end_date: str | None = Field(None, alias="endDate")
    current: bool = False
    description: str = ""

    model_config = {"populate_by_name": True}


class EducationEntryIn(BaseModel):
    id: str
    school: str = ""
    degree: str = ""
    field: str = ""
    start_date: str = Field("", alias="startDate")
    end_date: str | None = Field(None, alias="endDate")
    gpa: str | None = None
    highlights: str = ""

    model_config = {"populate_by_name": True}


class SkillGroupIn(BaseModel):
    id: str
    category: str = ""
    skills: list[str] = Field(default_factory=list)


class ProjectEntryIn(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    url: str | None = None
    technologies: list[str] = Field(default_factory=list)
    start_date: str = Field("", alias="startDate")
    end_date: str | None = Field(None, alias="endDate")

    model_config = {"populate_by_name": True}


class ResumeDataIn(BaseModel):
    full_name: str = Field("", alias="fullName")
    headline: str = ""
    summary: str = ""
    contact: ContactInfoIn = Field(default_factory=ContactInfoIn)
    experience: list[ExperienceEntryIn] = Field(default_factory=list)
    education: list[EducationEntryIn] = Field(default_factory=list)
    skill_groups: list[SkillGroupIn] = Field(default_factory=list, alias="skillGroups")
    projects: list[ProjectEntryIn] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


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
    # source_id ties this citation back to a specific entry in the input
    # resume (e.g. "experience.<uuid>" or "experience.<uuid>.bullet.3").
    # Optional for backwards compatibility with the original /v1/tailor flow
    # where ingested facts may not carry a structured source_id.
    source_id: str | None = None


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


class TailorResumeRequest(BaseModel):
    """Stateless tailoring: the FE sends the whole resume per request.

    No prior /v1/career/facts ingest is required. The service builds an
    in-memory retriever from `resume_data`, runs the agent, returns the
    bullets, and persists nothing. This is the integration shape used by
    the Next.js FE which keeps resumes in localStorage.
    """

    resume_data: ResumeDataIn
    jd_text: str
    target_sections: list[str] = Field(default_factory=lambda: ["summary"])
    strategy: str = "hybrid"


class TailorResumeResponse(TailorResponse):
    """Same shape as TailorResponse; declared distinctly so the FE can pin to it."""


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
