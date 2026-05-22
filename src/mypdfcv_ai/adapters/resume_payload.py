"""Adapt the FE ResumeData shape into a list of in-memory CareerFact records.

The FE (mypdfcv, Next.js) stores resumes in the browser and sends the whole
ResumeData on every tailoring request. We turn that structured shape into
the same `CareerFact` objects the retrievers normally read from the DB —
but unmanaged, never flushed. The retrievers don't care whether the facts
came from SQLAlchemy or from a request body; they only read attributes.

Each fact carries a `source_id` that ties it back to the originating
entry in the input resume. The FE uses these to highlight which experience
or project supports each tailored bullet (e.g. `experience.<uuid>`).
"""
from __future__ import annotations

import uuid

from mypdfcv_ai.api.schemas import ResumeDataIn
from mypdfcv_ai.db.models import CareerFact, SourceType


def _make_fact(*, content: str, source_type: str, source_id: str, metadata: dict) -> CareerFact:
    fact = CareerFact(
        id=str(uuid.uuid4()),
        user_id="__inline__",
        source_type=source_type,
        source_id=source_id,
        content=content,
        fact_metadata=metadata,
    )
    return fact


def _experience_header(exp) -> str:
    pieces = [p for p in [exp.title, exp.company] if p]
    header = " at ".join(pieces) if len(pieces) == 2 else (pieces[0] if pieces else "")
    span_bits = []
    if exp.start_date:
        span_bits.append(exp.start_date)
    if exp.current:
        span_bits.append("Present")
    elif exp.end_date:
        span_bits.append(exp.end_date)
    if span_bits:
        header = f"{header} ({'–'.join(span_bits)})" if header else "–".join(span_bits)
    if exp.location:
        header = f"{header}, {exp.location}" if header else exp.location
    return header


def _education_text(edu) -> str:
    pieces = [p for p in [edu.degree, edu.field] if p]
    degree = " in ".join(pieces) if len(pieces) == 2 else (pieces[0] if pieces else "")
    parts = [p for p in [degree, edu.school] if p]
    head = ", ".join(parts)
    span_bits = [b for b in [edu.start_date, edu.end_date] if b]
    if span_bits:
        head = f"{head} ({'–'.join(span_bits)})" if head else "–".join(span_bits)
    if edu.gpa:
        head = f"{head}, GPA {edu.gpa}"
    if edu.highlights:
        head = f"{head}. {edu.highlights}" if head else edu.highlights
    return head


def resume_to_facts(resume: ResumeDataIn) -> list[CareerFact]:
    """Flatten a ResumeData into a citation-friendly list of CareerFact records.

    Granularity rule of thumb: one fact per *thing the agent can cite as a
    single source*. A multi-line experience description becomes one fact per
    non-empty line, so the agent can cite a single bullet rather than a whole
    paragraph.
    """
    facts: list[CareerFact] = []

    if resume.summary.strip():
        facts.append(
            _make_fact(
                content=resume.summary.strip(),
                source_type=SourceType.FREEFORM.value,
                source_id="summary",
                metadata={"section": "summary"},
            )
        )

    if resume.headline.strip():
        headline = resume.headline.strip()
        if resume.full_name.strip():
            headline = f"{resume.full_name.strip()} — {headline}"
        facts.append(
            _make_fact(
                content=headline,
                source_type=SourceType.FREEFORM.value,
                source_id="headline",
                metadata={"section": "headline"},
            )
        )

    for exp in resume.experience:
        header = _experience_header(exp)
        if header:
            facts.append(
                _make_fact(
                    content=header,
                    source_type=SourceType.EXPERIENCE.value,
                    source_id=f"experience.{exp.id}",
                    metadata={
                        "company": exp.company,
                        "title": exp.title,
                        "kind": "header",
                    },
                )
            )
        # Bullets — split the description on newlines so the agent can cite
        # one specific bullet, not the whole paragraph.
        lines = [line.strip(" -•\t") for line in exp.description.splitlines()]
        lines = [line for line in lines if line]
        for idx, line in enumerate(lines):
            # Re-prefix with employer + title so the line stands alone and the
            # grounding regex sees the proper noun in context.
            context_prefix = ""
            if exp.title and exp.company:
                context_prefix = f"At {exp.company} as {exp.title}: "
            elif exp.company:
                context_prefix = f"At {exp.company}: "
            content = f"{context_prefix}{line}" if context_prefix else line
            facts.append(
                _make_fact(
                    content=content,
                    source_type=SourceType.EXPERIENCE.value,
                    source_id=f"experience.{exp.id}.bullet.{idx}",
                    metadata={
                        "company": exp.company,
                        "title": exp.title,
                        "kind": "bullet",
                        "bullet_index": idx,
                    },
                )
            )

    for proj in resume.projects:
        parts = [proj.name]
        if proj.description:
            parts.append(proj.description)
        if proj.technologies:
            parts.append("Technologies: " + ", ".join(proj.technologies))
        content = ". ".join(p.strip() for p in parts if p.strip())
        if content:
            facts.append(
                _make_fact(
                    content=content,
                    source_type=SourceType.PROJECT.value,
                    source_id=f"project.{proj.id}",
                    metadata={"name": proj.name, "technologies": proj.technologies},
                )
            )

    for group in resume.skill_groups:
        if not group.skills:
            continue
        head = f"{group.category}: " if group.category else ""
        content = head + ", ".join(s for s in group.skills if s)
        facts.append(
            _make_fact(
                content=content,
                source_type=SourceType.SKILL.value,
                source_id=f"skills.{group.id}",
                metadata={"category": group.category},
            )
        )

    for edu in resume.education:
        content = _education_text(edu)
        if content:
            facts.append(
                _make_fact(
                    content=content,
                    source_type=SourceType.EDUCATION.value,
                    source_id=f"education.{edu.id}",
                    metadata={"school": edu.school, "degree": edu.degree},
                )
            )

    return facts
