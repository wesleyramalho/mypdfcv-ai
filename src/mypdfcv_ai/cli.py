"""CLI for seeding the demo profile and running ad-hoc tailoring.

Designed so the interview demo can be driven from the command line as a
fallback if Streamlit has any issues.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from mypdfcv_ai.agents.tailor_agent import run_tailor_agent
from mypdfcv_ai.config import ROOT_DIR
from mypdfcv_ai.db.models import CareerFact
from mypdfcv_ai.db.repository import CareerFactsRepository
from mypdfcv_ai.db.session import get_sessionmaker, init_db
from mypdfcv_ai.ingestion.embedder import embed_texts
from mypdfcv_ai.retrieval.factory import build_retriever

DEMO_USER_ID = "demo-user"
DEMO_PROFILE_PATH = ROOT_DIR / "eval" / "datasets" / "demo_profile.json"

app = typer.Typer(no_args_is_help=True)


@app.command()
def seed(profile: Path = DEMO_PROFILE_PATH, user_id: str = DEMO_USER_ID) -> None:
    """Initialise the DB and load a demo career profile into it."""
    init_db()
    data = json.loads(profile.read_text())
    facts_in = data["facts"]

    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        repo = CareerFactsRepository(session)
        repo.delete_for_user(user_id)
        embeddings = embed_texts([f["content"] for f in facts_in])
        records = []
        for f, vec in zip(facts_in, embeddings, strict=True):
            fact = CareerFact(
                user_id=user_id,
                source_type=f.get("source_type", "experience"),
                content=f["content"],
                fact_metadata=f.get("metadata", {}),
            )
            fact.set_embedding(vec.tolist())
            records.append(fact)
        repo.add_many(records)
        session.commit()
    typer.echo(f"seeded {len(facts_in)} facts for user '{user_id}'")


@app.command()
def tailor(
    jd_file: Path,
    user_id: str = DEMO_USER_ID,
    strategy: str = "hybrid",
    sections: str = "summary,experience.current",
) -> None:
    """Run the tailoring agent on a JD file and pretty-print the bullets."""
    jd = jd_file.read_text()
    target_sections = [s.strip() for s in sections.split(",") if s.strip()]

    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        retriever = build_retriever(session, user_id, strategy=strategy)  # type: ignore[arg-type]
        result = run_tailor_agent(
            jd_text=jd,
            target_sections=target_sections,
            retriever=retriever,
            user_id=user_id,
        )

    typer.echo(f"\n=== Tailoring run ({result.iterations} iters, {result.duration_ms}ms, {result.model_used}) ===\n")
    if not result.bullets:
        typer.echo("(no bullets emitted)")
        sys.exit(1)
    for b in result.bullets:
        typer.echo(f"[{b.section}] (confidence {b.confidence})")
        typer.echo(f"  → {b.text}")
        for c in b.cited_facts:
            typer.echo(f"     · cite [{c.fact_id[:8]}] {c.content}")
        typer.echo("")
    if result.finish_summary:
        typer.echo(f"agent summary: {result.finish_summary}")


if __name__ == "__main__":
    app()
