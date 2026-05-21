"""Streamlit demo for screen-sharing in an interview.

Single page, three columns:
  - Left: the seeded career profile (facts, with IDs)
  - Center: paste a JD, hit Tailor, see emitted bullets with confidence bars
            and clickable citations
  - Right: live agent trace (iterations, tool calls, timing)

Why Streamlit, not Next.js: fastest path to a screen-shareable demo. The
real product surface is the FastAPI service; Streamlit is just a thin
inspector on top of it.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from mypdfcv_ai.agents.tailor_agent import run_tailor_agent
from mypdfcv_ai.config import ROOT_DIR
from mypdfcv_ai.db.repository import CareerFactsRepository
from mypdfcv_ai.db.session import get_sessionmaker, init_db
from mypdfcv_ai.retrieval.factory import build_retriever

DEMO_USER_ID = "demo-user"
DEMO_PROFILE_PATH = ROOT_DIR / "eval" / "datasets" / "demo_profile.json"
JD_DIR = ROOT_DIR / "eval" / "datasets" / "jds"

st.set_page_config(
    page_title="mypdfcv-ai — Grounded Resume Tailor",
    page_icon="🎯",
    layout="wide",
)

st.title("mypdfcv-ai — Grounded Resume Tailor")
st.caption(
    "Paste a job description on the right. The agent retrieves matching facts "
    "from the seeded career profile, drafts bullets, verifies grounding, and "
    "only emits bullets where every claim is cited back to the user's actual "
    "history."
)


init_db()
SessionLocal = get_sessionmaker()


# --- sidebar: model + strategy controls ---
with st.sidebar:
    st.subheader("Settings")
    strategy = st.selectbox(
        "Retrieval strategy",
        options=["hybrid", "dense", "bm25"],
        index=0,
        help="Hybrid (RRF of dense + BM25) is the default. Switch to see how it changes the agent's behaviour.",
    )
    target_sections_raw = st.text_input(
        "Target sections (comma-separated)",
        value="summary,experience.current",
    )
    target_sections = [s.strip() for s in target_sections_raw.split(",") if s.strip()]
    st.divider()
    st.subheader("Demo controls")
    if st.button("Re-seed demo profile"):
        from mypdfcv_ai.cli import seed

        seed(profile=DEMO_PROFILE_PATH, user_id=DEMO_USER_ID)
        st.success("Profile reloaded.")
        st.rerun()


# --- columns ---
col_facts, col_main = st.columns([1, 2])

with col_facts:
    st.subheader("Seeded career facts")
    with SessionLocal() as session:
        facts = CareerFactsRepository(session).list_for_user(DEMO_USER_ID)
    if not facts:
        st.warning("No facts seeded. Run `python -m mypdfcv_ai.cli seed` or click 'Re-seed' in the sidebar.")
    else:
        for f in facts:
            with st.expander(f"[{f.source_type}] {f.content[:80]}{'…' if len(f.content) > 80 else ''}"):
                st.markdown(f"**ID:** `{f.id[:8]}`")
                st.write(f.content)
                if f.fact_metadata:
                    st.json(f.fact_metadata, expanded=False)


with col_main:
    st.subheader("Job description")
    sample_jds = sorted(JD_DIR.glob("*.txt"))
    sample_choice = st.selectbox(
        "Or pick a sample JD:",
        options=["(custom)"] + [p.stem for p in sample_jds],
        index=1 if sample_jds else 0,
    )

    default_jd = ""
    if sample_choice != "(custom)":
        default_jd = next(p.read_text() for p in sample_jds if p.stem == sample_choice)

    jd_text = st.text_area("JD text", value=default_jd, height=260)

    run_tailor = st.button("Tailor resume bullets", type="primary", disabled=not jd_text.strip())

    if run_tailor:
        with st.spinner("Agent running — search → draft → verify → emit…"):
            with SessionLocal() as session:
                retriever = build_retriever(session, DEMO_USER_ID, strategy=strategy)  # type: ignore[arg-type]
                result = run_tailor_agent(
                    jd_text=jd_text,
                    target_sections=target_sections,
                    retriever=retriever,
                    user_id=DEMO_USER_ID,
                )

        st.success(
            f"Done in {result.iterations} iterations, {result.duration_ms} ms — model `{result.model_used}`"
        )

        if not result.bullets:
            st.warning("Agent finished without emitting any bullets.")
            if result.notes:
                with st.expander("Agent notes"):
                    for n in result.notes:
                        st.text(n)

        for b in result.bullets:
            with st.container(border=True):
                left, right = st.columns([3, 1])
                with left:
                    st.markdown(f"**[{b.section}]**  {b.text}")
                with right:
                    st.metric("confidence", f"{b.confidence:.2f}")
                    st.progress(b.confidence)
                with st.expander(f"Citations ({len(b.cited_facts)})"):
                    for c in b.cited_facts:
                        st.markdown(f"- `{c.fact_id[:8]}` — _{c.source_type}_ — score {c.score:.2f}")
                        st.markdown(f"  > {c.content}")

        if result.finish_summary:
            st.caption(f"Agent summary: {result.finish_summary}")

        with st.expander("Run JSON (for sharing / debugging)"):
            st.code(
                json.dumps(
                    {
                        "iterations": result.iterations,
                        "duration_ms": result.duration_ms,
                        "model": result.model_used,
                        "bullets": [
                            {
                                "section": b.section,
                                "text": b.text,
                                "confidence": b.confidence,
                                "citations": b.citations,
                            }
                            for b in result.bullets
                        ],
                        "notes": result.notes,
                    },
                    indent=2,
                ),
                language="json",
            )
