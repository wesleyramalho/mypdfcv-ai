"""End-to-end tailoring eval.

For each JD in the eval set, run the agent and measure:
  - bullets_emitted:  count
  - mean_confidence:  self-reported
  - hallucination_rate: post-hoc check, fraction of bullets whose numbers/
                       capitalised terms do not appear anywhere in the
                       *entire* user corpus (catches even cases where
                       verify_claim let something through)
  - groundedness_judge: LLM-judge score 0–4 averaged across bullets

The hallucination check uses the FULL corpus (not just cited facts) so a
bullet that cited the wrong fact but happens to be true is forgiven.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from mypdfcv_ai.agents.tailor_agent import TailoredBullet, run_tailor_agent
from mypdfcv_ai.config import ROOT_DIR, get_settings
from mypdfcv_ai.db.repository import CareerFactsRepository
from mypdfcv_ai.db.session import get_sessionmaker
from mypdfcv_ai.llm.client import get_llm_client
from mypdfcv_ai.retrieval.factory import build_retriever

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")
_STOPWORDS = {"The", "A", "An", "This", "That", "These", "Those", "We", "I", "They", "It"}


@dataclass(slots=True)
class TailoringReport:
    jd_name: str
    bullets_emitted: int
    iterations: int
    duration_ms: int
    mean_confidence: float
    hallucination_rate: float
    judge_score: float
    judge_rationale: str


def hallucination_rate(bullets: list[TailoredBullet], corpus_text: str) -> float:
    if not bullets:
        return 0.0
    corpus_lower = corpus_text.lower()
    bad = 0
    for b in bullets:
        suspects = set(_NUMBER_RE.findall(b.text)) | {
            p for p in _PROPER_RE.findall(b.text) if p not in _STOPWORDS
        }
        if not suspects:
            continue
        if any(s.lower() not in corpus_lower for s in suspects):
            bad += 1
    return bad / len(bullets)


JUDGE_SYSTEM = """You are evaluating resume bullets for a job application.

For each bullet, score 0-4 on a single dimension: GROUNDEDNESS — does every concrete claim (numbers, employer names, technologies, outcomes) plausibly come from the candidate's history?

Return JSON: {"score": <0-4 mean across bullets>, "rationale": "<one sentence>"}"""


def judge(bullets: list[TailoredBullet], corpus_text: str) -> tuple[float, str]:
    if not bullets:
        return 0.0, "no bullets to judge"
    client = get_llm_client()
    model = get_settings().judge_model

    payload = {
        "candidate_history": corpus_text,
        "bullets": [{"section": b.section, "text": b.text} for b in bullets],
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=300,
    )
    content = completion.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
        return float(data.get("score", 0)), str(data.get("rationale", ""))[:280]
    except (json.JSONDecodeError, ValueError):
        return 0.0, f"judge returned malformed output: {content[:80]}"


def run(
    *,
    user_id: str = "demo-user",
    jd_dir: Path | None = None,
    out_path: Path | None = None,
    strategy: str = "hybrid",
    sections: tuple[str, ...] = ("summary", "experience.current"),
) -> str:
    jd_dir = jd_dir or (ROOT_DIR / "eval" / "datasets" / "jds")
    reports: list[TailoringReport] = []

    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        all_facts = CareerFactsRepository(session).list_for_user(user_id)
        corpus_text = "\n".join(f.content for f in all_facts)

    for jd_path in sorted(jd_dir.glob("*.txt")):
        jd_text = jd_path.read_text()
        with SessionLocal() as session:
            retriever = build_retriever(session, user_id, strategy=strategy)  # type: ignore[arg-type]
            result = run_tailor_agent(
                jd_text=jd_text,
                target_sections=list(sections),
                retriever=retriever,
                user_id=user_id,
            )

        mean_conf = (
            sum(b.confidence for b in result.bullets) / len(result.bullets)
            if result.bullets
            else 0.0
        )
        hr = hallucination_rate(result.bullets, corpus_text)
        js, jr = judge(result.bullets, corpus_text)
        reports.append(
            TailoringReport(
                jd_name=jd_path.stem,
                bullets_emitted=len(result.bullets),
                iterations=result.iterations,
                duration_ms=result.duration_ms,
                mean_confidence=round(mean_conf, 3),
                hallucination_rate=round(hr, 3),
                judge_score=round(js, 2),
                judge_rationale=jr,
            )
        )

    md = _render_markdown(reports, strategy=strategy)
    if out_path:
        out_path.write_text(md)
    return md


def _render_markdown(reports: list[TailoringReport], *, strategy: str) -> str:
    lines = [f"# Tailoring eval (strategy={strategy})", ""]
    lines.append(
        "| JD | bullets | iters | ms | mean conf | halluc. rate | judge 0-4 | rationale |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in reports:
        lines.append(
            f"| {r.jd_name} | {r.bullets_emitted} | {r.iterations} | {r.duration_ms} | "
            f"{r.mean_confidence:.2f} | {r.hallucination_rate:.2f} | {r.judge_score:.2f} | {r.judge_rationale} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    out = ROOT_DIR / "eval" / "reports" / "tailoring.md"
    md = run(out_path=out)
    print(md)
    print(f"\nWrote {out}")
