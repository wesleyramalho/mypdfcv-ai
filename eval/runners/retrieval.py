"""Retrieval quality eval.

For each (query, ideal_fact_substrings) pair, run each strategy and compute:
  - P@k:   fraction of top-k hits that are relevant
  - R@k:   fraction of relevant facts that appear in top-k
  - MRR:   reciprocal rank of the first relevant hit (averaged)

"Relevant" = the hit's content contains at least one ideal substring.

This file is the single most under-loved portfolio artifact: most candidates
say "I built a RAG pipeline"; very few measure their pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mypdfcv_ai.config import ROOT_DIR
from mypdfcv_ai.db.session import get_sessionmaker
from mypdfcv_ai.retrieval.base import Hit, Strategy
from mypdfcv_ai.retrieval.factory import build_retriever

GOLD_PATH = ROOT_DIR / "eval" / "datasets" / "retrieval_gold.json"
DEFAULT_K = 5
STRATEGIES: list[Strategy] = ["dense", "bm25", "hybrid"]


@dataclass(slots=True)
class QueryReport:
    query: str
    strategy: str
    p_at_k: float
    r_at_k: float
    mrr: float
    relevant_hits: int
    total_relevant: int


def _is_relevant(hit: Hit, ideal_substrings: list[str]) -> bool:
    text = hit.content.lower()
    return any(sub.lower() in text for sub in ideal_substrings)


def evaluate_query(
    query: str,
    ideal_substrings: list[str],
    *,
    user_id: str,
    k: int = DEFAULT_K,
) -> list[QueryReport]:
    SessionLocal = get_sessionmaker()
    reports: list[QueryReport] = []
    with SessionLocal() as session:
        for strategy in STRATEGIES:
            retriever = build_retriever(session, user_id, strategy=strategy)
            hits = retriever.search(query, k=k)

            relevant_hit_count = sum(1 for h in hits if _is_relevant(h, ideal_substrings))

            # MRR
            first_rel_rank = next(
                (i + 1 for i, h in enumerate(hits) if _is_relevant(h, ideal_substrings)),
                None,
            )
            mrr = 1.0 / first_rel_rank if first_rel_rank else 0.0

            # R@k uses total relevant in the *full corpus*, not just in top-k.
            # We approximate by also pulling top 20 and counting relevant there.
            wide = retriever.search(query, k=20)
            total_relevant = max(
                sum(1 for h in wide if _is_relevant(h, ideal_substrings)),
                relevant_hit_count,
            )

            reports.append(
                QueryReport(
                    query=query,
                    strategy=strategy,
                    p_at_k=relevant_hit_count / max(len(hits), 1),
                    r_at_k=relevant_hit_count / max(total_relevant, 1),
                    mrr=mrr,
                    relevant_hits=relevant_hit_count,
                    total_relevant=total_relevant,
                )
            )
    return reports


def run(*, user_id: str = "demo-user", k: int = DEFAULT_K, out_path: Path | None = None) -> str:
    gold = json.loads(GOLD_PATH.read_text())
    all_reports: list[QueryReport] = []
    for q in gold["queries"]:
        all_reports.extend(
            evaluate_query(
                q["query"],
                q["ideal_fact_substrings"],
                user_id=user_id,
                k=k,
            )
        )

    md = _render_markdown(all_reports, k=k)
    if out_path:
        out_path.write_text(md)
    return md


def _render_markdown(reports: list[QueryReport], *, k: int) -> str:
    lines: list[str] = [f"# Retrieval eval (k={k})", ""]

    # Per-query table.
    lines.append("## Per-query results")
    lines.append("")
    lines.append("| Query | Strategy | P@k | R@k | MRR | Rel. in top-k |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in reports:
        lines.append(
            f"| {r.query} | {r.strategy} | {r.p_at_k:.2f} | {r.r_at_k:.2f} | {r.mrr:.2f} | {r.relevant_hits}/{r.total_relevant} |"
        )

    # Aggregate per strategy.
    lines += ["", "## Aggregate (mean across queries)", ""]
    lines.append("| Strategy | mean P@k | mean R@k | mean MRR |")
    lines.append("|---|---:|---:|---:|")
    by_strategy: dict[str, list[QueryReport]] = {}
    for r in reports:
        by_strategy.setdefault(r.strategy, []).append(r)
    for strategy, rs in by_strategy.items():
        mp = sum(r.p_at_k for r in rs) / len(rs)
        mr = sum(r.r_at_k for r in rs) / len(rs)
        mm = sum(r.mrr for r in rs) / len(rs)
        lines.append(f"| {strategy} | {mp:.3f} | {mr:.3f} | {mm:.3f} |")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    out = ROOT_DIR / "eval" / "reports" / "retrieval.md"
    md = run(out_path=out)
    print(md)
    print(f"\nWrote {out}")
