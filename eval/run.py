"""Run all eval suites and write reports under eval/reports/."""
from __future__ import annotations

from mypdfcv_ai.config import ROOT_DIR
from eval.runners import retrieval, tailoring


def main() -> None:
    reports_dir = ROOT_DIR / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("== retrieval ==")
    print(retrieval.run(out_path=reports_dir / "retrieval.md"))

    print("\n== tailoring ==")
    print(tailoring.run(out_path=reports_dir / "tailoring.md"))


if __name__ == "__main__":
    main()
