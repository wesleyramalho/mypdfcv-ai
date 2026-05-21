# mypdfcv-ai

A grounded resume tailoring service. Takes a user's full career history plus a
job description, returns rewritten resume bullets where every factual claim is
**cited back to the user's actual experience** — never invented.

Built as a portfolio piece for an AI Engineer role. Designed to demonstrate:

- Production-style Python (FastAPI, clean architecture, repository pattern)
- RAG with three retrieval strategies compared head-to-head (dense / BM25 / hybrid)
- Tool-calling agent loops with bounded iteration and structured output
- Hallucination control via citation tracking and confidence scoring
- Retrieval & tailoring evaluation harness with markdown reports
- Provider-agnostic LLM layer (runs on free OpenRouter models; swap to Claude/GPT with one env var)

> **Quickstart**, **architecture diagram**, and **eval results** below.

---

## Quickstart

```bash
git clone <repo> mypdfcv-ai && cd mypdfcv-ai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # add OPENROUTER_API_KEY

# initialise SQLite + seed demo data
python -m mypdfcv_ai.cli seed

# run the API
uvicorn mypdfcv_ai.main:app --reload

# in another terminal: the Streamlit demo
streamlit run demo/app.py

# run the eval harness
python -m mypdfcv_ai.eval.run
```

## Architecture

(Filled in by Phase 11.)

## Eval results

(Filled in after the eval runner produces its first report.)

## Why these choices

(Design rationale section — filled in last.)
