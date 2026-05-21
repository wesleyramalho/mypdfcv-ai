# mypdfcv-ai

A grounded resume-tailoring service. Given (1) a user's full career history
and (2) a target job description, it emits resume bullets where every
factual claim — numbers, employer names, technologies, outcomes — is cited
back to the user's actual experience. Hallucinated claims are blocked at
the tool layer, not just discouraged in a prompt.

Built as a focused portfolio piece in ~1 working day. Designed to
demonstrate the skill set on a senior AI Engineer JD:

- **Production-style Python** — FastAPI, SQLAlchemy, repository pattern, structlog
- **RAG with three strategies** — dense / BM25 / hybrid (RRF), measured head-to-head
- **Tool-calling agent** — bounded loop, structured output, deterministic grounding gate
- **Hallucination control** — citation tracking + confidence scoring + verify_claim tool
- **Eval harness** — retrieval P@k/R@k/MRR and tailoring quality with LLM-judge
- **Provider-agnostic LLM layer** — runs on free OpenRouter tier today; swap to Claude/GPT with one env var

## Architecture

```
┌─────────────────┐   HTTPS / JSON    ┌────────────────────────────────────┐
│  Streamlit UI   │ ────────────────► │   FastAPI service                  │
│  (demo/app.py)  │                   │                                    │
└─────────────────┘                   │  /v1/career/facts   ingest         │
                                      │  /v1/search          retrieve only │
                                      │  /v1/tailor          full agent    │
                                      │  /v1/healthz                       │
                                      │                                    │
                                      │  ┌──────────────────────────────┐  │
                                      │  │ tailor_agent (loop)          │  │
                                      │  │   ┌────────────────────────┐ │  │
                                      │  │   │ tools                  │ │  │
                                      │  │   │  search_jd_requirements│ │  │
                                      │  │   │  search_history        │ │  │
                                      │  │   │  verify_claim   ◄ HARD │ │  │
                                      │  │   │  emit_bullet     GATE  │ │  │
                                      │  │   │  finish                │ │  │
                                      │  │   └────────────────────────┘ │  │
                                      │  └──────────────┬───────────────┘  │
                                      │                 │                  │
                                      │  ┌──────────────▼───────────────┐  │
                                      │  │ Retrieval (Retriever)        │  │
                                      │  │   dense (sentence-transformers│ │
                                      │  │           + numpy cosine)    │  │
                                      │  │   bm25  (rank-bm25)          │  │
                                      │  │   hybrid (RRF, k=60)         │  │
                                      │  └──────────────┬───────────────┘  │
                                      │                 │                  │
                                      │  ┌──────────────▼───────────────┐  │
                                      │  │ Repository                   │  │
                                      │  │   SQLite (demo) / pgvector   │  │
                                      │  │   (production target)        │  │
                                      │  └──────────────────────────────┘  │
                                      │                                    │
                                      │  ┌──────────────────────────────┐  │
                                      │  │ LLM layer                    │  │
                                      │  │   OpenAI SDK → OpenRouter →  │  │
                                      │  │   Gemini Flash (free) /      │  │
                                      │  │   Claude / GPT (env-swap)    │  │
                                      │  └──────────────────────────────┘  │
                                      └────────────────────────────────────┘
```

## How the hallucination gate works

The agent has five tools. Two of them — `verify_claim` and `emit_bullet` —
are deterministic Python, not the LLM:

- `verify_claim` extracts every number, year, and proper noun from a draft
  bullet and checks whether each appears in the cited fact texts. If a
  claim is unsupported, it returns `grounded=false` with the offending
  terms, and the agent has to revise.
- `emit_bullet` re-runs the same check before persisting the bullet. The
  agent literally cannot return a bullet whose claims aren't anchored in
  retrieved facts.

This is the "validation strategies, graceful failure mechanisms" part of
the JD made concrete: the prompt asks the agent not to hallucinate, but
the *tool layer* enforces it.

## Quickstart

```bash
git clone <repo> mypdfcv-ai && cd mypdfcv-ai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# OpenRouter free-tier key from https://openrouter.ai/keys
cp .env.example .env  # then paste your key into OPENROUTER_API_KEY=

# seed demo profile
python -m mypdfcv_ai.cli seed

# run the API (terminal 1)
uvicorn mypdfcv_ai.main:app --reload

# run the Streamlit demo (terminal 2)
streamlit run demo/app.py
```

Or, end-to-end from the CLI:

```bash
python -m mypdfcv_ai.cli tailor \
  eval/datasets/jds/ai_engineer_latam.txt \
  --strategy hybrid \
  --sections summary,experience.current
```

## Eval

```bash
# retrieval only — no LLM, fast
python -m eval.runners.retrieval

# end-to-end (uses the agent + LLM judge)
python -m eval.runners.tailoring

# everything
python -m eval.run
```

Reports land in `eval/reports/*.md`. The retrieval table on the demo
profile shows BM25 winning on this small corpus because the queries share
a lot of vocabulary with the facts; dense and hybrid have higher MRR
(first-relevant rank), which matters more for agent retrieval. **A real
deployment would re-tune on production query logs** — the eval harness
makes that re-tune mechanical.

## Repo layout

```
src/mypdfcv_ai/
├── api/         # FastAPI router + Pydantic schemas
├── agents/      # Tailor agent + tool schemas
├── retrieval/   # Retriever protocol + dense / bm25 / hybrid
├── grounding/   # verify_claim + confidence formula
├── ingestion/   # sentence-transformers wrapper
├── llm/         # OpenRouter-backed OpenAI client
├── db/          # SQLAlchemy models + repository
├── config.py    # pydantic-settings
└── main.py      # FastAPI app
eval/
├── datasets/    # demo_profile.json + JDs + retrieval_gold.json
├── runners/     # retrieval.py + tailoring.py
└── reports/     # markdown reports
demo/app.py      # Streamlit UI
tests/           # pytest smoke tests
```

## Design rationale

**Why a separate AI service instead of putting this in the Next.js app?**
The resume builder is a static, client-heavy Next.js project. An AI service
needs Python (the JD requires 5+ years of it), a different deployment
cadence, separate cost accounting, and freedom to use heavy native deps
(sentence-transformers, torch). Splitting the repo is a production
discipline signal — the boundary is explicit.

**Why SQLite, not pgvector?** Demo portability. The reviewer can
`git clone && pip install && python -m mypdfcv_ai.cli seed && streamlit run` with
no Docker step. The `Repository` abstraction in `db/repository.py` means
the pgvector swap is one new file plus a migration; the agent loop and
retrievers do not change.

**Why hand-roll the agent loop instead of using LangChain / LangGraph?**
The loop is ~150 lines. Hand-rolling it makes the iteration bound,
structured-output contract, and tool dispatch visible to a reviewer in
ten minutes. Frameworks would hide the parts of the system that matter
most for the interview.

**Why OpenRouter + free Gemini Flash, not Claude?** Demonstrating the
provider abstraction is more valuable than demonstrating "I have Claude
credits." The exact same code targets `anthropic/claude-sonnet-4`,
`openai/gpt-4o`, or a local Ollama endpoint by changing one env var.

**Why deterministic grounding instead of LLM-as-judge?** Both, actually.
`verify_claim` runs synchronously inside the agent loop (cheap, fast,
catches the obvious cases). The eval harness adds an LLM judge for the
groundedness score — that's a slower, fuzzier signal we only need offline.

## What I'd do next

| Item | Why | Effort |
|---|---|---|
| Postgres + pgvector swap | Production scale, RLS for multi-tenant | 0.5 day |
| Skill-graph retrieval (O*NET) | The GraphRAG bullet on the JD | 1 day |
| Multimodal `/import` (vision LLM) | Phone photo → resume JSON | 0.5 day |
| Cloud Run + Supabase free-tier deploy | Public demo URL | 0.5 day |
| Confidence threshold tuning vs eval set | Make the weights non-arbitrary | 0.5 day |
| LLM-judge entailment in `verify_claim` | Catch hallucinations the regex misses | 0.5 day |
| Feedback-loop UI in the Next.js editor | Accept/Reject signals → fine-tune dataset | 1 day |

## License

MIT. See `LICENSE`.
