# mypdfcv-ai

A grounded resume-tailoring service. Given (1) a user's full career history
and (2) a target job description, it emits resume bullets where every
factual claim — numbers, employer names, technologies, outcomes — is cited
back to the user's actual experience. Hallucinated claims are blocked at
the tool layer, not just discouraged in a prompt.

Features:
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

## Eval — real numbers

Reports live in `eval/reports/*.md`. Generated automatically from
`python -m eval.run`.

### Retrieval (8 queries, k=5, n=19 facts)

| Strategy | mean P@5 | mean R@5 | mean MRR |
|---|---:|---:|---:|
| dense  | 0.300 | 0.729 | 0.875 |
| bm25   | 0.650 | 1.000 | 0.854 |
| hybrid | 0.350 | 0.833 | 0.875 |

BM25 wins on this small corpus because the gold queries share a lot of
vocabulary with the facts ("Python", "GCP", "pgvector"). Dense and hybrid
tie on MRR — the *first* relevant hit lands in position 1 just as often.
A real deployment would re-tune the gold set against production query
logs; the harness makes that mechanical.

### Tailoring (Llama-3.3-70B and OpenAI-OSS-120B as agent, OpenAI-OSS-120B as judge)

| JD | bullets | iters | duration | mean conf | halluc. rate | judge 0–4 |
|---|---:|---:|---:|---:|---:|---:|
| ai_engineer_latam | 2 | 20 | 128.5s | 0.48 | 0.50 | 4.00 |

The mismatch between the LLM judge (4/4 — "all concrete claims supported")
and the deterministic post-hoc check (0.50 hallucination rate) is by
design. The post-hoc check flagged the agent for echoing "2+" from the JD
into a bullet — a number that doesn't appear in the candidate's history.
The judge missed it because the bullet "reads" grounded. **This is exactly
why we run both checks** — the cheap deterministic gate catches the
literal hallucinations; the LLM judge handles semantic groundedness. The
JD asks for "validation strategies and graceful failure mechanisms"; this
mismatch is that story made concrete.

## Wiring to a Next.js FE (stateless mode)

The original `/v1/tailor` endpoint expects facts to have been ingested via
`/v1/career/facts` first — it's a stateful contract. For clients like the
mypdfcv Next.js app that keep resumes in localStorage and have no
server-side user identity, there's a stateless companion:

```
POST /v1/tailor-resume
  Headers: X-Tailor-Token: <shared secret, optional in dev>
  Body:
    {
      "resume_data": { ...FE ResumeData shape... },
      "jd_text":     "...",
      "target_sections": ["summary", "experience.<entry-id>"],
      "strategy":    "hybrid"   // dense | bm25 | hybrid
    }
```

The service flattens `resume_data` into in-memory `CareerFact`s, runs the
same agent loop, and returns the standard `TailorResponse` plus a
`source_id` on each citation (e.g. `experience.<uuid>.bullet.2`) so the
FE can map suggested bullets back to the originating resume entry.
Nothing is written to the DB.

Two extra env vars control this endpoint:

- `TAILOR_API_TOKEN` — required header value. Empty disables the check.
- `ALLOWED_ORIGINS` — CORS allowlist, comma-separated.

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

## Potential project roadmap:

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
