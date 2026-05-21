"""The grounded tailoring agent.

A bounded tool-calling loop. The agent has five tools; it can only emit a
bullet after verify_claim returns grounded=true. The system prompt is
strict about hallucination — but we don't rely on the prompt alone, the
verify_claim tool deterministically rejects ungrounded claims so the agent
literally cannot persist a fabrication into the response.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from mypdfcv_ai.agents.tools.schemas import TOOLS
from mypdfcv_ai.config import get_settings
from mypdfcv_ai.grounding.citations import verify_grounding
from mypdfcv_ai.grounding.confidence import confidence_score
from mypdfcv_ai.llm.client import get_llm_client
from mypdfcv_ai.logging import get_logger
from mypdfcv_ai.retrieval.base import Hit, Retriever

log = get_logger(__name__)


SYSTEM_PROMPT = """You are a resume tailor. Your job is to rewrite resume bullets so they match a target job description, using ONLY facts the user has actually provided.

HARD RULES — violating any of these is a failure:
1. Every factual claim (numbers, percentages, employer names, technologies, dates, outcomes) MUST come from a fact returned by search_history.
2. You MUST call verify_claim BEFORE emit_bullet for every bullet.
3. If verify_claim returns grounded=false, you must either find supporting facts via more search_history calls, or rewrite the bullet to remove unsupported claims.
4. NEVER invent numbers or percentages. If the user did not write "lifted revenue by 30%", you cannot write that.
5. Cite EVERY fact_id you used in the emit_bullet call.

WORKFLOW:
1. Call search_jd_requirements once.
2. For each target section requested:
   a. Call search_history to find relevant facts.
   b. Draft a bullet using only those facts.
   c. Call verify_claim with the bullet and the fact_ids.
   d. If grounded, call emit_bullet. If not, revise or move on.
3. When every section has either a bullet or a documented skip, call finish.

Be concise. Do not explain yourself in natural language between tool calls."""


@dataclass(slots=True)
class TailoredBullet:
    section: str
    text: str
    citations: list[str]
    cited_facts: list[Hit]
    confidence: float


@dataclass(slots=True)
class TailoringResult:
    bullets: list[TailoredBullet] = field(default_factory=list)
    iterations: int = 0
    duration_ms: int = 0
    model_used: str = ""
    finish_summary: str = ""
    notes: list[str] = field(default_factory=list)


def run_tailor_agent(
    *,
    jd_text: str,
    target_sections: list[str],
    retriever: Retriever,
    user_id: str,
    client: OpenAI | None = None,
) -> TailoringResult:
    settings = get_settings()
    llm = client or get_llm_client()
    model = settings.agent_model

    fact_cache: dict[str, Hit] = {}  # fact_id -> Hit (for citation resolution)

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _initial_user_message(jd_text, target_sections),
        },
    ]

    result = TailoringResult(model_used=model)
    started = time.perf_counter()

    for iteration in range(1, settings.agent_max_iterations + 1):
        result.iterations = iteration
        completion = llm.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=settings.agent_temperature,
            max_tokens=1500,
        )
        msg = completion.choices[0].message
        # Persist the assistant turn (tool calls + any reasoning).
        messages.append(msg.model_dump(exclude_none=True))

        tool_calls = msg.tool_calls or []
        if not tool_calls:
            # Model is talking instead of acting — prod it back into the loop.
            result.notes.append(f"iter {iteration}: model produced text, no tool call")
            messages.append(
                {
                    "role": "user",
                    "content": "Please continue using tool calls only. Do not respond with prose.",
                }
            )
            continue

        finished = False
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            log.info("tool_call", iter=iteration, name=name, args=args)
            tool_result = _dispatch_tool(
                name=name,
                args=args,
                jd_text=jd_text,
                retriever=retriever,
                fact_cache=fact_cache,
                result=result,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                }
            )
            if name == "finish":
                result.finish_summary = args.get("summary", "")
                finished = True

        if finished:
            break

    result.duration_ms = int((time.perf_counter() - started) * 1000)
    return result


def _initial_user_message(jd_text: str, target_sections: list[str]) -> str:
    return (
        f"Job description:\n---\n{jd_text}\n---\n\n"
        f"Target sections to tailor: {target_sections}\n\n"
        "Begin by calling search_jd_requirements."
    )


def _dispatch_tool(
    *,
    name: str,
    args: dict[str, Any],
    jd_text: str,
    retriever: Retriever,
    fact_cache: dict[str, Hit],
    result: TailoringResult,
) -> dict[str, Any]:
    if name == "search_jd_requirements":
        return {"jd_text": jd_text, "instruction": "Parse the JD into ranked requirements yourself, then call search_history per requirement."}

    if name == "search_history":
        query = args.get("query", "")
        k = min(max(int(args.get("k", 6)), 1), 10)
        hits = retriever.search(query, k=k)
        for h in hits:
            fact_cache[h.fact_id] = h
        return {
            "hits": [
                {"fact_id": h.fact_id, "content": h.content, "score": round(h.score, 3), "source": h.source_type}
                for h in hits
            ]
        }

    if name == "verify_claim":
        bullet = args.get("bullet", "")
        fact_ids = args.get("supporting_fact_ids", [])
        cited = [fact_cache[fid] for fid in fact_ids if fid in fact_cache]
        missing = [fid for fid in fact_ids if fid not in fact_cache]
        check = verify_grounding(bullet, [c.content for c in cited])
        if missing:
            return {
                "grounded": False,
                "reason": f"Unknown fact_ids: {missing}. Use IDs from search_history results.",
                "unsupported_terms": [],
            }
        return {
            "grounded": check.grounded,
            "reason": check.reason,
            "unsupported_terms": check.unsupported_terms,
        }

    if name == "emit_bullet":
        section = args.get("section", "")
        text = args.get("text", "")
        citations = args.get("citations", [])
        cited = [fact_cache[fid] for fid in citations if fid in fact_cache]
        # Re-verify on the way in — belt + suspenders, the agent should already have called verify_claim.
        check = verify_grounding(text, [c.content for c in cited])
        if not check.grounded:
            return {
                "accepted": False,
                "reason": f"Refused: {check.reason}",
            }
        score = confidence_score(
            bullet=text,
            cited_texts=[c.content for c in cited],
            retrieval_scores=[c.score for c in cited],
            jd_text=jd_text,
        )
        result.bullets.append(
            TailoredBullet(
                section=section,
                text=text,
                citations=citations,
                cited_facts=cited,
                confidence=score,
            )
        )
        return {"accepted": True, "confidence": score}

    if name == "finish":
        return {"acknowledged": True}

    return {"error": f"unknown tool: {name}"}
