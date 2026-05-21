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
from mypdfcv_ai.llm.resilient import chat_completion_with_fallback
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
6. fact_ids are 36-character UUIDs. ALWAYS pass them in FULL, character-for-character. Never abbreviate, truncate, or use ellipses. Wrong: '68c...'. Right: '68cb4c50-2044-4f7f-82ad-20a1c4ac433e'.
7. emit_bullet requires both `section` (non-empty) and `text` (a complete, finished bullet of >= 20 characters).

WORKFLOW:
1. Call search_jd_requirements once.
2. For each target section requested:
   a. Call search_history to find relevant facts.
   b. Draft a bullet using only those facts.
   c. Call verify_claim with the bullet and the fact_ids.
   d. If grounded=true, IMMEDIATELY call emit_bullet with the SAME bullet text. Do NOT rewrite — your job is to ship grounded bullets, not perfect them.
   e. If grounded=false, revise based on the unsupported_terms feedback or move on.
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

    fact_cache: dict[str, Hit] = {}  # fact_id -> Hit (for citation resolution)
    last_model_used = settings.agent_model

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _initial_user_message(jd_text, target_sections),
        },
    ]

    result = TailoringResult(model_used=last_model_used)
    started = time.perf_counter()

    for iteration in range(1, settings.agent_max_iterations + 1):
        result.iterations = iteration
        completion_result = chat_completion_with_fallback(
            llm,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=settings.agent_temperature,
            max_tokens=1500,
        )
        completion = completion_result.completion
        last_model_used = completion_result.model_used
        result.model_used = last_model_used
        if any(outcome != "ok" for _m, outcome in completion_result.attempts[:-1]):
            result.notes.append(
                f"iter {iteration}: fallback chain {completion_result.attempts}"
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
            if name == "finish" and tool_result.get("acknowledged"):
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
        bad_ids = [fid for fid in fact_ids if len(fid) != 36 or fid not in fact_cache]
        if bad_ids:
            return {
                "grounded": False,
                "reason": (
                    f"Unknown or truncated fact_ids: {bad_ids}. Use the full 36-char "
                    "UUIDs exactly as returned by search_history."
                ),
                "unsupported_terms": [],
            }
        cited = [fact_cache[fid] for fid in fact_ids]
        check = verify_grounding(bullet, [c.content for c in cited])
        response: dict[str, Any] = {
            "grounded": check.grounded,
            "reason": check.reason,
            "unsupported_terms": check.unsupported_terms,
        }
        if check.grounded:
            response["next_action"] = (
                "REQUIRED: your very next tool call MUST be emit_bullet with "
                "the same bullet text and the same fact_ids. Do not call "
                "verify_claim again, do not rewrite, do not call finish. "
                "Call emit_bullet now."
            )
        else:
            response["next_action"] = (
                "Either revise the bullet to remove the unsupported terms, "
                "or call search_history to find supporting facts."
            )
        return response

    if name == "emit_bullet":
        section = args.get("section", "").strip()
        text = args.get("text", "").strip()
        citations = args.get("citations", [])
        if not section or not text:
            return {
                "accepted": False,
                "reason": "Both 'section' and 'text' are required and must be non-empty.",
            }
        # Catch truncated / abbreviated fact_ids (e.g. '68c...'). Real IDs are
        # 36-char UUIDs.
        bad_ids = [fid for fid in citations if len(fid) != 36 or fid not in fact_cache]
        if bad_ids:
            return {
                "accepted": False,
                "reason": (
                    f"Unknown or truncated fact_ids: {bad_ids}. Use the full 36-char "
                    "UUIDs exactly as returned by search_history."
                ),
            }
        cited = [fact_cache[fid] for fid in citations]
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
        if not result.bullets:
            return {
                "acknowledged": False,
                "reason": (
                    "Refused: you have not emitted any bullets yet. Call "
                    "search_history → verify_claim → emit_bullet at least once "
                    "before calling finish."
                ),
            }
        return {"acknowledged": True}

    return {"error": f"unknown tool: {name}"}
