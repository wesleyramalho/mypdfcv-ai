"""Resilient chat-completion with a model fallback chain.

Real-world LLM workloads must survive provider rate limits and transient
errors. Rather than blowing up the agent loop on the first 429, we walk a
configured fallback chain. The "model_used" return value tells the caller
which model actually answered — important for the eval harness so we
attribute scores to the right model.

This is intentionally simple (no jitter, no circuit breaker) — that's
production v2. v1 demonstrates the pattern.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
from openai import APIStatusError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion

from mypdfcv_ai.config import get_settings
from mypdfcv_ai.logging import get_logger

log = get_logger(__name__)

_RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(slots=True)
class CompletionResult:
    completion: ChatCompletion
    model_used: str
    attempts: list[tuple[str, str]]  # (model, outcome) per try


def _model_chain() -> list[str]:
    s = get_settings()
    chain = [s.agent_model]
    for m in (s.agent_fallback_models or "").split(","):
        m = m.strip()
        if m and m not in chain:
            chain.append(m)
    return chain


def chat_completion_with_fallback(client: OpenAI, **kwargs: Any) -> CompletionResult:
    """Call client.chat.completions.create, walking the fallback chain on retryable errors.

    Caller passes the same kwargs they would pass to the SDK directly,
    minus `model` (which is supplied from the chain).
    """
    chain = _model_chain()
    attempts: list[tuple[str, str]] = []
    last_exc: Exception | None = None

    for model in chain:
        try:
            completion = client.chat.completions.create(model=model, **kwargs)
            attempts.append((model, "ok"))
            if len(attempts) > 1:
                log.warning("fallback_succeeded", attempts=attempts)
            return CompletionResult(completion=completion, model_used=model, attempts=attempts)
        except RateLimitError as e:
            attempts.append((model, "rate_limited"))
            log.warning("model_rate_limited", model=model)
            last_exc = e
            continue
        except APIStatusError as e:
            if e.status_code in _RETRYABLE_STATUSES:
                attempts.append((model, f"http_{e.status_code}"))
                log.warning("model_failed_retryable", model=model, status=e.status_code)
                last_exc = e
                # Tiny backoff before next provider tries.
                time.sleep(0.5)
                continue
            attempts.append((model, f"http_{e.status_code}_nonretry"))
            raise
        except httpx.HTTPError as e:
            attempts.append((model, f"transport_{type(e).__name__}"))
            log.warning("model_transport_error", model=model, err=str(e))
            last_exc = e
            time.sleep(0.5)
            continue

    # Exhausted chain.
    log.error("all_models_exhausted", attempts=attempts)
    raise RuntimeError(
        f"All models in fallback chain failed: {attempts}. "
        "Last error: " + (str(last_exc) if last_exc else "unknown")
    ) from last_exc
