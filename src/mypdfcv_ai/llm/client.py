"""Thin LLM client.

We use the official OpenAI SDK pointed at OpenRouter's OpenAI-compatible
endpoint. This keeps the interface familiar (`chat.completions.create` +
`tools=...`) while letting the demo run on free-tier models. To swap to
Anthropic, OpenAI, or any other provider, change `agent_model` env var —
no code changes needed for the agent loop.
"""
from __future__ import annotations

from openai import OpenAI

from mypdfcv_ai.config import get_settings

_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env "
                "and add a key from https://openrouter.ai/keys"
            )
        _client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            # OpenRouter recommends these headers for free-tier accounting and
            # leaderboard visibility. Safe to leave on a portfolio repo.
            default_headers={
                "HTTP-Referer": "https://github.com/wesleyramalho/mypdfcv-ai",
                "X-Title": "mypdfcv-ai",
            },
        )
    return _client
