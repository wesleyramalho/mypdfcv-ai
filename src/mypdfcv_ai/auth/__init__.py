"""Shared-secret auth for FE-originated endpoints.

The /v1/tailor-resume endpoint is meant to be called by the Next.js FE's
server-side proxy, which holds the token in env. We check the
X-Tailor-Token header against the configured value. When the token is
empty (local dev default), we skip the check.
"""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from mypdfcv_ai.config import get_settings


def require_tailor_token(x_tailor_token: str | None = Header(default=None)) -> None:
    expected = get_settings().tailor_api_token
    if not expected:
        # Local dev — no token configured, so anyone on this host can call.
        return
    if x_tailor_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Tailor-Token",
        )
