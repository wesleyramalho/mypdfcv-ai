from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mypdfcv_ai.api.routes import router
from mypdfcv_ai.config import get_settings
from mypdfcv_ai.db.session import init_db
from mypdfcv_ai.logging import configure_logging


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    init_db()
    yield


app = FastAPI(
    title="mypdfcv-ai",
    description="Grounded resume tailoring via RAG + tool-calling agents.",
    version="0.1.0",
    lifespan=lifespan,
)

_origins = [o.strip() for o in get_settings().allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Tailor-Token"],
)

app.include_router(router)
