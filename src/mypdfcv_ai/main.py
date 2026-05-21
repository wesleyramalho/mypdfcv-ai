from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from mypdfcv_ai.api.routes import router
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
app.include_router(router)
