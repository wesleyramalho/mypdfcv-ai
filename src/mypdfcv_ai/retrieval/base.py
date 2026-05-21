"""Common Retriever interface.

All retrievers share the same shape so the agent and the eval harness can
swap between strategies via a literal: `"dense" | "bm25" | "hybrid"`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

Strategy = Literal["dense", "bm25", "hybrid"]


@dataclass(slots=True)
class Hit:
    fact_id: str
    content: str
    score: float
    source_type: str
    metadata: dict


class Retriever(Protocol):
    def search(self, query: str, k: int = 8) -> list[Hit]: ...
