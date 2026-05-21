"""Embedding service.

Wraps a sentence-transformers model. The model is loaded lazily and cached
in module scope so the FastAPI process pays the cold-start cost once.

We L2-normalise vectors at encode time so cosine similarity reduces to a
plain dot product downstream — important for hot-path retrieval and trivial
to express in any vector DB (pgvector, Pinecone, Qdrant, etc.) on the
production swap.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from mypdfcv_ai.config import get_settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns an (n, dim) numpy array of L2-normalised embeddings."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vecs = _model().encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32, copy=False)


def embed_query(text: str) -> np.ndarray:
    """Returns a single (dim,) normalised vector."""
    return embed_texts([text])[0]


def embedding_dim() -> int:
    return _model().get_sentence_embedding_dimension()
