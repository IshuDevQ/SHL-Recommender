from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import List, Optional, Set

from .catalog import Assessment, load_catalog
from .retrieval import Filters, _passes_filters


@lru_cache(maxsize=1)
def _model():
    """Load once at startup, reuse forever."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _catalog_embeddings():
    """
    Pre-compute embeddings for all catalog items at startup.
    Each item is represented as:
      '{name}. {test_type_name}. {description}'
    This gives the model enough context to match abstract queries.
    """
    model = _model()
    catalog = load_catalog()

    texts = []
    for a in catalog:
        # Rich text representation — name + type + description
        parts = [a.name, a.test_type_name]
        if a.description:
            parts.append(a.description)
        texts.append(". ".join(parts))

    # normalize_embeddings=True means we can use dot product as cosine similarity
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    return np.array(embeddings), list(catalog)


def vector_search(
    query: str,
    *,
    filters: Optional[Filters] = None,
    k: int = 10,
    min_score: float = 0.25,
) -> List[Assessment]:
    """
    Semantic similarity search over catalog embeddings.
    Returns up to k assessments that pass filters, ordered by similarity.
    Only returns items with cosine similarity >= min_score to avoid noise.
    """
    if not query.strip():
        return []

    from .retrieval import Filters as F
    filters = filters or F()

    model = _model()
    embeddings, catalog = _catalog_embeddings()

    query_emb = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Cosine similarity (dot product of normalized vectors)
    scores = (embeddings @ query_emb.T).squeeze()

    ranked = sorted(
        zip(scores.tolist(), catalog),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, a in ranked:
        if score < min_score:
            break
        if _passes_filters(a, filters):
            results.append(a)
        if len(results) >= k:
            break

    return results


def warm_embeddings() -> int:
    """Call at startup to pre-compute everything. Returns catalog size."""
    _, catalog = _catalog_embeddings()
    return len(catalog)