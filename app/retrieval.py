from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Set

from rank_bm25 import BM25Okapi

from .catalog import Assessment, load_catalog

_TOKEN_RE = __import__("re").compile(r"[a-z0-9.+#]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _bm25_index():
    docs = [
        f"{a.name} {a.test_type_name} {a.description}"
        for a in load_catalog()
    ]
    tokenised = [_tokenize(d) for d in docs]
    return BM25Okapi(tokenised), list(load_catalog())


@dataclass
class Filters:
    max_duration: Optional[float] = None
    include_types: Optional[Set[str]] = None
    exclude_types: Optional[Set[str]] = None
    require_remote: bool = False


def _passes_filters(a: Assessment, f: Filters) -> bool:
    if f.max_duration is not None and a.duration_minutes is not None:
        if a.duration_minutes > f.max_duration:
            return False
    if f.exclude_types:
        if any(t in f.exclude_types for t in a.test_types):
            return False
    if f.include_types:
        if not any(t in f.include_types for t in a.test_types):
            return False
    if f.require_remote and not a.remote_testing:
        return False
    return True


def keyword_candidates(skill_terms: Iterable[str]) -> List[Assessment]:
    """Exact/substring matches on catalog skill terms — highest-precision path."""
    terms = [t for t in skill_terms if t]
    if not terms:
        return []
    out = []
    for a in load_catalog():
        if any(t in a.skill_term or a.skill_term in t for t in terms):
            out.append(a)
    return out


def search(
    query_text: str,
    *,
    skill_terms: Sequence[str] = (),
    filters: Optional[Filters] = None,
    k: int = 10,
) -> List[Assessment]:
    """
    Hybrid retrieval:
    1. Grounded keyword matches (catalog vocabulary) ranked first.
    2. BM25 fills remaining slots with a score threshold to block noise.
    Hard filters enforced inside — no violating row ever reaches the reply layer.
    """
    filters = filters or Filters()
    bm25, docs = _bm25_index()

    ranked: List[Assessment] = []
    seen: Set[str] = set()

    for a in keyword_candidates(skill_terms):
        if _passes_filters(a, filters) and a.id not in seen:
            ranked.append(a)
            seen.add(a.id)

    remaining = k - len(ranked)
    if remaining > 0 and query_text.strip():
        scores = bm25.get_scores(_tokenize(query_text))
        max_score = max(scores) if len(scores) > 0 else 0
        threshold = max_score * 0.20 if max_score > 0 else 0
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        for score, a in scored:
            if score < threshold or score <= 0:
                break
            if a.id in seen or not _passes_filters(a, filters):
                continue
            ranked.append(a)
            seen.add(a.id)
            if len(ranked) >= k:
                break

    # Last resort: filter-only, no relevance ranking
    if not ranked:
        for a in docs:
            if _passes_filters(a, filters):
                ranked.append(a)
            if len(ranked) >= k:
                break

    return ranked[:k]


def find_by_name(query: str) -> Optional[Assessment]:
    """Best-effort fuzzy match for compare queries. Returns None if uncertain."""
    from .catalog import by_name_lookup
    from .nlu import ASSESSMENT_ALIASES

    q = query.strip().lower()
    if not q:
        return None

    exact = by_name_lookup().get(q)
    if exact:
        return exact

    if q in ASSESSMENT_ALIASES:
        resolved = ASSESSMENT_ALIASES[q]
        match = by_name_lookup().get(resolved)
        if match:
            return match

    candidates = [
        a for a in load_catalog()
        if q in a.name.lower() or a.name.lower() in q
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return min(candidates, key=lambda a: len(a.name))
    return None
