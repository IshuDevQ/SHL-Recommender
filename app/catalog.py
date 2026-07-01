from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"

TYPE_CODE_TO_NAME = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

_NOISE = re.compile(
    r"\b(new|next generation|[0-9]+\.[0-9]+|report|form \d+|\(.*?\))\b", re.I
)
_NON_ALNUM = re.compile(r"[^a-z0-9.+# ]+")


def _normalise(text: str) -> str:
    text = text.lower()
    text = _NOISE.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Assessment:
    id: str
    slug: str
    name: str
    url: str
    test_type: str
    test_types: Tuple[str, ...]
    test_type_name: str
    remote_testing: bool
    adaptive_irt: bool
    duration_minutes: Optional[float]
    description: str
    job_levels: Tuple[str, ...]
    skill_term: str


@lru_cache(maxsize=1)
def load_catalog() -> List[Assessment]:
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    out = []
    for r in raw:
        out.append(
            Assessment(
                id=r["id"],
                slug=r["slug"],
                name=r["name"],
                url=r["url"],
                test_type=r["test_type"],
                test_types=tuple(r.get("test_types", [r["test_type"]])),
                test_type_name=r.get("test_type_name")
                or TYPE_CODE_TO_NAME.get(r["test_type"], ""),
                remote_testing=bool(r.get("remote_testing")),
                adaptive_irt=bool(r.get("adaptive_irt")),
                duration_minutes=r.get("duration_minutes"),
                description=r.get("description") or "",
                job_levels=tuple(r.get("job_levels", [])),
                skill_term=_normalise(r["name"]),
            )
        )
    return out


@lru_cache(maxsize=1)
def by_name_lookup() -> Dict[str, Assessment]:
    return {a.name.lower(): a for a in load_catalog()}


@lru_cache(maxsize=1)
def skill_vocabulary() -> List[str]:
    terms = {a.skill_term for a in load_catalog() if len(a.skill_term) >= 2}
    return sorted(terms, key=len, reverse=True)


# ── Head-word index ───────────────────────────────────────────────────────────

_ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "from", "up", "down", "out",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "this", "that", "these", "those", "i", "we",
    "you", "he", "she", "it", "they", "them", "their", "our", "your",
    "his", "her", "its", "who", "whom", "which", "what", "as", "do",
    "does", "did", "have", "has", "had", "having", "job", "role", "work",
    "working", "team", "company", "candidate", "candidates", "assessment",
    "assessments", "time", "minutes", "minute", "hiring", "hire", "need",
    "needs", "want", "wants", "looking", "look", "skills", "skill",
    "technical", "business", "able", "also", "must", "good", "strong",
    "person", "people", "experience", "years", "year", "required",
    "fundamentals", "basics", "principles", "concepts", "knowledge",
    "ability", "abilities", "proficiency", "proficient", "expert",
    "perform", "design", "build", "write", "develop", "manage", "lead",
}

_GENERIC_WORD_STOPLIST = _ENGLISH_STOPWORDS | {
    "new", "test", "tests", "testing", "general", "basic", "essentials",
    "entry", "next", "report", "reports", "level", "form", "split",
    "screen", "verify", "shl", "global", "data", "customer", "sales",
    "manufacturing", "retail", "microsoft", "universal", "occupational",
    "situational", "sample", "interactive", "standard", "advanced",
    "development", "candidate", "user", "profile", "guide", "core",
    "enterprise", "service", "services",
}


@lru_cache(maxsize=1)
def head_word_index() -> Dict[str, int]:
    from collections import Counter

    counts: Counter = Counter()
    for a in load_catalog():
        for word in set(a.skill_term.split()):
            if len(word) < 2 or word in _GENERIC_WORD_STOPLIST:
                continue
            counts[word] += 1
    return dict(counts)


def is_specific_head_word(word: str, max_fanout: int = 6) -> bool:
    counts = head_word_index()
    return word in counts and counts[word] <= max_fanout
