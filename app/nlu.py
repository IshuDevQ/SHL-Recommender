from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from .catalog import is_specific_head_word, load_catalog, skill_vocabulary

# ── Scope guards ──────────────────────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore (all|the|your|previous|above) (instructions|prompt|rules)",
    r"disregard (the|all|your) (above|previous|prior)",
    r"you are now",
    r"act as (a|an)\s",
    r"pretend (you are|to be)",
    r"reveal (your|the) (system|hidden)? ?prompt",
    r"system prompt",
    r"new instructions",
    r"jailbreak",
    r"do anything now",
    r"\bDAN\b",
    r"forget (your|all) (previous )?instructions",
    r"override your",
]

LEGAL_PATTERNS = [
    r"\bis it legal\b",
    r"\blegally required\b",
    r"\blegal(ly)? (obligation|requirement|compliance|requirement)\b",
    r"\blawsuit\b",
    r"\bsue\b|\bsued\b",
    r"discriminat\w* (claim|lawsuit|complaint)",
    r"\bEEOC\b",
    r"can i be fired for",
    r"employment law",
    r"\battorney\b|\blawyer\b",
    r"\bdoes (this|the) (test|assessment) (satisfy|fulfill|meet) (a |the )?(legal|regulatory|compliance)",
    r"\bregulatory (obligation|requirement|compliance)\b",
    r"\brequired (by|under) (law|hipaa|gdpr|regulation)\b",
]

GENERAL_HIRING_PATTERNS = [
    r"how (do|should) i (write|draft) a job (posting|description|ad)\b",
    r"what salary should i (offer|pay)",
    r"how (do|should) i (interview|negotiate)",
    r"write (me )?a job (posting|description|ad)\b",
    r"how to (fire|terminate|onboard) (an? )?employee",
    r"recruiting strategy",
    r"employer branding",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_LEGAL_RE = [re.compile(p, re.IGNORECASE) for p in LEGAL_PATTERNS]
_GENERAL_HIRING_RE = [re.compile(p, re.IGNORECASE) for p in GENERAL_HIRING_PATTERNS]


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_RE)


def detect_off_topic(text: str) -> Optional[str]:
    if any(p.search(text) for p in _LEGAL_RE):
        return "legal_advice"
    if any(p.search(text) for p in _GENERAL_HIRING_RE):
        return "general_hiring_advice"
    return None


# ── Slot definitions ──────────────────────────────────────────────────────────

SENIORITY_TERMS = {
    "intern": "Entry-Level",
    "entry level": "Entry-Level",
    "entry-level": "Entry-Level",
    "graduate": "Graduate",
    "fresher": "Graduate",
    "junior": "Entry-Level",
    "mid level": "Mid-Professional",
    "mid-level": "Mid-Professional",
    "mid-professional": "Mid-Professional",
    "senior": "Professional Individual Contributor",
    "lead": "Professional Individual Contributor",
    "manager": "Manager",
    "managerial": "Manager",
    "director": "Director",
    "executive": "Executive",
    "cxo": "Executive",
    "c-level": "Executive",
    "vp": "Executive",
    "supervisor": "Supervisor",
    "front line manager": "Front Line Manager",
}

TEST_TYPE_TERMS: List[tuple] = [
    ("situational judgement", "B"),
    ("situational judgment", "B"),
    ("biodata", "B"),
    ("personality", "P"),
    ("behavioral", "P"),
    ("behavioural", "P"),
    ("cognitive", "A"),
    ("aptitude", "A"),
    ("reasoning", "A"),
    ("numerical", "A"),
    ("verbal ability", "A"),
    ("inductive", "A"),
    ("deductive", "A"),
    ("simulation", "S"),
    ("role play", "S"),
    ("competency", "C"),
    ("competencies", "C"),
    ("360", "D"),
    ("development report", "D"),
    ("knowledge test", "K"),
    ("skills test", "K"),
    ("technical test", "K"),
    ("coding test", "K"),
    ("assessment exercise", "E"),
    ("in-tray", "E"),
    ("in tray", "E"),
]

DURATION_RE = re.compile(
    r"(\d{1,3})\s*(?:-|to)?\s*(?:\d{1,3})?\s*(minutes?|mins?|hour|hr)s?",
    re.IGNORECASE,
)

NO_PREFERENCE_RE = re.compile(
    r"\b(no preference|don'?t know|not sure|doesn'?t matter|"
    r"any (is fine|works)|whatever (works|is fine)|no specific|no constraint)\b",
    re.IGNORECASE,
)

# Confirmation: user is locking in the shortlist.
# eoc=true only fires when this is detected AND we already have recs on table.
CONFIRM_RE = re.compile(
    r"\b("
    r"confirm(?:ed)?|"
    r"that'?s (it|all|good|perfect|great|what we need|settled|done|confirmed|correct)|"
    r"that (covers?|works?|is all|is it|is what we need)|"
    r"this (covers?|works?|is all)|"
    r"(looks?|sounds?) good|"
    r"perfect[,.]?|"
    r"locking? (it )?in|lock(?:ed)? in|locked|"
    r"thanks?,?\s*(that'?s all|we'?re done)?|"
    r"keep (it|the (list|shortlist|battery|stack))( as.is| as is)?|"
    r"go (with|ahead)( with (this|that|these))?|"
    r"we'?ll (take|go with|use) (that|this|these|it)|"
    r"(that|this) is (our|the) final|"
    r"final (list|battery|shortlist)[,.]?(\s+confirmed)?|"
    r"(we'?re|i'?m) (good|done|set|happy with this)|"
    r"understood[,.]?\s*(keep|confirmed)?|"
    r"no (more )?changes?|"
    r"(great|done|settled|approved|proceed)[.,!]?\s*$|"
    r"keep the shortlist as.is"
    r")\b",
    re.IGNORECASE,
)

_COMPARE_AND_VS = re.compile(
    r"(?:difference between|compare)\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)([?.!]|$)",
    re.IGNORECASE,
)

_QUESTION_STARTERS = re.compile(
    r"^(what|how|why|when|where|who|which|is|are|can|could|should|do|does)\b",
    re.IGNORECASE,
)

ROLE_HINTS = {
    "stakeholder": "communication collaboration personality",
    "collaborat": "communication teamwork personality",
    "leadership": "leadership management personality",
    "manage people": "management leadership personality",
    "team lead": "leadership management personality",
    "customer service": "customer service simulation situational judgement",
    "call center": "contact center customer service simulation",
    "contact center": "contact center customer service simulation",
    "sales": "sales personality behavior",
    "data analy": "numerical reasoning sql excel",
    "graduate": "graduate cognitive ability",
    "fresher": "graduate cognitive ability",
    "plant operator": "safety dependability manufacturing",
    "safety": "safety dependability",
    "healthcare": "hipaa medical terminology",
    "financial analyst": "numerical reasoning financial accounting",
    r"full.stack": "java spring sql angular",
    "backend": "java spring sql",
    "frontend": "angular javascript react",
    "devops": "docker aws linux",
    "cloud": "aws azure docker",
}

# Maps lowercased user alias → fragment of catalog name/slug to match
ASSESSMENT_ALIASES = {
    "opq": "occupational personality questionnaire opq32r",
    "opq32r": "occupational personality questionnaire opq32r",
    "opq32": "occupational personality questionnaire opq32r",
    "gsa": "global skills assessment",
    "verify g+": "verify interactive g",
    "g+": "verify interactive g",
    "dsi": "dependability and safety instrument",
    "mq": "motivation questionnaire",
    "sjt": "graduate scenarios",
    "rest": "restful web services",
    "restful": "restful web services",
    "aws": "amazon web services",
    "svar": "svar spoken english",
    "linux": "linux programming",
    "numerical reasoning": "verify interactive numerical reasoning",
    "verify g": "verify interactive g",
}


def _resolve_alias(term: str) -> Optional[str]:
    low = term.strip().lower()
    return ASSESSMENT_ALIASES.get(low)


def _extract_compare_targets(text: str) -> List[str]:
    m = _COMPARE_AND_VS.search(text)
    if not m:
        return []
    a = re.sub(r"^(the|a|an)\s+", "", m.group(1).strip(" '\""), flags=re.IGNORECASE)
    b = re.sub(r"^(the|a|an)\s+", "", m.group(2).strip(" '\""), flags=re.IGNORECASE)
    return [a, b]


# ── Directive ─────────────────────────────────────────────────────────────────

@dataclass
class Directive:
    add_skills: Set[str] = field(default_factory=set)
    remove_skills: Set[str] = field(default_factory=set)
    remove_names: Set[str] = field(default_factory=set)
    add_test_types: Set[str] = field(default_factory=set)
    remove_test_types: Set[str] = field(default_factory=set)
    role_terms: List[str] = field(default_factory=list)
    duration_max: Optional[float] = None
    seniority: Optional[str] = None
    jd_text: Optional[str] = None
    no_preference: bool = False
    compare_targets: List[str] = field(default_factory=list)
    off_topic_reason: Optional[str] = None
    injection: bool = False
    confirmation: bool = False


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_directive(text: str) -> Directive:
    d = Directive()
    text_stripped = text.strip()
    if not text_stripped:
        return d

    if detect_injection(text_stripped):
        d.injection = True
        return d

    reason = detect_off_topic(text_stripped)
    if reason:
        d.off_topic_reason = reason
        return d

    if CONFIRM_RE.search(text_stripped):
        d.confirmation = True

    if NO_PREFERENCE_RE.search(text_stripped):
        d.no_preference = True

    d.compare_targets = _extract_compare_targets(text_stripped)

    # Duration
    dur_m = DURATION_RE.search(text_stripped)
    if dur_m:
        value = float(dur_m.group(1))
        unit = dur_m.group(2).lower()
        if unit.startswith("hour") or unit == "hr":
            value *= 60
        d.duration_max = value

    low = text_stripped.lower()

    # Seniority
    for term, label in SENIORITY_TERMS.items():
        if term in low:
            d.seniority = label
            break

    # ── Test type add / remove ────────────────────────────────────────────────

    _JUST_ONLY = bool(re.search(r"\b(just|only)\b", low))

    def _type_negated(phrase: str) -> bool:
        pattern = (
            r"\b(no|not|remove|drop|exclude|excluding|without|"
            r"don'?t need|do not need|skip)\b"
            r"[^.;,]{0,50}?\b" + re.escape(phrase) + r"s?\b"
        )
        alt = (
            r"\b" + re.escape(phrase) + r"s?\b"
            + r"[^.;,]{0,20}?\b(not|no longer|removed|excluded|dropped)\b"
        )
        return bool(re.search(pattern, low) or re.search(alt, low))

    desired_types: Set[str] = set()
    for term, code in TEST_TYPE_TERMS:
        if re.search(r"\b" + re.escape(term) + r"s?\b", low):
            if not _type_negated(term):
                desired_types.add(code)

    for term, code in TEST_TYPE_TERMS:
        in_text = bool(re.search(r"\b" + re.escape(term) + r"s?\b", low))
        if not in_text:
            continue
        if _type_negated(term):
            d.remove_test_types.add(code)
            d.add_test_types.discard(code)
        else:
            d.add_test_types.add(code)

    if _JUST_ONLY and desired_types:
        all_codes = {code for _, code in TEST_TYPE_TERMS}
        for code in all_codes - desired_types:
            d.remove_test_types.add(code)
        d.add_test_types = desired_types

    # ── Named assessment removal ──────────────────────────────────────────────

    _REMOVE_VERB_RE = re.compile(
        r"\b(drop|remove|exclude|skip|take out|without|get rid of)\b"
        r"[^.;,]{0,5}?(the\s+)?([a-z0-9][a-z0-9\s.+#&\-]{1,40}?)(?=[,;.!?\n]|$|\s+and\b)",
        re.IGNORECASE,
    )
    for m in _REMOVE_VERB_RE.finditer(low):
        target = m.group(3).strip()
        alias = _resolve_alias(target)
        if alias:
            d.remove_names.add(alias)
        else:
            clean_target = re.sub(r"[^a-z0-9.+# ]+", " ", target).strip()
            if clean_target and len(clean_target) >= 2:
                d.remove_names.add(clean_target)

    # ── Catalog skill terms ───────────────────────────────────────────────────

    matched: Set[str] = set()
    clean = re.sub(r"[^a-z0-9.+# ]+", " ", low)
    clean = re.sub(r"\s+", " ", clean).strip()
    remaining = f" {clean} "

    def _is_negated(term: str) -> bool:
        return bool(re.search(
            r"\b(no|not|remove|drop|exclude|excluding|without|"
            r"don'?t need|do not need|skip)\b"
            r"[^.;,]{0,40}?" + re.escape(term),
            low,
        ))

    for term in skill_vocabulary():
        pad = f" {term} "
        if pad in remaining:
            if _is_negated(term):
                d.remove_skills.add(term)
            else:
                matched.add(term)
            remaining = remaining.replace(pad, "  ")

    # Head-word fallback for bare single-word mentions
    for word in set(re.findall(r"[a-z][a-z0-9.+#]{1,}", remaining)):
        if is_specific_head_word(word):
            if _is_negated(word):
                d.remove_skills.add(word)
            else:
                matched.add(word)

    d.add_skills = matched

    # Role / soft-skill hints
    for trigger, hint in ROLE_HINTS.items():
        if re.search(trigger, low):
            d.role_terms.append(hint)

    # Job description heuristic
    word_count = len(text_stripped.split())
    if word_count >= 35 and not _QUESTION_STARTERS.match(text_stripped):
        d.jd_text = text_stripped
    elif word_count >= 6:
        d.role_terms.append(text_stripped)

    return d
