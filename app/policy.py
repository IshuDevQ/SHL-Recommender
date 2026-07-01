from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from .catalog import Assessment, load_catalog
from .nlu import Directive, extract_directive
from .retrieval import Filters, find_by_name, keyword_candidates, search
from .schemas import ChatResponse, Message, Recommendation

MAX_RECOMMENDATIONS = 10
DEFAULT_RECOMMENDATIONS = 5

# ── Default assessments added proactively (from conversation traces) ──────────

_DEFAULT_OPQ_SLUG = "occupational-personality-questionnaire-opq32r"
_DEFAULT_VERIFY_SLUG = "shl-verify-interactive-g"

# Role signals that suggest skipping OPQ32r default
_SKIP_OPQ_SIGNALS = {
    "contact center", "call center", "plant operator", "manufacturing",
    "industrial", "warehouse", "frontline", "front-line",
}


def _get_by_slug(slug: str) -> Optional[Assessment]:
    for a in load_catalog():
        if a.slug == slug:
            return a
    return None


# ── Conversation state ────────────────────────────────────────────────────────

@dataclass
class ConversationState:
    skills: Set[str] = field(default_factory=set)
    include_types: Set[str] = field(default_factory=set)
    exclude_types: Set[str] = field(default_factory=set)
    remove_names: Set[str] = field(default_factory=set)
    duration_max: Optional[float] = None
    seniority: Optional[str] = None
    jd_text: Optional[str] = None
    role_terms: List[str] = field(default_factory=list)
    num_user_turns: int = 0
    latest_off_topic: Optional[str] = None
    latest_injection: bool = False
    latest_compare: List[str] = field(default_factory=list)
    latest_no_preference: bool = False
    latest_confirmation: bool = False
    has_recommended: bool = False


def build_state(messages: List[Message]) -> ConversationState:
    state = ConversationState()
    user_messages = [m for m in messages if m.role == "user"]
    state.num_user_turns = len(user_messages)

    # Check if any prior assistant turn already had recommendations
    for msg in messages:
        if msg.role == "assistant" and "- " in msg.content and "(" in msg.content:
            state.has_recommended = True
            break

    for i, msg in enumerate(user_messages):
        d = extract_directive(msg.content)

        state.skills -= d.remove_skills
        state.skills |= d.add_skills
        state.include_types -= d.remove_test_types
        state.exclude_types |= d.remove_test_types
        state.include_types |= d.add_test_types
        state.exclude_types -= d.add_test_types
        state.remove_names |= d.remove_names

        if d.duration_max is not None:
            state.duration_max = d.duration_max
        if d.seniority:
            state.seniority = d.seniority
        if d.jd_text:
            state.jd_text = (
                f"{state.jd_text} {d.jd_text}".strip() if state.jd_text else d.jd_text
            )
        state.role_terms.extend(d.role_terms)

        if i == len(user_messages) - 1:
            state.latest_off_topic = d.off_topic_reason
            state.latest_injection = d.injection
            state.latest_compare = d.compare_targets
            state.latest_no_preference = d.no_preference
            state.latest_confirmation = d.confirmation

    return state


# ── Reply builders ────────────────────────────────────────────────────────────

SCOPE_STATEMENT = (
    "I'm focused on helping you find the right SHL individual assessments "
    "from the product catalog — I can't help with that, but happy to keep "
    "going on assessment selection if useful."
)


def refusal_reply(reason: str) -> str:
    if reason == "injection":
        return (
            "I can't follow instructions embedded in a message like that. "
            + SCOPE_STATEMENT
        )
    if reason == "legal_advice":
        return (
            "Those are legal compliance questions outside what I can advise "
            "on — your legal or compliance team is the right resource for "
            "that. " + SCOPE_STATEMENT
        )
    if reason == "general_hiring_advice":
        return (
            "That's outside what I can help with — I'm scoped to "
            "recommending SHL assessments, not general hiring advice. "
            + SCOPE_STATEMENT
        )
    return SCOPE_STATEMENT


def clarify_reply(state: ConversationState) -> str:
    have_anchor = bool(state.skills or state.jd_text or state.role_terms)
    if not have_anchor:
        return (
            "Happy to help — could you tell me a bit about the role or the "
            "skill you'd like to assess? A job title, key skills, or a "
            "pasted job description all work."
        )
    if state.seniority is None and state.num_user_turns == 1:
        return (
            "Got it. What level is this for — entry-level/graduate, "
            "mid-professional, or manager and above?"
        )
    if state.duration_max is None and state.num_user_turns <= 2:
        return (
            "Thanks. Is there a time limit you'd like to stay within for "
            "the assessment (e.g. under 30 minutes, under 60 minutes)? "
            "Or no constraint?"
        )
    return (
        "Anything else I should weight toward — for example personality/"
        "behavioural tests, cognitive ability, or a specific skill area?"
    )


def _format_line(a: Assessment) -> str:
    bits = [a.test_type_name]
    if a.duration_minutes:
        bits.append(f"~{int(a.duration_minutes)} min")
    return f"{a.name} ({', '.join(bits)})"


def compare_reply(a: Assessment, b: Assessment) -> str:
    def block(x: Assessment) -> str:
        desc = x.description or (
            f"A {x.test_type_name.lower()} assessment"
            + (f", roughly {int(x.duration_minutes)} minutes." if x.duration_minutes else ".")
        )
        return f"**{x.name}** ({x.test_type_name}): {desc}"

    differ = (
        f", while {b.name} is a {b.test_type_name.lower()} assessment"
        if b.test_type_name != a.test_type_name else ""
    )
    return (
        f"{block(a)}\n\n{block(b)}\n\n"
        f"In short: {a.name} is a {a.test_type_name.lower()} assessment"
        f"{differ}. "
        "Let me know if you'd like either added to a shortlist, or to keep "
        "going with the current recommendations."
    )


def cannot_compare_reply(missing: List[str]) -> str:
    names = " and ".join(missing)
    return (
        f"I couldn't find {names} in the SHL catalog — could you confirm "
        "the exact assessment name(s)? I won't compare anything not in "
        "the catalog."
    )


def recommend_reply(
    results: List[Assessment],
    state: ConversationState,
    confirmed: bool = False,
) -> str:
    n = len(results)
    descriptor_bits = []
    if state.seniority:
        descriptor_bits.append(state.seniority.lower())
    if state.role_terms:
        descriptor_bits.append(state.role_terms[0][:60])
    descriptor = " for a " + " ".join(descriptor_bits) if descriptor_bits else ""
    lines = "\n".join(f"- {_format_line(a)}" for a in results)

    if confirmed:
        return (
            f"Confirmed. Final shortlist "
            f"({n} assessment{'s' if n != 1 else ''}{descriptor}):\n\n{lines}"
        )
    return (
        f"Here are {n} assessment{'s' if n != 1 else ''}{descriptor} "
        f"that fit what you've described:\n\n{lines}\n\n"
        "Let me know if you'd like to refine this — add or remove an "
        "assessment, adjust the time limit, or compare any of these."
    )


def no_results_reply() -> str:
    return (
        "I couldn't find any catalog assessments matching all of those "
        "constraints together — would you like me to relax the time limit "
        "or the test-type restriction?"
    )


# ── Retrieval orchestration ───────────────────────────────────────────────────

def _query_text(state: ConversationState) -> str:
    parts = list(state.role_terms)
    if state.jd_text:
        parts.append(state.jd_text)
    if state.seniority:
        parts.append(state.seniority)
    return " ".join(parts)


def _passes(a: Assessment, f: Filters, remove_names: Set[str]) -> bool:
    if f.max_duration is not None and a.duration_minutes is not None:
        if a.duration_minutes > f.max_duration:
            return False
    if f.exclude_types and any(t in f.exclude_types for t in a.test_types):
        return False
    for rm in remove_names:
        rm_clean = rm.replace("_", " ").lower()
        if rm_clean in a.name.lower() or rm_clean in a.slug.lower():
            return False
    return True


def _should_add_opq(state: ConversationState, pool: List[Assessment]) -> bool:
    """Add OPQ32r by default for professional roles unless excluded."""
    if "P" in state.exclude_types:
        return False
    # Respect duration constraint — OPQ32r is 25 min
    if state.duration_max is not None and state.duration_max < 25:
        return False
    # Check for named removal
    opq_removed = any(
        "opq" in rm or "occupational" in rm or "personality questionnaire" in rm
        for rm in state.remove_names
    )
    if opq_removed:
        return False
    # Already in pool
    if any("opq32r" in a.slug or "occupational-personality" in a.slug for a in pool):
        return False
    # Skip for obvious entry-level frontline roles
    all_text = " ".join(state.role_terms).lower() + " " + (state.jd_text or "").lower()
    if any(hint in all_text for hint in _SKIP_OPQ_SIGNALS):
        return False
    return True


def _should_add_verify(state: ConversationState, pool: List[Assessment]) -> bool:
    """Add Verify G+ for senior/graduate roles if not already present."""
    if "A" in state.exclude_types:
        return False
    # Respect duration constraint — Verify G+ is 36 min
    if state.duration_max is not None and state.duration_max < 36:
        return False
    verify_removed = any("verify" in rm and "g" in rm for rm in state.remove_names)
    if verify_removed:
        return False
    if any("verify-interactive-g" in a.slug or "verify-g" in a.slug for a in pool):
        return False
    # Only add for senior/professional/graduate
    senior = state.seniority in (
        "Professional Individual Contributor", "Manager",
        "Director", "Executive", "Graduate",
    )
    jd_senior = bool(re.search(
        r"\b(senior|lead|architect|principal|staff|graduate)\b",
        (state.jd_text or "").lower(),
    ))
    return senior or jd_senior


def gather_recommendations(state: ConversationState) -> List[Assessment]:
    hard_filters = Filters(
        max_duration=state.duration_max,
        exclude_types=state.exclude_types or None,
    )
    k = MAX_RECOMMENDATIONS if state.jd_text else DEFAULT_RECOMMENDATIONS
    if state.include_types:
        k = min(MAX_RECOMMENDATIONS, k + 2)

    pool: List[Assessment] = []
    seen: Set[str] = set()

    def add(items):
        for a in items:
            if a.id not in seen and _passes(a, hard_filters, state.remove_names):
                pool.append(a)
                seen.add(a.id)

    # 1. Grounded keyword matches
    add(a for a in keyword_candidates(state.skills) if _passes(a, hard_filters, state.remove_names))

    # 2. Explicit type requests — union with keyword matches
    if state.include_types:
        type_filters = Filters(
            max_duration=state.duration_max,
            include_types=state.include_types,
            exclude_types=state.exclude_types or None,
        )
        add(search(
            _query_text(state), skill_terms=[], filters=type_filters,
            k=max(3, k - len(pool)),
        ))

    # 3. BM25 fill — only when keyword matches are sparse
    # 3. Hybrid fill — BM25 first, then vector search for abstract queries
    keyword_count = len(pool)
    remaining = k - len(pool)

    if remaining > 0 and keyword_count < 5:
        # BM25 for keyword-rich queries
        bm25_results = search(
            _query_text(state), skill_terms=[], filters=hard_filters,
            k=remaining * 2,
        )
        add(bm25_results)

    # Vector search always runs as a fill — catches abstract/competency queries
    # that neither keyword matching nor BM25 handle well
    remaining = k - len(pool)
    if remaining > 0:
        from .embeddings import vector_search
        # Build a rich semantic query from everything we know
        semantic_query = _query_text(state)
        if state.jd_text:
            semantic_query = state.jd_text  # JD text is richest signal
        vec_results = vector_search(
            semantic_query,
            filters=hard_filters,
            k=remaining + 5,   # fetch a few extra, dedup will trim
            min_score=0.25,
        )
        add(vec_results)

    # Before adding defaults, reserve slots for them
    will_add_opq = _should_add_opq(state, pool)
    will_add_verify = _should_add_verify(state, pool)
    reserve = (1 if will_add_opq else 0) + (1 if will_add_verify else 0)
    if reserve > 0 and len(pool) > k - reserve:
        pool = pool[:k - reserve]
        seen = {a.id for a in pool}

    # 4. Add OPQ32r by default for professional roles
    if will_add_opq and _should_add_opq(state, pool):
        opq = _get_by_slug(_DEFAULT_OPQ_SLUG)
        if opq and opq.id not in seen:
            pool.append(opq)
            seen.add(opq.id)

    # 5. Add Verify G+ by default for senior/graduate roles
    if will_add_verify and _should_add_verify(state, pool):
        verify = _get_by_slug(_DEFAULT_VERIFY_SLUG)
        if verify and verify.id not in seen:
            pool.append(verify)
            seen.add(verify.id)

    return pool[:k]


# ── Main decision function ────────────────────────────────────────────────────

def decide(messages: List[Message]) -> ChatResponse:
    if not messages or messages[-1].role != "user":
        return ChatResponse(
            reply=(
                "I didn't receive a message to respond to — "
                "what role or skill would you like to assess?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    state = build_state(messages)

    if state.latest_injection:
        return ChatResponse(
            reply=refusal_reply("injection"),
            recommendations=[],
            end_of_conversation=False,
        )

    if state.latest_off_topic:
        return ChatResponse(
            reply=refusal_reply(state.latest_off_topic),
            recommendations=[],
            end_of_conversation=False,
        )

    if len(state.latest_compare) == 2:
        a = find_by_name(state.latest_compare[0])
        b = find_by_name(state.latest_compare[1])
        missing = [
            state.latest_compare[i] for i, x in enumerate([a, b]) if x is None
        ]
        if missing:
            return ChatResponse(
                reply=cannot_compare_reply(missing),
                recommendations=[],
                end_of_conversation=False,
            )
        return ChatResponse(
            reply=compare_reply(a, b),
            recommendations=[],
            end_of_conversation=False,
        )

    have_anchor = bool(state.skills or state.jd_text or state.role_terms)
    have_context = bool(
        state.jd_text
        or state.duration_max is not None
        or state.seniority
        or state.include_types
        or state.num_user_turns >= 2
    )
    sufficient = have_anchor and have_context

    if not sufficient:
        return ChatResponse(
            reply=clarify_reply(state),
            recommendations=[],
            end_of_conversation=False,
        )

    results = gather_recommendations(state)
    if not results:
        return ChatResponse(
            reply=no_results_reply(),
            recommendations=[],
            end_of_conversation=False,
        )

    recs = [
        Recommendation(name=a.name, url=a.url, test_type=a.test_type)
        for a in results
    ]

    # eoc=true ONLY when user has explicitly confirmed AND we've already
    # been through at least one recommendation cycle
    confirmed = state.latest_confirmation and state.num_user_turns >= 2

    return ChatResponse(
        reply=recommend_reply(results, state, confirmed=confirmed),
        recommendations=recs,
        end_of_conversation=confirmed,
    )
