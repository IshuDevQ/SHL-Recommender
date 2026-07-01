"""
Run once after downloading assessments.json.

Usage:
    python data/clean_catalog.py

Input:  data/assessments.json   (downloaded from SHL)
Output: data/catalog.json       (used by the running service)
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
RAW_JSON = HERE / "assessments.json"
OUT_JSON = HERE / "catalog.json"

TYPE_NAME_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgement": "B",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Personality & Behaviour": "P",
    "Simulations": "S",
}

# When an item has multiple keys, pick in this priority order
TYPE_PRIORITY = ["K", "S", "A", "P", "B", "E", "C", "D"]

# Pre-packaged Job Solutions — excluded (not Individual Test Solutions)
EXCLUDE_NAMES = {
    "Customer Service Phone Solution",
    "Entry Level Cashier Solution",
    "Entry Level Customer Service (General) Solution",
    "Entry Level Customer Serv-Retail & Contact Center",
    "Entry Level Hotel Front Desk Solution",
    "Entry Level Sales Solution",
    "Entry Level Technical Support Solution",
    "Sales & Service Phone Solution",
}

# Verified descriptions for key flagship products
VERIFIED_DESCRIPTIONS = {
    "Occupational Personality Questionnaire OPQ32r": (
        "OPQ32r is SHL's flagship occupational personality questionnaire. "
        "It measures 32 personality characteristics relevant to workplace "
        "behaviour, organised into the Big 5-aligned domains of Relationships "
        "with People, Thinking Style, and Feelings & Emotions, used to predict "
        "job performance, leadership potential and team fit."
    ),
    "Global Skills Assessment": (
        "The Global Skills Assessment (GSA) measures 96 discrete behavioural "
        "skills aligned to SHL's Universal Competency Framework (UCF) in a "
        "single 15-minute forced-choice assessment. Unlike OPQ32r's trait-based "
        "personality profile, GSA reports self-reported current behaviour against "
        "specific job-relevant skill scales, suited to skills-based hiring, "
        "internal mobility, and development use cases across any role or level."
    ),
}

DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)")
_NOISE = re.compile(
    r"\b(new|next generation|[0-9]+\.[0-9]+|report|form \d+|\(.*?\))\b", re.I
)
_NON_ALNUM = re.compile(r"[^a-z0-9.+# ]+")


def normalise(text: str) -> str:
    text = text.lower()
    text = _NOISE.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_duration(row: dict):
    """
    Returns float minutes or None.
    Handles: '30 minutes', 'Approximate Completion Time in minutes = 30',
             'Variable', 'Untimed', '0 minutes', ''
    """
    for field_name in ("duration", "duration_raw"):
        val = str(row.get(field_name, "")).strip()
        if not val:
            continue
        low = val.lower()
        if "variable" in low or "untimed" in low:
            return None
        m = DURATION_RE.search(val)
        if m:
            value = float(m.group(1))
            return value if value > 0 else None
    return None


def slugify(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def pick_primary_type(codes: list) -> str:
    for code in TYPE_PRIORITY:
        if code in codes:
            return code
    return codes[0] if codes else ""


def main():
    if not RAW_JSON.exists():
        print(f"ERROR: {RAW_JSON} not found.")
        print("Run this first:")
        print(
            '  curl -o data/assessments.json '
            '"https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"'
        )
        return

    raw_text = RAW_JSON.read_bytes().decode("utf-8", errors="replace")
    raw = json.loads(raw_text, strict=False)
    catalog = []
    seen_ids = set()
    excluded = 0

    for row in raw:
        if row.get("status", "ok") != "ok":
            excluded += 1
            continue

        name = row.get("name", "").strip()
        if not name or name in EXCLUDE_NAMES:
            excluded += 1
            continue

        # Map keys list to type codes
        keys = row.get("keys", [])
        type_codes = []
        type_names = []
        seen_codes: set = set()
        for key in keys:
            code = TYPE_NAME_TO_CODE.get(key.strip())
            if code and code not in seen_codes:
                type_codes.append(code)
                type_names.append(key.strip())
                seen_codes.add(code)

        if not type_codes:
            excluded += 1
            continue

        primary_code = pick_primary_type(type_codes)
        primary_name = next(
            (n for n, c in zip(type_names, type_codes) if c == primary_code),
            type_names[0],
        )

        entity_id = str(row.get("entity_id", "")).strip()
        if entity_id and entity_id in seen_ids:
            continue
        if entity_id:
            seen_ids.add(entity_id)

        url = row.get("link", "").strip()
        remote = str(row.get("remote", "")).strip().lower() == "yes"
        adaptive = str(row.get("adaptive", "")).strip().lower() == "yes"
        duration = parse_duration(row)
        description = row.get("description", "").strip()
        description = re.sub(r"\r\n|\r", "\n", description)
        description = re.sub(r"\n{3,}", "\n\n", description).strip()

        if name in VERIFIED_DESCRIPTIONS:
            description = VERIFIED_DESCRIPTIONS[name]

        catalog.append({
            "id": entity_id,
            "slug": slugify(url),
            "name": name,
            "url": url,
            "test_type": primary_code,
            "test_types": type_codes,
            "test_type_name": primary_name,
            "test_type_names": type_names,
            "remote_testing": remote,
            "adaptive_irt": adaptive,
            "duration_minutes": duration,
            "description": description,
            "job_levels": row.get("job_levels", []),
        })

    catalog.sort(key=lambda r: r["name"])
    OUT_JSON.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done. Wrote {len(catalog)} assessments to {OUT_JSON}  (excluded {excluded})")


if __name__ == "__main__":
    main()
