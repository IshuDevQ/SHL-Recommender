from __future__ import annotations
import os
import httpx

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
LLM_ENABLED = os.environ.get("LLM_REPHRASE_ENABLED", "true").lower() == "true"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT_SECONDS", "4"))


_SYSTEM = (
    "You lightly polish a chat reply from an SHL assessment recommender so "
    "it reads naturally. Keep every fact, every assessment name, every number, "
    "and the overall meaning EXACTLY the same — only improve phrasing and flow. "
    "Do not add new claims, assessments, or URLs. "
    "Return only the rewritten reply, nothing else."
)


def maybe_rephrase(reply: str) -> str:
    """Optional cosmetic LLM polish. Disabled by default. Never touches recommendations."""
    if not LLM_ENABLED or not GROQ_API_KEY:
        return reply
    try:
        with httpx.Client(timeout=LLM_TIMEOUT) as client:
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "temperature": 0.3,
                    "max_tokens": 300,
                    "messages": [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": reply},
                    ],
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return text or reply
    except Exception:
        return reply
