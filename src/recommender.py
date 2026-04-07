"""
recommender.py — Proactive recommendation engine.

On employee login:
1. Queries ChromaDB for corpus chunks relevant to the employee's profile
   (age, family status, enrolled plans, interests, tenure).
2. Filters out topics already in ``seen_recommendations``.
3. Passes the top candidates to GPT-4o to generate a personalised opening
   greeting with 1-2 tailored benefit nudges.
"""

from __future__ import annotations

import openai
from dotenv import load_dotenv

from . import vector_store as vs
from .prompts import build_recommendation_prompt, format_employee_context

load_dotenv(override=True)

CHAT_MODEL = "gpt-4o"
_RETRIEVE_TOP_K = 6  # fetch more than we need so filtering has headroom


def _build_profile_query(state: dict) -> str:
    """Construct a natural-language query that captures the employee's profile.

    The query is designed to surface corpus chunks most relevant to the
    employee's situation so that the recommender has good candidates to work
    with.
    """
    parts: list[str] = []

    age = state.get("age")
    if age:
        parts.append(f"employee age {age}")

    family_status = state.get("family_status", "")
    if family_status:
        parts.append(family_status)

    enrolled = state.get("enrolled_benefits", {})
    health = enrolled.get("health_plan")
    if health:
        parts.append(f"health plan {health}")

    contrib = enrolled.get("retirement_401k_contribution_pct", 0)
    if contrib == 0:
        parts.append("not contributing to 401k retirement savings")
    else:
        parts.append(f"401k contribution {contrib}%")

    fsa = enrolled.get("fsa_election")
    hsa = enrolled.get("hsa_election")
    if not fsa and not hsa:
        parts.append("no FSA HSA account")

    interests: list[str] = []
    for item in state.get("interests", []):
        if isinstance(item, dict):
            interests.append(item.get("text", ""))
        else:
            interests.append(str(item))
    if interests:
        parts.append(f"interested in {', '.join(interests)}")

    hire_date = state.get("hire_date", "")
    if hire_date:
        parts.append(f"hired {hire_date}")

    return " | ".join(parts) if parts else "employee benefits overview"


def _filter_seen(chunks: list[dict], seen: list[str]) -> list[dict]:
    """Remove chunks whose source topic is already in *seen*."""
    seen_set = set(seen)
    return [c for c in chunks if c.get("source") not in seen_set]


def generate_greeting(state: dict) -> str:
    """Generate a personalised opening greeting for the employee.

    Queries ChromaDB with a profile-derived query, filters already-seen
    topics, then calls GPT-4o to craft a warm, personalised greeting with
    1-2 concrete benefit recommendations.

    Args:
        state: The loaded employee state dict.

    Returns:
        A markdown-formatted greeting string ready to display in the chatbot.
    """
    # 1. Build a profile-based query and retrieve candidates
    query = _build_profile_query(state)
    all_chunks = vs.query(query, n_results=_RETRIEVE_TOP_K)

    # 2. Filter out topics already surfaced to this employee
    seen = state.get("seen_recommendations", [])
    unseen_chunks = _filter_seen(all_chunks, seen)

    # Fall back to all chunks if everything has been seen
    candidates = unseen_chunks if unseen_chunks else all_chunks

    # Limit to top-3 for brevity
    candidates = candidates[:3]

    # 3. Build the GPT-4o prompt and call the API
    messages = build_recommendation_prompt(state, candidates)
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=300,
    )

    greeting = response.choices[0].message.content.strip()

    # 4. Record the surfaced topic sources so they won't be repeated
    surfaced = list({c["source"] for c in candidates})
    seen_set = set(seen)
    for src in surfaced:
        if src not in seen_set:
            state.setdefault("seen_recommendations", []).append(src)

    return greeting
