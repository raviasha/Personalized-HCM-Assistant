"""
prompts.py — System prompts and message-building helpers.
"""

from __future__ import annotations

import json
from . import vector_store as vs
# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are the Acme Corp Benefits Assistant — a knowledgeable, friendly, and
empathetic HR benefits advisor.

Your job is to help Acme Corp employees understand and make the most of their
benefits. You answer questions using only the official Acme Corp benefits
documentation provided in each conversation.

Guidelines:
- Be concise, warm, and professional.
- Base every answer strictly on the provided documentation context.
  Do not invent plan details, dollar amounts, dates, or rules.
- If a question is not covered by the provided context, say so clearly and
  direct the employee to HR at benefits@acme.com.
- Use bullet points or numbered lists when they improve clarity.
- Encourage employees to take action where appropriate (e.g., enroll before
  the deadline, contact Fidelity to change contribution rate, etc.).
- Never ask the employee for sensitive personal or financial data.
- When an employee profile is provided, tailor your response to their
  specific situation (age, enrolled plans, interests, tenure, family status).

Source citation rules:
- The documentation excerpts provided to you are each labelled [DOC-1], [DOC-2], etc.
- At the end of EVERY response, after a blank line and a "---" separator, add a
  "**📄 Sources:**" section and list each document label you actually used,
  one per line, exactly as the label (e.g. [DOC-1]).
- Only list labels you genuinely drew information from. Omit unused ones.
- Do not add any extra text to the sources list — just the labels.
"""


# ---------------------------------------------------------------------------
# Employee context formatter
# ---------------------------------------------------------------------------

def _extract_text_items(items: list) -> list[str]:
    """Return plain text strings from an array that may contain dicts or strings."""
    result = []
    for item in items:
        if isinstance(item, dict):
            result.append(item.get("text", str(item)))
        else:
            result.append(str(item))
    return result


def format_employee_context(state: dict) -> str:
    """Format an employee state dict as a concise context block for the LLM.

    Only includes fields that are relevant to personalisation; omits session
    bookkeeping fields like ``session_counter`` and ``seen_recommendations``.
    """
    lines: list[str] = [
        "=== EMPLOYEE PROFILE ===",
        f"Name: {state.get('name', 'Unknown')}",
        f"Age: {state.get('age', 'Unknown')}",
        f"Department: {state.get('department', 'Unknown')}",
        f"Tenure: hired {state.get('hire_date', 'Unknown')}",
        f"Family status: {state.get('family_status', 'Unknown')}",
    ]

    enrolled = state.get("enrolled_benefits", {})
    if enrolled:
        lines.append("Enrolled benefits:")
        lines.append(f"  - Health plan: {enrolled.get('health_plan', 'None')}")
        lines.append(f"  - 401k contribution: {enrolled.get('retirement_401k_contribution_pct', 0)}%")
        lines.append(f"  - FSA election: {enrolled.get('fsa_election') or 'None'}")
        lines.append(f"  - HSA election: {enrolled.get('hsa_election') or 'None'}")

    interests = _extract_text_items(state.get("interests", []))
    if interests:
        lines.append(f"Known interests: {', '.join(interests)}")

    questions = _extract_text_items(state.get("unresolved_questions", []))
    if questions:
        lines.append("Open questions:")
        for q in questions:
            lines.append(f"  - {q}")

    actions = _extract_text_items(state.get("pending_actions", []))
    if actions:
        lines.append("Pending actions:")
        for a in actions:
            lines.append(f"  - {a}")

    history = state.get("interaction_history", [])
    if history:
        lines.append("Recent interaction history:")
        for entry in history:
            if isinstance(entry, dict):
                session_num = entry.get("session", "?")
                summary = entry.get("summary", "")
                lines.append(f"  [Session {session_num}] {summary}")
            else:
                lines.append(f"  {entry}")

    historical_summary = state.get("historical_summary", "")
    if historical_summary:
        lines.append(f"Older history summary: {historical_summary}")

    lines.append("=========================")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def build_rag_messages(
    context_chunks: list[dict],
    conversation_history: list[dict],
    user_message: str,
    employee_state: dict | None = None,
) -> tuple[list[dict], dict]:
    """Assemble the messages list for a RAG chat completion request.

    Each context chunk is labelled [DOC-N] so the LLM can cite sources.

    Args:
        context_chunks: List of dicts with at least ``"text"`` and ``"source"``
            keys, as returned by ``vector_store.query()``.
        conversation_history: The Gradio chatbot history (list of
            ``{"role": ..., "content": ...}`` dicts). The current user
            message must **not** be included here — it is passed separately.
        user_message: The sanitised user message for the current turn.
        employee_state: Optional employee state dict for personalisation.

    Returns:
        A ``(messages, doc_map)`` tuple where ``messages`` is the list of
        OpenAI-compatible message dicts and ``doc_map`` maps label strings
        like ``"DOC-1"`` to ``{"title": str, "source": str}`` dicts.
    """
    title_map = vs.get_source_title_map()

    # Build labelled context sections and the doc_map simultaneously
    doc_map: dict[str, dict] = {}
    labelled_sections: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        label = f"DOC-{i}"
        source = chunk.get("source", "")
        title = title_map.get(source, source.replace("_", " ").title())
        doc_map[label] = {"title": title, "source": source}
        labelled_sections.append(f"[{label}: {title}]\n\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(labelled_sections)

    # Build the context injection content, including employee profile if available
    context_parts = [
        "Here is the relevant Acme Corp benefits documentation for this conversation:\n\n"
        f"{context_text}"
    ]
    if employee_state:
        context_parts.append(
            "\nHere is the profile of the employee you are speaking with:\n\n"
            + format_employee_context(employee_state)
        )

    messages: list[dict] = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n".join(context_parts),
        },
        {
            "role": "assistant",
            "content": (
                "Thank you — I have reviewed the relevant benefits "
                "documentation and the employee's profile. I am ready to "
                "provide personalised guidance."
            ),
        },
    ]

    # Append previous turns (skip the initial system greeting from the UI)
    for turn in conversation_history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # Append the current user message
    messages.append({"role": "user", "content": user_message})

    return messages, doc_map


# ---------------------------------------------------------------------------
# Recommendation / greeting prompt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Memory update prompt
# ---------------------------------------------------------------------------

MEMORY_UPDATE_SYSTEM_PROMPT = """\
You are an assistant that extracts structured memory updates from an HR benefits
chat conversation.

Analyze the conversation transcript and the employee's current state, then
produce a JSON delta describing what changed.

Output ONLY a single valid JSON object — no markdown fences, no commentary.

Schema:
{
  "add": {
    "interests":            ["plain-text string", ...],
    "unresolved_questions": ["plain-text string", ...],
    "pending_actions":      ["plain-text string", ...],
    "interaction_summary":  "single string summarising this conversation (max 200 chars)"
  },
  "remove": {
    "interests":            ["exact text of existing item to remove", ...],
    "unresolved_questions": ["exact text of existing item to remove", ...],
    "pending_actions":      ["exact text of existing item to remove", ...]
  },
  "update": {
    "last_interaction_timestamp": "ISO-8601 UTC datetime string"
  }
}

Rules:
- All keys in "add", "remove", and "update" are OPTIONAL. Omit keys whose
  arrays / values would be empty or unchanged.
- "interaction_summary" must always be present in "add" if any real conversation
  occurred (at least one user message).
- For "add.interests": only add topics genuinely explored in THIS conversation
  that are not already in the employee's interests list.
- For "add.unresolved_questions": only add questions that were asked but NOT
  fully answered in this conversation.
- For "add.pending_actions": only add action items the employee explicitly
  expressed intent to take.
- For "remove.unresolved_questions": only remove items present in the current
  state whose question was clearly and fully answered in this conversation.
- For "remove.interests": remove an interest only if it is now fully addressed
  and the employee expressed no further curiosity.
- For "remove.pending_actions": remove an action only if the transcript shows
  it was completed or explicitly abandoned.
- NEVER change enrolled_benefits, personal info fields, or invent facts.
- NEVER add PII to any field.
- Use the exact text from the current unresolved_questions / interests /
  pending_actions lists when referring to items to remove (for exact matching).
"""


def build_memory_update_messages(
    transcript: str,
    employee_state: dict,
    current_timestamp: str,
) -> list[dict]:
    """Assemble the messages for the post-conversation memory-update call.

    Args:
        transcript: The full conversation as a plain-text string (role: text).
        employee_state: The current employee state dict.
        current_timestamp: ISO-8601 UTC string for ``last_interaction_timestamp``.

    Returns:
        A list of OpenAI-compatible message dicts.
    """
    from . import vector_store as vs  # local import to avoid circular deps

    # Summarise relevant mutable fields so the LLM can reason about deltas
    interests = [
        (i.get("text", str(i)) if isinstance(i, dict) else str(i))
        for i in employee_state.get("interests", [])
    ]
    questions = [
        (q.get("text", str(q)) if isinstance(q, dict) else str(q))
        for q in employee_state.get("unresolved_questions", [])
    ]
    actions = [
        (a.get("text", str(a)) if isinstance(a, dict) else str(a))
        for a in employee_state.get("pending_actions", [])
    ]

    state_summary = (
        f"Employee: {employee_state.get('name', 'Unknown')}\n"
        f"Current interests: {interests or '(none)'}\n"
        f"Current unresolved questions: {questions or '(none)'}\n"
        f"Current pending actions: {actions or '(none)'}\n"
        f"Current session counter: {employee_state.get('session_counter', 0)}\n"
        f"Current timestamp (use for update.last_interaction_timestamp): {current_timestamp}"
    )

    user_content = (
        "CURRENT EMPLOYEE STATE SUMMARY:\n"
        f"{state_summary}\n\n"
        "CONVERSATION TRANSCRIPT:\n"
        f"{transcript}\n\n"
        "Produce the JSON memory delta now."
    )

    return [
        {"role": "system", "content": MEMORY_UPDATE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------

_RECOMMENDATION_SYSTEM_PROMPT = """\
You are the Acme Corp Benefits Assistant — a knowledgeable, friendly, and
empathetic HR benefits advisor.

Your task is to greet an employee at the start of a session and proactively
surface 1-2 benefit topics that are specifically relevant to their profile and
situation.

Guidelines:
- Address the employee by their first name.
- Be warm, concise, and actionable — no more than 3-4 sentences total.
- Reference at least one concrete detail from their profile (e.g., their
  enrollment status, tenure, age, family situation) to show personalisation.
- Briefly mention 1-2 specific benefit opportunities or reminders drawn from
  the provided documentation excerpts.
- End with an open invitation for them to ask questions.
- Do NOT include a sources section.
- Do NOT invent numbers or rules not present in the documentation excerpts.
"""


def build_recommendation_prompt(
    employee_state: dict,
    candidate_chunks: list[dict],
) -> list[dict]:
    """Build the messages list for the proactive greeting completion call.

    Args:
        employee_state: The loaded employee state dict.
        candidate_chunks: Up to 3 corpus chunks most relevant to this employee
            and not yet surfaced (as returned by ``recommender``).

    Returns:
        A list of OpenAI-compatible message dicts.
    """
    title_map = vs.get_source_title_map()

    # Format candidate chunks as brief excerpts
    excerpt_parts: list[str] = []
    for i, chunk in enumerate(candidate_chunks, start=1):
        source = chunk.get("source", "")
        title = title_map.get(source, source.replace("_", " ").title())
        # Keep excerpts short so the prompt stays within budget
        excerpt = chunk["text"][:600]
        excerpt_parts.append(f"[Excerpt {i}: {title}]\n{excerpt}")

    excerpts_text = "\n\n---\n\n".join(excerpt_parts)

    employee_context = format_employee_context(employee_state)

    user_content = (
        "Here are some relevant benefits documentation excerpts:\n\n"
        f"{excerpts_text}\n\n"
        "Here is the employee's profile:\n\n"
        f"{employee_context}\n\n"
        "Please write a personalised opening greeting for this employee."
    )

    return [
        {"role": "system", "content": _RECOMMENDATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
