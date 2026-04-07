"""
memory_updater.py — Post-conversation bidirectional memory update.

After a chat session ends, this module:
  1. Formats the conversation transcript.
  2. Calls GPT-4o to extract a structured add/remove/update delta.
  3. Validates the delta schema.
  4. Merges the delta into the employee state and returns the updated state.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import openai
from dotenv import load_dotenv

from .prompts import build_memory_update_messages

load_dotenv(override=True)

_MODEL = "gpt-4o"
_MUTABLE_ARRAY_FIELDS = ("interests", "unresolved_questions", "pending_actions")


# ---------------------------------------------------------------------------
# Transcript formatter
# ---------------------------------------------------------------------------

def _format_transcript(history: list[dict]) -> str:
    """Convert Gradio message history into a plain-text transcript string."""
    lines: list[str] = []
    for turn in history:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Delta extraction
# ---------------------------------------------------------------------------

def _extract_delta(transcript: str, employee_state: dict, current_timestamp: str) -> dict:
    """Call GPT-4o to extract an add/remove/update delta from the transcript.

    Returns:
        The parsed delta dict (may be empty on failure).
    """
    messages = build_memory_update_messages(transcript, employee_state, current_timestamp)
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=600,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Delta validation
# ---------------------------------------------------------------------------

def _validate_delta(delta: dict) -> dict:
    """Validate and sanitise the delta dict returned by the LLM.

    - Ensures top-level keys are only "add", "remove", "update".
    - Ensures array fields contain only strings.
    - Ensures "interaction_summary" is a string ≤ 200 chars.
    - Removes unrecognised nested keys.
    - Returns a cleaned delta dict.

    Raises:
        ValueError: if the delta is structurally invalid beyond repair.
    """
    if not isinstance(delta, dict):
        raise ValueError("Delta must be a JSON object.")

    allowed_top = {"add", "remove", "update"}
    for key in list(delta.keys()):
        if key not in allowed_top:
            del delta[key]

    # Validate "add"
    add = delta.get("add", {})
    if not isinstance(add, dict):
        delta["add"] = {}
        add = delta["add"]

    for field in _MUTABLE_ARRAY_FIELDS:
        if field in add:
            if not isinstance(add[field], list):
                del add[field]
            else:
                add[field] = [str(item) for item in add[field] if item]

    if "interaction_summary" in add:
        summary = add["interaction_summary"]
        if not isinstance(summary, str) or not summary.strip():
            del add["interaction_summary"]
        else:
            add["interaction_summary"] = summary.strip()[:200]

    # Disallow enrollment changes in "add"
    for forbidden in ("enrolled_benefits", "name", "employee_id", "age",
                      "department", "hire_date", "family_status"):
        add.pop(forbidden, None)

    # Validate "remove"
    remove = delta.get("remove", {})
    if not isinstance(remove, dict):
        delta["remove"] = {}
        remove = delta["remove"]

    for field in _MUTABLE_ARRAY_FIELDS:
        if field in remove:
            if not isinstance(remove[field], list):
                del remove[field]
            else:
                remove[field] = [str(item) for item in remove[field] if item]

    # Validate "update"
    update = delta.get("update", {})
    if not isinstance(update, dict):
        delta["update"] = {}
        update = delta["update"]

    allowed_update_keys = {"last_interaction_timestamp"}
    for key in list(update.keys()):
        if key not in allowed_update_keys:
            del update[key]

    return delta


# ---------------------------------------------------------------------------
# Fuzzy remove helper
# ---------------------------------------------------------------------------

def _find_matching_item(items: list, target_text: str, threshold: float = 0.6) -> int | None:
    """Return the index of the best-matching item in *items* for *target_text*.

    Matching strategy:
      1. Exact text match (case-insensitive).
      2. Substring containment.
      3. Token overlap ratio ≥ *threshold*.

    Returns:
        Index of the matching item, or ``None`` if no match found.
    """
    target_lower = target_text.strip().lower()

    for i, item in enumerate(items):
        text = (
            item.get("text", str(item)).lower()
            if isinstance(item, dict)
            else str(item).lower()
        )
        # Exact match
        if text == target_lower:
            return i
        # Substring containment (either direction)
        if target_lower in text or text in target_lower:
            return i

    # Token overlap
    target_tokens = set(re.findall(r"\w+", target_lower))
    best_ratio = 0.0
    best_idx = None
    for i, item in enumerate(items):
        text = (
            item.get("text", str(item)).lower()
            if isinstance(item, dict)
            else str(item).lower()
        )
        item_tokens = set(re.findall(r"\w+", text))
        if not item_tokens and not target_tokens:
            continue
        union = item_tokens | target_tokens
        if not union:
            continue
        overlap = len(item_tokens & target_tokens) / len(union)
        if overlap > best_ratio:
            best_ratio = overlap
            best_idx = i

    if best_ratio >= threshold:
        return best_idx
    return None


# ---------------------------------------------------------------------------
# Delta merge
# ---------------------------------------------------------------------------

def _apply_delta(state: dict, delta: dict) -> dict:
    """Merge *delta* into *state* and return the updated state.

    - Additions: new items are wrapped in the ``{text, added_session,
      last_referenced_session}`` metadata format used by the compactor.
    - Removals: items are fuzzy-matched by text and removed.
    - Updates: fields in ``delta["update"]`` overwrite their counterparts.
    - ``interaction_summary`` is appended to ``interaction_history``.
    """
    session = state.get("session_counter", 0)

    # --- ADDITIONS ---
    add = delta.get("add", {})
    for field in _MUTABLE_ARRAY_FIELDS:
        new_texts: list[str] = add.get(field, [])  # type: ignore[assignment]
        if not new_texts:
            continue
        existing: list = state.setdefault(field, [])
        # Collect existing plain texts to avoid duplicates
        existing_texts = {
            (item.get("text", "").lower() if isinstance(item, dict) else str(item).lower())
            for item in existing
        }
        for text in new_texts:
            if text.lower() not in existing_texts:
                existing.append({
                    "text": text,
                    "added_session": session,
                    "last_referenced_session": session,
                })
                existing_texts.add(text.lower())

    # Append interaction summary to history
    summary_text: str = add.get("interaction_summary", "")  # type: ignore[assignment]
    if summary_text:
        history: list = state.setdefault("interaction_history", [])
        history.append({"session": session, "summary": summary_text})

    # --- REMOVALS ---
    remove = delta.get("remove", {})
    for field in _MUTABLE_ARRAY_FIELDS:
        remove_texts: list[str] = remove.get(field, [])  # type: ignore[assignment]
        if not remove_texts:
            continue
        existing = state.get(field, [])
        for target in remove_texts:
            idx = _find_matching_item(existing, target)
            if idx is not None:
                existing.pop(idx)

    # --- UPDATES ---
    update = delta.get("update", {})
    for key, value in update.items():
        state[key] = value

    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_state(history: list[dict], state: dict) -> dict:
    """Analyse the conversation and return an updated employee state dict.

    Skips the update and returns the original state unchanged if the
    conversation contains no real user messages.

    Args:
        history: Gradio chatbot history (list of ``{"role": ..., "content": ...}``).
        state:   Current employee state dict (modified in place and returned).

    Returns:
        The updated state dict.
    """
    # Only process if there is at least one real user turn
    user_turns = [t for t in history if t.get("role") == "user"]
    if not user_turns:
        return state

    transcript = _format_transcript(history)
    current_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        raw_delta = _extract_delta(transcript, state, current_timestamp)
        delta = _validate_delta(raw_delta)
        state = _apply_delta(state, delta)
    except Exception as exc:  # noqa: BLE001
        # Log but don't crash — fall back to saving unchanged state
        print(f"[MemoryUpdater] Delta extraction/merge failed: {exc}")

    return state
