"""
state_compactor.py — Token budget enforcement, field-level compaction, and expiry.

Runs on login (after state load, before recommender) to keep the employee state
JSON lean and within the 1500-token prompt budget.

Compaction rules (from plan FR-8):
  - interaction_history:  cap 5  — oldest entries summarized into historical_summary via GPT-4o
  - interests:            cap 10 — excess items dropped oldest-first, prioritising items
                                   not referenced in last 3 sessions
  - unresolved_questions: cap 10 — auto-expire items not referenced in 3+ sessions
  - pending_actions:      cap 5  — auto-expire items not referenced in 3+ sessions

After per-field compaction: measure state tokens; if still >1500, trigger
aggressive historical_summary compression + trim longest array items.
"""

from __future__ import annotations

import json

import openai
import tiktoken
from dotenv import load_dotenv

load_dotenv(override=True)

_MODEL = "gpt-4o"
_TOKEN_BUDGET = 1500
_HISTORY_CAP = 5
_INTERESTS_CAP = 10
_QUESTIONS_CAP = 10
_ACTIONS_CAP = 5
_EXPIRY_SESSIONS = 3  # sessions without reference → stale


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    enc = tiktoken.encoding_for_model(_MODEL)
    return len(enc.encode(text))


def _state_token_count(state: dict) -> int:
    return _count_tokens(json.dumps(state, default=str))


# ---------------------------------------------------------------------------
# History compaction (oldest → historical_summary via GPT-4o)
# ---------------------------------------------------------------------------

def _summarize_entries(entries: list[dict], existing_summary: str) -> str:
    """Call GPT-4o to fold *entries* into *existing_summary*.

    Returns a concise summary string of at most 500 characters.
    """
    entry_text = "\n".join(
        f"[Session {e.get('session', '?')}] {e.get('summary', '')}"
        for e in entries
    )
    user_content = (
        "You are a concise memory summarizer. Given an existing historical summary "
        "and new session entries to fold in, produce a single updated summary string "
        "of at most 500 characters. Preserve key facts and be concise.\n\n"
        f"EXISTING SUMMARY:\n{existing_summary or '(none)'}\n\n"
        f"NEW ENTRIES TO FOLD IN:\n{entry_text}\n\n"
        "Output ONLY the updated summary string — no commentary, no quotes."
    )
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": user_content}],
        temperature=0.0,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()[:500]


def _compact_interaction_history(state: dict) -> dict:
    """Summarize overflow history entries into historical_summary and trim to cap."""
    history: list[dict] = state.get("interaction_history", [])
    if len(history) <= _HISTORY_CAP:
        return state

    # Oldest entries are at the front of the list
    overflow = history[: len(history) - _HISTORY_CAP]
    keep = history[len(history) - _HISTORY_CAP :]

    try:
        new_summary = _summarize_entries(overflow, state.get("historical_summary", ""))
        state["historical_summary"] = new_summary
    except Exception as exc:
        print(f"[Compactor] History summarization failed: {exc}")

    state["interaction_history"] = keep
    return state


# ---------------------------------------------------------------------------
# Array field expiry and cap enforcement
# ---------------------------------------------------------------------------

def _expire_array_field(
    state: dict,
    field: str,
    cap: int,
    current_session: int,
) -> dict:
    """Apply session-based expiry and cap to a mutable metadata array field.

    Items are dicts with ``{ "text": str, "added_session": int,
    "last_referenced_session": int }`` (as written by memory_updater).

    Expiry: items whose ``last_referenced_session`` is ≥ _EXPIRY_SESSIONS behind
    ``current_session`` are removed (assumed stale / resolved elsewhere).

    Cap: if still over *cap* after expiry, drop the items that were referenced
    least recently (and added earliest as a tiebreaker) until at cap.
    """
    items: list = state.get(field, [])
    if not items:
        return state

    def is_stale(item: object) -> bool:
        if not isinstance(item, dict):
            return False
        last_ref = item.get("last_referenced_session", 0)
        return (current_session - last_ref) >= _EXPIRY_SESSIONS

    fresh = [item for item in items if not is_stale(item)]

    if len(fresh) > cap:
        def sort_key(item: object) -> tuple[int, int]:
            if isinstance(item, dict):
                return (
                    item.get("last_referenced_session", 0),
                    item.get("added_session", 0),
                )
            return (0, 0)

        fresh.sort(key=sort_key)
        fresh = fresh[-cap:]  # keep the *cap* most-recently-referenced items

    state[field] = fresh
    return state


# ---------------------------------------------------------------------------
# Token budget gate — aggressive trim
# ---------------------------------------------------------------------------

def _aggressive_trim(state: dict) -> dict:
    """Last-resort trimming when state is still over the token budget.

    Strategy:
      1. Aggressively compress historical_summary to ≤200 chars via GPT-4o.
      2. Drop the single array item with the longest text, one at a time, until
         the state fits within the budget.
    """
    summary = state.get("historical_summary", "")
    if summary and _state_token_count(state) > _TOKEN_BUDGET:
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Summarize the following in at most 200 characters, "
                            f"preserving key facts:\n\n{summary}"
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=60,
            )
            state["historical_summary"] = response.choices[0].message.content.strip()[:200]
        except Exception as exc:
            print(f"[Compactor] Aggressive summary compression failed: {exc}")
            state["historical_summary"] = summary[:200]

    mutable_fields = [
        "interaction_history",
        "interests",
        "unresolved_questions",
        "pending_actions",
    ]

    while _state_token_count(state) > _TOKEN_BUDGET:
        best_field: str | None = None
        best_idx: int | None = None
        best_len = 0

        for field in mutable_fields:
            for i, item in enumerate(state.get(field, [])):
                if isinstance(item, dict):
                    text = item.get("summary", "") or item.get("text", "")
                else:
                    text = str(item)
                if len(text) > best_len:
                    best_len = len(text)
                    best_field = field
                    best_idx = i

        if best_field is None or best_idx is None:
            break  # nothing more to trim

        state[best_field].pop(best_idx)

    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compact(state: dict) -> dict:
    """Run all compaction passes on *state* and return the (possibly modified) state.

    Compaction order:
      1. Compact interaction_history — summarize overflow into historical_summary.
      2. Expire stale items from interests, unresolved_questions, pending_actions
         and enforce per-field caps.
      3. Token budget gate — aggressive trim if state is still >1500 tokens.

    This function never raises; errors in individual passes are logged and skipped
    so login is never blocked by a compaction failure.

    Args:
        state: The employee state dict (modified in place and returned).

    Returns:
        The compacted state dict.
    """
    current_session: int = state.get("session_counter", 0)

    # 1. History compaction
    try:
        state = _compact_interaction_history(state)
    except Exception as exc:
        print(f"[Compactor] History compaction error: {exc}")

    # 2. Per-field expiry + cap
    for field, cap in [
        ("interests", _INTERESTS_CAP),
        ("unresolved_questions", _QUESTIONS_CAP),
        ("pending_actions", _ACTIONS_CAP),
    ]:
        try:
            state = _expire_array_field(state, field, cap, current_session)
        except Exception as exc:
            print(f"[Compactor] Field expiry error for '{field}': {exc}")

    # 3. Token budget gate
    try:
        if _state_token_count(state) > _TOKEN_BUDGET:
            state = _aggressive_trim(state)
    except Exception as exc:
        print(f"[Compactor] Token budget enforcement error: {exc}")

    return state
