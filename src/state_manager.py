"""
state_manager.py — Load and save employee state.

Priority:
  1. Supabase Postgres — when SUPABASE_URL + SUPABASE_KEY are set in the environment.
  2. Local JSON files in employees/ — fallback / dev mode.

Each employee has a row in Supabase's ``employee_states`` table (columns:
``employee_id`` TEXT PK, ``name`` TEXT, ``state`` JSONB, ``updated_at`` TIMESTAMPTZ).
Local JSON files are always written as a backup mirror so the app works
without Supabase during local development.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EMPLOYEES_DIR = Path(__file__).parent.parent / "employees"

# Supabase table name
_TABLE = "employee_states"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name_to_filename(name: str) -> str:
    """Convert a display name like 'Maria Chen' to 'maria_chen.json'."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{slug}.json"


def _filepath(name: str) -> Path:
    return _EMPLOYEES_DIR / _name_to_filename(name)


# ---------------------------------------------------------------------------
# Supabase client — lazy-initialised, only when env vars are present
# ---------------------------------------------------------------------------

_supabase_client = None
_supabase_initialised = False


def _get_supabase():
    """Return a Supabase client if SUPABASE_URL and SUPABASE_KEY are set, else None."""
    global _supabase_client, _supabase_initialised
    if _supabase_initialised:
        return _supabase_client
    _supabase_initialised = True

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        return None

    try:
        from supabase import create_client  # type: ignore
        _supabase_client = create_client(url, key)
        print("[StateManager] Supabase client initialised.")
    except Exception as exc:
        print(f"[StateManager] Failed to initialise Supabase client: {exc}")
        _supabase_client = None

    return _supabase_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_employee_names() -> list[str]:
    """Return a sorted list of employee display names (from local JSON files)."""
    names: list[str] = []
    for path in sorted(_EMPLOYEES_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            names.append(data["name"])
        except (json.JSONDecodeError, KeyError):
            pass  # skip malformed files
    return names


def load(employee_name: str) -> dict:
    """Load and return the employee state dict for *employee_name*.

    Tries Supabase first (if configured), seeding from the local JSON file
    if no row exists yet.  Falls back entirely to local JSON when Supabase is
    not configured.

    Raises:
        FileNotFoundError: if the employee cannot be found anywhere.
    """
    client = _get_supabase()
    if client is not None:
        state = _load_from_supabase(client, employee_name)
        if state is not None:
            return state
        # Supabase is configured but this employee has no row yet — seed it.
        print(f"[StateManager] No Supabase row for '{employee_name}'; seeding from local file.")
        state = _load_from_file(employee_name)
        _save_to_supabase(client, state)
        return state

    return _load_from_file(employee_name)


def save(state: dict) -> None:
    """Persist *state*.

    Writes to Supabase when configured; always mirrors to local JSON as a
    dev-friendly backup.

    Raises:
        KeyError: if *state* does not contain a ``"name"`` key.
    """
    client = _get_supabase()
    if client is not None:
        _save_to_supabase(client, state)
    _save_to_file(state)


# ---------------------------------------------------------------------------
# Local JSON helpers
# ---------------------------------------------------------------------------

def _load_from_file(employee_name: str) -> dict:
    path = _filepath(employee_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No state file found for employee '{employee_name}' "
            f"(expected: {path})"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_to_file(state: dict) -> None:
    name = state["name"]
    path = _filepath(name)
    _EMPLOYEES_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def _load_from_supabase(client, employee_name: str) -> Optional[dict]:
    """Fetch employee state JSONB from Supabase by display name. Returns None if missing."""
    try:
        response = (
            client.table(_TABLE)
            .select("state")
            .eq("name", employee_name)
            .limit(1)
            .execute()
        )
        rows = response.data
        if rows:
            return rows[0]["state"]
        return None
    except Exception as exc:
        print(f"[StateManager] Supabase load error for '{employee_name}': {exc}")
        return None


def _save_to_supabase(client, state: dict) -> None:
    """Upsert employee state to Supabase (keyed on employee_id)."""
    employee_id = state.get("employee_id", state["name"])
    try:
        client.table(_TABLE).upsert(
            {
                "employee_id": employee_id,
                "name": state["name"],
                "state": state,
            },
            on_conflict="employee_id",
        ).execute()
    except Exception as exc:
        print(f"[StateManager] Supabase save error for '{state['name']}': {exc}")
