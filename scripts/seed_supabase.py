#!/usr/bin/env python3
"""
scripts/seed_supabase.py — Seed Supabase with initial employee state from local JSON files.

Usage:
    SUPABASE_URL=https://xxx.supabase.co SUPABASE_KEY=your-key python scripts/seed_supabase.py

Run this once after creating the Supabase project and table.  Re-running is safe:
the upsert will overwrite existing rows, restoring state to the repo defaults.

Required Supabase table (run in the SQL editor):

    CREATE TABLE employee_states (
        employee_id TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        state       JSONB NOT NULL,
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE OR REPLACE FUNCTION _update_updated_at()
    RETURNS TRIGGER LANGUAGE plpgsql AS $$
    BEGIN
        NEW.updated_at = now();
        RETURN NEW;
    END;
    $$;

    CREATE TRIGGER trg_employee_states_updated_at
    BEFORE UPDATE ON employee_states
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_EMPLOYEES_DIR = _ROOT / "employees"


def main() -> None:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()

    if not url or not key:
        print(
            "ERROR: SUPABASE_URL and SUPABASE_KEY must be set as environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from supabase import create_client  # type: ignore
    except ImportError:
        print(
            "ERROR: supabase package not installed.  Run: pip install supabase",
            file=sys.stderr,
        )
        sys.exit(1)

    client = create_client(url, key)

    employee_files = sorted(_EMPLOYEES_DIR.glob("*.json"))
    if not employee_files:
        print(f"No employee JSON files found in {_EMPLOYEES_DIR}", file=sys.stderr)
        sys.exit(1)

    seeded = 0
    for path in employee_files:
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        employee_id = state.get("employee_id")
        name = state.get("name")

        if not employee_id or not name:
            print(f"  SKIP {path.name}: missing employee_id or name")
            continue

        try:
            client.table("employee_states").upsert(
                {"employee_id": employee_id, "name": name, "state": state},
                on_conflict="employee_id",
            ).execute()
            print(f"  OK   {name} ({employee_id})")
            seeded += 1
        except Exception as exc:
            print(f"  FAIL {name}: {exc}", file=sys.stderr)

    print(f"\nSeeded {seeded}/{len(employee_files)} employees.")


if __name__ == "__main__":
    main()
