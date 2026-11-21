"""Append-only audit log: who asked what, which sources they saw, what was decided.

Postgres table audit_log when DATABASE_URL is set, else data/runtime/audit.jsonl.
"""

import json
import time
from pathlib import Path

from . import config, db

_AUDIT_FILE = "audit.jsonl"


def audit_path() -> Path:
    """Location of the audit log (inside the data folder; tests swap it)."""
    return config.DATA_DIR / _AUDIT_FILE


def record(event: str, **fields: object) -> None:
    """Write one entry per event."""
    if db.enabled():
        user = fields.get("user")
        db.insert_audit(event, str(user) if user else None, fields)
        return

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
    with audit_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
