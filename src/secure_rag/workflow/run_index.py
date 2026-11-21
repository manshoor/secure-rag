"""Run index: one row per workflow run, for the "waiting on me" and "my cases" lists.

A JSON file in RAG_DATA_DIR under a file lock (the budget ledger's pattern),
so the CLI and the API can both write it. Rows hold ids, statuses, roles and
amounts, never text. A row can lag after a crash; the run's checkpoint is the
source of truth, and every row is checked against it before anyone sees it.
"""

import fcntl
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from ..core import config

_INDEX_FILE = "runs.json"
_INDEX_LOCK = "runs.lock"
_GUARD = threading.Lock()  # flock is per process; this covers threads in one process


def record(row: dict) -> None:
    """Add or replace the row for row['run_id']."""
    with _index() as index:
        index[row["run_id"]] = row


def rows() -> list[dict]:
    """Every row, most recently updated first."""
    with _index() as index:
        return sorted(index.values(), key=lambda row: row.get("updated", ""), reverse=True)


def _read(path) -> dict:
    """The index, or empty if missing or damaged: it is a cache of the runs, rebuilt as runs change."""
    if not path.exists():
        return {}
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return index if isinstance(index, dict) else {}


@contextmanager
def _index() -> Iterator[dict]:
    """The index under an exclusive lock; changes are written back atomically."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / _INDEX_FILE
    with _GUARD, (config.DATA_DIR / _INDEX_LOCK).open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            index = _read(path)
            before = json.dumps(index, sort_keys=True)
            yield index
            if json.dumps(index, sort_keys=True) == before:
                return

            # Temp file, then rename: a crash never leaves half an index.
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temp, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
