"""Demo reset: clear the runs a demo leaves behind, so the next rehearsal starts empty.

Deletes only these named files in RAG_DATA_DIR, never a folder:

    runs.json, runs.lock                  run list            (run_index.py)
    budget_ledger.json, budget_ledger.lock budget holds        (systems.py)
    outbox.jsonl                          quote requests, POs (workflow.py)
    checkpoints.sqlite (+ -wal, -shm)     saved runs          (agent.py)

The audit log stays: it is a record, not demo state. Runs only with RAG_DEMO=on,
and not with Postgres, whose checkpoints and data this does not touch.
Refused while a server runs on the same folder: it holds the SQLite file open.
"""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from . import config, db

RESET_FILES = (
    "runs.json", "runs.lock",
    "budget_ledger.json", "budget_ledger.lock",
    "outbox.jsonl",
    "checkpoints.sqlite", "checkpoints.sqlite-wal", "checkpoints.sqlite-shm",
)


class ResetRefused(Exception):
    """Reset is not allowed in this setup; the message says why."""


def reset_targets() -> list[Path]:
    """The files reset would delete now: regular files only, never a link or folder."""
    _check_allowed()
    found = [config.DATA_DIR / name for name in RESET_FILES]
    return [path for path in found if path.is_file() and not path.is_symlink()]


def reset() -> list[Path]:
    """Delete the demo state, with no server running. Returns the files deleted."""
    targets = reset_targets()
    with _no_server():
        for path in targets:
            path.unlink(missing_ok=True)
    return targets


def _check_allowed() -> None:
    if config.demo_mode() is not config.DemoMode.ON:
        raise ResetRefused("demo reset runs only with RAG_DEMO=on")
    if db.enabled():
        raise ResetRefused("demo reset does not clear Postgres; unset DATABASE_URL or reset the database yourself")


# ---- the server lock --------------------------------------------------------------------------
#
#   API process ── shared lock on server.lock for its whole life
#   rag demo reset ── exclusive lock, without waiting: refused while any server holds it

_SERVER_LOCK = "server.lock"


@contextmanager
def server_running() -> Iterator[None]:
    """Mark a running server for as long as the block runs."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (config.DATA_DIR / _SERVER_LOCK).open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def _no_server() -> Iterator[None]:
    """Hold the exclusive lock while resetting; refuse if a server holds the shared one."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (config.DATA_DIR / _SERVER_LOCK).open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ResetRefused("a server is running on this data folder; stop it first") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
