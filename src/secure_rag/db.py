"""Postgres driver: the only module that writes SQL.

    identity.py ──► users, document access   (who may read what, read on every request)
    audit.py    ──► audit_log                (one row per question and agent step)
    agent.py    ──► LangGraph checkpoints    (tables created by PostgresSaver.setup())
    evaluate.py ──► eval_runs                (one row per golden-set run)

Active only when DATABASE_URL is set. Callers check enabled() first.
"""

import atexit
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from . import config

# Settings PostgresSaver needs (it manages its own transactions); the rest of
# this module writes single statements, so autocommit suits it too.
_CONNECTION_KWARGS = {"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}
_POOL_MAX_SIZE = 10

# Advisory lock key for schema changes, so an API and a CLI starting together
# on a fresh database do not race on CREATE TABLE or checkpoint migrations.
_SCHEMA_LOCK_KEY = 7_310_442

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id   text PRIMARY KEY,
    name      text NOT NULL,
    companies text[] NOT NULL DEFAULT '{}',
    roles     text[] NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id  text PRIMARY KEY,
    title   text NOT NULL,
    company text NOT NULL,
    roles   text[] NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS audit_log (
    id      bigserial PRIMARY KEY,
    ts      timestamptz NOT NULL DEFAULT now(),
    event   text NOT NULL,
    user_id text,
    data    jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_log_user_ts ON audit_log (user_id, ts);
CREATE TABLE IF NOT EXISTS eval_runs (
    id     bigserial PRIMARY KEY,
    ts     timestamptz NOT NULL DEFAULT now(),
    passed integer NOT NULL,
    total  integer NOT NULL,
    leaks  integer NOT NULL,
    report jsonb NOT NULL
);
"""

_pools: dict[str, ConnectionPool] = {}


class Overwrite(str, Enum):
    """What seeding does with rows that already exist."""

    KEEP = "keep"  # insert missing rows only; live changes survive restarts
    RESTORE = "restore"  # reset existing rows to the seed values


@dataclass(frozen=True)
class UserRow:
    user_id: str
    name: str
    companies: list[str]
    roles: list[str]


@dataclass(frozen=True)
class DocumentRow:
    doc_id: str
    title: str
    company: str
    roles: list[str]


def enabled() -> bool:
    """True when DATABASE_URL is set."""
    return bool(config.database_url())


def pool() -> ConnectionPool:
    """Shared connection pool for the current DATABASE_URL (thread-safe)."""
    url = config.database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    if url not in _pools:
        _pools[url] = ConnectionPool(url, kwargs=_CONNECTION_KWARGS, max_size=_POOL_MAX_SIZE, open=True)
    return _pools[url]


@atexit.register
def _close_pools() -> None:
    """Close connections cleanly when the process exits."""
    for open_pool in _pools.values():
        open_pool.close()
    _pools.clear()


@contextmanager
def _advisory_lock(key_sql: str, key: object) -> Iterator[None]:
    """Hold a session advisory lock on its own connection, outside the pool.

    The locked work uses the pool. Taking the lock from the pool too would
    let lock holders use up every connection and stall each other.
    Closing the connection releases the lock even if the unlock is never reached.
    """
    with psycopg.connect(config.database_url(), autocommit=True) as conn:
        conn.execute(f"SELECT pg_advisory_lock({key_sql})", (key,))
        try:
            yield
        finally:
            conn.execute(f"SELECT pg_advisory_unlock({key_sql})", (key,))


@contextmanager
def schema_lock() -> Iterator[None]:
    """Hold a Postgres advisory lock while creating or migrating tables."""
    with _advisory_lock("%s", _SCHEMA_LOCK_KEY):
        yield


@contextmanager
def run_lock(run_id: str) -> Iterator[None]:
    """Postgres advisory lock for one agent run, so two processes cannot decide it at once.

    No-op when Postgres is off (the runner's in-process lock still applies).
    """
    if not enabled():
        yield
        return
    with _advisory_lock("hashtextextended(%s, 0)", run_id):
        yield


def create_schema() -> None:
    """Create the app's tables if missing. Never drops or changes data."""
    # One statement per call: prepared statements (prepare_threshold=0) reject scripts.
    statements = [part.strip() for part in _SCHEMA.split(";") if part.strip()]
    with schema_lock(), pool().connection() as conn:
        for statement in statements:
            conn.execute(statement)


# ---- users ------------------------------------------------------------------


def fetch_user(user_id: str) -> UserRow | None:
    with pool().connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = %s", (user_id,)).fetchone()
    return UserRow(**row) if row else None


def fetch_users() -> list[UserRow]:
    with pool().connection() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY user_id").fetchall()
    return [UserRow(**row) for row in rows]


def insert_users(rows: list[UserRow], overwrite: Overwrite) -> None:
    """Add users. KEEP leaves existing rows alone (safe on every start); RESTORE resets them."""
    conflict = "DO NOTHING" if overwrite is Overwrite.KEEP else (
        "DO UPDATE SET name = EXCLUDED.name, companies = EXCLUDED.companies, roles = EXCLUDED.roles"
    )
    with pool().connection() as conn:
        for row in rows:
            conn.execute(
                f"INSERT INTO users (user_id, name, companies, roles) VALUES (%s, %s, %s, %s) "
                f"ON CONFLICT (user_id) {conflict}",
                (row.user_id, row.name, row.companies, row.roles),
            )


def add_user_role(user_id: str, role: str) -> bool:
    """Grant a role. False when the user does not exist."""
    with pool().connection() as conn:
        result = conn.execute(
            "UPDATE users SET roles = array_append(roles, %s) "
            "WHERE user_id = %s AND NOT (%s = ANY(roles))",
            (role, user_id, role),
        )
        if result.rowcount:
            return True
        return conn.execute("SELECT 1 FROM users WHERE user_id = %s", (user_id,)).fetchone() is not None


def remove_user_role(user_id: str, role: str) -> bool:
    """Revoke a role. False when the user does not exist."""
    with pool().connection() as conn:
        result = conn.execute(
            "UPDATE users SET roles = array_remove(roles, %s) WHERE user_id = %s", (role, user_id)
        )
    return result.rowcount > 0


# ---- document access --------------------------------------------------------


def insert_documents(rows: list[DocumentRow], overwrite: Overwrite) -> None:
    """Register documents. KEEP leaves existing access alone; RESTORE resets it to the given tags."""
    conflict = "DO NOTHING" if overwrite is Overwrite.KEEP else (
        "DO UPDATE SET title = EXCLUDED.title, company = EXCLUDED.company, roles = EXCLUDED.roles"
    )
    with pool().connection() as conn:
        for row in rows:
            conn.execute(
                f"INSERT INTO documents (doc_id, title, company, roles) VALUES (%s, %s, %s, %s) "
                f"ON CONFLICT (doc_id) {conflict}",
                (row.doc_id, row.title, row.company, row.roles),
            )


def fetch_documents(doc_ids: list[str]) -> list[DocumentRow]:
    """Current access rows for these documents (missing ids are left out)."""
    if not doc_ids:
        return []
    with pool().connection() as conn:
        rows = conn.execute("SELECT * FROM documents WHERE doc_id = ANY(%s)", (doc_ids,)).fetchall()
    return [DocumentRow(**row) for row in rows]


# ---- audit and eval ---------------------------------------------------------


def insert_audit(event: str, user_id: str | None, data: dict) -> None:
    with pool().connection() as conn:
        conn.execute(
            "INSERT INTO audit_log (event, user_id, data) VALUES (%s, %s, %s)",
            (event, user_id, Jsonb(data, dumps=_dumps)),
        )


def insert_eval_run(passed: int, total: int, leaks: int, report: dict) -> None:
    with pool().connection() as conn:
        conn.execute(
            "INSERT INTO eval_runs (passed, total, leaks, report) VALUES (%s, %s, %s, %s)",
            (passed, total, leaks, Jsonb(report, dumps=_dumps)),
        )


def _dumps(value: object) -> str:
    """JSON that tolerates sets, enums and dataclass values by falling back to str()."""
    return json.dumps(value, ensure_ascii=False, default=str)
