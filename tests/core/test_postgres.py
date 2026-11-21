"""Integration tests against a real Postgres and Qdrant (docker compose).

Skipped unless RAG_TEST_DATABASE_URL (and, for Qdrant, RAG_TEST_QDRANT_URL) is set:

    RAG_TEST_DATABASE_URL=postgresql://rag:rag@127.0.0.1:5433/rag_test \
    RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -q

These truncate tables, so point them at rag_test, never the main database.
"""

import os

import pytest
from qdrant_client import QdrantClient

from secure_rag.core import db
from secure_rag.workflow.agent import AgentRunner, Decision, KeywordPlanner, Status, open_checkpointer
from secure_rag.rag.answer import ask
from secure_rag.rag.embed import HashEmbedder
from secure_rag.rag.evaluate import EvalReport, save_run
from secure_rag.core.identity import DocAccess, RoleChange, get_user, seed_directory, set_role
from secure_rag.rag.store import KnowledgeStore

_TEST_DB = os.environ.get("RAG_TEST_DATABASE_URL", "")
_TEST_QDRANT = os.environ.get("RAG_TEST_QDRANT_URL", "")
_TEST_DB_SUFFIX = "_test"
_LEAVE_QUESTION = "How many leave days can I carry over?"
_LEAVE_DOC = "group-hr-annual-leave"
_BOARD_DOC = "group-board-memo"
_PURCHASE = "Raise a purchase of AED 40,000 from Falcon Steel Fabrication LLC for steel beams"

pytestmark = pytest.mark.skipif(not _TEST_DB, reason="RAG_TEST_DATABASE_URL not set")


@pytest.fixture
def pg(monkeypatch, system):
    """Point the app at the test database and reset it to the seed."""
    if not _TEST_DB.rsplit("/", 1)[-1].split("?")[0].endswith(_TEST_DB_SUFFIX):
        pytest.fail(f"RAG_TEST_DATABASE_URL must name a database ending in '{_TEST_DB_SUFFIX}'; these tests truncate tables")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB)
    db.create_schema()
    with db.pool().connection() as conn:
        conn.execute("TRUNCATE users, documents, audit_log, eval_runs")
    seed_directory(db.Overwrite.RESTORE, _seed_documents(system))
    return system


def _seed_documents(system) -> dict[str, tuple[str, DocAccess]]:
    return {d.doc_id: (d.title, d.access) for d in system.report.documents}


def _audit_count() -> int:
    with db.pool().connection() as conn:
        return conn.execute("SELECT count(*) AS n FROM audit_log").fetchone()["n"]


# ---- live access changes ------------------------------------------------------


def test_revoked_role_refuses_next_question_without_rebuild(pg):
    assert not ask(pg.store, pg.composer, get_user("sara"), _LEAVE_QUESTION).refused

    set_role("sara", "employee", RoleChange.REVOKE)
    set_role("sara", "hr", RoleChange.REVOKE)

    answer = ask(pg.store, pg.composer, get_user("sara"), _LEAVE_QUESTION)
    assert answer.refused
    assert not answer.citations


def test_narrowed_document_access_applies_at_once(pg):
    # The index still carries the old tags; the post-fusion check reads Postgres.
    db.insert_documents([db.DocumentRow(_LEAVE_DOC, "Annual leave", "group", ["executive"])], db.Overwrite.RESTORE)

    answer = ask(pg.store, pg.composer, get_user("sara"), _LEAVE_QUESTION)
    assert _LEAVE_DOC not in {c.doc_id for c in answer.citations}


def test_document_without_access_row_is_unreadable(pg):
    with db.pool().connection() as conn:
        conn.execute("DELETE FROM documents WHERE doc_id = %s", (_LEAVE_DOC,))

    hits = pg.store.search(get_user("sara"), _LEAVE_QUESTION, 4)
    assert _LEAVE_DOC not in {hit.chunk.doc_id for hit in hits}


def test_missing_board_row_keeps_canary_hidden_even_from_executive(pg):
    with db.pool().connection() as conn:
        conn.execute("DELETE FROM documents WHERE doc_id = %s", (_BOARD_DOC,))

    assert pg.store.hidden_canaries(get_user("noura")) == ["CANARY-BOARD-7731"]


def test_setup_keeps_revocation_and_reset_restores_it(pg):
    set_role("sara", "hr", RoleChange.REVOKE)

    seed_directory(db.Overwrite.KEEP, _seed_documents(pg))
    assert "hr" not in get_user("sara").roles

    seed_directory(db.Overwrite.RESTORE, _seed_documents(pg))
    assert "hr" in get_user("sara").roles


def test_widened_row_does_not_override_front_matter(pg):
    # Effective access is front matter AND Postgres: widening by SQL alone opens nothing,
    # and the board canary stays in the output check for users the front matter blocks.
    db.insert_documents([db.DocumentRow(_BOARD_DOC, "Board memo", "group", ["employee"])], db.Overwrite.RESTORE)

    hits = pg.store.search(get_user("sara"), "What does the board memo say about the merger?", 8)
    assert _BOARD_DOC not in {hit.chunk.doc_id for hit in hits}
    assert pg.store.hidden_canaries(get_user("sara")) == ["CANARY-BOARD-7731"]


def test_deleted_user_stays_deleted_after_setup(pg):
    with db.pool().connection() as conn:
        conn.execute("DELETE FROM users WHERE user_id = 'sara'")

    seed_directory(db.Overwrite.KEEP, _seed_documents(pg))
    with pytest.raises(KeyError):
        get_user("sara")


def test_unknown_user_cannot_change_roles(pg):
    with pytest.raises(KeyError):
        set_role("nobody", "hr", RoleChange.GRANT)


# ---- audit, eval and checkpoints ------------------------------------------------


def test_audit_goes_to_postgres(pg, temp_data_dir):
    before = _audit_count()
    ask(pg.store, pg.composer, get_user("sara"), _LEAVE_QUESTION)

    assert _audit_count() == before + 1
    assert not (temp_data_dir / "audit.jsonl").exists()


def test_eval_run_is_saved(pg):
    assert save_run(EvalReport())
    with db.pool().connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM eval_runs").fetchone()["n"] == 1


def test_paused_run_survives_a_new_runner(pg):
    first = AgentRunner(pg.store, pg.composer, KeywordPlanner(), open_checkpointer())
    paused = first.start(get_user("omar"), _PURCHASE)
    assert paused.status is Status.AWAITING_APPROVAL

    # A fresh runner stands in for a restarted API process.
    second = AgentRunner(pg.store, pg.composer, KeywordPlanner(), open_checkpointer())
    assert second.resume(paused.thread_id, get_user("faisal"), Decision.APPROVE).status is Status.DONE


def test_approver_who_lost_role_cannot_approve(pg):
    runner = AgentRunner(pg.store, pg.composer, KeywordPlanner(), open_checkpointer())
    paused = runner.start(get_user("omar"), _PURCHASE)
    set_role("faisal", "finance", RoleChange.REVOKE)

    still = runner.resume(paused.thread_id, get_user("faisal"), Decision.APPROVE)
    assert still.status is Status.AWAITING_APPROVAL


def test_requester_who_lost_role_while_paused_is_not_executed(pg, temp_data_dir):
    runner = AgentRunner(pg.store, pg.composer, KeywordPlanner(), open_checkpointer())
    paused = runner.start(get_user("omar"), _PURCHASE)
    set_role("omar", "procurement", RoleChange.REVOKE)

    outcome = runner.resume(paused.thread_id, get_user("faisal"), Decision.APPROVE)
    assert outcome.status is Status.DENIED
    assert not (temp_data_dir / "outbox.jsonl").exists()


# ---- Qdrant server ---------------------------------------------------------------


class _CountingEmbedder(HashEmbedder):
    """Counts embedded texts, to prove a restart reuses stored vectors."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return super().embed(texts)


@pytest.mark.skipif(not _TEST_QDRANT, reason="RAG_TEST_QDRANT_URL not set")
def test_qdrant_server_reuses_collection_on_restart(monkeypatch, system):
    monkeypatch.setenv("QDRANT_URL", _TEST_QDRANT)
    chunks, documents = system.report.chunks, system.report.documents

    KnowledgeStore(_CountingEmbedder(), chunks, documents)  # builds or reuses
    restarted = _CountingEmbedder()
    store = KnowledgeStore(restarted, chunks, documents)

    # Only the query below is embedded; no chunk is re-embedded.
    hits = store.search(get_user("sara"), _LEAVE_QUESTION, 4)
    assert restarted.calls == 1
    assert hits and hits[0].chunk.doc_id == _LEAVE_DOC

    # The test embedder's collection is its own; remove it from the shared server.
    QdrantClient(url=_TEST_QDRANT).delete_collection(store._collection)


def test_locks_do_not_take_pool_connections(pg):
    # A lock held from the pool, while the locked work needs the pool too,
    # stalls once every connection is a lock holder.
    from contextlib import ExitStack

    with ExitStack() as held:
        for _ in range(db._POOL_MAX_SIZE - 1):
            held.enter_context(db.pool().connection())
        for lock in (db.run_lock("run-1"), db.schema_lock()):
            with lock, db.pool().connection(timeout=2) as conn:
                assert conn.execute("SELECT 1 AS one").fetchone()["one"] == 1
