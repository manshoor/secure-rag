"""CLI for agent-mode runs and the demo: boq --mode agent, deliver, extend/drop, demo reset."""

import re

import pytest

from secure_rag import cli, config, run_index

ISSUES = config.SAMPLES_DIR / "boq_with_issues.csv"
QUOTES = config.SAMPLES_DIR / "quotes"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")
    monkeypatch.setenv("RAG_EMBEDDER", "hash")
    monkeypatch.setenv("RAG_LLM", "offline")


def _run(capsys, *argv) -> tuple[int, str]:
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def _run_id(out: str) -> str:
    return re.search(r"^run (\S+)", out, re.M).group(1)


def _status(out: str) -> str:
    return re.search(r"status: (\S+)", out).group(1)


# ---- agent runs from the command line ------------------------------------------------------


def test_boq_mode_defaults_to_classic(capsys):
    code, out = _run(capsys, "workflow", "boq", "--user", "khalid", str(ISSUES))
    assert code == 0 and _status(out) == "awaiting_procurement"


def test_agent_run_end_to_end(capsys):
    _, out = _run(capsys, "workflow", "boq", "--user", "khalid", "--mode", "agent", str(ISSUES))
    run_id = _run_id(out)
    assert _status(out) == "awaiting_rfq"

    _, out = _run(capsys, "workflow", "approve", run_id, "--approver", "omar")
    assert _status(out) == "awaiting_quotes"

    # The file name stands for the mailbox sender (the vendor id).
    _, out = _run(capsys, "workflow", "deliver", run_id, "--user", "omar", *map(str, sorted((QUOTES / "round1").glob("*.csv"))))
    assert _status(out) == "awaiting_quote_decision"

    _, out = _run(capsys, "workflow", "approve", run_id, "--approver", "omar", "--extend")
    assert _status(out) == "awaiting_quotes"

    _, out = _run(capsys, "workflow", "deliver", run_id, "--user", "omar", *map(str, sorted((QUOTES / "round2").glob("*.csv"))))
    assert _status(out) == "awaiting_procurement"
    assert "33,015.00" in out


def test_deadline_needs_no_files(capsys):
    _, out = _run(capsys, "workflow", "boq", "--user", "khalid", "--mode", "agent", str(ISSUES))
    run_id = _run_id(out)
    _run(capsys, "workflow", "approve", run_id, "--approver", "omar")
    code, out = _run(capsys, "workflow", "deliver", run_id, "--user", "omar", "--deadline")
    assert code == 0 and _status(out) == "awaiting_quote_decision"


def test_deliver_refuses_the_requester(capsys):
    _, out = _run(capsys, "workflow", "boq", "--user", "khalid", "--mode", "agent", str(ISSUES))
    run_id = _run_id(out)
    _run(capsys, "workflow", "approve", run_id, "--approver", "omar")
    code, out = _run(capsys, "workflow", "deliver", run_id, "--user", "khalid", "--deadline")
    assert code == 1 and out.startswith("error:")


def test_deliver_refuses_a_missing_file_and_keeps_the_round(capsys, tmp_path):
    _, out = _run(capsys, "workflow", "boq", "--user", "khalid", "--mode", "agent", str(ISSUES))
    run_id = _run_id(out)
    _run(capsys, "workflow", "approve", run_id, "--approver", "omar")
    code, out = _run(capsys, "workflow", "deliver", run_id, "--user", "omar", str(tmp_path / "V-GLB-1.csv"))
    assert code == 1 and "V-GLB-1.csv" in out
    _, out = _run(capsys, "workflow", "show", run_id, "--user", "omar")
    assert _status(out) == "awaiting_quotes"


def test_deliver_needs_files_or_deadline(capsys):
    with pytest.raises(SystemExit):
        cli.main(["workflow", "deliver", "abc", "--user", "omar"])


# ---- demo reset -----------------------------------------------------------------------------


def _runtime_files(data_dir):
    """The files a demo run leaves behind, plus files reset must keep."""
    for name in ("runs.json", "runs.lock", "budget_ledger.json", "budget_ledger.lock", "outbox.jsonl",
                 "checkpoints.sqlite", "checkpoints.sqlite-wal", "checkpoints.sqlite-shm", "audit.jsonl", "notes.txt"):
        (data_dir / name).write_text("x", encoding="utf-8")
    (data_dir / "uploads").mkdir()


def test_demo_reset_lists_and_keeps_without_yes(capsys, monkeypatch, temp_data_dir):
    monkeypatch.setenv("RAG_DEMO", "on")
    _runtime_files(temp_data_dir)
    code, out = _run(capsys, "demo", "reset")
    assert code == 0
    assert "runs.json" in out and "--yes" in out
    assert (temp_data_dir / "runs.json").exists()


def test_demo_reset_deletes_only_demo_state(capsys, monkeypatch, temp_data_dir):
    monkeypatch.setenv("RAG_DEMO", "on")
    _runtime_files(temp_data_dir)
    code, _ = _run(capsys, "demo", "reset", "--yes")
    assert code == 0
    left = sorted(path.name for path in temp_data_dir.iterdir())
    assert left == ["audit.jsonl", "notes.txt", "server.lock", "uploads"]  # the audit log is a record; it stays


def test_demo_reset_refuses_while_the_server_runs(capsys, monkeypatch, temp_data_dir):
    from secure_rag import demo

    monkeypatch.setenv("RAG_DEMO", "on")
    _runtime_files(temp_data_dir)
    with demo.server_running():
        code, out = _run(capsys, "demo", "reset", "--yes")
    assert code == 1 and "server" in out
    assert (temp_data_dir / "runs.json").exists()
    assert _run(capsys, "demo", "reset", "--yes")[0] == 0  # stopped: reset works


def test_demo_reset_needs_demo_mode(capsys, temp_data_dir):
    _runtime_files(temp_data_dir)
    code, out = _run(capsys, "demo", "reset", "--yes")
    assert code == 1 and "RAG_DEMO=on" in out
    assert (temp_data_dir / "runs.json").exists()


def test_demo_reset_refuses_postgres(capsys, monkeypatch, temp_data_dir):
    monkeypatch.setenv("RAG_DEMO", "on")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/db")
    _runtime_files(temp_data_dir)
    code, out = _run(capsys, "demo", "reset", "--yes")
    assert code == 1 and "Postgres" in out
    assert (temp_data_dir / "runs.json").exists()


def test_reset_list_matches_the_files_modules_write():
    """The reset list names the same files the owning modules write."""
    from secure_rag import agent, demo, systems, workflow

    owned = {run_index._INDEX_FILE, run_index._INDEX_LOCK, systems._LEDGER_FILE, systems._LEDGER_LOCK, workflow._OUTBOX_FILE, agent._CHECKPOINT_FILE}
    assert owned <= set(demo.RESET_FILES)
