"""Run mode and the keyed outbox: the groundwork for agent-mode BOQ runs.

Classic runs and runs saved before these fields existed must behave as before.
"""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import config
from secure_rag.core.identity import get_user
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.workflow import RunMode, WfStatus, WorkflowRunner, _thread, _write_outbox

VILLA_XLSX = config.SAMPLES_DIR / "boq_villa_blockwork.xlsx"


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


@pytest.fixture
def wf(system) -> WorkflowRunner:
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())


def _outbox() -> list[dict]:
    path = config.DATA_DIR / "outbox.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ---- outbox key ----------------------------------------------------------------


def test_outbox_keeps_one_record_per_key():
    # One quote request per vendor: different keys land, a repeated key does not.
    assert _write_outbox("r1", "rfq", {"to": "V-GLB-1"}, key="V-GLB-1:glass_block")
    assert _write_outbox("r1", "rfq", {"to": "V-GLB-2"}, key="V-GLB-2:glass_block")
    assert not _write_outbox("r1", "rfq", {"to": "V-GLB-1"}, key="V-GLB-1:glass_block")
    assert [record["key"] for record in _outbox()] == ["V-GLB-1:glass_block", "V-GLB-2:glass_block"]


def test_outbox_without_key_writes_the_old_record_shape():
    # Classic runs: no key field, so records match what older versions wrote.
    assert _write_outbox("r1", "purchase_order", {"total_aed": "1.00"})
    assert _outbox() == [{"run_id": "r1", "kind": "purchase_order", "total_aed": "1.00"}]


def test_old_record_without_key_still_blocks_a_repeat():
    # A record written before keys existed counts as key "".
    path = config.DATA_DIR / "outbox.jsonl"
    path.write_text(json.dumps({"run_id": "r1", "kind": "purchase_order"}) + "\n")
    assert not _write_outbox("r1", "purchase_order", {})
    assert _write_outbox("r1", "purchase_order", {}, key="V-GRC-2")


def test_key_does_not_leak_across_runs_or_kinds():
    assert _write_outbox("r1", "rfq", {}, key="V-1")
    assert _write_outbox("r2", "rfq", {}, key="V-1")
    assert _write_outbox("r1", "reminder", {}, key="V-1")


# ---- run mode -------------------------------------------------------------------


def test_run_mode_values():
    assert [mode.value for mode in RunMode] == ["classic", "agent"]


def test_boq_run_defaults_to_classic(wf):
    run = wf.start_boq(get_user("khalid"), VILLA_XLSX)
    assert run.mode is RunMode.CLASSIC
    assert run.status is WfStatus.AWAITING_PROCUREMENT


def test_boq_run_records_agent_mode(wf):
    run = wf.start_boq(get_user("khalid"), VILLA_XLSX, mode=RunMode.AGENT)
    assert run.mode is RunMode.AGENT


def test_request_run_is_classic(wf):
    run = wf.start_request(get_user("sara"), "3 site engineers join on 1 Dec. Order laptops and safety kits.")
    assert run.mode is RunMode.CLASSIC


def test_run_saved_without_mode_reads_as_classic(wf):
    # A checkpoint from before the mode field: same shape _start wrote, minus "mode".
    state = {
        "run_id": "legacy000001",
        "entry": "boq",
        "requester": "khalid",
        "company": "alpha-construction",
        "department": "projects",
        "boq_file": str(VILLA_XLSX),
        "privacy": "normal",
        "steps": [],
        "approvals": {},
        "flags": [],
        "refused_vendors": [],
        "summaries": {},
        "usage": {"input": 0, "output": 0, "cost_usd": 0.0},
        "status": "running",
    }
    wf._graph.invoke(state, _thread("legacy000001"))
    view = wf.view("legacy000001", get_user("khalid"))
    assert view.mode is RunMode.CLASSIC
    assert view.status is WfStatus.AWAITING_PROCUREMENT
