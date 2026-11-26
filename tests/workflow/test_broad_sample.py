"""The broad fit-out sample: catalogue lines, then lines in all five quotation trades."""

import importlib.util

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import config
from secure_rag.core.identity import get_user
from secure_rag.workflow import workflow
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.workflow import RunMode, WfStatus, WorkflowRunner

BROAD_CSV = config.SAMPLES_DIR / "boq_fitout_broad.csv"
_SCRIPT = config.PROJECT_ROOT / "scripts" / "make_samples.py"
_PACKAGES = {
    "glass_block": ["1.4"],
    "grc": ["1.5"],
    "joinery": ["2.1", "2.2", "2.3"],
    "aluminium_glazing": ["3.1", "3.2"],
    "stone_cladding": ["4.1", "4.2", "4.3"],
}


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


def _script():
    spec = importlib.util.spec_from_file_location("make_samples", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_file_is_what_the_script_writes(tmp_path):
    script = _script()
    script._csv(script.BROAD, tmp_path / "broad.csv")
    assert (tmp_path / "broad.csv").read_bytes() == BROAD_CSV.read_bytes()


def test_the_broad_boq_has_five_packages_and_three_priced_lines(system):
    wf = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    run = wf.start_boq(get_user("khalid"), BROAD_CSV, mode=RunMode.AGENT)
    values = wf._graph.get_state(workflow._thread(run.run_id)).values
    assert run.status is WfStatus.AWAITING_RFQ
    assert {package["id"]: package["items"] for package in values["packages"]} == _PACKAGES
    assert [(line["item_no"], line["code"]) for line in values["lines"]] == [("1.1", "BLK-200-HOL"), ("1.2", "PLS-INT-15"), ("1.3", "PNT-EMU-2C")]
    assert values["flags"] == [] and len(values["rfq"]) == sum(len(items) for items in _PACKAGES.values())
