"""Fixes from the security and money review: each test pins one way a wrong total or a leak got through."""

import io
import zipfile
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag import config, identity
from secure_rag.agent import Decision
from secure_rag.api import MAX_UPLOAD_BYTES, create_app
from secure_rag.boq_parser import is_private
from secure_rag.identity import User, get_user
from secure_rag.intake import ClaudeSummarizer, OfflineIntake, TemplateSummarizer
from secure_rag.llm import Usage
from secure_rag.workflow import Gate, WfStatus, WorkflowRunner

REQUEST = "3 site engineers join on 1 Dec. Order laptops and safety kits."
ISSUES = config.SAMPLES_DIR / "boq_with_issues.csv"
HEADER = "Item,Description,Unit,Qty\n"


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


@pytest.fixture
def wf(system) -> WorkflowRunner:
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())


def _csv(tmp_path: Path, rows: str, name: str = "boq.csv") -> Path:
    path = tmp_path / name
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


class FakeCanonicalizer:
    def __init__(self, rewrites: list[str]) -> None:
        self.rewrites = rewrites
        self.calls = 0

    def suggest(self, descriptions: list[str]) -> tuple[list[str], Usage]:
        self.calls += 1
        return self.rewrites, Usage()


class FakeClient:
    def __init__(self, reply: dict) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete_json(self, system: str, prompt: str, schema: dict) -> tuple[dict, Usage]:
        self.prompts.append(prompt)
        return self.reply, Usage()


class BrokenSummarizer:
    def summarize(self, facts: dict):
        raise RuntimeError("model unavailable")


# ---- 1: a model rewrite never prices a line --------------------------------------------


def test_model_suggestion_is_a_hint_not_a_price(system):
    canon = FakeCanonicalizer(["Solid concrete block wall 100mm thick", "Precast concrete lintel over openings"])
    wf = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver(), canonicalizer=canon)
    run = wf.start_boq(get_user("khalid"), ISSUES)
    assert canon.calls == 1
    assert run.total == Decimal("21240.00")
    rfq = {item["item_no"]: item for item in run.rfq}
    assert sorted(rfq) == ["1.4", "2.2"]
    assert rfq["1.4"]["suggested_code"] == "BLK-100-SOL"


def test_model_suggestions_with_wrong_count_are_dropped(system):
    canon = FakeCanonicalizer(["Precast concrete lintel over openings"])  # two weak lines, one rewrite
    wf = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver(), canonicalizer=canon)
    run = wf.start_boq(get_user("khalid"), ISSUES)
    assert all("suggested_code" not in item for item in run.rfq)


# ---- 2: the matcher does not price a different item ----------------------------------------


@pytest.mark.parametrize(
    "description",
    [
        "Solid concrete block wall 200mm thick",  # solid, but only hollow comes in 200mm
        "External cement plaster 15mm thick",  # external, but 15mm is the internal plaster
        "Internal cement plaster 15mm thick to hollow concrete block wall 200mm thick",  # two items tie
        "Demolish and cart away hollow concrete block wall 200mm thick",  # removal, not supply
        "Hollow concrete block wall 200mm thick - EXCLUDED, by others",
    ],
)
def test_ambiguous_or_wrong_lines_are_not_priced(wf, tmp_path, description):
    path = _csv(tmp_path, f'1,"{description}",m2,100\n2,Precast concrete lintel over openings,lm,10\n')
    run = wf.start_boq(get_user("khalid"), path)
    assert [line["item_no"] for line in run.lines] == ["2"]
    assert "1" in {item["item_no"] for item in run.review}


# ---- 3: one hire is one equipment set ------------------------------------------------------


def test_named_hire_inside_the_count_is_not_counted_twice(wf):
    run = wf.start_request(get_user("sara"), "3 site engineers join on 1 Dec. Order laptops and safety kits for HR-1041.")
    assert len(run.hires) == 3 and run.total == Decimal("18150.00")


# ---- 5: data/private is private whatever the letter case -------------------------------------


def test_private_check_ignores_letter_case(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    (private / "real.csv").write_text("x")
    if not (tmp_path / "PRIVATE" / "real.csv").exists():
        pytest.skip("case-sensitive file system")
    monkeypatch.setattr(config, "PRIVATE_DIR", private)
    assert is_private(tmp_path / "PRIVATE" / "real.csv")


# ---- 6: a decision names the gate it is for ------------------------------------------------


def test_decision_for_another_gate_is_refused(wf):
    run = wf.start_request(get_user("sara"), REQUEST)
    with pytest.raises(ValueError):
        wf.decide(run.run_id, get_user("omar"), Decision.APPROVE, Gate.FINANCE)
    assert wf.decide(run.run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT).status is WfStatus.AWAITING_FINANCE


# ---- 7: the company is the run's, not the requester's current one ------------------------------


def test_requester_moving_company_does_not_open_the_gate(wf, monkeypatch):
    run = wf.start_request(get_user("sara"), REQUEST)
    wf.decide(run.run_id, get_user("omar"), Decision.APPROVE)
    moved = User("sara", "Sara", frozenset({"beta-properties"}), frozenset({"employee", "hr"}))
    monkeypatch.setitem(identity._USERS, "sara", moved)
    assert not wf.may_view(run.run_id, get_user("lina"))
    assert wf.decide(run.run_id, get_user("lina"), Decision.APPROVE).status is WfStatus.AWAITING_FINANCE


# ---- 8 and 15: notes come from facts; a failed or wrong note falls back to the template ----------


def test_failed_summarizer_falls_back_to_template(system):
    wf = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver(), BrokenSummarizer())
    run = wf.start_request(get_user("sara"), REQUEST)
    after = wf.decide(run.run_id, get_user("omar"), Decision.APPROVE)
    assert after.status is WfStatus.AWAITING_FINANCE and "18,150.00" in after.summaries["finance"]


def test_note_with_numbers_not_in_the_facts_is_replaced(system):
    client = FakeClient({"line1": "Total AED 1.", "line2": "No approval needed."})
    wf = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver(), ClaudeSummarizer(client))
    run = wf.start_request(get_user("sara"), REQUEST)
    assert "18,150.00" in run.summaries["procurement"] and "AED 1." not in run.summaries["procurement"]


# ---- 9 and 11: unreadable rows and files stop or queue, never vanish or crash ---------------------


@pytest.mark.parametrize("quantity", ["120 m2", "TBC", "1.200,50", "1e30", "1 200"])
def test_unreadable_quantity_goes_to_review(wf, tmp_path, quantity):
    path = _csv(tmp_path, f'1,Stainless steel wall ties,nos,"{quantity}"\n2,Precast concrete lintel over openings,lm,10\n')
    run = wf.start_boq(get_user("khalid"), path)
    assert [line["item_no"] for line in run.lines] == ["2"]
    assert "1" in {item["item_no"] for item in run.review}


def test_zip_named_xlsx_stops_the_run(wf, tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("note.txt", "not a workbook")
    path = tmp_path / "fake.xlsx"
    path.write_bytes(buffer.getvalue())
    run = wf.start_boq(get_user("khalid"), path)
    assert run.status is WfStatus.STOPPED and run.stopped_at == "boq_parse"


@pytest.mark.parametrize(
    "text",
    ["3 site engineers join on 2026-02-30. Order laptops.", "3 site engineers join. Order laptops for HR-1041."],
)
def test_bad_dates_stop_at_understand(wf, text):
    run = wf.start_request(get_user("sara"), text)
    assert run.status is WfStatus.STOPPED and run.stopped_at == "understand"


# ---- 10: nothing priced, nothing to approve ---------------------------------------------------


def test_boq_with_no_priced_line_stops(wf, tmp_path):
    path = _csv(tmp_path, "1,Decorative GRC cornice,lm,45\n")
    run = wf.start_boq(get_user("khalid"), path)
    assert run.status is WfStatus.STOPPED and run.stopped_at == "boq_price" and run.pending_gate is None


# ---- 12: private runs hide section names and never reach the model ------------------------------


def test_private_run_hides_sections_and_skips_the_model(system, tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    path = private / "real.csv"
    path.write_bytes(ISSUES.read_bytes())
    monkeypatch.setattr(config, "PRIVATE_DIR", private)
    client = FakeClient({"line1": "x", "line2": "y"})
    wf = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver(), ClaudeSummarizer(client))
    run = wf.start_boq(get_user("khalid"), path)
    assert client.prompts == []
    with TestClient(create_app(system, None, wf)) as client_api:
        body = client_api.get(f"/workflows/{run.run_id}", headers={"X-User": "khalid"}).json()
    assert "sections" not in body["boq_stats"] and body["lines"] == []


# ---- 14: an oversized upload is refused before the form is read --------------------------------


def test_oversized_upload_refused_even_for_unknown_user(system, wf):
    with TestClient(create_app(system, None, wf)) as client:
        files = {"file": ("big.csv", b"a" * (2 * MAX_UPLOAD_BYTES), "text/csv")}
        assert client.post("/workflows/boq", files=files, headers={"X-User": "mallory"}).status_code == 413


def test_template_summarizer_is_the_offline_default():
    assert TemplateSummarizer().summarize({"what": "x", "company": "c", "total": "1.00", "vendors": "v", "budget_after": "0.00", "needs": "n"})
