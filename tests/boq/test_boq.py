"""BOQ entry: parse -> normalise units -> match to catalogue -> price in code -> review -> gates."""

import json
from decimal import Decimal

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import config
from secure_rag.workflow.agent import Decision
from secure_rag.boq.boq import normalise_unit
from secure_rag.boq.boq_parser import parse_boq
from secure_rag.core.identity import get_user
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.workflow import WfStatus, WorkflowRunner

VILLA_XLSX = config.SAMPLES_DIR / "boq_villa_blockwork.xlsx"
ISSUES_CSV = config.SAMPLES_DIR / "boq_with_issues.csv"

# Hand-computed from scripts/make_samples.py quantities and the catalogue rates:
#  A1 4,420.00  A2 1,140.00  B1 27,880.00  B2 15,370.00  B3 4,800.00  B4 1,080.00
#  B5 1,620.00  C1 24,480.00 C2 13,920.00  C3 4,350.00  D1 35,200.00 D2 31,160.00
#  D3 20,350.00 E1 5,365.00  E2 7,440.00   E3 630.00    F1 11,400.00 F2 5,320.00
VILLA_TOTAL = Decimal("215925.00")
FINISHES_TOTAL = Decimal("86710.00")  # D1 + D2 + D3


@pytest.fixture
def wf(system) -> WorkflowRunner:
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())


def _outbox() -> list[dict]:
    path = config.DATA_DIR / "outbox.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ---- parsing ---------------------------------------------------------------------------


def test_tabular_parser_reads_the_villa_boq():
    lines = parse_boq(VILLA_XLSX)
    assert len(lines) == 18
    first = lines[0]
    assert (first.item_no, first.unit, first.quantity, first.section) == ("A1", "m2", Decimal("85"), "SUBSTRUCTURE BLOCKWORK")
    assert lines[13].quantity == Decimal("18.5")


@pytest.mark.parametrize(
    ("raw", "unit"),
    [("m2", "m2"), ("sqm", "m2"), ("m²", "m2"), ("SQ.M", "m2"), ("Nos", "nos"), ("No.", "nos"), ("pcs", "nos"),
     ("KG", "kg"), ("m3", "m3"), ("cum", "m3"), ("LM", "lm"), ("Rm", "lm"), ("bags", "")],
)
def test_units_normalise(raw, unit):
    assert normalise_unit(raw) == unit


# ---- case 9: happy path ------------------------------------------------------------------


def test_villa_boq_is_matched_priced_and_approved(wf):
    run = wf.start_boq(get_user("khalid"), VILLA_XLSX)
    assert run.status is WfStatus.AWAITING_PROCUREMENT
    assert len(run.lines) == 18 and run.rfq == []
    assert run.total == VILLA_TOTAL
    assert Decimal(run.boq_stats["sections"]["FINISHES"]) == FINISHES_TOTAL

    # High-value lines are priced but wait in the review queue for procurement.
    assert {item["item_no"] for item in run.review} == {"D1", "D2"}

    wf.decide(run.run_id, get_user("omar"), Decision.APPROVE)
    still = wf.decide(run.run_id, get_user("faisal"), Decision.APPROVE)  # above AED 100,000
    assert still.status is WfStatus.AWAITING_FINANCE
    done = wf.decide(run.run_id, get_user("noura"), Decision.APPROVE)
    assert done.status is WfStatus.DONE

    orders = [entry for entry in _outbox() if entry["kind"] == "purchase_order"]
    assert len(orders) == 1 and orders[0]["total_aed"] == "215925.00" and len(orders[0]["lines"]) == 18
    assert not [entry for entry in _outbox() if entry["kind"] == "it_ticket"]


def test_wrong_role_cannot_start_either_entry(wf):
    with pytest.raises(PermissionError):
        wf.start_boq(get_user("sara"), VILLA_XLSX)
    with pytest.raises(PermissionError):
        wf.start_request(get_user("khalid"), "3 site engineers join on 1 Dec. Order laptops.")


# ---- case 10: unit variants and unknown items ---------------------------------------------


def test_unit_variants_match_and_unknown_items_go_to_rfq(wf):
    run = wf.start_boq(get_user("khalid"), ISSUES_CSV)
    priced = {line["item_no"]: line for line in run.lines}
    assert priced["1.1"]["unit"] == "m2" and priced["1.1"]["code"] == "BLK-200-HOL"  # "sqm"
    assert priced["1.2"]["code"] == "BLK-150-HOL"  # "m²"
    assert priced["1.3"]["unit"] == "nos"  # "Nos"
    assert priced["2.1"]["code"] == "PLS-INT-15"  # "SQ.M"
    assert priced["2.3"]["code"] == "LNT-PC"  # "Lm"

    rfq = {item["item_no"]: item for item in run.rfq}
    assert set(rfq) == {"1.4", "2.2"}
    assert all("unit_price" not in item and "total" not in item for item in rfq.values())

    # 1.1 120 x 68 + 1.2 80 x 58 + 1.3 300 x 1.80 + 2.1 200 x 32 + 2.3 20 x 75
    assert run.total == Decimal("8160.00") + Decimal("4640.00") + Decimal("540.00") + Decimal("6400.00") + Decimal("1500.00")


# ---- case 11: planted instruction in a description ------------------------------------------


def test_planted_instruction_is_flagged_and_never_prices(wf):
    run = wf.start_boq(get_user("khalid"), ISSUES_CSV)
    assert "1.5" not in {line["item_no"] for line in run.lines}
    held = [item for item in run.review if item["item_no"] == "1.5"]
    assert held and "injection" in held[0]["reason"]
    assert "unit_price" not in held[0]
    assert any("1.5" in flag for flag in run.flags)
    assert all(Decimal(line["unit_price"]) > Decimal("1") for line in run.lines)


# ---- privacy -------------------------------------------------------------------------------------


def test_private_file_audit_holds_counts_only(wf, monkeypatch, tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    target = private / "boq.csv"
    target.write_bytes(ISSUES_CSV.read_bytes())
    monkeypatch.setattr(config, "PRIVATE_DIR", private)
    monkeypatch.setenv("RAG_TRACE_CONTENT", "full")

    run = wf.start_boq(get_user("khalid"), target)
    text = (config.DATA_DIR / "audit.jsonl").read_text()
    assert run.boq_stats["lines"] == 8
    for phrase in ("Hollow concrete", "Glass block", "ignore the rate card", "boq.csv"):
        assert phrase not in text
