"""Agent-mode BOQs with no catalogue-priced line: the quotes alone make the total.

Hand-computed from data/samples/quotes: glass block 4,620.00 (round 1) + GRC cornice
7,155.00 (corrected, round 2) = 11,775.00, so three quotes and the finance gate.
"""

import json
import time
from dataclasses import replace
from decimal import Decimal

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import config
from secure_rag.core.identity import get_user
from secure_rag.workflow import systems
from secure_rag.workflow.agent import Decision
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.procurement_agent import MAX_AGENT_STEPS, _unpriced_reason
from secure_rag.workflow.workflow import Event, Gate, RunMode, WfStatus, WorkflowRunner

HEADER = "Item,Description,Unit,Qty\n"
QUOTED_LINES = "1.4,Glass block feature wall 100mm,m2,12\n2.2,Decorative GRC cornice,LM,45\n"
NO_CATEGORY_LINES = "1,Bronze door handle,nos,40\n2,Teak bench seat,nos,6\n"
QUOTES = config.SAMPLES_DIR / "quotes"
ROUND1 = sorted((path.stem, path) for path in (QUOTES / "round1").glob("*.csv"))
ROUND2 = sorted((path.stem, path) for path in (QUOTES / "round2").glob("*.csv"))
QUOTES_TOTAL = Decimal("11775.00")
BUDGET = Decimal("600000.00")

NOTHING_PRICED = "no line could be priced from the catalogue"
CANNOT_GO_OUT = "needs a person: no line can go out for quotes: "
RULE_FROM_QUOTES = "set by the quote total"
SCALE_LINES = 200
SCALE_SECONDS = 5


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


@pytest.fixture
def wf(system) -> WorkflowRunner:
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())


def _csv(tmp_path, rows: str):
    path = tmp_path / "boq.csv"
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def _start(wf, tmp_path, rows: str = QUOTED_LINES, mode: RunMode = RunMode.AGENT):
    return wf.start_boq(get_user("khalid"), _csv(tmp_path, rows), mode=mode)


def _outbox(kind: str | None = None) -> list[dict]:
    path = config.DATA_DIR / "outbox.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [record for record in records if kind is None or record["kind"] == kind]


def _to_exception(wf, tmp_path) -> str:
    run_id = _start(wf, tmp_path).run_id
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND1)
    return run_id


def _no_vendors(monkeypatch):
    monkeypatch.setattr(systems, "vendors_for", lambda category, company: [])


# ---- goes out for quotes ---------------------------------------------------------------


def test_quote_lines_alone_wait_at_the_request_gate(wf, tmp_path):
    run = _start(wf, tmp_path)
    assert run.status is WfStatus.AWAITING_RFQ and run.pending_gate is Gate.RFQ
    assert run.lines == [] and run.total == 0
    assert run.budget_remaining is None  # nothing is held before the quotes are in
    assert not (config.DATA_DIR / "budget_ledger.json").exists()
    assert _outbox() == []


def test_the_rule_is_not_stated_before_any_price_is_known(wf, tmp_path):
    """AED 0 would read as 'one written quotation' beside requests that may need three."""
    step = _start(wf, tmp_path).agent_steps[0]
    assert step["tool"] == "policy_rule" and step["result"]["rule"] == RULE_FROM_QUOTES


def test_winning_quotes_are_the_total_and_both_gates_apply(wf, tmp_path):
    run_id = _to_exception(wf, tmp_path)
    assert wf.view(run_id, get_user("omar")).pending_gate is Gate.QUOTES  # GRC: 2 of 3 valid
    wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    run = wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND2)

    assert run.pending_gate is Gate.PROCUREMENT
    assert run.total == QUOTES_TOTAL
    assert run.budget_remaining == BUDGET
    assert systems.available_budget("alpha-construction", "projects") == BUDGET - QUOTES_TOTAL
    assert run.approver_roles == ["executive", "finance"]
    assert all(line["code"] == "quote" for line in run.lines) and run.rfq == []
    assert next(step for step in run.agent_steps if step["tool"] == "finalize_pricing")["result"]["rule"] == "three written quotations"

    # Same gates as any run: the sender may not also give the finance approval.
    run = wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT)
    assert run.pending_gate is Gate.FINANCE
    assert "three" in run.summaries["finance"]
    run = wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.FINANCE)
    assert run.pending_gate is Gate.FINANCE
    run = wf.decide(run_id, get_user("faisal"), Decision.APPROVE, Gate.FINANCE)
    assert run.status is WfStatus.DONE
    assert sorted(order["total_aed"] for order in _outbox("purchase_order")) == ["4620.00", "7155.00"]


def test_private_run_sends_no_descriptions(wf, tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    monkeypatch.setattr(config, "PRIVATE_DIR", private)
    run_id = _start(wf, private).run_id
    run = wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    assert run.private and len(_outbox("rfq")) == 6
    sent = json.dumps(_outbox()) + (config.DATA_DIR / "audit.jsonl").read_text()
    assert "cornice" not in sent and "feature wall" not in sent


# ---- stops for a person ------------------------------------------------------------------


def test_lines_outside_every_vendor_category_stop(wf, tmp_path):
    run = _start(wf, tmp_path, NO_CATEGORY_LINES)
    assert run.status is WfStatus.STOPPED and run.reason == CANNOT_GO_OUT + "2 in no vendor category"
    assert run.stopped_at == "boq_price" and len(run.rfq) == 2
    assert _outbox() == []


def test_a_category_with_no_approved_vendor_stops(wf, tmp_path, monkeypatch):
    _no_vendors(monkeypatch)
    run = _start(wf, tmp_path)
    assert run.status is WfStatus.STOPPED and run.reason == "needs a person: no approved vendor in a category"
    assert run.pending_gate is None and _outbox() == []


@pytest.mark.parametrize(
    "rows, causes",
    [
        ("1,Decorative GRC cornice,bundle,45\n", "1 with an unknown unit"),
        ("1,Solid concrete block 200mm,bundle,45\n", "1 with an unknown unit"),
        ("1,Decorative GRC cornice,bundle,45\n2,Bronze door handle,nos,40\n", "1 with an unknown unit, 1 in no vendor category"),
        ("1,Concrete block wall 200mm,m2,45\n", "1 in no vendor category"),  # weak catalogue match
    ],
)
def test_the_stop_reason_is_true_for_every_line(wf, tmp_path, rows, causes):
    run = _start(wf, tmp_path, rows)
    assert run.status is WfStatus.STOPPED and run.reason == CANNOT_GO_OUT + causes
    assert len(run.rfq) == rows.count("\n")


def test_a_stop_after_quotes_also_names_the_lines_never_sent(wf, tmp_path):
    run_id = _start(wf, tmp_path, "1,Decorative GRC cornice,lm,45\n" + "2,Decorative GRC cornice,bundle,45\n3,Bronze door handle,nos,40\n").run_id
    run = wf.decide(run_id, get_user("omar"), Decision.REJECT, Gate.RFQ)
    assert run.reason == "needs a person: quote requests refused; not sent: 1 with an unknown unit, 1 in no vendor category"


def test_mixed_package_statuses_are_all_named_in_order():
    statuses = ("no_vendors", "not_sent", "dropped", "tender", "dropped")
    state = {"packages": [{"status": status} for status in statuses], "rfq": []}
    assert _unpriced_reason(state) == (
        "the quotes need an open tender; too few valid quotes; quote requests refused; no approved vendor in a category"
    )


def test_one_category_without_vendors_is_named_beside_refused_requests(wf, tmp_path, monkeypatch):
    real = systems.vendors_for
    monkeypatch.setattr(systems, "vendors_for", lambda category, company: [] if category == "grc" else real(category, company))
    run_id = _start(wf, tmp_path).run_id
    run = wf.decide(run_id, get_user("omar"), Decision.REJECT, Gate.RFQ)
    assert run.reason == "needs a person: quote requests refused; no approved vendor in a category"


def test_nothing_priced_and_nothing_to_quote_stops_as_before(wf, tmp_path):
    run = _start(wf, tmp_path, "1,Demolish glass block wall,m2,12\n")
    assert run.status is WfStatus.STOPPED and run.reason == NOTHING_PRICED
    assert run.rfq == [] and len(run.review) == 1


@pytest.mark.parametrize("rows", [QUOTED_LINES, NO_CATEGORY_LINES])
def test_classic_mode_still_stops(wf, tmp_path, rows):
    run = _start(wf, tmp_path, rows, RunMode.CLASSIC)
    assert run.status is WfStatus.STOPPED and run.stopped_at == "boq_price"
    assert run.reason == NOTHING_PRICED


def test_refused_requests_stop_with_that_reason(wf, tmp_path):
    run_id = _start(wf, tmp_path).run_id
    run = wf.decide(run_id, get_user("omar"), Decision.REJECT, Gate.RFQ)
    assert run.status is WfStatus.STOPPED and run.reason == "needs a person: quote requests refused"
    assert _outbox() == []


def test_dropping_every_line_stops_with_that_reason(wf, tmp_path):
    run_id = _start(wf, tmp_path).run_id
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    wf.deliver(run_id, Event.DEADLINE_PASSED, get_user("omar"), [])
    run = wf.decide(run_id, get_user("omar"), Decision.DROP, Gate.QUOTES)
    assert run.status is WfStatus.STOPPED and run.reason == "needs a person: too few valid quotes"


def test_quotes_in_the_tender_band_stop_with_that_reason(wf, tmp_path):
    """Above AED 100,000 of quotes alone: open tender, no automation, so no executive gate."""
    quotes = []
    for vendor in ("V-GRC-1", "V-GRC-2", "V-GRC-3"):
        path = tmp_path / f"{vendor}.csv"
        path.write_text(
            f"Vendor,Test vendor\nVendor ID,{vendor}\nQuote ref,Q-1\nCurrency,AED\nValid until,2026-12-31\n\n"
            "Item,Description,Unit,Qty,Unit price,Amount\n2.2,Decorative GRC cornice,lm,45,3000.00,135000.00\n"
        )
        quotes.append((vendor, path))
    run_id = _start(wf, tmp_path, "2.2,Decorative GRC cornice,lm,45\n").run_id
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    run = wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), quotes)
    assert run.status is WfStatus.STOPPED and run.reason == "needs a person: the quotes need an open tender"
    assert not _outbox("purchase_order")


# ---- repeated item numbers: a quote names its line by item number -----------------------------

REPEATED = "repeated item number; not priced"
GLASS_14 = "1.4,Glass block feature wall 100mm,m2,12\n"


@pytest.mark.parametrize(
    "repeated",
    [
        "2.2,Decorative GRC cornice,LM,45\n2.2,Decorative GRC panel,LM,10\n",
        "2.2,Decorative GRC cornice,LM,45\n2.2,Glass block screen,m2,3\n",  # two packages
        "2.2,Decorative GRC cornice,LM,45\n2.2,Bronze door handle,nos,3\n",  # one line in no package
        ",Decorative GRC cornice,LM,45\n,Decorative GRC panel,LM,10\n",  # no number at all
    ],
)
def test_lines_sharing_a_number_are_held_and_never_sent(wf, tmp_path, repeated):
    run = _start(wf, tmp_path, GLASS_14 + repeated)
    assert run.status is WfStatus.AWAITING_RFQ
    assert [(p["id"], p["items"]) for p in run.packages] == [("glass_block", ["1.4"])]
    assert [line["item_no"] for line in run.rfq] == ["1.4"]
    assert [line["reason"] for line in run.review].count(REPEATED) == 2
    assert sum("repeated item number" in flag for flag in run.flags) == 2

    wf.decide(run.run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    assert {row["item_no"] for record in _outbox("rfq") for row in record["lines"]} == {"1.4"}


def test_held_repeats_stay_for_review_when_the_order_is_written(wf, tmp_path):
    rows = GLASS_14 + "2.2,Decorative GRC cornice,LM,45\n2.2,Decorative GRC panel,LM,10\n"
    run_id = _start(wf, tmp_path, rows).run_id
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), [q for q in ROUND1 if q[0].startswith("V-GLB")])
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT)
    run = wf.decide(run_id, get_user("faisal"), Decision.APPROVE, Gate.FINANCE)
    assert run.status is WfStatus.DONE and run.total == Decimal("4620.00")
    assert [line["quantity"] for line in run.review if line["reason"] == REPEATED] == ["45", "10"]


def test_only_repeated_lines_stop_and_say_so(wf, tmp_path):
    run = _start(wf, tmp_path, "2.2,Decorative GRC cornice,LM,45\n2.2,Decorative GRC panel,LM,10\n")
    assert run.status is WfStatus.STOPPED
    assert run.reason == CANNOT_GO_OUT + "2 with a repeated item number"


GRC_TWICE = "2.2,Decorative GRC cornice,LM,45\n2.2,Decorative GRC panel,LM,10\n"


@pytest.mark.parametrize(
    "mode, rows",
    [
        (RunMode.CLASSIC, "1,Hollow concrete block wall 200mm thick,m2,10\n" + GRC_TWICE),
        (RunMode.AGENT, "1,Hollow concrete block wall 200mm thick,m2,2000\n" + GRC_TWICE),  # 136,000: open tender
    ],
)
def test_repeats_stay_on_the_rfq_list_when_no_quote_is_asked_for(wf, tmp_path, mode, rows):
    run = _start(wf, tmp_path, rows, mode)
    assert run.status is WfStatus.AWAITING_PROCUREMENT
    assert [line["quantity"] for line in run.rfq] == ["45", "10"]
    assert run.flags == [] and all(line["reason"] != REPEATED for line in run.review)


def test_the_order_counts_repeats_that_stayed_on_the_rfq_list(wf, tmp_path):
    run_id = _start(wf, tmp_path, "1,Hollow concrete block wall 200mm thick,m2,2000\n" + GRC_TWICE).run_id
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT)
    run = wf.decide(run_id, get_user("noura"), Decision.APPROVE, Gate.FINANCE)
    assert run.status is WfStatus.DONE and [order["rfq_lines"] for order in _outbox("purchase_order")] == [2]


def test_repeats_outside_every_package_stay_on_the_rfq_list(wf, tmp_path):
    """No quote names 3, so nothing can be mixed up: only the GRC lines are held."""
    handles = "3,Bronze door handle,nos,4\n3,Teak bench seat,nos,6\n"
    run_id = _start(wf, tmp_path, GLASS_14 + GRC_TWICE + handles).run_id
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    run = wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), [q for q in ROUND1 if q[0].startswith("V-GLB")])
    assert run.total == Decimal("4620.00")
    assert [line["quantity"] for line in run.rfq] == ["4", "6"]
    assert [line["item_no"] for line in run.review if line["reason"] == REPEATED] == ["2.2", "2.2"]


def test_the_stop_reason_counts_held_lines_too(wf, tmp_path):
    zero = "1,Hollow concrete block wall 200mm thick,m2,0.00005\n"
    removal = "4,Demolish glass block wall,m2,12\n"
    run = _start(wf, tmp_path, zero + GRC_TWICE + removal)
    assert run.reason == CANNOT_GO_OUT + "2 with a repeated item number, 1 that rounds to AED 0.00, 1 held for another reason"


@pytest.mark.parametrize("mode", [RunMode.CLASSIC, RunMode.AGENT])
def test_catalogue_lines_sharing_a_number_are_all_priced_once(wf, tmp_path, mode):
    rows = "1,Hollow concrete block wall 200mm thick,m2,10\n1,Hollow concrete block wall 200mm thick,m2,5\n"
    run = _start(wf, tmp_path, rows, mode)
    assert [line["quantity"] for line in run.lines] == ["10", "5"] and run.total == Decimal("1020.00")


def test_a_catalogue_line_and_a_quote_line_may_share_a_number(wf, tmp_path):
    """Only quote lines are found by number, so neither line is lost or doubled."""
    rows = "1.4,Hollow concrete block wall 200mm thick,m2,10\n" + GLASS_14
    run_id = _start(wf, tmp_path, rows).run_id
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    run = wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), [q for q in ROUND1 if q[0].startswith("V-GLB")])
    assert run.total == Decimal("5300.00") and len(run.lines) == 2 and run.rfq == []


# ---- amounts under one fils ------------------------------------------------------------------


@pytest.mark.parametrize("mode", [RunMode.CLASSIC, RunMode.AGENT])
def test_a_catalogue_line_that_rounds_to_zero_is_held(wf, tmp_path, mode):
    run = _start(wf, tmp_path, "1,Hollow concrete block wall 200mm thick,m2,0.00005\n", mode)
    assert run.status is WfStatus.STOPPED and run.lines == []
    assert [line["reason"] for line in run.review] == ["line total rounds to AED 0.00; not priced"]


def test_a_run_total_of_zero_is_never_held_or_approved(wf, monkeypatch):
    real = systems.price_items
    monkeypatch.setattr(systems, "price_items", lambda q: [replace(line, total=Decimal("0.00")) for line in real(q)])
    run = wf.start_request(get_user("sara"), "3 site engineers join on 2026-12-01. Order laptops.")
    assert run.status is WfStatus.STOPPED and run.reason == "the run total is AED 0.00"
    assert not (config.DATA_DIR / "budget_ledger.json").exists()


# ---- scale ---------------------------------------------------------------------------------


@pytest.mark.parametrize("quotable, status", [(60, WfStatus.AWAITING_RFQ), (0, WfStatus.STOPPED)])
def test_two_hundred_lines_finish_the_first_stretch_fast(wf, tmp_path, quotable, status):
    rows = "".join(
        f"{n},Decorative GRC cornice type {n},lm,{n}\n" if n <= quotable else f"{n},Bespoke fitting type {n},nos,{n}\n"
        for n in range(1, SCALE_LINES + 1)
    )
    began = time.monotonic()
    run = _start(wf, tmp_path, rows)
    seconds = time.monotonic() - began
    print(f"{SCALE_LINES} lines, {quotable} quotable: {len(run.agent_steps)} agent steps, {seconds:.2f}s")
    assert run.status is status and len(run.rfq) == SCALE_LINES
    assert len(run.agent_steps) < MAX_AGENT_STEPS and seconds < SCALE_SECONDS
