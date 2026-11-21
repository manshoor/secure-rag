"""Agent-mode BOQ runs: quote requests, vendor replies, the exception gate and one PO per vendor.

Numbers are hand-computed from data/systems/vendor_catalogue.json and the
quote files in data/samples/quotes (see scripts/make_samples.py):

    catalogue lines 1.1, 1.2, 1.3, 2.1, 2.3            21,240.00
    1.4 glass block: Gulf Blocks Factory LLC wins       4,620.00
    2.2 GRC cornice: Cobalt Dune, corrected, round 2    7,155.00
    run total                                          33,015.00  -> three quotes, finance approves
"""

import json
import threading
from dataclasses import replace
from decimal import Decimal

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag import config, procurement_agent, workflow
from secure_rag.agent import Decision
from secure_rag.identity import User, get_user
from secure_rag.intake import OfflineIntake
from secure_rag.llm import ModelRefused, Usage
from secure_rag.procurement_agent import MAX_REFUSALS, Proposal, ScriptedPlanner, Tool
from secure_rag.workflow import Event, Gate, RunMode, WfStatus, WorkflowRunner

ISSUES_CSV = config.SAMPLES_DIR / "boq_with_issues.csv"
QUOTES = config.SAMPLES_DIR / "quotes"
ROUND1 = sorted((path.stem, path) for path in (QUOTES / "round1").glob("*.csv"))
ROUND2 = sorted((path.stem, path) for path in (QUOTES / "round2").glob("*.csv"))
GLASS_ONLY = [(sender, path) for sender, path in ROUND1 if sender.startswith("V-GLB")]

CATALOGUE_TOTAL = Decimal("21240.00")
RUN_TOTAL = Decimal("33015.00")
AGENT_USER = User("agent:procurement", "Procurement agent", frozenset({"alpha-construction"}), frozenset({"employee", "procurement"}))


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


@pytest.fixture
def wf(system) -> WorkflowRunner:
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())


def _outbox(kind: str | None = None) -> list[dict]:
    path = config.DATA_DIR / "outbox.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [record for record in records if kind is None or record["kind"] == kind]


def _start(wf) -> str:
    return wf.start_boq(get_user("khalid"), ISSUES_CSV, mode=RunMode.AGENT).run_id


def _to_exception(wf) -> str:
    """Upload, approve sending, deliver round 1: GRC has 2 of 3 valid quotes."""
    run_id = _start(wf)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND1)
    return run_id


def _to_procurement_gate(wf) -> str:
    run_id = _to_exception(wf)
    wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND2)
    return run_id


# ---- end to end ---------------------------------------------------------------------


def test_upload_drafts_quote_requests_and_waits_for_procurement(wf):
    run = wf.start_boq(get_user("khalid"), ISSUES_CSV, mode=RunMode.AGENT)
    assert run.status is WfStatus.AWAITING_RFQ
    assert run.pending_gate is Gate.RFQ
    assert run.total == CATALOGUE_TOTAL
    assert run.budget_remaining is None  # nothing is held before the quotes are in
    assert [step["tool"] for step in run.agent_steps] == ["policy_rule", "pick_vendors", "draft_rfqs", "request_rfq_approval"]
    assert {pkg["id"]: pkg["items"] for pkg in run.packages} == {"glass_block": ["1.4"], "grc": ["2.2"]}
    assert {pkg["id"]: pkg["vendors"] for pkg in run.packages} == {
        "glass_block": ["V-GLB-1", "V-GLB-2", "V-GLB-3"],
        "grc": ["V-GRC-1", "V-GRC-2", "V-GRC-3"],
    }
    assert _outbox() == []  # nothing goes out before Omar approves


def test_approving_sends_one_request_per_vendor(wf):
    run_id = _start(wf)
    run = wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    assert run.status is WfStatus.AWAITING_QUOTES
    assert run.pending_gate is None
    keys = sorted(record["key"] for record in _outbox("rfq"))
    assert keys == ["V-GLB-1:glass_block", "V-GLB-2:glass_block", "V-GLB-3:glass_block", "V-GRC-1:grc", "V-GRC-2:grc", "V-GRC-3:grc"]
    assert all(record["email"].endswith(".invalid") for record in _outbox("rfq"))


def test_replies_are_checked_and_too_few_valid_quotes_ask_procurement(wf):
    run_id = _to_exception(wf)
    run = wf.view(run_id, get_user("omar"))
    assert run.status is WfStatus.AWAITING_QUOTE_DECISION
    assert run.pending_gate is Gate.QUOTES
    by_vendor = {quote["vendor_id"]: quote for quote in run.quotes}
    assert len(by_vendor) == 7
    assert by_vendor["V-GRC-2"]["problems"] == ["amount_differs"]
    assert by_vendor["V-GLB-2"]["valid"] and by_vendor["V-GLB-2"]["flags"] == ["planted_instruction"]
    assert set(by_vendor["V-GRC-9"]["problems"]) >= {"vendor_not_approved", "not_requested"}
    status = {pkg["id"]: pkg["status"] for pkg in run.packages}
    assert status == {"glass_block": "won", "grc": "short"}


def test_extension_chases_the_vendor_and_the_corrected_quote_wins(wf):
    run_id = _to_exception(wf)
    run = wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    assert run.status is WfStatus.AWAITING_QUOTES
    assert [record["key"] for record in _outbox("reminder")] == ["V-GRC-2:grc:2"]

    run = wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND2)
    assert run.status is WfStatus.AWAITING_PROCUREMENT
    assert run.total == RUN_TOTAL
    assert run.budget_remaining - run.total == Decimal("566985.00")
    assert run.approver_roles == ["executive", "finance"]
    winners = {row["package"]: (row["winner"], row["total"]) for row in run.comparison}
    assert winners == {"glass_block": ("V-GLB-3", "4620.00"), "grc": ("V-GRC-2", "7155.00")}
    assert run.rfq == []  # both quoted lines are now priced


def test_both_gates_end_in_one_purchase_order_per_vendor(wf):
    run_id = _to_procurement_gate(wf)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT)
    done = wf.decide(run_id, get_user("faisal"), Decision.APPROVE, Gate.FINANCE)
    assert done.status is WfStatus.DONE

    orders = {record["key"]: Decimal(record["total_aed"]) for record in _outbox("purchase_order")}
    assert orders == {
        "Gulf Blocks Factory LLC": Decimal("18920.00"),
        "Falcon Steel Fabrication LLC": Decimal("540.00"),
        "Al Noor Building Materials LLC": Decimal("6400.00"),
        "Cobalt Dune Facades LLC": Decimal("7155.00"),
    }
    assert sum(orders.values()) == RUN_TOTAL
    assert len(_outbox("rfq")) == 6 and len(_outbox("reminder")) == 1


def test_refused_quote_requests_take_the_classic_path(wf):
    run_id = _start(wf)
    run = wf.decide(run_id, get_user("omar"), Decision.REJECT, Gate.RFQ)
    assert run.status is WfStatus.AWAITING_PROCUREMENT
    assert run.total == CATALOGUE_TOTAL
    assert sorted(line["item_no"] for line in run.rfq) == ["1.4", "2.2"]
    assert any("quote requests refused" in flag for flag in run.flags)
    assert _outbox("rfq") == []


def test_dropping_lines_keeps_them_on_the_rfq_list(wf):
    run_id = _to_exception(wf)
    run = wf.decide(run_id, get_user("omar"), Decision.DROP, Gate.QUOTES)
    assert run.status is WfStatus.AWAITING_PROCUREMENT
    assert run.total == CATALOGUE_TOTAL + Decimal("4620.00")
    assert [line["item_no"] for line in run.rfq] == ["2.2"]
    assert {pkg["id"]: pkg["status"] for pkg in run.packages} == {"glass_block": "won", "grc": "dropped"}


def test_a_second_shortfall_drops_the_lines(wf):
    run_id = _to_exception(wf)
    wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    run = wf.deliver(run_id, Event.DEADLINE_PASSED, get_user("omar"))
    assert run.status is WfStatus.AWAITING_PROCUREMENT
    assert {pkg["id"]: pkg["status"] for pkg in run.packages} == {"glass_block": "won", "grc": "dropped"}


def test_partial_replies_wait_until_the_deadline(wf):
    run_id = _start(wf)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    run = wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), GLASS_ONLY)
    assert run.status is WfStatus.AWAITING_QUOTES
    run = wf.deliver(run_id, Event.DEADLINE_PASSED, get_user("omar"))
    assert run.status is WfStatus.AWAITING_QUOTE_DECISION
    assert {pkg["id"]: pkg["status"] for pkg in run.packages} == {"glass_block": "won", "grc": "short"}


def test_open_tender_band_takes_quoted_lines_out(wf, monkeypatch):
    # Above AED 100,000 the policy asks for an open tender: no automation.
    monkeypatch.setattr(procurement_agent, "quotes_needed", lambda total: None)
    run_id = _to_exception(wf)
    run = wf.view(run_id, get_user("omar"))
    assert run.status is WfStatus.AWAITING_PROCUREMENT
    assert run.total == CATALOGUE_TOTAL
    assert {pkg["status"] for pkg in run.packages} == {"tender"}


# ---- who may do what ----------------------------------------------------------------------


def test_requester_finance_and_agents_cannot_approve_sending(wf):
    run_id = _start(wf)
    for person in (get_user("khalid"), get_user("faisal"), AGENT_USER):
        run = wf.decide(run_id, person, Decision.APPROVE, Gate.RFQ)
        assert run.status is WfStatus.AWAITING_RFQ
    assert "agent" in wf.view(run_id, get_user("khalid")).reason
    assert _outbox() == []


def test_agent_id_cannot_approve_a_spend_gate(wf):
    run_id = _to_procurement_gate(wf)
    run = wf.decide(run_id, AGENT_USER, Decision.APPROVE, Gate.PROCUREMENT)
    assert run.status is WfStatus.AWAITING_PROCUREMENT


def test_deliver_checks_the_person_and_the_pause(wf):
    run_id = _start(wf)
    with pytest.raises(ValueError):
        wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND1)  # still at the RFQ gate
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    with pytest.raises(PermissionError):
        wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("khalid"), ROUND1)  # requester
    with pytest.raises(LookupError):
        wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("lina"), ROUND1)  # other company
    with pytest.raises(ValueError):
        wf.decide(run_id, get_user("omar"), Decision.APPROVE)  # a wait is not a gate
    with pytest.raises(ValueError):
        wf.resume(run_id, get_user("khalid"))


def test_procurement_sees_quote_prices_and_the_requester_does_not(wf):
    run_id = _to_exception(wf)
    omar = wf.view(run_id, get_user("omar"))
    khalid = wf.view(run_id, get_user("khalid"))
    assert all("total" in quote for quote in omar.quotes)
    assert all("total" not in quote and "lines" not in quote for quote in khalid.quotes)
    assert [quote["vendor_id"] for quote in khalid.quotes] == [quote["vendor_id"] for quote in omar.quotes]
    assert all("total" not in row for row in khalid.comparison)
    with pytest.raises(LookupError):
        wf.view(run_id, get_user("lina"))


def test_procurement_who_sent_requests_can_see_the_waiting_run(wf):
    run_id = _start(wf)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    assert wf.may_view(run_id, get_user("omar"))
    assert not wf.may_view(run_id, get_user("faisal"))


# ---- crash, resume, concurrency ---------------------------------------------------------------


def test_crash_while_sending_requests_never_sends_twice(wf, monkeypatch):
    run_id = _start(wf)
    real = workflow._write_outbox
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(workflow, "_write_outbox", flaky)
    with pytest.raises(OSError):
        wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    monkeypatch.setattr(workflow, "_write_outbox", real)

    # Omar's approval is recorded: a later click cannot replace it, and resume finishes it,
    # sending only the requests that were not written yet.
    assert len(_outbox("rfq")) == 3
    with pytest.raises(ValueError):
        wf.decide(run_id, get_user("omar"), Decision.REJECT, Gate.RFQ)
    run = wf.resume(run_id, get_user("khalid"))
    assert run.status is WfStatus.AWAITING_QUOTES
    assert len(_outbox("rfq")) == 6
    assert len({record["key"] for record in _outbox("rfq")}) == 6
    assert [event["decision"] for event in run.events] == ["approve"]


def test_crash_after_a_classic_approval_is_finished_by_resume(wf, monkeypatch):
    # Before the fix a later reject was silently replayed as the stored approve.
    run_id = wf.start_boq(get_user("khalid"), config.SAMPLES_DIR / "boq_villa_blockwork.xlsx").run_id
    real = workflow.audit.record

    def broken(kind, **fields):
        if fields.get("step") == "gate_procurement":
            raise OSError("disk full")
        return real(kind, **fields)

    monkeypatch.setattr(workflow.audit, "record", broken)
    with pytest.raises(OSError):
        wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT)
    monkeypatch.setattr(workflow.audit, "record", real)

    with pytest.raises(ValueError):
        wf.decide(run_id, get_user("omar"), Decision.REJECT, Gate.PROCUREMENT)
    run = wf.resume(run_id, get_user("khalid"))
    assert run.status is WfStatus.AWAITING_FINANCE
    assert run.approvals == {"procurement": "omar"}


def test_a_reply_that_breaks_the_reader_does_not_stick_the_run(wf, monkeypatch):
    def broken(path):
        raise RuntimeError("reader bug")

    run_id = _start(wf)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    monkeypatch.setattr(procurement_agent, "parse_quote", broken)
    run = wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), GLASS_ONLY)
    assert {quote["problems"][0] for quote in run.quotes} == {"unreadable_file"}
    run = wf.deliver(run_id, Event.DEADLINE_PASSED, get_user("omar"))
    assert run.status is WfStatus.AWAITING_QUOTE_DECISION


def test_crash_while_writing_orders_resumes_with_each_order_once(wf, monkeypatch):
    run_id = _to_procurement_gate(wf)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT)
    real = workflow._write_outbox
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(workflow, "_write_outbox", flaky)
    with pytest.raises(OSError):
        wf.decide(run_id, get_user("faisal"), Decision.APPROVE, Gate.FINANCE)
    monkeypatch.setattr(workflow, "_write_outbox", real)

    run = wf.resume(run_id, get_user("khalid"))
    assert run.status is WfStatus.DONE
    assert sorted(record["key"] for record in _outbox("purchase_order")) == sorted(
        ["Gulf Blocks Factory LLC", "Falcon Steel Fabrication LLC", "Al Noor Building Materials LLC", "Cobalt Dune Facades LLC"]
    )


def test_two_deliveries_at_once_have_one_effect(wf):
    run_id = _start(wf)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    outcomes: list[str] = []

    def deliver():
        try:
            wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND1)
            outcomes.append("ok")
        except ValueError:
            outcomes.append("refused")

    threads = [threading.Thread(target=deliver) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["ok", "refused"]
    assert len(wf.view(run_id, get_user("omar")).quotes) == 7


# ---- planner misbehaviour ---------------------------------------------------------------------


class _Fixed:
    """A planner that always proposes the same step."""

    def __init__(self, tool: str, args: dict | None = None, why: str = "testing") -> None:
        self.proposal = Proposal(tool, args or {}, why)

    def next_step(self, facts: dict, allowed: list[str]) -> tuple[Proposal, Usage]:
        return self.proposal, Usage()


def _runner(system, planner) -> WorkflowRunner:
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver(), step_planner=planner)


def test_tool_not_allowed_is_refused_and_three_refusals_stop(system):
    wf = _runner(system, _Fixed(Tool.FINISH.value))
    run = wf.start_boq(get_user("khalid"), ISSUES_CSV, mode=RunMode.AGENT)
    assert run.status is WfStatus.STOPPED
    assert "needs a person" in run.reason
    assert [step["status"] for step in run.agent_steps] == ["refused"] * MAX_REFUSALS
    assert _outbox() == []


def test_bad_arguments_are_refused(system):
    wf = _runner(system, _Fixed(Tool.CHASE_VENDOR.value, {"vendor_id": "V-GRC-9", "package": "grc"}))
    run = wf.start_boq(get_user("khalid"), ISSUES_CSV, mode=RunMode.AGENT)
    assert run.status is WfStatus.STOPPED
    assert _outbox() == []


def test_a_repeated_step_with_no_change_stops(system):
    wf = _runner(system, _Fixed(Tool.POLICY_RULE.value))
    run = wf.start_boq(get_user("khalid"), ISSUES_CSV, mode=RunMode.AGENT)
    assert run.status is WfStatus.STOPPED
    assert "repeated" in run.reason
    assert len(run.agent_steps) == 2


def test_step_limit_stops_the_loop(system, monkeypatch):
    monkeypatch.setattr(procurement_agent, "MAX_AGENT_STEPS", 2)
    wf = _runner(system, ScriptedPlanner())
    run = wf.start_boq(get_user("khalid"), ISSUES_CSV, mode=RunMode.AGENT)
    assert run.status is WfStatus.STOPPED
    assert "step limit" in run.reason


def test_a_declined_planner_step_falls_back_to_the_rules(system):
    class Declines:
        def next_step(self, facts, allowed):
            raise ModelRefused("model declined the request")

    wf = _runner(system, Declines())
    run = wf.start_boq(get_user("khalid"), ISSUES_CSV, mode=RunMode.AGENT)
    assert run.status is WfStatus.AWAITING_RFQ
    assert all(step["fallback"] for step in run.agent_steps)


def test_a_why_line_with_an_unknown_number_is_dropped(system):
    class Inventive(ScriptedPlanner):
        def next_step(self, facts, allowed):
            proposal, usage = super().next_step(facts, allowed)
            return Proposal(proposal.tool, proposal.args, "Vendor offers AED 999 off"), usage

    wf = _runner(system, Inventive())
    run = wf.start_boq(get_user("khalid"), ISSUES_CSV, mode=RunMode.AGENT)
    assert all(step["why"] == "" for step in run.agent_steps)


def test_private_runs_always_use_the_rules(system, monkeypatch, tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    target = private / "boq.csv"
    target.write_bytes(ISSUES_CSV.read_bytes())
    monkeypatch.setattr(config, "PRIVATE_DIR", private)

    wf = _runner(system, _Fixed(Tool.FINISH.value))
    run = wf.start_boq(get_user("khalid"), target, mode=RunMode.AGENT)
    assert run.status is WfStatus.AWAITING_RFQ
    wf.decide(run.run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    assert all("description" not in record for record in _outbox("rfq"))
    text = (config.DATA_DIR / "audit.jsonl").read_text()
    assert "Glass block" not in text and "GRC" not in text


# ---- review fixes -------------------------------------------------------------------------------


def _boq(tmp_path, rows: list[str]):
    path = tmp_path / "boq.csv"
    path.write_text("\n".join(["Item,Description,Unit,Qty", "1,WORKS,,", *rows]) + "\n", encoding="utf-8")
    return path


def test_quote_rule_counts_packages_already_won(wf, tmp_path):
    # 1,500 lintel + 4,620 glass + GRC at least 7,155 is over 10,000: GRC needs three quotes.
    boq = _boq(tmp_path, ["1.4,Glass block feature wall 100mm,m2,12", "2.2,Decorative GRC cornice,LM,45", "2.3,Precast concrete lintel over openings,Lm,20"])
    run_id = wf.start_boq(get_user("khalid"), boq, mode=RunMode.AGENT).run_id
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND1)
    wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    run = wf.deliver(run_id, Event.DEADLINE_PASSED, get_user("omar"))
    assert {pkg["id"]: pkg["status"] for pkg in run.packages} == {"glass_block": "won", "grc": "dropped"}


def test_late_quotes_that_pass_the_tender_threshold_take_every_package_out(wf, tmp_path):
    # 88,400 blockwork + 4,620 glass + 7,560 GRC = 100,580: open tender, no automation.
    boq = _boq(tmp_path, ["1.1,Hollow concrete block wall 200mm thick,m2,1300", "1.4,Glass block feature wall 100mm,m2,12", "2.2,Decorative GRC cornice,LM,45"])
    run_id = wf.start_boq(get_user("khalid"), boq, mode=RunMode.AGENT).run_id
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), GLASS_ONLY)
    wf.deliver(run_id, Event.DEADLINE_PASSED, get_user("omar"))
    wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    grc = [(sender, path) for sender, path in ROUND1 if sender.startswith("V-GRC")]
    run = wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), grc)
    assert {pkg["status"] for pkg in run.packages} == {"tender"}
    assert run.total == Decimal("88400.00")


def test_vendor_struck_off_after_quoting_stops_before_any_order(wf, monkeypatch):
    run_id = _to_procurement_gate(wf)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT)
    real = workflow.systems.vendors
    monkeypatch.setattr(
        workflow.systems, "vendors",
        lambda: [replace(v, approved=False) if v.vendor_id == "V-GRC-2" else v for v in real()],
    )
    run = wf.decide(run_id, get_user("faisal"), Decision.APPROVE, Gate.FINANCE)
    assert run.status is WfStatus.STOPPED
    assert "V-GRC-2" in run.reason
    assert _outbox("purchase_order") == []
    assert not workflow.systems.is_spent(run_id)


def test_quote_expired_before_finance_approves_stops_before_any_order(wf, monkeypatch):
    run_id = _to_procurement_gate(wf)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT)
    monkeypatch.setenv("RAG_TODAY", "2027-03-01")
    run = wf.decide(run_id, get_user("faisal"), Decision.APPROVE, Gate.FINANCE)
    assert run.status is WfStatus.STOPPED
    assert "expired" in run.reason
    assert _outbox("purchase_order") == []


def test_requester_does_not_see_quote_prices_in_the_priced_lines(wf):
    run_id = _to_procurement_gate(wf)
    quoted = [line for line in wf.view(run_id, get_user("khalid")).lines if line["code"] == "quote"]
    assert quoted and all("unit_price" not in line and "total" not in line and "quote_ref" not in line for line in quoted)
    omar = [line for line in wf.view(run_id, get_user("omar")).lines if line["code"] == "quote"]
    assert all("unit_price" in line for line in omar)


def test_an_empty_delivery_is_refused(wf):
    run_id = _start(wf)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    with pytest.raises(ValueError):
        wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), [])


def test_step_limit_counts_steps_since_the_last_pause(wf, monkeypatch):
    # The longest stretch without a pause in the sample case is 4 steps.
    monkeypatch.setattr(procurement_agent, "MAX_AGENT_STEPS", 4)
    run_id = _to_procurement_gate(wf)
    assert wf.view(run_id, get_user("omar")).status is WfStatus.AWAITING_PROCUREMENT


def test_the_approver_can_finish_their_own_crashed_decision(wf, monkeypatch):
    # Classic finance gate: the reject crashes while releasing the hold; Faisal finishes it himself.
    run_id = wf.start_boq(get_user("khalid"), ISSUES_CSV).run_id
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT)
    real = workflow.systems.release_budget
    calls = {"n": 0}

    def flaky(run):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("ledger locked")
        return real(run)

    monkeypatch.setattr(workflow.systems, "release_budget", flaky)
    with pytest.raises(OSError):
        wf.decide(run_id, get_user("faisal"), Decision.REJECT, Gate.FINANCE)
    assert wf.may_view(run_id, get_user("faisal"))
    run = wf.resume(run_id, get_user("faisal"))
    assert run.status is WfStatus.REJECTED
    assert not wf.may_view(run_id, get_user("sara"))


def test_procurement_can_finish_a_crashed_send(wf, monkeypatch):
    run_id = _start(wf)
    real = workflow._write_outbox

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(workflow, "_write_outbox", broken)
    with pytest.raises(OSError):
        wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    monkeypatch.setattr(workflow, "_write_outbox", real)
    assert wf.resume(run_id, get_user("omar")).status is WfStatus.AWAITING_QUOTES


def test_cli_prints_a_requester_view_without_quote_prices(wf, capsys):
    from secure_rag import cli

    run_id = _to_procurement_gate(wf)
    cli._print_run(wf.view(run_id, get_user("khalid")))
    assert "Cobalt Dune Facades LLC" in capsys.readouterr().out
