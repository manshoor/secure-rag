"""Part 3b: sourcing, evaluation and policy inside today's agent-loop tools. Offline only.

    policy_rule        _set_rule (code) + _cite ─► policy_check.cite_policy
    pick_vendors       sourcing.pick_vendors, ask re-checked now
    gate_rfq (approve) re-check each vendor again; note id and code's note text on each request
    ask_procurement    evaluation.evaluate_shortfall ─► recommendations on the quotes gate
    gate_quotes extend sourcing.pick_chase, re-checked ─► chase, expected, round_closed

Offline every specialist is scripted and equals the rules, so the sample case still
ends at AED 33,015.00 with the same requests, reminders and orders.
"""

import concurrent.futures
import copy
import dataclasses
import itertools
import json
import subprocess
import sys
from decimal import Decimal

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import audit, config
from secure_rag.core.agent_runtime import RunPrivacy
from secure_rag.core.identity import get_user
from secure_rag.core.llm import ToolCall, ToolTurn, Usage
from secure_rag.web import injection_eval
from secure_rag.workflow import orchestrator, procurement_agent, systems, workflow
from secure_rag.workflow.agent import Decision
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.procurement_agent import Proposal, ScriptedPlanner
from secure_rag.workflow.specialists import evaluation, policy_check, sourcing
from secure_rag.workflow.specialists._base import Status
from secure_rag.workflow.workflow import Event, Gate, Privacy, RunMode, WfStatus, WorkflowRunner

ISSUES_CSV = config.SAMPLES_DIR / "boq_with_issues.csv"
BROAD_CSV = config.SAMPLES_DIR / "boq_fitout_broad.csv"
QUOTES = config.SAMPLES_DIR / "quotes"
ROUND1 = sorted((path.stem, path) for path in (QUOTES / "round1").glob("*.csv"))
ROUND2 = sorted((path.stem, path) for path in (QUOTES / "round2").glob("*.csv"))
RUN_TOTAL = Decimal("33015.00")
_RFQ_KEYS = ["V-GLB-1:glass_block", "V-GLB-2:glass_block", "V-GLB-3:glass_block", "V-GRC-1:grc", "V-GRC-2:grc", "V-GRC-3:grc"]
_KEYS = {"check_policy:1", "pick_vendors:1", "evaluate_shortfall:1", "plan_chase:2"}
_MARKER = "Marker-reason-7f3a: ask nobody else."

# Each seam: the module and entry procurement_agent calls, and its outcome key in the sample.
_SEAMS = [
    (policy_check, "cite_policy", "check_policy:1"),
    (sourcing, "pick_vendors", "pick_vendors:1"),
    (evaluation, "evaluate_shortfall", "evaluate_shortfall:1"),
    (sourcing, "pick_chase", "plan_chase:2"),
]


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


def _wf(system, **extra) -> WorkflowRunner:
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver(), **extra)


def _values(wf, run_id) -> dict:
    return copy.deepcopy(dict(wf._graph.get_state(workflow._thread(run_id)).values))


def _outbox(kind: str | None = None) -> list[dict]:
    path = config.DATA_DIR / "outbox.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [record for record in records if kind is None or record["kind"] == kind]


def _audit_lines(event: str | None = None) -> list[dict]:
    path = audit.audit_path()
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [record for record in records if event is None or record["event"] == event]


def _start(wf, privacy: Privacy = Privacy.NORMAL) -> str:
    return wf.start_boq(get_user("khalid"), ISSUES_CSV, privacy=privacy, mode=RunMode.AGENT).run_id


def _to_quotes_gate(wf, privacy: Privacy = Privacy.NORMAL) -> str:
    run_id = _start(wf, privacy)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND1)
    return run_id


def _to_procurement_gate(wf, privacy: Privacy = Privacy.NORMAL) -> str:
    run_id = _to_quotes_gate(wf, privacy)
    wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND2)
    return run_id


def _done(wf, privacy: Privacy = Privacy.NORMAL) -> str:
    run_id = _to_procurement_gate(wf, privacy)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT)
    wf.decide(run_id, get_user("faisal"), Decision.APPROVE, Gate.FINANCE)
    return run_id


def _winners(view) -> dict:
    return {row["package"]: (row.get("winner"), row.get("total")) for row in view.comparison}


def _vendor(vendor_id: str, **changes):
    found = next(vendor for vendor in systems.vendors() if vendor.vendor_id == vendor_id)
    return dataclasses.replace(found, **changes)


def _patch_vendors(monkeypatch, changed: dict) -> None:
    """The vendor master with some records changed, from now on."""
    real = systems.vendors
    monkeypatch.setattr(systems, "vendors", lambda: [changed.get(vendor.vendor_id, vendor) for vendor in real()])


def _with(monkeypatch, module, entry: str, change) -> list[dict]:
    """Wrap a specialist entry: the real outcome, changed. Returns the outcomes it gave."""
    real = getattr(module, entry)
    given = []

    def wrapped(*args, **kwargs):
        found = change(real(*args, **kwargs))
        given.append(found)
        return found

    monkeypatch.setattr(module, entry, wrapped)
    return given


# ---- the sample case: unchanged, plus notes -------------------------------------------------------


def test_the_sample_ends_at_the_same_total_with_a_note_on_each_request(system):
    wf = _wf(system)
    run_id = _done(wf)
    view = wf.view(run_id, get_user("omar"))
    assert view.status is WfStatus.DONE and view.total == RUN_TOTAL
    assert _winners(view) == {"glass_block": ("V-GLB-3", "4620.00"), "grc": ("V-GRC-2", "7155.00")}

    # The same requests, each with the note id and code's words for it.
    requests = _outbox("rfq")
    assert sorted(entry["key"] for entry in requests) == _RFQ_KEYS
    for entry in requests:
        assert set(entry) == {"run_id", "kind", "key", "to", "email", "company", "package", "lines", "note", "note_text"}
        assert entry["note"] == "standard" and entry["note_text"] == sourcing.note_text("standard")
    assert [entry["key"] for entry in _outbox("reminder")] == ["V-GRC-2:grc:2"]
    assert len(_outbox("purchase_order")) == 4


def test_each_specialist_outcome_is_saved_by_kind_and_round(system):
    wf = _wf(system)
    values = _values(wf, _done(wf))
    outcomes = values["outcomes"]
    assert set(outcomes) == _KEYS
    assert {key: found["status"] for key, found in outcomes.items()} == dict.fromkeys(_KEYS, Status.SCRIPTED.value)
    assert outcomes["pick_vendors:1"]["specialist"] == "sourcing_vendors" and outcomes["plan_chase:2"]["specialist"] == "sourcing_chase"
    for module, _, key in _SEAMS:
        assert module.decision_holds(outcomes[key]["decision"], outcomes[key]["reason_codes"]), key

    # One audit line per specialist call: name, status, codes and counts.
    lines = _audit_lines("specialist")
    assert [line["name"] for line in lines] == ["policy_check", "sourcing_vendors", "evaluation", "sourcing_chase"]
    assert all(set(line) >= {"name", "status", "reason_codes", "turns", "tool_calls"} for line in lines)
    assert all(isinstance(line["turns"], int) and isinstance(line["tool_calls"], int) for line in lines)


def test_scripted_outcomes_equal_today_s_choice(system):
    wf = _wf(system)
    run_id = _start(wf)
    at_rfq = _values(wf, run_id)

    # pick_vendors: every approved vendor of the category, the standard note.
    for package in at_rfq["packages"]:
        assert package["vendors"] == [vendor.vendor_id for vendor in systems.vendors_for(package["category"], "alpha-construction")]
        assert package["note"] == "standard"
    assert procurement_agent.rfq_payload(at_rfq)["notes"] == {"glass_block": "standard", "grc": "standard"}
    policy = next(step for step in at_rfq["agent_steps"] if step["tool"] == "policy_rule")
    assert policy["result"] == {"rule": at_rfq["agent"]["policy"], "cited": True}
    assert at_rfq["outcomes"]["check_policy:1"]["decision"] == {"cite": ["alpha-procurement-policy#purchase-thresholds-0"]}

    # evaluate_shortfall: a recommendation per short package on the quotes gate.
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND1)
    short = _values(wf, run_id)
    assert procurement_agent.quotes_payload(short)["recommendations"] == [{"package": "grc", "recommend": "extend", "problems_that_matter": ["amount_differs", "vendor_not_approved", "not_requested"]}]

    # pick_chase: code's whole list, in its order.
    extended = wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    assert extended.status is WfStatus.AWAITING_QUOTES
    assert _values(wf, run_id)["outcomes"]["plan_chase:2"]["decision"]["chase"] == [{"vendor_id": "V-GRC-2", "package": "grc"}]
    assert _values(wf, run_id)["agent"]["expected"] == ["V-GRC-2"]


def test_payloads_have_no_recommendation_or_note_without_an_outcome(system):
    """A run saved before 3b: no outcomes key, no note on a package."""
    wf = _wf(system)
    state = _values(wf, _start(wf))
    old = {**state, "packages": [{key: value for key, value in package.items() if key != "note"} for package in state["packages"]]}
    del old["outcomes"]
    assert procurement_agent.rfq_payload(old)["notes"] == {"glass_block": "standard", "grc": "standard"}
    assert procurement_agent.quotes_payload(old)["recommendations"] == []


# ---- a hostile tool client changes nothing --------------------------------------------------------


def test_a_hostile_tool_client_changes_no_price_vendor_winner_total_or_approver(system):
    clean = _wf(system)
    expected = clean.view(_done(clean), get_user("omar"))
    expected_outbox = sorted((entry["kind"], entry.get("key", "")) for entry in _outbox())
    (config.DATA_DIR / "outbox.jsonl").unlink()

    # One run per first attack, so every seam meets both: fitting the schema (code's validator refuses it)
    # and invented ids (the schema refuses it).
    ends: dict[str, set[str]] = {key: set() for key in _KEYS}
    for attack in injection_eval._Attack:
        (config.DATA_DIR / "outbox.jsonl").unlink(missing_ok=True)
        client = injection_eval._HostileToolClient(attack)
        wf = _wf(system, agent_client=client)
        run_id = _done(wf)
        view = wf.view(run_id, get_user("omar"))
        values = _values(wf, run_id)
        assert client.tried == 8
        assert view.status is WfStatus.DONE and view.total == expected.total == RUN_TOTAL
        assert _winners(view) == _winners(expected)
        assert [line["unit_price"] for line in view.lines] == [line["unit_price"] for line in expected.lines]
        assert {p["id"]: p["vendors"] for p in view.packages} == {p["id"]: p["vendors"] for p in expected.packages}
        assert view.approvals == expected.approvals and view.approver_roles == expected.approver_roles
        assert sorted((entry["kind"], entry.get("key", "")) for entry in _outbox()) == expected_outbox
        assert {key: found["status"] for key, found in values["outcomes"].items()} == dict.fromkeys(_KEYS, Status.FALLBACK.value)
        assert injection_eval._HostileToolClient.REASON not in json.dumps(values, default=str)
        for key, found in values["outcomes"].items():
            ends[key].add(found["audit"]["end_reason"])
    assert ends == dict.fromkeys(_KEYS, {"completed", "broken_result"})


def test_the_hostile_quote_case_meets_every_seam_with_both_attacks(system, monkeypatch):
    seen = []
    real = injection_eval._HostileToolClient.tool_turn

    def recorded(self, system_text, messages, tools, max_tokens):
        turn = real(self, system_text, messages, tools, max_tokens)
        if turn.calls[0].name == "finish":
            seen.append((tools[0]["name"], "steel" in json.dumps(turn.calls[0].arguments)))
        return turn

    monkeypatch.setattr(injection_eval._HostileToolClient, "tool_turn", recorded)
    with injection_eval._sandbox() as folder:
        models = injection_eval._models(system, injection_eval.Kit.HOSTILE)
        injection_eval._quote_planted(system, models, folder)
    agents = {"policy_sections", "packages", "short_packages", "chase_packages"}
    assert {(agent, invented) for agent, invented in seen} == {(agent, invented) for agent in agents for invented in (True, False)}


def test_the_hostile_kit_wires_its_tool_client_into_every_run(system):
    models = injection_eval._models(system, injection_eval.Kit.HOSTILE)
    assert isinstance(models.agent_client, injection_eval._HostileToolClient)
    assert injection_eval._runner(system, models)._agent_client is models.agent_client
    assert injection_eval._models(system, injection_eval.Kit.OFFLINE).agent_client is None

    # The quote case runs twice (one per attack) and crosses all four seams each time: two turns each.
    with injection_eval._sandbox() as folder:
        models = injection_eval._models(system, injection_eval.Kit.HOSTILE)
        injection_eval._quote_planted(system, models, folder)
    assert models.agent_client.runs == 8 and models.agent_client.tried == 16
    boq = {r.case: r.note for r in injection_eval.run_injection_eval(system, injection_eval.Kit.HOSTILE) if r.case.startswith("boq_")}
    assert boq and all("specialists tried" in note for note in boq.values())


# ---- privacy --------------------------------------------------------------------------------------


class _NeverCalled:
    def __init__(self) -> None:
        self.calls = 0

    def tool_turn(self, system, messages, tools, max_tokens):
        self.calls += 1
        raise AssertionError("a private run called the model")


def test_a_counts_only_run_makes_no_client_call_and_saves_private_outcomes(system):
    client = _NeverCalled()
    wf = _wf(system, agent_client=client)
    run_id = _to_procurement_gate(wf, Privacy.COUNTS_ONLY)
    values = _values(wf, run_id)
    assert client.calls == 0
    assert values["privacy"] == Privacy.COUNTS_ONLY.value
    assert {key: found["status"] for key, found in values["outcomes"].items()} == dict.fromkeys(_KEYS, Status.PRIVATE.value)
    assert Decimal(values["total"]) == RUN_TOTAL


def test_an_open_run_passes_open_privacy_and_the_run_id(system, monkeypatch):
    seen = []
    real = sourcing.pick_vendors
    monkeypatch.setattr(sourcing, "pick_vendors", lambda state, client, **kw: seen.append(kw) or real(state, client, **kw))
    wf = _wf(system)
    run_id = _start(wf)
    _start(wf, Privacy.COUNTS_ONLY)
    assert seen == [{"privacy": RunPrivacy.OPEN, "case_id": run_id}, {"privacy": RunPrivacy.PRIVATE, "case_id": seen[1]["case_id"]}]


# ---- re-checks when applying ----------------------------------------------------------------------


@pytest.mark.parametrize("change", [{"approved": False}, {"company": "beta-properties"}, {"categories": frozenset({"glass_block"})}])
def test_a_vendor_struck_off_before_approval_gets_no_request(system, monkeypatch, change):
    wf = _wf(system)
    run_id = _start(wf)
    _patch_vendors(monkeypatch, {"V-GRC-3": _vendor("V-GRC-3", **change)})
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    values = _values(wf, run_id)
    assert sorted(entry["key"] for entry in _outbox("rfq")) == [key for key in _RFQ_KEYS if key != "V-GRC-3:grc"]
    assert "V-GRC-3" not in values["agent"]["expected"]
    assert next(p for p in values["packages"] if p["id"] == "grc")["vendors"] == ["V-GRC-1", "V-GRC-2"]


def test_a_licence_expired_before_approval_gets_no_request(system, monkeypatch):
    wf = _wf(system)
    run_id = _start(wf)
    _patch_vendors(monkeypatch, {"V-GRC-3": _vendor("V-GRC-3", licence_expiry=config.today() - config.today().resolution)})
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    assert "V-GRC-3:grc" not in {entry["key"] for entry in _outbox("rfq")}
    assert "V-GRC-3" not in _values(wf, run_id)["agent"]["expected"]


def test_a_licence_that_ends_today_still_counts(system, monkeypatch):
    wf = _wf(system)
    run_id = _start(wf)
    _patch_vendors(monkeypatch, {"V-GRC-3": _vendor("V-GRC-3", licence_expiry=config.today())})
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    assert "V-GRC-3:grc" in {entry["key"] for entry in _outbox("rfq")}


def test_a_package_left_with_no_vendor_gets_no_vendors(system, monkeypatch):
    wf = _wf(system)
    run_id = _start(wf)
    _patch_vendors(monkeypatch, {vendor: _vendor(vendor, approved=False) for vendor in ("V-GLB-1", "V-GLB-2", "V-GLB-3")})
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    values = _values(wf, run_id)
    glass = next(p for p in values["packages"] if p["id"] == "glass_block")
    assert glass["status"] == "no_vendors" and glass["vendors"] == []
    assert {entry["package"] for entry in _outbox("rfq")} == {"grc"}
    assert values["agent"]["expected"] == ["V-GRC-1", "V-GRC-2", "V-GRC-3"]


def test_pick_vendors_drops_an_ask_that_fails_the_check_now(system, monkeypatch):
    """The outcome holds the vendor master as read at entry: code checks the ask again when applying."""
    def invented(found):
        rows = [{**row, "ask": [*row["ask"], "V-GRC-9", "V-GLB-1"]} if row["package"] == "grc" else row for row in found["decision"]["packages"]]
        return {**found, "decision": {"packages": rows}}

    _with(monkeypatch, sourcing, "pick_vendors", invented)
    wf = _wf(system)
    values = _values(wf, _start(wf))
    assert next(p for p in values["packages"] if p["id"] == "grc")["vendors"] == ["V-GRC-1", "V-GRC-2", "V-GRC-3"]
    assert next(step for step in values["agent_steps"] if step["tool"] == "pick_vendors")["result"] == {"packages": 2, "vendors": 6}


def test_a_package_missing_from_the_decision_gets_no_vendors(system, monkeypatch):
    _with(monkeypatch, sourcing, "pick_vendors", lambda found: {**found, "decision": {"packages": found["decision"]["packages"][1:]}})
    wf = _wf(system)
    values = _values(wf, _start(wf))
    status = {p["id"]: (p["status"], p["vendors"]) for p in values["packages"]}
    assert status == {"glass_block": ("no_vendors", []), "grc": ("open", ["V-GRC-1", "V-GRC-2", "V-GRC-3"])}
    assert procurement_agent.rfq_payload(values)["notes"] == {"grc": "standard"}


class _Narrowing:
    """A model that asks the first three candidates of each package with the lead-time note; nothing else valid."""

    def __init__(self, facts: dict) -> None:
        self.facts = facts
        self.calls = 0

    def tool_turn(self, system, messages, tools, max_tokens):
        self.calls += 1
        answer = tools[-1]["input_schema"]["properties"]
        if "packages" in answer and "ask" in answer["packages"]["items"]["properties"]:
            rows = [{"package": p["package"], "ask": [row["vendor_id"] for row in p["candidates"][:3]], "note": "confirm_lead_time"} for p in self.facts["packages"]]
            output = {"packages": rows, "reason_codes": ["asked_fewer"], "reason": _MARKER}
        else:
            output = {}
        return ToolTurn("", (ToolCall(f"n{self.calls}", "finish", output),), "tool_use", Usage(), ())


def test_an_accepted_narrower_ask_is_what_goes_out(system):
    probe = _wf(system)
    facts = sourcing.vendor_facts(_values(probe, probe.start_boq(get_user("khalid"), BROAD_CSV, mode=RunMode.AGENT).run_id))
    (config.DATA_DIR / "outbox.jsonl").unlink(missing_ok=True)
    wf = _wf(system, agent_client=_Narrowing(facts))
    run_id = wf.start_boq(get_user("khalid"), BROAD_CSV, mode=RunMode.AGENT).run_id
    values = _values(wf, run_id)
    assert values["outcomes"]["pick_vendors:1"]["status"] == Status.AGENT.value
    asked = {p["package"]: [row["vendor_id"] for row in p["candidates"][:3]] for p in facts["packages"]}
    assert {p["id"]: p["vendors"] for p in values["packages"]} == asked
    assert procurement_agent.rfq_payload(values)["notes"] == dict.fromkeys(asked, "confirm_lead_time")

    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    requests = _outbox("rfq")
    assert sorted(entry["key"] for entry in requests) == sorted(f"{vendor}:{package}" for package, vendors in asked.items() for vendor in vendors)
    assert {entry["note_text"] for entry in requests} == {sourcing.note_text("confirm_lead_time")}
    assert sorted(_values(wf, run_id)["agent"]["expected"]) == sorted({vendor for vendors in asked.values() for vendor in vendors})


def _chased(change):
    return lambda found: {**found, "decision": {**found["decision"], "chase": change(found["decision"]["chase"])}}


def test_a_chase_below_the_minimum_takes_code_s_list(system, monkeypatch):
    """The guard (QA F2): a package left below its minimum after the re-check is chased in full."""
    _with(monkeypatch, sourcing, "pick_chase", _chased(lambda chase: []))
    wf = _wf(system)
    run_id = _to_quotes_gate(wf)
    wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    values = _values(wf, run_id)
    assert [entry["key"] for entry in _outbox("reminder")] == ["V-GRC-2:grc:2"]
    assert values["agent"]["round"] == 2


def _four_grc(system, monkeypatch) -> tuple[WorkflowRunner, dict]:
    """By hand (deep review D1): grc invited V-GRC-1..4, valid quotes from 1 and 3, needed 3; V-GRC-4 is a copy of V-GRC-3."""
    wf = _wf(system)
    state = _values(wf, _to_quotes_gate(wf))
    real = systems.vendors
    extra = dataclasses.replace(next(v for v in real() if v.vendor_id == "V-GRC-3"), vendor_id="V-GRC-4", email="v-grc-4@example.invalid")
    monkeypatch.setattr(systems, "vendors", lambda: [*real(), extra])
    packages = [{**p, "vendors": ["V-GRC-1", "V-GRC-2", "V-GRC-3", "V-GRC-4"]} if p["id"] == "grc" else p for p in state["packages"]]
    quotes = [q for q in state["quotes"] if q["package"] != "grc" or q["vendor_id"] in {"V-GRC-1", "V-GRC-2", "V-GRC-3"}]
    quotes = [{**q, "valid": True, "problems": []} if q["vendor_id"] in {"V-GRC-1", "V-GRC-3"} else q for q in quotes]
    return wf, {**state, "packages": packages, "quotes": quotes, "reply": {"approver": "omar", "decision": "extend"}}


class _FirstTarget:
    """A model that chases only the first target it is offered. offered: the vendor ids it saw."""

    def __init__(self) -> None:
        self.offered: list[str] = []

    def tool_turn(self, system, messages, tools, max_tokens):
        answer = tools[-1]["input_schema"]["properties"]
        entry = answer["chase"]["items"]["properties"]
        self.offered = list(entry["vendor_id"]["enum"])
        output = {"chase": [{"vendor_id": entry["vendor_id"]["enum"][0], "package": entry["package"]["enum"][0]}], "reason_codes": [], "reason": ""}
        return ToolTurn("", (ToolCall("f1", "finish", output),), "tool_use", Usage(), ())


def test_a_struck_off_target_is_never_offered_to_the_chase_agent(system, monkeypatch):
    """D1: V-GRC-2 struck off since round 1; the agent sees only V-GRC-4 and its pick counts."""
    wf, state = _four_grc(system, monkeypatch)
    _patch_vendors(monkeypatch, {"V-GRC-2": _vendor("V-GRC-2", approved=False)})
    wf._agent_client = client = _FirstTarget()
    update = procurement_agent.quotes_apply(wf, state)
    assert client.offered == ["V-GRC-4"]
    assert update["outcomes"]["plan_chase:2"]["status"] == Status.AGENT.value
    assert update["agent"]["chase"] == [{"vendor_id": "V-GRC-4", "package": "grc"}]
    assert update["agent"]["expected"] == ["V-GRC-4"]


class _Reversed:
    """A model that chases every target it is offered, last first."""

    def tool_turn(self, system, messages, tools, max_tokens):
        entry = tools[-1]["input_schema"]["properties"]["chase"]["items"]["properties"]
        chase = [{"vendor_id": vendor_id, "package": entry["package"]["enum"][0]} for vendor_id in reversed(entry["vendor_id"]["enum"])]
        return ToolTurn("", (ToolCall("r1", "finish", {"chase": chase, "reason_codes": [], "reason": ""}),), "tool_use", Usage(), ())


def test_the_agent_s_order_stands_at_the_minimum(system, monkeypatch):
    """grc with no valid quote, V-GRC-2 struck off: the minimum counts the two offered, so their order is kept."""
    wf = _wf(system)
    state = _values(wf, _to_quotes_gate(wf))
    quotes = [{**q, "valid": False, "problems": ["amount_differs"]} if q["package"] == "grc" else q for q in state["quotes"]]
    state = {**state, "quotes": quotes, "reply": {"approver": "omar", "decision": "extend"}}
    _patch_vendors(monkeypatch, {"V-GRC-2": _vendor("V-GRC-2", approved=False)})
    wf._agent_client = _Reversed()
    update = procurement_agent.quotes_apply(wf, state)
    assert update["outcomes"]["plan_chase:2"]["status"] == Status.AGENT.value
    assert update["agent"]["chase"] == [{"vendor_id": "V-GRC-3", "package": "grc"}, {"vendor_id": "V-GRC-1", "package": "grc"}]


def test_the_chase_minimum_counts_the_policy_s_most_quotes():
    """D2: round 1's needed can be lower than round 2's; the minimum never asks fewer than three can reach."""
    assert sourcing._min_chase(1, 0, 3) == 3
    assert sourcing._min_chase(1, 2, 3) == 1
    assert sourcing._min_chase(3, 2, 1) == 1
    assert sourcing._min_chase(3, 1, 4) == 2
    assert sourcing._min_chase(None, 0, 2) == 2
    assert sourcing._min_chase(1, 5, 3) == 0


def test_a_low_round_one_needed_still_chases_enough(system, monkeypatch):
    """D2 probe: grc with no valid quote and round 1's needed at 1; chasing one vendor is refused, all three are chased."""
    wf = _wf(system)
    state = _values(wf, _to_quotes_gate(wf))
    quotes = [{**q, "valid": False, "problems": ["amount_differs"]} if q["package"] == "grc" else q for q in state["quotes"]]
    comparison = [{**row, "needed": 1} for row in state["comparison"]]
    state = {**state, "quotes": quotes, "comparison": comparison, "reply": {"approver": "omar", "decision": "extend"}}
    assert sourcing.chase_facts(state)["packages"][0]["min_chase"] == 3
    wf._agent_client = _FirstTarget()
    update = procurement_agent.quotes_apply(wf, state)
    assert update["outcomes"]["plan_chase:2"]["status"] == Status.FALLBACK.value
    assert [target["vendor_id"] for target in update["agent"]["chase"]] == ["V-GRC-1", "V-GRC-2", "V-GRC-3"]


# ---- the vendor master is read before the paid call (D3) ----------------------------------------


def _master_closed_after(monkeypatch, module, entry: str) -> None:
    """Once the specialist returns, any read of the vendor master raises."""
    real_entry, real_vendors = getattr(module, entry), systems.vendors
    closed = []

    def vendors():
        if closed:
            raise RuntimeError("vendor master read after the specialist")
        return real_vendors()

    def wrapped(*args, **kwargs):
        found = real_entry(*args, **kwargs)
        closed.append(True)
        return found

    monkeypatch.setattr(systems, "vendors", vendors)
    monkeypatch.setattr(module, entry, wrapped)


def test_pick_vendors_reads_the_vendor_master_before_the_specialist(system, monkeypatch):
    wf = _wf(system)
    state = _values(wf, _start(wf))
    agent = {**state["agent"], "picked": False}
    _master_closed_after(monkeypatch, sourcing, "pick_vendors")
    update, result = procurement_agent._pick_vendors(wf, {**state, "agent": agent}, agent, {})
    assert result == {"packages": 2, "vendors": 6}


def test_the_extend_reads_the_vendor_master_before_the_specialist(system, monkeypatch):
    wf = _wf(system)
    state = _values(wf, _to_quotes_gate(wf))
    _master_closed_after(monkeypatch, sourcing, "pick_chase")
    update = procurement_agent.quotes_apply(wf, {**state, "reply": {"approver": "omar", "decision": "extend"}})
    assert update["agent"]["chase"] == [{"vendor_id": "V-GRC-2", "package": "grc"}]


def test_a_tool_error_after_the_specialist_keeps_its_usage(system, monkeypatch):
    """D3: code that raises after a paid call still counts the call; the retry pays again."""
    _with(monkeypatch, sourcing, "pick_vendors", _paid)
    real, failed = procurement_agent._applied_ask, []

    def once(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise RuntimeError("after the call")
        return real(*args, **kwargs)

    monkeypatch.setattr(procurement_agent, "_applied_ask", once)
    planner = _PaidPlanner()
    wf = _wf(system, step_planner=planner)
    view = wf.view(_start(wf), get_user("omar"))
    assert [step["status"] for step in view.agent_steps if step["tool"] == "pick_vendors"] == ["error", "done"]
    assert view.tokens == 15 * planner.calls + 2 * 120


def test_the_real_error_surfaces_at_the_error_cap(system, monkeypatch):
    """A raise after the call that keeps failing reaches the caller as itself, not as the usage carrier."""
    def broken(*args, **kwargs):
        raise LookupError("after the call")

    monkeypatch.setattr(procurement_agent, "_applied_ask", broken)
    wf = _wf(system)
    with pytest.raises(LookupError):
        _start(wf)


# ---- an outcome that fails its shape check is not applied (QA F4) ------------------------------


def test_an_unknown_note_is_not_applied(system, monkeypatch):
    def odd(found):
        rows = [{**row, "note": "pay_now"} for row in found["decision"]["packages"]]
        return {**found, "decision": {"packages": rows}}

    _with(monkeypatch, sourcing, "pick_vendors", odd)
    wf = _wf(system)
    run_id = _start(wf)
    values = _values(wf, run_id)
    assert values["outcomes"]["pick_vendors:1"]["status"] == Status.FALLBACK.value
    assert {p["note"] for p in values["packages"]} == {"standard"}
    assert [(line["name"], line["error"]) for line in _audit_lines("specialist_error")] == [("sourcing_vendors", "invalid_outcome")]
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    assert sorted(entry["key"] for entry in _outbox("rfq")) == _RFQ_KEYS


@pytest.mark.parametrize(("module", "entry", "key"), _SEAMS)
def test_an_outcome_of_the_wrong_shape_falls_back(system, monkeypatch, module, entry, key):
    _with(monkeypatch, module, entry, lambda found: {**found, "decision": {"approve": True}})
    wf = _wf(system)
    values = _values(wf, _to_procurement_gate(wf))
    assert values["outcomes"][key]["status"] == Status.FALLBACK.value
    assert Decimal(values["total"]) == RUN_TOTAL
    assert [entry["key"] for entry in _outbox("reminder")] == ["V-GRC-2:grc:2"]


def test_a_pick_saved_in_the_chase_shape_falls_back(system, monkeypatch):
    """Sourcing's check takes either shape: the seam also checks the task's own keys."""
    _with(monkeypatch, sourcing, "pick_vendors", lambda found: {**found, "decision": {"chase": [], "skipped": []}, "reason_codes": []})
    wf = _wf(system)
    values = _values(wf, _start(wf))
    assert values["outcomes"]["pick_vendors:1"]["status"] == Status.FALLBACK.value
    assert sum(len(p["vendors"]) for p in values["packages"]) == 6


def test_a_paid_outcome_of_the_wrong_shape_still_counts(system, monkeypatch):
    _with(monkeypatch, sourcing, "pick_vendors", lambda found: {**_paid(found), "decision": {"approve": True}})
    planner = _PaidPlanner()
    wf = _wf(system, step_planner=planner)
    view = wf.view(_start(wf), get_user("omar"))
    assert view.tokens == 15 * planner.calls + 120


def test_a_repeated_ask_is_asked_once(system, monkeypatch):
    def doubled(found):
        return {**found, "decision": {"packages": [{**row, "ask": row["ask"] + row["ask"][:1]} for row in found["decision"]["packages"]]}}

    _with(monkeypatch, sourcing, "pick_vendors", doubled)
    wf = _wf(system)
    values = _values(wf, _start(wf))
    assert {p["id"]: p["vendors"] for p in values["packages"]} == {"glass_block": ["V-GLB-1", "V-GLB-2", "V-GLB-3"], "grc": ["V-GRC-1", "V-GRC-2", "V-GRC-3"]}
    assert procurement_agent.rfq_payload(values)["requests"] == 6


def test_the_fallback_ask_leaves_out_an_expired_licence(system, monkeypatch):
    """QA F12: the saved fallback decision names only askable vendors."""
    monkeypatch.setattr(sourcing, "pick_vendors", lambda *args, **kwargs: 1 / 0)
    _patch_vendors(monkeypatch, {"V-GRC-3": _vendor("V-GRC-3", licence_expiry=config.today() - config.today().resolution)})
    wf = _wf(system)
    values = _values(wf, _start(wf))
    rows = {row["package"]: row["ask"] for row in values["outcomes"]["pick_vendors:1"]["decision"]["packages"]}
    assert rows == {"glass_block": ["V-GLB-1", "V-GLB-2", "V-GLB-3"], "grc": ["V-GRC-1", "V-GRC-2"]}


def test_a_chase_target_not_on_code_s_list_is_dropped(system, monkeypatch):
    invented = [{"vendor_id": "V-GRC-9", "package": "grc"}, {"vendor_id": "V-GRC-1", "package": "grc"}, {"vendor_id": "V-GRC-2", "package": "glass_block"}]
    _with(monkeypatch, sourcing, "pick_chase", _chased(lambda chase: invented + chase))
    wf = _wf(system)
    run_id = _to_quotes_gate(wf)
    wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    assert [entry["key"] for entry in _outbox("reminder")] == ["V-GRC-2:grc:2"]
    assert _values(wf, run_id)["agent"]["expected"] == ["V-GRC-2"]


def test_a_chase_target_struck_off_since_is_dropped(system, monkeypatch):
    wf = _wf(system)
    run_id = _to_quotes_gate(wf)
    _patch_vendors(monkeypatch, {"V-GRC-2": _vendor("V-GRC-2", approved=False)})
    wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    assert _outbox("reminder") == []
    assert _values(wf, run_id)["agent"]["expected"] == []


def test_a_repeated_chase_target_is_kept_once(system, monkeypatch):
    _with(monkeypatch, sourcing, "pick_chase", _chased(lambda chase: chase + chase))
    wf = _wf(system)
    state = _values(wf, _to_quotes_gate(wf))
    update = procurement_agent.quotes_apply(wf, {**state, "reply": {"approver": "omar", "decision": "extend"}})
    assert update["agent"]["chase"] == [{"vendor_id": "V-GRC-2", "package": "grc"}]


def test_quotes_apply_sets_chase_and_expected_before_any_planner_step(system):
    wf = _wf(system)
    state = _values(wf, _to_quotes_gate(wf))
    update = procurement_agent.quotes_apply(wf, {**state, "reply": {"approver": "omar", "decision": "extend"}})
    agent = update["agent"]
    assert agent["chase"] == [{"vendor_id": "V-GRC-2", "package": "grc"}]
    assert (agent["expected"], agent["round_closed"], agent["round"], agent["extended"]) == (["V-GRC-2"], False, 2, True)
    assert set(update["outcomes"]) == {"check_policy:1", "pick_vendors:1", "evaluate_shortfall:1", "plan_chase:2"}


# ---- a specialist that raises -------------------------------------------------------------------


@pytest.mark.parametrize(("module", "entry", "key"), _SEAMS)
def test_a_raising_specialist_falls_back_to_today_s_result(system, monkeypatch, module, entry, key):
    def broken(*args, **kwargs):
        raise RuntimeError("secret line text")

    monkeypatch.setattr(module, entry, broken)
    wf = _wf(system)
    run_id = _done(wf)
    view = wf.view(run_id, get_user("omar"))
    values = _values(wf, run_id)
    assert view.status is WfStatus.DONE and view.total == RUN_TOTAL
    assert sorted(entry["key"] for entry in _outbox("rfq")) == _RFQ_KEYS
    assert [entry["key"] for entry in _outbox("reminder")] == ["V-GRC-2:grc:2"]

    # Saved as a fallback in the specialist's own shape; no tool error, no error row.
    found = values["outcomes"][key]
    assert found["status"] == Status.FALLBACK.value and found["audit"]["end_reason"] == "unavailable" and found["reason"] == ""
    assert module.decision_holds(found["decision"], found["reason_codes"])
    assert values["agent"]["errors"] == 0 and all(step["status"] != "error" for step in values["agent_steps"])
    errors = _audit_lines("specialist_error")
    assert [(line["name"], line["error"]) for line in errors] == [(found["specialist"], "RuntimeError")]
    assert "secret line text" not in audit.audit_path().read_text()


def test_fallback_decisions_are_today_s_rules(system, monkeypatch):
    for module, entry, _ in _SEAMS:
        monkeypatch.setattr(module, entry, lambda *args, **kwargs: 1 / 0)
    wf = _wf(system)
    values = _values(wf, _to_procurement_gate(wf))
    outcomes = values["outcomes"]
    assert outcomes["check_policy:1"]["decision"] == {"cite": []}
    assert outcomes["pick_vendors:1"]["decision"] == {"packages": [
        {"package": "glass_block", "ask": ["V-GLB-1", "V-GLB-2", "V-GLB-3"], "note": "standard", "left_out": []},
        {"package": "grc", "ask": ["V-GRC-1", "V-GRC-2", "V-GRC-3"], "note": "standard", "left_out": []},
    ]}
    assert outcomes["evaluate_shortfall:1"]["decision"] == {"packages": []}
    assert outcomes["plan_chase:2"]["decision"] == {"chase": [{"vendor_id": "V-GRC-2", "package": "grc"}], "skipped": []}
    policy = next(step for step in values["agent_steps"] if step["tool"] == "policy_rule")
    assert policy["result"]["cited"] is False and values["agent"]["policy"]


# ---- usage --------------------------------------------------------------------------------------


class _PaidPlanner(ScriptedPlanner):
    def __init__(self) -> None:
        self.calls = 0

    def next_step(self, facts: dict, allowed: list[str]) -> tuple[Proposal, Usage]:
        self.calls += 1
        proposal, _ = super().next_step(facts, allowed)
        return proposal, Usage(10, 5)


def _paid(found: dict) -> dict:
    return {**found, "usage": {**found["usage"], "input_tokens": 100, "output_tokens": 20, "cost_usd": 0.25}}


def test_specialist_usage_adds_to_the_run(system, monkeypatch):
    planner = _PaidPlanner()
    plain = _wf(system, step_planner=planner)
    before = plain.view(_to_procurement_gate(plain), get_user("omar"))
    assert before.tokens == 15 * planner.calls  # a specialist step keeps its planner call's usage

    for module, entry, _ in _SEAMS:
        _with(monkeypatch, module, entry, _paid)
    wf = _wf(system, step_planner=_PaidPlanner())
    after = wf.view(_to_procurement_gate(wf), get_user("omar"))
    assert after.tokens == before.tokens + 4 * 120
    assert after.cost_usd == pytest.approx(before.cost_usd + 4 * 0.25)


# ---- model text stays in outcomes ---------------------------------------------------------------


def test_a_marker_reason_appears_only_in_outcomes(system, monkeypatch):
    for module, entry, _ in _SEAMS:
        _with(monkeypatch, module, entry, lambda found: {**found, "reason": _MARKER})
    wf = _wf(system)
    run_id = _to_quotes_gate(wf)
    short = _values(wf, run_id)
    payloads = [procurement_agent.quotes_payload(short)]
    wf.decide(run_id, get_user("omar"), Decision.EXTEND, Gate.QUOTES)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, get_user("omar"), ROUND2)
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.PROCUREMENT)
    wf.decide(run_id, get_user("faisal"), Decision.APPROVE, Gate.FINANCE)
    values = _values(wf, run_id)
    payloads.append(procurement_agent.rfq_payload(values))

    assert all(found["reason"] == _MARKER for found in values["outcomes"].values())
    rest = {key: value for key, value in values.items() if key != "outcomes"}
    view = wf.view(run_id, get_user("omar"))
    for shown in (rest, dataclasses.asdict(view), payloads):
        assert _MARKER not in json.dumps(shown, default=str)
    assert _MARKER not in audit.audit_path().read_text()
    assert _MARKER not in (config.DATA_DIR / "outbox.jsonl").read_text()


# ---- the policy tool ----------------------------------------------------------------------------


class _PolicyTwice(ScriptedPlanner):
    """Checks the policy again once the vendors are picked."""

    def __init__(self) -> None:
        self.steps = 0

    def next_step(self, facts: dict, allowed: list[str]) -> tuple[Proposal, Usage]:
        self.steps += 1
        if self.steps == 3:
            return Proposal("policy_rule", {}, ""), Usage()
        return super().next_step(facts, allowed)


def test_a_repeated_policy_rule_cites_once(system, monkeypatch):
    given = _with(monkeypatch, policy_check, "cite_policy", lambda found: found)
    wf = _wf(system, step_planner=_PolicyTwice())
    values = _values(wf, _start(wf))
    assert [step["tool"] for step in values["agent_steps"]][:3] == ["policy_rule", "pick_vendors", "policy_rule"]
    assert len(given) == 1
    assert [step["result"]["cited"] for step in values["agent_steps"] if step["tool"] == "policy_rule"] == [True, True]


def test_set_rule_alone_sets_code_s_rule():
    agent = {"catalogue_total": "21240.00", "policy": ""}
    assert procurement_agent._set_rule(agent)["policy"] == procurement_agent._rule_for(Decimal("21240.00"))
    assert procurement_agent._set_rule({**agent, "catalogue_total": "0"})["policy"] == "set by the quote total"


def test_the_policy_tool_no_longer_calls_ask():
    assert not hasattr(procurement_agent, "ask")


# ---- names and keys -----------------------------------------------------------------------------


def test_outcome_keys_are_task_kinds_and_rounds(system):
    kinds = {kind.value for kind in orchestrator.TaskKind}
    assert {kind.value for kind in procurement_agent._Kind} <= kinds
    state = {"agent": {"round": 2}}
    assert procurement_agent._outcome_key(state, procurement_agent._Kind.PLAN_CHASE) == "plan_chase:2"


def test_restated_names_match_the_specialists():
    assert procurement_agent._SPECIALISTS == {
        procurement_agent._Kind.CHECK_POLICY: policy_check._AGENT_NAME,
        procurement_agent._Kind.PICK_VENDORS: sourcing._VENDORS_AGENT,
        procurement_agent._Kind.EVALUATE_SHORTFALL: evaluation._AGENT_NAME,
        procurement_agent._Kind.PLAN_CHASE: sourcing._CHASE_AGENT,
    }
    assert procurement_agent._DEFAULT_NOTE == sourcing._Note.STANDARD.value


def test_the_runner_takes_an_agent_client(system):
    client = _NeverCalled()
    assert _wf(system, agent_client=client)._agent_client is client
    assert _wf(system)._agent_client is None


# ---- import order -------------------------------------------------------------------------------


_MODULES = [
    "secure_rag.workflow.workflow",
    "secure_rag.workflow.procurement_agent",
    "secure_rag.workflow.orchestrator",
    "secure_rag.workflow.specialists.boq_intake",
    "secure_rag.workflow.specialists.policy_check",
    "secure_rag.workflow.specialists.sourcing",
    "secure_rag.workflow.specialists.evaluation",
]


def _imports(pair: tuple[str, str]) -> tuple[tuple[str, str], int, str]:
    script = f"import {pair[0]}\nimport {pair[1]}\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    return pair, done.returncode, done.stderr[-500:]


def test_every_module_pair_imports_in_a_fresh_process_in_both_orders():
    pairs = list(itertools.permutations(_MODULES, 2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        failed = [(pair, error) for pair, code, error in pool.map(_imports, pairs) if code != 0]
    assert failed == []


def test_procurement_agent_imports_no_specialist_at_module_level():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(procurement_agent))
    top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [getattr(node, "module", None) or "" for node in top] + [alias.name for node in top for alias in node.names]
    assert not [name for name in names if "specialists" in name or "orchestrator" in name]
