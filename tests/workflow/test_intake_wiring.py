"""Part 3c: the intake specialist wired before pricing (agent mode only). Offline only.

    boq_match ─► boq_review (agent mode) ─► boq_price
                 boq_intake.review_weak_lines, re-checked by code:
                   hint   ─► suggested_code + the catalogue item's own description, never priced
                   trade  ─► proposed_trade, only where line_category found none: joins that package
                   review ─► intake_review
                 every proposal on rfq_payload["proposals"]; procurement's RFQ decision settles it

Classic mode keeps boq_match ─► boq_price and the canonicalizer. Offline the
specialist is scripted and equals the rules, so the sample case is unchanged.
Sample weak lines: L3 (1.4, glass block) and L6 (2.2, GRC); L4 (1.5) is held.
"""

import copy
import dataclasses
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.boq import boq
from secure_rag.core import audit, config
from secure_rag.core.agent_runtime import RunPrivacy
from secure_rag.core.identity import get_user
from secure_rag.core.llm import ToolCall, ToolTurn, Usage
from secure_rag.web import injection_eval
from secure_rag.workflow import procurement_agent, systems, workflow
from secure_rag.workflow.agent import Decision
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.specialists import boq_intake
from secure_rag.workflow.specialists._base import Status
from secure_rag.workflow.workflow import Gate, Privacy, RunMode, WfStatus, WorkflowRunner

ISSUES_CSV = config.SAMPLES_DIR / "boq_with_issues.csv"
BROAD_CSV = config.SAMPLES_DIR / "boq_fitout_broad.csv"
KEY = "review_lines:1"
GLASS, HELD, GRC, MATCHED = "L3", "L4", "L6", "L0"
_CLEARS = "Solid concrete block wall 100mm thick"  # a rewrite of 1.4 that clears for BLK-100-SOL
_HINT = "BLK-100-SOL"
_PRICED_TOTAL = Decimal("21240.00")
_MARKER = "Marker-reason-intake: ask nobody else."
# The sample with one more weak line that names no trade: a joinery item in other words (L8).
_WARDROBE = "3.1,Hardwood veneer wardrobe,nos,6\n"
WARDROBE = "L8"
# Before 3c (captured on df85758): the steps up to pricing, in both modes.
_CLASSIC_STEPS = ["boq_parse", "boq_match", "boq_price"]
_AGENT_STEPS = ["boq_parse", "boq_match", "boq_review", "boq_price"]


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


def _wf(system, **extra) -> WorkflowRunner:
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver(), **extra)


def _values(wf, run_id) -> dict:
    return copy.deepcopy(dict(wf._graph.get_state(workflow._thread(run_id)).values))


def _start(wf, path=ISSUES_CSV, privacy: Privacy = Privacy.NORMAL, mode: RunMode = RunMode.AGENT) -> str:
    return wf.start_boq(get_user("khalid"), path, privacy=privacy, mode=mode).run_id


def _outbox(kind: str | None = None) -> list[dict]:
    path = config.DATA_DIR / "outbox.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [record for record in records if kind is None or record["kind"] == kind]


def _audit_lines(event: str) -> list[dict]:
    path = audit.audit_path()
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [record for record in records if record["event"] == event]


def _with_wardrobe(tmp_path):
    path = tmp_path / "boq_wardrobe.csv"
    path.write_text(ISSUES_CSV.read_text(encoding="utf-8") + _WARDROBE, encoding="utf-8")
    return path


def _with(monkeypatch, change) -> list[dict]:
    """Wrap the intake entry: the real outcome, changed. Returns the outcomes it gave."""
    real = boq_intake.review_weak_lines
    given = []

    def wrapped(*args, **kwargs):
        found = change(real(*args, **kwargs))
        given.append(found)
        return found

    monkeypatch.setattr(boq_intake, "review_weak_lines", wrapped)
    return given


def _row(line, action="quote", code="", review="none", trade="none") -> dict:
    return {"line": line, "action": action, "suggested_code": code, "review": review, "trade": trade}


def _decided(rows: list[dict]):
    """An outcome with these rows as its decision (skips the specialist's own validator)."""
    return lambda found: {**found, "status": Status.AGENT.value, "decision": {"lines": rows}}


class _Intake:
    """The intake model: optional rescores, then a finish with these rows. Other agents get an empty finish
    (their schema refuses it, so they fall back to the rules)."""

    def __init__(self, rows: list[dict], rescores: tuple = (), reason: str = "Two lines need a look.") -> None:
        self.rows = rows
        self.rescores = rescores
        self.reason = reason
        self.intake_calls = 0

    def tool_turn(self, system, messages, tools, max_tokens):
        if tools[0]["name"] != "weak_lines":
            return _turn([("finish", {})])
        self.intake_calls += 1
        if not messages[1:] and self.rescores:
            return _turn([("rescore", {"line": line, "description": text}) for line, text in self.rescores])
        return _turn([("finish", {"lines": self.rows, "reason_codes": [], "reason": self.reason})], Usage(300, 40, 0.1))


def _turn(calls, usage=Usage()) -> ToolTurn:
    named = tuple(ToolCall(f"t{n}", name, args) for n, (name, args) in enumerate(calls))
    echo = tuple({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments} for call in named)
    return ToolTurn("", named, "tool_use", usage, echo)


class _NeverCalled:
    def __init__(self) -> None:
        self.calls = 0

    def tool_turn(self, system, messages, tools, max_tokens):
        self.calls += 1
        raise AssertionError("a private run called the model")


class _Recording:
    """A canonicalizer that records whether it was asked."""

    def __init__(self) -> None:
        self.asked = 0

    def suggest(self, descriptions):
        self.asked += 1
        return [], Usage()


def _rules_rows(state: dict) -> list[dict]:
    return [_row(line["line"], trade=line["trade"]) for line in boq_intake.weak_line_facts(state)["lines"]]


# ---- the graph: agent mode adds boq_review only -----------------------------------------------------


def test_classic_mode_keeps_its_steps_and_never_reviews(system):
    wf = _wf(system)
    values = _values(wf, _start(wf, mode=RunMode.CLASSIC))
    assert [step["name"] for step in values["steps"]] == _CLASSIC_STEPS
    assert "outcomes" not in values
    assert all("proposed_trade" not in line and "intake_review" not in line for line in values["rfq"])


def test_agent_mode_adds_boq_review_between_match_and_price(system):
    wf = _wf(system)
    values = _values(wf, _start(wf))
    assert [step["name"] for step in values["steps"]] == _AGENT_STEPS
    review = values["steps"][2]
    assert review["status"] == "done"
    assert review["detail"] == {"weak": 2, "hints": 0, "trades": 0, "reviews": 0, "outcome": Status.SCRIPTED.value}


def test_the_sample_ends_at_the_same_total_and_outbox(system):
    wf = _wf(system)
    run_id = _start(wf)
    omar = get_user("omar")
    wf.decide(run_id, omar, Decision.APPROVE, Gate.RFQ)
    quotes = config.SAMPLES_DIR / "quotes"
    wf.deliver(run_id, workflow.Event.QUOTES_DELIVERED, omar, sorted((p.stem, p) for p in (quotes / "round1").glob("*.csv")))
    wf.decide(run_id, omar, Decision.EXTEND, Gate.QUOTES)
    wf.deliver(run_id, workflow.Event.QUOTES_DELIVERED, omar, sorted((p.stem, p) for p in (quotes / "round2").glob("*.csv")))
    wf.decide(run_id, omar, Decision.APPROVE, Gate.PROCUREMENT)
    view = wf.decide(run_id, get_user("faisal"), Decision.APPROVE, Gate.FINANCE)
    assert view.status is WfStatus.DONE and view.total == Decimal("33015.00")
    assert sorted(entry["key"] for entry in _outbox("rfq")) == [
        "V-GLB-1:glass_block", "V-GLB-2:glass_block", "V-GLB-3:glass_block", "V-GRC-1:grc", "V-GRC-2:grc", "V-GRC-3:grc"]
    assert [entry["key"] for entry in _outbox("reminder")] == ["V-GRC-2:grc:2"]
    assert len(_outbox("purchase_order")) == 4


# ---- scripted outcome equals the rules ---------------------------------------------------------------


@pytest.mark.parametrize("path", [ISSUES_CSV, BROAD_CSV])
def test_the_scripted_outcome_equals_the_rules(system, path):
    probe = _wf(system)
    classic = _values(probe, _start(probe, path, mode=RunMode.CLASSIC))
    wf = _wf(system)
    values = _values(wf, _start(wf, path))
    found = values["outcomes"][KEY]
    assert found["status"] == Status.SCRIPTED.value and found["specialist"] == "boq_intake"
    assert all(row["action"] == "quote" and row["suggested_code"] == "" and row["review"] == "none" for row in found["decision"]["lines"])
    assert procurement_agent.rfq_payload(values)["proposals"] == []

    # The RFQ lines are those of a classic run: no new field.
    assert values["rfq"] == classic["rfq"]


def test_the_broad_sample_reviews_its_ten_weak_lines(system):
    wf = _wf(system)
    values = _values(wf, _start(wf, BROAD_CSV))
    assert len(values["outcomes"][KEY]["decision"]["lines"]) == 10
    assert values["steps"][2]["detail"]["weak"] == 10


@pytest.mark.parametrize("privacy", [Privacy.NORMAL, Privacy.COUNTS_ONLY])
@pytest.mark.parametrize("path", [ISSUES_CSV, BROAD_CSV])
def test_real_intake_outcomes_hold_their_shape(system, path, privacy):
    wf = _wf(system)
    found = _values(wf, _start(wf, path, privacy))["outcomes"][KEY]
    assert boq_intake.decision_holds(found["decision"], found["reason_codes"]) is True


# ---- accepted proposals ----------------------------------------------------------------------------


def test_an_accepted_hint_is_shown_never_priced(system):
    client = _Intake([_row(GLASS, "hint", _HINT, trade="glass_block"), _row(GRC, trade="grc")], rescores=((GLASS, _CLEARS),))
    wf = _wf(system, agent_client=client)
    values = _values(wf, _start(wf))
    assert values["outcomes"][KEY]["status"] == Status.AGENT.value
    assert values["steps"][2]["detail"]["hints"] == 1

    # Still on the RFQ list and in its package; the price list and total unchanged.
    glass = next(line for line in values["rfq"] if line["item_no"] == "1.4")
    assert glass["suggested_code"] == _HINT
    assert "1.4" not in {line["item_no"] for line in values["lines"]}
    assert Decimal(values["agent"]["catalogue_total"]) == _PRICED_TOTAL
    assert "1.4" in next(p for p in values["packages"] if p["id"] == "glass_block")["items"]
    assert procurement_agent.rfq_payload(values)["proposals"] == [
        {"line": GLASS, "item": "1.4", "trade": "none", "suggested_code": _HINT, "review": "none"}]


def test_a_hint_carries_the_catalogue_item_s_own_description(system):
    state = _matched_state()
    runner = SimpleNamespace(_agent_client=_Intake([_row(GLASS, "hint", _HINT, trade="glass_block"), _row(GRC, trade="grc")], rescores=((GLASS, _CLEARS),)))
    update = procurement_agent.review_step(runner, state)
    line = update["boq_raw"][3]
    item = next(item for item in systems.catalogue() if item.code == _HINT)
    assert line["suggested_code"] == _HINT and line["suggested"] == item.description[:120]
    assert line["status"] == state["boq_raw"][3]["status"]


def test_an_accepted_trade_joins_that_package_as_a_proposal(system, tmp_path):
    path = _with_wardrobe(tmp_path)
    plain = _wf(system)
    before = _values(plain, _start(plain, path))
    assert {p["id"] for p in before["packages"]} == {"glass_block", "grc"}

    rows = [_row(GLASS, trade="glass_block"), _row(GRC, trade="grc"), _row(WARDROBE, trade="joinery")]
    wf = _wf(system, agent_client=_Intake(rows))
    values = _values(wf, _start(wf, path))
    joinery = next(p for p in values["packages"] if p["id"] == "joinery")
    assert joinery["items"] == ["3.1"] and joinery["vendors"]
    assert values["steps"][2]["detail"] == {"weak": 3, "hints": 0, "trades": 1, "reviews": 0, "outcome": Status.AGENT.value}
    wardrobe = next(line for line in values["rfq"] if line["item_no"] == "3.1")
    assert wardrobe["proposed_trade"] == "joinery"
    assert procurement_agent.rfq_payload(values)["proposals"] == [
        {"line": WARDROBE, "item": "3.1", "trade": "joinery", "suggested_code": "", "review": "none"}]
    assert Decimal(values["agent"]["catalogue_total"]) == _PRICED_TOTAL


def test_a_proposed_trade_package_goes_through_the_other_specialists(system, monkeypatch, tmp_path):
    """The specialists after intake take the new package as any other: scripted, no error, every askable vendor."""
    _with(monkeypatch, _decided([_row(GLASS, trade="glass_block"), _row(GRC, trade="grc"), _row(WARDROBE, trade="joinery")]))
    wf = _wf(system)
    values = _values(wf, _start(wf, _with_wardrobe(tmp_path)))
    assert {key: values["outcomes"][key]["status"] for key in ("check_policy:1", "pick_vendors:1")} == dict.fromkeys(("check_policy:1", "pick_vendors:1"), Status.SCRIPTED.value)
    assert _audit_lines("specialist_error") == []
    joinery = next(p for p in values["packages"] if p["id"] == "joinery")
    assert joinery["vendors"] == [vendor.vendor_id for vendor in systems.vendors_for("joinery", "alpha-construction")]
    assert procurement_agent.rfq_payload(values)["requests"] == 6 + len(joinery["vendors"])


def test_a_refused_rfq_gate_sends_nothing_for_a_proposed_trade(system, tmp_path):
    rows = [_row(GLASS, trade="glass_block"), _row(GRC, trade="grc"), _row(WARDROBE, trade="joinery")]
    wf = _wf(system, agent_client=_Intake(rows))
    run_id = _start(wf, _with_wardrobe(tmp_path))
    wf.decide(run_id, get_user("omar"), Decision.REJECT, Gate.RFQ)
    assert _outbox("rfq") == []
    assert next(p for p in _values(wf, run_id)["packages"] if p["id"] == "joinery")["status"] == "not_sent"


def test_an_approved_rfq_gate_asks_the_proposed_trade_s_vendors(system, tmp_path):
    rows = [_row(GLASS, trade="glass_block"), _row(GRC, trade="grc"), _row(WARDROBE, trade="joinery")]
    wf = _wf(system, agent_client=_Intake(rows))
    run_id = _start(wf, _with_wardrobe(tmp_path))
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    joinery = [entry for entry in _outbox("rfq") if entry["package"] == "joinery"]
    assert joinery and all(entry["lines"] == [{"item_no": "3.1", "quantity": "6", "unit": "nos", "description": "Hardwood veneer wardrobe"}] for entry in joinery)


def test_an_accepted_review_is_saved_on_the_line(system):
    client = _Intake([_row(GLASS, "review", review="unit_doubt", trade="glass_block"), _row(GRC, trade="grc")])
    wf = _wf(system, agent_client=client)
    values = _values(wf, _start(wf))
    glass = next(line for line in values["rfq"] if line["item_no"] == "1.4")
    assert glass["intake_review"] == "unit_doubt"
    assert values["steps"][2]["detail"]["reviews"] == 1
    assert procurement_agent.rfq_payload(values)["proposals"] == [
        {"line": GLASS, "item": "1.4", "trade": "none", "suggested_code": "", "review": "unit_doubt"}]


def test_a_line_with_no_trade_and_no_proposal_stays_rfq_only(system, tmp_path):
    wf = _wf(system)
    values = _values(wf, _start(wf, _with_wardrobe(tmp_path)))
    assert "joinery" not in {p["id"] for p in values["packages"]}
    assert "3.1" in {line["item_no"] for line in values["rfq"]}


# ---- code re-checks each row when applying ----------------------------------------------------------


@pytest.mark.parametrize("rows", [
    [_row(GLASS, trade="joinery"), _row(GRC, trade="grc")],  # a trade on a line code categorised
    [_row(HELD, "hint", _HINT), _row(HELD, "review", review="unclear", trade="grc")],  # a held line
    [_row(MATCHED, "hint", _HINT), _row(MATCHED, "review", review="unclear", trade="joinery")],  # a line that is not weak
    [_row("L999", "hint", _HINT, trade="joinery"), _row("L999", "review", review="unclear")],  # no such line
    [_row(GLASS, "quote", _HINT, trade="glass_block"), _row(GRC, "quote", review="unclear", trade="grc")],  # fields without their action
    [_row(GLASS, "review", trade="glass_block"), _row(GRC, "review", trade="grc")],  # a review with no reason
], ids=["categorised", "held", "not_weak", "invented", "no_action", "no_reason"])
def test_a_row_that_fails_the_check_now_is_not_applied(system, monkeypatch, rows):
    _with(monkeypatch, _decided(rows))
    wf = _wf(system)
    values = _values(wf, _start(wf))
    assert values["outcomes"][KEY]["status"] == Status.AGENT.value  # shape held: code's apply refused the rows
    assert values["steps"][2]["detail"] == {"weak": 2, "hints": 0, "trades": 0, "reviews": 0, "outcome": Status.AGENT.value}
    for line in [*values["rfq"], *values["review"]]:
        assert not {"suggested_code", "proposed_trade", "intake_review"} & set(line)
    assert procurement_agent.rfq_payload(values)["proposals"] == []
    assert {p["id"] for p in values["packages"]} == {"glass_block", "grc"}


def test_a_line_answered_twice_takes_the_first_answer(system, monkeypatch):
    _with(monkeypatch, _decided([_row(GLASS, "review", review="unclear", trade="glass_block"), _row(GLASS, "review", review="unit_doubt", trade="glass_block")]))
    wf = _wf(system)
    values = _values(wf, _start(wf))
    assert next(line for line in values["rfq"] if line["item_no"] == "1.4")["intake_review"] == "unclear"


@pytest.mark.parametrize("change", [
    lambda found: {**found, "decision": {"lines": [_row(GLASS, trade="steel")]}},  # an unknown trade
    lambda found: {**found, "decision": {"lines": [_row(GLASS, "hint", "FAKE-1")]}},  # a code not in the catalogue
    lambda found: {**found, "decision": {"lines": [{**_row(GLASS), "price": "1"}]}},  # an extra key
    lambda found: {**found, "decision": {"packages": []}},  # another specialist's shape
], ids=["trade", "code", "extra_key", "other_shape"])
def test_an_outcome_of_the_wrong_shape_falls_back_to_the_rules(system, monkeypatch, change):
    _with(monkeypatch, change)
    wf = _wf(system)
    values = _values(wf, _start(wf))
    found = values["outcomes"][KEY]
    assert found["status"] == Status.FALLBACK.value
    assert found["decision"] == {"lines": [_row(GLASS, trade="glass_block"), _row(GRC, trade="grc")]}
    assert [(line["name"], line["error"]) for line in _audit_lines("specialist_error")] == [("boq_intake", "invalid_outcome")]


# ---- hostile client, privacy, failure, usage, model text ---------------------------------------------


@pytest.mark.parametrize("attack", list(injection_eval._Attack))
def test_a_hostile_intake_changes_no_price_package_or_total(system, attack):
    plain = _wf(system)
    expected = _values(plain, _start(plain))
    client = injection_eval._HostileToolClient(attack)
    wf = _wf(system, agent_client=client)
    values = _values(wf, _start(wf))
    found = values["outcomes"][KEY]
    assert found["status"] == Status.FALLBACK.value
    assert found["audit"]["end_reason"] == {"fitting": "completed", "invented": "broken_result"}[attack.value]
    assert values["lines"] == expected["lines"] and values["total"] == expected["total"]
    assert values["rfq"] == expected["rfq"]
    assert {p["id"]: p["vendors"] for p in values["packages"]} == {p["id"]: p["vendors"] for p in expected["packages"]}
    assert injection_eval._HostileToolClient.REASON not in json.dumps({k: v for k, v in values.items() if k != "outcomes"}, default=str)


def test_the_hostile_fitting_finish_fits_the_intake_schema(system):
    """The fitting attack must pass the schema, so code's validator is what refuses it."""
    client = injection_eval._HostileToolClient(injection_eval._Attack.FITTING)
    wf = _wf(system, agent_client=client)
    found = _values(wf, _start(wf))["outcomes"][KEY]
    assert found["audit"]["end_reason"] == "completed"


def test_a_counts_only_run_makes_no_intake_call(system):
    client = _NeverCalled()
    wf = _wf(system, agent_client=client)
    values = _values(wf, _start(wf, privacy=Privacy.COUNTS_ONLY))
    assert client.calls == 0
    assert values["outcomes"][KEY]["status"] == Status.PRIVATE.value


def test_intake_gets_open_privacy_only_for_a_normal_run(system, monkeypatch):
    seen = []
    real = boq_intake.review_weak_lines
    monkeypatch.setattr(boq_intake, "review_weak_lines", lambda state, client, **kw: seen.append(kw) or real(state, client, **kw))
    wf = _wf(system)
    run_id = _start(wf)
    _start(wf, privacy=Privacy.COUNTS_ONLY)
    assert seen == [{"privacy": RunPrivacy.OPEN, "case_id": run_id}, {"privacy": RunPrivacy.PRIVATE, "case_id": seen[1]["case_id"]}]


def test_a_raising_intake_falls_back_to_the_rules_with_no_tool_error(system, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("secret line text")

    monkeypatch.setattr(boq_intake, "review_weak_lines", broken)
    wf = _wf(system)
    run_id = _start(wf)
    values = _values(wf, run_id)
    assert wf.view(run_id, get_user("khalid")).status is WfStatus.AWAITING_RFQ
    found = values["outcomes"][KEY]
    assert found["status"] == Status.FALLBACK.value and found["audit"]["end_reason"] == "unavailable" and found["reason"] == ""
    assert found["decision"] == {"lines": [_row(GLASS, trade="glass_block"), _row(GRC, trade="grc")]}
    assert boq_intake.decision_holds(found["decision"], found["reason_codes"])
    assert [(line["name"], line["error"]) for line in _audit_lines("specialist_error")] == [("boq_intake", "RuntimeError")]
    assert "secret line text" not in audit.audit_path().read_text()
    assert values["agent"]["errors"] == 0


def test_the_fallback_rules_equal_the_scripted_rules(system):
    state = _matched_state()
    decision, codes = procurement_agent._review_rules(state)
    assert decision == {"lines": _rules_rows(state)}
    assert boq_intake.decision_holds(decision, codes)


def test_one_specialist_audit_line_per_intake_call(system):
    wf = _wf(system)
    _start(wf)
    lines = [line for line in _audit_lines("specialist") if line["name"] == "boq_intake"]
    assert len(lines) == 1 and lines[0]["status"] == Status.SCRIPTED.value


def test_intake_usage_adds_to_the_run(system, monkeypatch):
    plain = _wf(system)
    before = plain.view(_start(plain), get_user("khalid"))
    _with(monkeypatch, lambda found: {**found, "usage": {**found["usage"], "input_tokens": 100, "output_tokens": 20, "cost_usd": 0.25}})
    wf = _wf(system)
    after = wf.view(_start(wf), get_user("khalid"))
    assert after.tokens == before.tokens + 120
    assert after.cost_usd == pytest.approx(before.cost_usd + 0.25)


def test_a_marker_reason_appears_only_in_outcomes(system, tmp_path):
    rows = [_row(GLASS, "review", review="unclear", trade="glass_block"), _row(GRC, trade="grc"), _row(WARDROBE, trade="joinery")]
    wf = _wf(system, agent_client=_Intake(rows, reason=_MARKER))
    run_id = _start(wf, _with_wardrobe(tmp_path))
    values = _values(wf, run_id)
    assert values["outcomes"][KEY]["reason"] == _MARKER
    rest = {key: value for key, value in values.items() if key != "outcomes"}
    shown = [rest, dataclasses.asdict(wf.view(run_id, get_user("omar"))), procurement_agent.rfq_payload(values)]
    assert _MARKER not in json.dumps(shown, default=str)
    assert _MARKER not in audit.audit_path().read_text()
    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)
    assert _MARKER not in (config.DATA_DIR / "outbox.jsonl").read_text()


# ---- canonicalizer, resume ---------------------------------------------------------------------------


def test_the_canonicalizer_is_not_called_in_agent_mode(system):
    canon = _Recording()
    wf = _wf(system, canonicalizer=canon)
    _start(wf)
    assert canon.asked == 0


def test_the_canonicalizer_is_still_called_in_classic_mode(system):
    canon = _Recording()
    wf = _wf(system, canonicalizer=canon)
    _start(wf, mode=RunMode.CLASSIC)
    assert canon.asked == 1


def test_a_saved_outcome_is_reused_with_no_second_call(system, monkeypatch):
    state = _matched_state()
    client = _Intake([_row(GLASS, "review", review="unclear", trade="glass_block"), _row(GRC, trade="grc")])
    first = procurement_agent.review_step(SimpleNamespace(_agent_client=client), state)
    calls = client.intake_calls
    monkeypatch.setattr(boq_intake, "review_weak_lines", lambda *args, **kwargs: 1 / 0)
    again = procurement_agent.review_step(SimpleNamespace(_agent_client=client), {**state, **first, "boq_raw": state["boq_raw"]})
    assert client.intake_calls == calls
    assert again["boq_raw"] == first["boq_raw"]
    assert again.get("usage", first["usage"]) == first["usage"]
    assert "outcomes" not in again or again["outcomes"] == first["outcomes"]


def test_the_review_step_keeps_earlier_outcomes_and_usage(system):
    state = {**_matched_state(), "outcomes": {"other:1": {"x": 1}}, "usage": {"input": 7, "output": 3, "cost_usd": 0.5}}
    update = procurement_agent.review_step(SimpleNamespace(_agent_client=None), state)
    assert set(update["outcomes"]) == {"other:1", KEY}
    assert update["usage"] == {"input": 7, "output": 3, "cost_usd": 0.5}


def _matched_state() -> dict:
    """The run state as match_step leaves it in agent mode."""
    runner = SimpleNamespace(_boq_reader=None, _canonicalizer=None)
    state = {"boq_file": str(ISSUES_CSV), "flags": [], "steps": [], "usage": {"input": 0, "output": 0, "cost_usd": 0.0},
             "privacy": "normal", "mode": "agent", "requester": "khalid", "run_id": "r-1"}
    state.update(boq.parse_step(runner, state))
    state.update(boq.match_step(runner, state))
    return state


@pytest.mark.parametrize("case", ["boq_known_pattern", "boq_reworded", "boq_arabic"])
def test_every_hostile_boq_case_attacks_intake(system, monkeypatch, case):
    """The BOQ cases no longer reach the hostile canonicaliser in agent mode: the hostile intake takes its place."""
    finished = []
    real = injection_eval._HostileToolClient.tool_turn

    def recorded(self, system_text, messages, tools, max_tokens):
        turn = real(self, system_text, messages, tools, max_tokens)
        if turn.calls[0].name == "finish":
            finished.append(tools[0]["name"])
        return turn

    monkeypatch.setattr(injection_eval._HostileToolClient, "tool_turn", recorded)
    run = dict(injection_eval._CASES)[case]
    with injection_eval._sandbox() as folder:
        models = injection_eval._models(system, injection_eval.Kit.HOSTILE)
        note, _ = run(system, models, folder)
    assert "weak_lines" in finished, finished
    assert models.canonicalizer.tried == 0
    assert "specialists tried" in note


def test_a_row_on_a_line_with_no_unit_is_not_applied(system, monkeypatch, tmp_path):
    """Weak needs a unit code read: an unknown unit stays RFQ-only, whatever a row says (L8)."""
    path = tmp_path / "boq_no_unit.csv"
    path.write_text(ISSUES_CSV.read_text(encoding="utf-8") + "3.1,Hardwood veneer wardrobe,xyz,6\n", encoding="utf-8")
    _with(monkeypatch, _decided([_row(WARDROBE, "review", review="unclear", trade="joinery")]))
    wf = _wf(system)
    values = _values(wf, _start(wf, path))
    wardrobe = next(line for line in values["rfq"] if line["item_no"] == "3.1")
    assert not {"proposed_trade", "intake_review", "line"} & set(wardrobe)
    assert "joinery" not in {p["id"] for p in values["packages"]}


@pytest.mark.parametrize("held", [{"note": "possible injection in description"}, {"injection": True}, {"status": "held"}])
def test_a_row_on_a_line_held_since_is_not_applied(system, held):
    """Defence: a line marked held (by note, flag or status) after the specialist ran gets nothing."""
    state = _matched_state()
    state["boq_raw"][3] = {**state["boq_raw"][3], **held}
    found = {"status": Status.AGENT.value, "decision": {"lines": [_row(GLASS, "review", review="unclear", trade="glass_block")]}}
    update = procurement_agent.review_step(SimpleNamespace(_agent_client=None), {**state, "outcomes": {KEY: found}})
    assert "intake_review" not in update["boq_raw"][3]


# ---- after QA (F1 to F6) ------------------------------------------------------------------------------


def _drive_to_done(wf, run_id) -> "workflow.RunView":
    omar = get_user("omar")
    quotes = config.SAMPLES_DIR / "quotes"
    wf.decide(run_id, omar, Decision.APPROVE, Gate.RFQ)
    wf.deliver(run_id, workflow.Event.QUOTES_DELIVERED, omar, sorted((p.stem, p) for p in (quotes / "round1").glob("*.csv")))
    wf.decide(run_id, omar, Decision.EXTEND, Gate.QUOTES)
    wf.deliver(run_id, workflow.Event.QUOTES_DELIVERED, omar, sorted((p.stem, p) for p in (quotes / "round2").glob("*.csv")))
    wf.decide(run_id, omar, Decision.APPROVE, Gate.PROCUREMENT)
    return wf.decide(run_id, get_user("faisal"), Decision.APPROVE, Gate.FINANCE)


def test_a_proposed_trade_never_joins_a_package_code_formed(system, monkeypatch, tmp_path):
    """F1 (user's ruling: never join): grc proposed on 3.1 leaves code's grc package as it was."""
    _with(monkeypatch, _decided([_row(GLASS, trade="glass_block"), _row(GRC, trade="grc"), _row(WARDROBE, trade="grc")]))
    wf = _wf(system)
    run_id = _start(wf, _with_wardrobe(tmp_path))
    values = _values(wf, run_id)
    assert next(p for p in values["packages"] if p["id"] == "grc")["items"] == ["2.2"]
    assert "3.1" not in {item for p in values["packages"] for item in p["items"]}
    wardrobe = next(line for line in values["rfq"] if line["item_no"] == "3.1")
    assert wardrobe["proposed_trade"] == "grc"
    assert procurement_agent.rfq_payload(values)["proposals"] == [
        {"line": WARDROBE, "item": "3.1", "trade": "grc", "suggested_code": "", "review": "none"}]
    view = _drive_to_done(wf, run_id)
    assert view.status is WfStatus.DONE and view.total == Decimal("33015.00")


def test_proposed_lines_still_form_a_package_of_a_new_trade(system):
    rfq = [
        {"item_no": "2.2", "description": "Decorative GRC cornice", "unit": "lm", "quantity": "45"},
        {"item_no": "3.1", "description": "Hardwood veneer wardrobe", "unit": "nos", "quantity": "6", "proposed_trade": "joinery"},
        {"item_no": "3.2", "description": "Veneer shelf", "unit": "nos", "quantity": "4", "proposed_trade": "joinery"},
        {"item_no": "3.3", "description": "Cornice return", "unit": "lm", "quantity": "5", "proposed_trade": "grc"},
    ]
    packages = {p["id"]: p["items"] for p in procurement_agent._packages(rfq)}
    assert packages == {"grc": ["2.2"], "joinery": ["3.1", "3.2"]}


def test_a_left_out_proposed_line_counts_as_in_no_vendor_category():
    """With no package at all, a proposed line kept out of code's trade is counted as in no vendor category."""
    rfq = [
        {"item_no": "2.2", "description": "Decorative GRC cornice", "unit": "lm", "quantity": "", "reason": "no catalogue item"},
        {"item_no": "3.1", "description": "Cornice return", "unit": "lm", "quantity": "5", "proposed_trade": "grc", "reason": "no catalogue item"},
    ]
    assert procurement_agent._packages(rfq) == []
    assert procurement_agent._unquoted_causes(rfq, []) == "1 in no vendor category"


def test_code_s_category_wins_over_a_proposed_trade():
    """F6: _trade on a categorised line that carries a proposed_trade."""
    line = {"description": "Glass block feature wall 100mm", "proposed_trade": "joinery"}
    assert procurement_agent._trade(line, set()) == "glass_block"
    assert procurement_agent._trade({"description": "Wardrobe", "proposed_trade": "joinery"}, set()) == "joinery"
    assert procurement_agent._trade({"description": "Wardrobe", "proposed_trade": "joinery"}, {"joinery"}) == ""


def test_rows_for_one_line_with_leading_zeros_are_applied_once():
    """F2: L03 and L3 name the same line; the first counts."""
    state = _matched_state()
    rows = [_row("L03", "review", review="unclear", trade="glass_block"), _row(GLASS, "review", review="unit_doubt", trade="glass_block")]
    found = {"status": Status.AGENT.value, "decision": {"lines": rows}}
    update = procurement_agent.review_step(SimpleNamespace(_agent_client=None), {**state, "outcomes": {KEY: found}})
    assert update["boq_raw"][3]["intake_review"] == "unclear" and update["boq_raw"][3]["line"] == GLASS
    assert update["steps"][-1]["detail"]["reviews"] == 1


def test_a_row_just_past_the_last_line_is_skipped():
    """F3: L<len(boq_raw)> is out of range: skipped, never raised."""
    state = _matched_state()
    past = f"L{len(state['boq_raw'])}"
    found = {"status": Status.AGENT.value, "decision": {"lines": [_row(past, "review", review="unclear")]}}
    update = procurement_agent.review_step(SimpleNamespace(_agent_client=None), {**state, "outcomes": {KEY: found}})
    assert update["boq_raw"] == state["boq_raw"]
    assert update["steps"][-1]["detail"]["reviews"] == 0


def test_the_hostile_fitting_trade_is_not_none_and_not_code_s():
    """F4: the fitting attack's trade must break the facts on a categorised line."""
    state = _matched_state()
    facts = boq_intake.weak_line_facts(state)["lines"]
    row = boq_intake._output_schema([line["line"] for line in facts])["properties"]["lines"]["items"]["properties"]
    answers = injection_eval._HostileToolClient._fitting_lines(row)
    code_trades = {line["line"]: line["trade"] for line in facts}
    assert answers and all(answer["trade"] != "none" for answer in answers)
    assert all(answer["trade"] != code_trades[answer["line"]] for answer in answers if code_trades[answer["line"]] != "none")


@pytest.mark.parametrize("path", [ISSUES_CSV, BROAD_CSV])
def test_the_restated_weak_test_equals_intake_s(path):
    """Departure 6: procurement_agent._still_weak mirrors boq_intake._is_weak."""
    runner = SimpleNamespace(_boq_reader=None, _canonicalizer=None)
    state = {"boq_file": str(path), "flags": [], "steps": [], "usage": {}, "privacy": "normal", "mode": "agent", "requester": "khalid", "run_id": "r-1"}
    state.update(boq.parse_step(runner, state))
    state.update(boq.match_step(runner, state))
    assert [procurement_agent._still_weak(line) for line in state["boq_raw"]] == [boq_intake._is_weak(line) for line in state["boq_raw"]]
    assert any(boq_intake._is_weak(line) for line in state["boq_raw"])
