"""Chat front door, run list, demo events and resume over HTTP."""

import json
import threading

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import config

from secure_rag.workflow import run_index
from secure_rag.web.api import create_app
from secure_rag.web.chat import Intent, classify
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.workflow import WorkflowRunner

ISSUES = config.SAMPLES_DIR / "boq_with_issues.csv"


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


@pytest.fixture
def client(system, runner):
    workflows = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    with TestClient(create_app(system, runner, workflows)) as test_client:
        yield test_client


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setenv("RAG_DEMO", "on")


def _as(user: str) -> dict:
    return {"X-User": user}


def _upload(client, mode: str | None = "agent", user: str = "khalid"):
    data = {"mode": mode} if mode else {}
    return client.post("/workflows/boq", files={"file": ("boq.csv", ISSUES.read_bytes(), "text/csv")}, data=data, headers=_as(user))


def _decide(client, run_id: str, user: str, decision: str, gate: str):
    return client.post(f"/workflows/{run_id}/decision", json={"decision": decision, "gate": gate}, headers=_as(user))


def _event(client, run_id: str, user: str = "omar", event: str = "quotes_delivered", sample: str | None = "round1"):
    body = {"event": event, **({"sample": sample} if sample else {})}
    return client.post(f"/workflows/{run_id}/events", json=body, headers=_as(user))


def _chat(client, user: str, message: str, run_id: str | None = None):
    body = {"message": message, **({"run_id": run_id} if run_id else {})}
    return client.post("/chat", json=body, headers=_as(user))


def _agent_run(client) -> str:
    return _upload(client).json()["run_id"]


# ---- /me ------------------------------------------------------------------------------


def test_me_says_offline_and_demo(client, monkeypatch):
    body = client.get("/me", headers=_as("omar")).json()
    assert (body["llm"], body["demo"]) == ("offline", False)
    monkeypatch.setenv("RAG_DEMO", "on")
    assert client.get("/me", headers=_as("omar")).json()["demo"] is True


# ---- upload mode ------------------------------------------------------------------------


def test_upload_defaults_to_classic(client):
    body = _upload(client, mode=None).json()
    assert (body["mode"], body["status"]) == ("classic", "awaiting_procurement")


def test_upload_in_agent_mode_waits_for_quote_requests(client):
    body = _upload(client).json()
    assert (body["mode"], body["status"], body["pending_gate"]) == ("agent", "awaiting_rfq", "rfq")
    assert [step["tool"] for step in body["agent_steps"]][0] == "policy_rule"
    assert {pkg["id"] for pkg in body["packages"]} == {"glass_block", "grc"}


def test_unknown_mode_is_refused(client):
    assert _upload(client, mode="autopilot").status_code == 422


# ---- decisions per gate ------------------------------------------------------------------


def test_quote_gate_takes_extend_or_drop_and_other_gates_do_not(client, demo):
    run_id = _agent_run(client)
    assert _decide(client, run_id, "omar", "extend", "rfq").status_code == 409
    assert _decide(client, run_id, "omar", "approve", "rfq").json()["status"] == "awaiting_quotes"
    assert _event(client, run_id).json()["status"] == "awaiting_quote_decision"
    assert _decide(client, run_id, "omar", "approve", "quotes").status_code == 409
    assert _decide(client, run_id, "omar", "extend", "quotes").json()["status"] == "awaiting_quotes"
    body = _event(client, run_id, sample="round2").json()
    assert (body["status"], body["total"]) == ("awaiting_procurement", "33015.00")


def test_the_older_agent_runner_refuses_quote_choices(runner):
    from secure_rag.workflow.agent import Decision
    from secure_rag.core.identity import get_user

    for choice in (Decision.EXTEND, Decision.DROP):
        with pytest.raises(ValueError):
            runner.resume("any-run", get_user("faisal"), choice)


# ---- demo events ---------------------------------------------------------------------------


def test_events_need_demo_mode(client):
    run_id = _agent_run(client)
    _decide(client, run_id, "omar", "approve", "rfq")
    assert _event(client, run_id).status_code == 404


def test_events_check_the_person_and_the_sample(client, demo):
    run_id = _agent_run(client)
    assert _event(client, run_id).status_code == 409  # still at the RFQ gate
    _decide(client, run_id, "omar", "approve", "rfq")
    assert _event(client, run_id, user="khalid").status_code == 403
    assert _event(client, run_id, user="lina").status_code == 404
    assert _event(client, run_id, sample="../../private").status_code == 422
    assert _event(client, run_id, event="deadline_passed", sample=None).json()["status"] == "awaiting_quote_decision"


# ---- resume --------------------------------------------------------------------------------


def test_resume_route(client):
    run_id = _agent_run(client)
    assert client.post(f"/workflows/{run_id}/resume", headers=_as("khalid")).status_code == 409
    assert client.post(f"/workflows/{run_id}/resume", headers=_as("lina")).status_code == 404


# ---- run list ------------------------------------------------------------------------------


def _ids(client, user: str, scope: str) -> list[str]:
    return [row["run_id"] for row in client.get(f"/workflows?scope={scope}", headers=_as(user)).json()]


def test_waiting_list_follows_the_gate(client):
    run_id = _agent_run(client)
    assert _ids(client, "omar", "waiting") == [run_id]
    assert _ids(client, "faisal", "waiting") == []
    assert _ids(client, "khalid", "waiting") == []  # the requester cannot decide
    assert _ids(client, "khalid", "mine") == [run_id]
    assert _ids(client, "lina", "mine") == [] and _ids(client, "lina", "waiting") == []

    _decide(client, run_id, "omar", "reject", "rfq")  # classic path: procurement gate next
    _decide(client, run_id, "omar", "approve", "procurement")
    assert _ids(client, "omar", "waiting") == []
    assert _ids(client, "faisal", "waiting") == [run_id]


def test_run_list_rows_hold_no_text(client):
    _agent_run(client)
    [row] = client.get("/workflows?scope=mine", headers=_as("khalid")).json()
    assert set(row) == {"run_id", "entry", "mode", "status", "pending_gate", "requester", "total", "updated"}
    stored = (config.DATA_DIR / "runs.json").read_text()
    assert "Glass block" not in stored and "boq.csv" not in stored


def test_unknown_scope_is_refused(client):
    assert client.get("/workflows?scope=everything", headers=_as("noura")).status_code == 422


def test_run_index_keeps_every_row_under_concurrent_writes():
    def write(n: int) -> None:
        run_index.record({"run_id": f"r{n:02d}", "status": "running"})

    threads = [threading.Thread(target=write, args=(n,)) for n in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(row["run_id"] for row in run_index.rows()) == [f"r{n:02d}" for n in range(20)]


# ---- chat: reading a message -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("What is waiting on me?", Intent.WAITING_ON_ME),
        ("anything waiting for my approval", Intent.WAITING_ON_ME),
        ("Show my cases", Intent.MY_CASES),
        ("approve it", Intent.DECIDE_REQUEST),
        ("Reject this please", Intent.DECIDE_REQUEST),
        ("What is the status of this case?", Intent.CASE_STATUS),
        ("Why is it waiting?", Intent.EXPLAIN_CASE),
        ("Please price this BOQ", Intent.START_CASE_HELP),
        ("help", Intent.HELP),
        ("How many leave days can I carry over?", Intent.ASK_DOCUMENTS),
    ],
)
def test_keyword_rules(message, intent):
    assert classify(message) is intent


# ---- chat: over HTTP ---------------------------------------------------------------------------


def test_waiting_on_me_shows_the_approval_card(client):
    run_id = _agent_run(client)
    body = _chat(client, "omar", "What is waiting on me?").json()
    assert body["intent"] == "waiting_on_me"
    [card] = [card for card in body["cards"] if card["kind"] == "rfq"]
    assert (card["run_id"], card["gate"], card["can_decide"], card["choices"]) == (run_id, "rfq", True, ["approve", "reject"])


def test_typed_approval_only_shows_the_card(client):
    run_id = _agent_run(client)
    body = _chat(client, "omar", "approve it", run_id).json()
    assert body["intent"] == "decide_request"
    assert any(card["kind"] == "rfq" for card in body["cards"])
    assert client.get(f"/workflows/{run_id}", headers=_as("omar")).json()["status"] == "awaiting_rfq"


def test_requester_sees_why_they_cannot_decide(client):
    run_id = _agent_run(client)
    [card] = [c for c in _chat(client, "khalid", "status", run_id).json()["cards"] if c["kind"] == "rfq"]
    assert card["can_decide"] is False
    assert "requester" in card["reason"]


def test_starting_a_case_depends_on_the_role(client):
    khalid = _chat(client, "khalid", "Please price this BOQ").json()
    sara = _chat(client, "sara", "Please price this BOQ").json()
    assert khalid["intent"] == sara["intent"] == "start_case_help"
    assert any(card["kind"] == "notice" and card.get("samples") for card in khalid["cards"])
    assert not any(card.get("samples") for card in sara["cards"])


def test_not_visible_looks_like_unknown(client):
    run_id = _agent_run(client)
    hidden = _chat(client, "lina", "status", run_id).json()
    unknown = _chat(client, "lina", "status", "000000000000").json()
    assert hidden == unknown


def test_documents_answer_with_citations_and_refusals_stay_plain(client):
    answer = _chat(client, "sara", "How many leave days can I carry over?").json()
    assert answer["intent"] == "ask_documents"
    [card] = answer["cards"]
    assert card["kind"] == "answer" and card["citations"]
    refused = _chat(client, "lina", "What does the board memo say?").json()
    assert [card["kind"] for card in refused["cards"]] == ["refusal"]
    assert "memo" not in refused["text"].lower()


def test_message_limit(client):
    assert _chat(client, "sara", "x" * 2001).status_code == 422
    assert _chat(client, "sara", "").status_code == 422


def test_chat_turn_audit_holds_no_message_text(client):
    # Document questions go through ask(), which audits the question as /ask always has.
    # Every other message leaves only ids, the intent and card kinds.
    _chat(client, "khalid", "show my cases, zebra-striped stapler")
    lines = [json.loads(line) for line in (config.DATA_DIR / "audit.jsonl").read_text().splitlines()]
    [turn] = [line for line in lines if line["event"] == "chat_turn"]
    assert set(turn) == {"ts", "event", "user", "intent", "run", "cards"}
    assert "zebra" not in (config.DATA_DIR / "audit.jsonl").read_text()


def test_done_case_shows_one_order_per_vendor(client, demo):
    run_id = _agent_run(client)
    _decide(client, run_id, "omar", "approve", "rfq")
    _event(client, run_id)
    _decide(client, run_id, "omar", "extend", "quotes")
    _event(client, run_id, sample="round2")
    _decide(client, run_id, "omar", "approve", "procurement")
    _decide(client, run_id, "faisal", "approve", "finance")
    cards = _chat(client, "faisal", "status", run_id).json()["cards"]
    [po] = [card for card in cards if card["kind"] == "po"]
    assert {order["vendor"]: order["total"] for order in po["orders"]}["Cobalt Dune Facades LLC"] == "7155.00"
    khalid = [card for card in _chat(client, "khalid", "status", run_id).json()["cards"] if card["kind"] == "po"][0]
    # The requester sees each PO's total (accepted: it shows the winning quote, never a losing one),
    # but still no quoted line prices.
    assert {order["vendor"]: order["total"] for order in khalid["orders"]}["Cobalt Dune Facades LLC"] == "7155.00"
    view = client.get(f"/workflows/{run_id}", headers=_as("khalid")).json()
    assert all("unit_price" not in line for line in view["lines"] if line.get("code") == "quote")


# ---- review fixes ---------------------------------------------------------------------------------


@pytest.fixture
def wf(system) -> WorkflowRunner:
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())


def test_run_list_reads_no_checkpoint_for_rows_that_cannot_match(wf, monkeypatch):
    from secure_rag.core.identity import get_user
    from secure_rag.workflow.workflow import RunMode, RunScope

    for _ in range(3):
        wf.start_boq(get_user("khalid"), ISSUES, mode=RunMode.AGENT)
    reads = {"n": 0}
    real = wf._graph.get_state

    def counted(*args, **kwargs):
        reads["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(wf._graph, "get_state", counted)
    assert wf.runs_for(get_user("lina"), RunScope.WAITING) == []
    assert wf.runs_for(get_user("lina"), RunScope.MINE) == []
    assert wf.runs_for(get_user("faisal"), RunScope.WAITING) == []
    assert reads["n"] == 0


def test_a_run_that_crashed_at_start_is_in_my_cases(wf, monkeypatch, tmp_path):
    from secure_rag.boq import boq
    from secure_rag.core.identity import get_user
    from secure_rag.workflow.workflow import RunScope

    def broken(runner, state):
        raise RuntimeError("matcher bug")

    # Undo this patch alone: monkeypatch.undo() would also drop the temp data folder.
    with monkeypatch.context() as patch:
        patch.setattr(boq, "match_step", broken)
        with pytest.raises(RuntimeError):
            wf.start_boq(get_user("khalid"), ISSUES)
    [row] = wf.runs_for(get_user("khalid"), RunScope.MINE)
    assert config.DATA_DIR == tmp_path  # the resume must not hold budget in the real data folder
    assert wf.resume(row.run_id, get_user("khalid")).status.value == "awaiting_procurement"


def test_private_run_cards_show_counts_only(system, wf, monkeypatch, tmp_path):
    from secure_rag.workflow.agent import Decision
    from secure_rag.web.chat import route_message
    from secure_rag.core.identity import get_user

    private = tmp_path / "private"
    private.mkdir()
    target = private / "boq.csv"
    target.write_bytes(ISSUES.read_bytes())
    monkeypatch.setattr(config, "PRIVATE_DIR", private)
    run_id = wf.start_boq(get_user("khalid"), target).run_id
    wf.decide(run_id, get_user("omar"), Decision.APPROVE)
    wf.decide(run_id, get_user("faisal"), Decision.APPROVE)
    [po] = [card for card in route_message(system, wf, get_user("faisal"), "status", run_id).cards if card["kind"] == "po"]
    assert po["orders"] == [] and po["count"] == 1  # classic: one purchase order for the run


def test_waiting_chat_skips_a_case_that_moved_on(system, wf, monkeypatch):
    from secure_rag.web.chat import route_message
    from secure_rag.core.identity import get_user
    from secure_rag.workflow.workflow import RunMode

    run_id = wf.start_boq(get_user("khalid"), ISSUES, mode=RunMode.AGENT).run_id
    rows = wf.runs_for(get_user("omar"), __import__("secure_rag.workflow.workflow", fromlist=["RunScope"]).RunScope.WAITING)
    from secure_rag.workflow.agent import Decision
    from secure_rag.workflow.workflow import Gate

    wf.decide(run_id, get_user("omar"), Decision.APPROVE, Gate.RFQ)  # moved on after the list was read
    monkeypatch.setattr(wf, "runs_for", lambda viewer, scope: rows)
    reply = route_message(system, wf, get_user("omar"), "what is waiting on me?")
    assert reply.cards == []


def test_a_damaged_run_index_does_not_break_decisions_or_lists(client):
    run_id = _agent_run(client)
    (config.DATA_DIR / "runs.json").write_text("{not json")
    response = _decide(client, run_id, "omar", "approve", "rfq")
    assert (response.status_code, response.json()["status"]) == (200, "awaiting_quotes")
    assert client.get("/workflows?scope=mine", headers=_as("khalid")).status_code == 200


def test_chat_run_id_must_look_like_an_id(client):
    assert _chat(client, "khalid", "status", "zebra-striped stapler; DROP TABLE x").status_code == 422


def test_a_decision_that_leaves_the_caller_unable_to_see_the_run_is_a_404(client, monkeypatch):
    run_id = _agent_run(client)
    workflows = client.app.state.workflows
    answers = iter([True, False])
    monkeypatch.setattr(workflows, "may_view", lambda run, user: next(answers))
    assert _decide(client, run_id, "omar", "approve", "rfq").status_code == 404


def test_private_run_gate_card_hides_packages(system, wf, monkeypatch, tmp_path):
    """Chat cards follow GET /workflows/{id}: no packages (items, vendors) for private files."""
    from secure_rag.web.chat import route_message
    from secure_rag.core.identity import get_user
    from secure_rag.workflow.workflow import RunMode

    private = tmp_path / "private"
    private.mkdir()
    target = private / "boq.csv"
    target.write_bytes(ISSUES.read_bytes())
    monkeypatch.setattr(config, "PRIVATE_DIR", private)
    wf.start_boq(get_user("khalid"), target, mode=RunMode.AGENT)
    [card] = route_message(system, wf, get_user("omar"), "what is waiting on me").cards
    assert card["kind"] == "rfq" and card["packages"] == []
