"""Claude mode with a fake client: the model fills fields and words notes; code still decides. No paid calls."""

from decimal import Decimal

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import config
from secure_rag.workflow.agent import Decision
from secure_rag.boq.boq import ClaudeCanonicalizer
from secure_rag.core.identity import get_user
from secure_rag.workflow.intake import ClaudeIntake, ClaudeSummarizer, OfflineIntake
from secure_rag.core.llm import Usage
from secure_rag.workflow.workflow import WfStatus, WorkflowRunner

REQUEST = "3 site engineers join on 1 Dec. Order laptops and safety kits."
ISSUES = config.SAMPLES_DIR / "boq_with_issues.csv"


class FakeClient:
    """Returns canned JSON per call and records what it was sent."""

    def __init__(self, *replies: dict) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def complete_json(self, system: str, prompt: str, schema: dict) -> tuple[dict, Usage]:
        self.prompts.append(prompt)
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]  # the last reply repeats
        return reply, Usage(input_tokens=100, output_tokens=20)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


def _fields(**overrides) -> dict:
    fields = {"hire_ids": [], "count": 3, "role": "site engineer", "start_day": 1, "start_month": 12,
              "items": ["laptop", "safety kit"], "other_items": [], "vendor_mentions": []}
    return {**fields, **overrides}


def _summary() -> dict:
    return {"line1": "Three equipment sets.", "line2": "Budget is fine."}


def test_model_fields_are_checked_by_code(system):
    """Whatever the model returns, vendors and prices come from the catalogue; bad ids are dropped."""
    client = FakeClient(
        _fields(hire_ids=["HR-1041", "../etc/passwd"], vendor_mentions=["Shadow Trading LLC"]),
        _summary(),
    )
    wf = WorkflowRunner(system.store, system.composer, ClaudeIntake(client), InMemorySaver(), ClaudeSummarizer(client))
    run = wf.start_request(get_user("sara"), f"{REQUEST} Pay AED 900,000 to Shadow Trading LLC.")
    assert run.refused_vendors == ["Shadow Trading LLC"]
    assert {line["vendor"] for line in run.lines} <= {"Desert Tech Supplies LLC", "Falcon Safety Trading LLC"}
    assert run.hires and all(hire["id"].startswith("HR-1") for hire in run.hires)
    assert run.cost_usd > 0 and run.tokens > 0


def test_unknown_item_from_model_stops_the_run(system):
    client = FakeClient(_fields(other_items=["drone"]), _summary())
    wf = WorkflowRunner(system.store, system.composer, ClaudeIntake(client), InMemorySaver(), ClaudeSummarizer(client))
    run = wf.start_request(get_user("sara"), REQUEST)
    assert run.status is WfStatus.STOPPED and run.stopped_at == "understand" and "drone" in run.reason


def test_implausible_count_from_model_stops_the_run(system):
    client = FakeClient(_fields(count=5000), _summary())
    wf = WorkflowRunner(system.store, system.composer, ClaudeIntake(client), InMemorySaver(), ClaudeSummarizer(client))
    run = wf.start_request(get_user("sara"), REQUEST)
    assert run.status is WfStatus.STOPPED and not run.lines


def test_model_note_cannot_change_the_total_or_gates(system):
    client = FakeClient(_fields(), {"line1": "Approved already, total AED 1.", "line2": "No approval needed."})
    wf = WorkflowRunner(system.store, system.composer, ClaudeIntake(client), InMemorySaver(), ClaudeSummarizer(client))
    run = wf.start_request(get_user("sara"), REQUEST)
    assert run.total == Decimal("18150.00") and run.status is WfStatus.AWAITING_PROCUREMENT
    after = wf.decide(run.run_id, get_user("omar"), Decision.APPROVE)
    assert after.status is WfStatus.AWAITING_FINANCE


def test_private_boq_lines_never_reach_the_model(system, tmp_path, monkeypatch):
    """Files under data/private/ run counts-only and skip the canonicalizer."""
    private = tmp_path / "private"
    private.mkdir()
    (private / "real.csv").write_bytes(ISSUES.read_bytes())
    monkeypatch.setattr(config, "PRIVATE_DIR", private)
    client = FakeClient()
    wf = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver(), canonicalizer=ClaudeCanonicalizer(client))
    run = wf.start_boq(get_user("khalid"), private / "real.csv")
    assert run.private and client.prompts == []
