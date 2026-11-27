"""Part 3b traces (QA F6): what a specialist sends to LangSmith in full trace mode.

A counts-only agent run is driven to the procurement gate and viewed: the policy
lookup still opens its spans, no agent span opens and no text is sent. A normal run
is the positive control: text is sent, and a marker reason in the scripted finish
stays in the saved outcomes, never in what is sent.
"""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import agent_runtime, config, tracing
from secure_rag.core.identity import get_user
from secure_rag.workflow import workflow
from secure_rag.workflow.agent import Decision
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.workflow import Event, Gate, Privacy, RunMode, WorkflowRunner

_ISSUES = config.SAMPLES_DIR / "boq_with_issues.csv"
_QUOTES = config.SAMPLES_DIR / "quotes"
_MARKER = "Tracemarker: chase nobody."  # no digits, so the screen keeps it
_AGENT_SPANS = {"agent_turn", "agent_tool"}


class _Sink:
    """Records what a langsmith.Client would send; never touches the network."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.updated: list[dict] = []

    def create_run(self, **run) -> None:
        self.created.append(run)

    def update_run(self, **run) -> None:
        self.updated.append(run)

    def flush(self, timeout: float | None = None) -> None:
        return None

    def names(self) -> set[str]:
        return {run["name"] for run in self.created}

    def texts(self) -> list:
        return [run[key] for run in self.created + self.updated for key in ("inputs", "outputs") if run.get(key)]

    def sent(self) -> str:
        return " ".join(str(run) for run in self.created + self.updated)


@pytest.fixture
def sink(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")
    monkeypatch.setenv("RAG_TRACE_CONTENT", "full")
    found = _Sink()
    tracing.configure_langsmith(found)
    return found


@pytest.fixture
def marked(monkeypatch):
    """Every scripted finish carries the marker as its reason."""
    real = agent_runtime.ScriptedClient.from_steps.__func__

    def with_marker(cls, steps, output):
        return real(cls, steps, {**output, "reason": _MARKER})

    monkeypatch.setattr(agent_runtime.ScriptedClient, "from_steps", classmethod(with_marker))


def _round(name: str) -> list:
    return sorted((path.stem, path) for path in (_QUOTES / name).glob("*.csv"))


def _to_procurement(wf, privacy: Privacy) -> str:
    """Start, view at each gate, approve, both rounds of quotes, extend once."""
    omar = get_user("omar")
    run_id = wf.start_boq(get_user("khalid"), _ISSUES, privacy, RunMode.AGENT).run_id
    wf.view(run_id, omar)
    wf.decide(run_id, omar, Decision.APPROVE, Gate.RFQ)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, omar, _round("round1"))
    wf.view(run_id, omar)
    wf.decide(run_id, omar, Decision.EXTEND, Gate.QUOTES)
    wf.deliver(run_id, Event.QUOTES_DELIVERED, omar, _round("round2"))
    wf.view(run_id, omar)
    return run_id


def _outcomes(wf, run_id) -> dict:
    return dict(wf._graph.get_state(workflow._thread(run_id)).values)["outcomes"]


def test_a_private_agent_run_sends_no_text_in_full_mode(system, sink, marked):
    wf = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    run_id = _to_procurement(wf, Privacy.COUNTS_ONLY)
    outcomes = _outcomes(wf, run_id)
    assert len(outcomes) == 4 and {found["status"] for found in outcomes.values()} == {"private"}
    assert {"ask", "retrieve"} <= sink.names()
    assert not _AGENT_SPANS & sink.names()
    assert sink.texts() == []
    assert _MARKER not in sink.sent()


def test_a_normal_agent_run_sends_text_but_no_reason(system, sink, marked):
    """The positive control: the same drive in a normal run opens agent spans and sends text."""
    wf = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    run_id = _to_procurement(wf, Privacy.NORMAL)
    outcomes = _outcomes(wf, run_id)
    assert len(outcomes) == 4 and {found["reason"] for found in outcomes.values()} == {_MARKER}
    assert _AGENT_SPANS <= sink.names()
    assert sink.texts() != []
    assert _MARKER not in sink.sent()
    assert _MARKER not in json.dumps(sink.created + sink.updated, default=str)
