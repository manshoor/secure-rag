"""LangSmith runs mirror the Langfuse spans: ids and counts, no text unless asked,
and LangChain's own tracing (whole graph state) never turns on."""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.tracers.context import _tracing_v2_is_enabled
from langgraph.checkpoint.memory import InMemorySaver
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from secure_rag.core import config, tracing
from secure_rag.rag.answer import ask
from secure_rag.core.identity import get_user
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.workflow import Decision, Privacy, RunMode, WorkflowRunner

_QUESTION = "How many leave days can I carry over?"
_ISSUES = config.SAMPLES_DIR / "boq_with_issues.csv"


class _FakeLangSmith:
    """Records what a langsmith.Client would send; never touches the network."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.flushed = 0
        self.flush_timeouts: list[float | None] = []

    def create_run(self, **run) -> None:
        self.created.append(run)

    def update_run(self, **run) -> None:
        self.updated.append(run)

    def flush(self, timeout: float | None = None) -> None:
        self.flushed += 1
        self.flush_timeouts.append(timeout)

    def sent(self) -> str:
        return " ".join(str(run) for run in self.created + self.updated)


@pytest.fixture
def runs(monkeypatch):
    monkeypatch.delenv("RAG_TRACE_CONTENT", raising=False)
    fake = _FakeLangSmith()
    tracing.configure_langsmith(fake)
    return fake


# ---- switching on -------------------------------------------------------------------------------


@pytest.fixture
def no_langfuse(monkeypatch):
    """reset() re-reads env; keys from the shell or .env must not turn Langfuse on."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_off_without_the_key(monkeypatch, no_langfuse):
    monkeypatch.delenv("LANGSMITH_KEY", raising=False)
    tracing.reset()
    assert not tracing.enabled()


def test_key_turns_it_on(monkeypatch, no_langfuse):
    seen = {}
    monkeypatch.setenv("LANGSMITH_KEY", "lsv2-test")
    monkeypatch.setattr(tracing, "make_langsmith_client", lambda key: seen.setdefault("key", key) and _FakeLangSmith())
    tracing.reset()
    assert tracing.enabled() and seen["key"] == "lsv2-test"


def test_langchain_auto_tracing_stays_off(runs, monkeypatch):
    """LangChain's tracer would send whole graph state (BOQ lines) to LangSmith."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    assert not _tracing_v2_is_enabled()
    with tracing.observe("ask"):
        assert not _tracing_v2_is_enabled()

    # LangGraph runs nodes on pool threads, which see the global switch, not this context.
    with ThreadPoolExecutor(1) as pool:
        assert not pool.submit(_tracing_v2_is_enabled).result()
    thread_result = []
    thread = threading.Thread(target=lambda: thread_result.append(_tracing_v2_is_enabled()))
    thread.start()
    thread.join()
    assert thread_result == [False]


def test_a_bad_langsmith_setting_turns_it_off(monkeypatch, no_langfuse):
    """A tracing typo must not stop asks and runs."""
    def broken(key):
        raise ValueError("bad LANGSMITH_* setting")

    monkeypatch.setenv("LANGSMITH_KEY", "lsv2-test")
    monkeypatch.setattr(tracing, "make_langsmith_client", broken)
    tracing.reset()
    assert not tracing.enabled()
    with tracing.observe("ask") as span:
        span.update(metadata={"ok": True})


def test_flush_waits_a_bounded_time(runs):
    tracing.flush()
    assert runs.flush_timeouts == [tracing.FLUSH_SECONDS]


# ---- what is sent -------------------------------------------------------------------------------


def test_ask_sends_nested_runs_without_text(system, runs):
    answer = ask(system.store, system.composer, get_user("sara"), _QUESTION)
    tracing.flush()
    by_name = {run["name"]: run for run in runs.created}
    assert {"ask", "retrieve", "output_check"} <= set(by_name)
    assert len({run["trace_id"] for run in runs.created}) == 1
    assert by_name["retrieve"]["parent_run_id"] == by_name["ask"]["id"]
    assert by_name["retrieve"]["run_type"] == "retriever"
    assert len(runs.updated) == len(runs.created) and runs.flushed == 1

    sent = runs.sent()
    assert "group-hr-annual-leave" in sent  # chunk ids are traced
    assert "sara" in sent  # user id is traced
    assert _QUESTION not in sent and answer.text not in sent  # text is not


def test_full_content_mode_adds_text(system, runs, monkeypatch):
    monkeypatch.setenv("RAG_TRACE_CONTENT", "full")
    ask(system.store, system.composer, get_user("sara"), _QUESTION)
    assert _QUESTION in runs.sent()


def test_refused_question_sends_no_forbidden_chunks(system, runs):
    answer = ask(system.store, system.composer, get_user("omar"), "What does the board memo say?")
    assert answer.refused and "group-board-memo" not in runs.sent()


def test_workflow_runs_hold_no_line_text(system, runs, monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")
    workflows = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    workflows.start_boq(get_user("khalid"), _ISSUES)
    names = {run["name"] for run in runs.created}
    assert "workflow_start" in names
    assert "Decorative GRC cornice" not in runs.sent()


def test_an_error_sends_its_type_only(runs):
    with pytest.raises(ValueError):
        with tracing.observe("ask"):
            raise ValueError("secret line text")
    assert "ValueError" in runs.sent() and "secret line text" not in runs.sent()


def test_both_backends_together(system, runs):
    exporter = InMemorySpanExporter()
    tracing.configure(
        tracing.make_client(
            public_key=f"pk-lf-{uuid.uuid4().hex}",
            secret_key="sk-lf-test",
            base_url="http://127.0.0.1:9",
            tracer_provider=TracerProvider(),
            span_exporter=exporter,
        )
    )
    answer = ask(system.store, system.composer, get_user("sara"), _QUESTION)
    tracing.flush()
    assert answer.trace_id  # the Langfuse trace id, as before
    assert "ask" in {span.name for span in exporter.get_finished_spans()}
    assert "ask" in {run["name"] for run in runs.created}


def test_child_runs_carry_their_own_user(runs):
    """The agent's policy lookup runs as the agent inside a person's run."""
    with tracing.attributes("omar", ["workflow"]), tracing.observe("workflow_decide", metadata={"decision": "approve"}):
        with tracing.attributes("agent", ["rag"]), tracing.observe("ask"):
            pass
    by_name = {run["name"]: run for run in runs.created}
    child = by_name["ask"]
    assert child["parent_run_id"] == by_name["workflow_decide"]["id"]
    assert child["extra"]["metadata"]["user_id"] == "agent" and child["tags"] == ["rag"]
    assert "decision" not in child["extra"]["metadata"]  # no parent fields copied down


# ---- text from runs: reasons, and private runs -------------------------------------------------
# Both backends get the same fields; errors are checked in each.

_UNKNOWN = "3 site engineers join on 1 Dec. Order laptops and unobtainium widgets."


def test_a_stop_reason_is_text(system, runs, monkeypatch):
    """A stop reason can quote the request; it is text, sent only in full mode."""
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")
    workflows = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    view = workflows.start_request(get_user("sara"), _UNKNOWN)
    assert "unobtainium" in view.reason
    assert "unobtainium" not in runs.sent()

    monkeypatch.setenv("RAG_TRACE_CONTENT", "full")
    workflows.start_request(get_user("sara"), _UNKNOWN)
    assert "unobtainium" in runs.sent()


def _texts(runs) -> list:
    """Every input and output sent, empty ones left out."""
    return [run[key] for run in runs.created + runs.updated for key in ("inputs", "outputs") if run.get(key)]


@pytest.mark.parametrize("mode", [RunMode.CLASSIC, RunMode.AGENT])
def test_a_private_run_sends_no_text_even_in_full_mode(system, runs, monkeypatch, mode):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")
    monkeypatch.setenv("RAG_TRACE_CONTENT", "full")
    workflows = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    view = workflows.start_boq(get_user("khalid"), _ISSUES, Privacy.COUNTS_ONLY, mode)
    workflows.view(view.run_id, get_user("omar"))  # an approver's view runs the policy lookup too
    assert {run["name"] for run in runs.created} >= {"workflow_start", "ask"}
    assert _texts(runs) == []
    assert {_session(run) for run in runs.created} == {view.run_id}  # the case id is still sent


def test_a_private_decision_sends_no_reason(system, runs, monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")
    monkeypatch.setenv("RAG_TRACE_CONTENT", "full")
    workflows = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    view = workflows.start_boq(get_user("khalid"), _ISSUES, Privacy.COUNTS_ONLY)
    view = workflows.decide(view.run_id, get_user("omar"), Decision.REJECT)
    assert view.reason and view.reason not in runs.sent()


def test_langfuse_error_sends_its_type_only():
    exporter = InMemorySpanExporter()
    client = tracing.make_client(
        public_key=f"pk-lf-{uuid.uuid4().hex}",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    tracing.configure(client)
    with pytest.raises(ValueError, match="secret line text"):  # the app still sees the message
        with tracing.observe("ask"):
            raise ValueError("secret line text")
    client.flush()
    (span,) = exporter.get_finished_spans()
    sent = f"{dict(span.attributes)} {span.status} {[(e.name, dict(e.attributes)) for e in span.events]}"
    assert "ValueError" in sent and "secret line text" not in sent


# ---- one case, one thread ----------------------------------------------------------------------


def _session(run) -> str | None:
    return run["extra"]["metadata"].get("session_id")


def test_every_run_of_a_case_shares_one_thread(system, runs, monkeypatch):
    """Start, views and decisions are separate traces; session_id joins them as one LangSmith thread."""
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")
    workflows = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    view = workflows.start_boq(get_user("khalid"), _ISSUES, mode=RunMode.AGENT)
    workflows.view(view.run_id, get_user("omar"))
    workflows.decide(view.run_id, get_user("omar"), Decision.APPROVE)
    assert {run["name"] for run in runs.created} >= {"workflow_start", "workflow_decide", "agent_step", "ask", "retrieve"}
    assert len({run["trace_id"] for run in runs.created}) > 1
    assert {_session(run) for run in runs.created} == {view.run_id}


def test_two_cases_are_two_threads(system, runs, monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")
    workflows = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    first = workflows.start_boq(get_user("khalid"), _ISSUES, mode=RunMode.AGENT)
    second = workflows.start_boq(get_user("khalid"), _ISSUES, mode=RunMode.AGENT)
    assert {_session(run) for run in runs.created} == {first.run_id, second.run_id}


def test_a_plain_question_has_no_thread(system, runs):
    ask(system.store, system.composer, get_user("sara"), _QUESTION)
    assert {_session(run) for run in runs.created} == {None}


# ---- model calls: LangSmith's own token and cost columns -----------------------------------------


def _generate(prompt: str = "secret prompt") -> None:
    """The calls ClaudeClient.complete_json makes around one model call."""
    with tracing.observe("generate", as_type="generation", model="claude-opus-5-5", input=tracing.content(prompt)) as span:
        span.update(
            model="claude-opus-5-5",
            usage_details={"input": 1000, "output": 50},
            cost_details={"total": 0.005},
            metadata={"stop_reason": "end_turn"},
        )
        span.update(output=tracing.content("secret reply"))


def test_a_model_call_fills_the_token_and_cost_columns(runs):
    _generate()
    (run,) = runs.updated
    sent = run["extra"]["metadata"]
    assert sent["usage_metadata"] == {
        "input_tokens": 1000,
        "output_tokens": 50,
        "total_tokens": 1050,
        "input_cost": pytest.approx(0.004),
        "output_cost": pytest.approx(0.001),
        "total_cost": 0.005,
    }
    assert (sent["ls_provider"], sent["ls_model_name"]) == ("anthropic", "claude-opus-5-5")
    assert runs.created[0]["run_type"] == "llm"

    # The keys sent before stay.
    assert sent["usage_details"] == {"input": 1000, "output": 50} and sent["cost_details"] == {"total": 0.005}
    assert sent["model"] == "claude-opus-5-5" and sent["stop_reason"] == "end_turn"


def test_usage_without_cost_sends_tokens_only(runs):
    with tracing.observe("generate", as_type="generation") as span:
        span.update(usage_details={"input": 3, "output": 2})
    (run,) = runs.updated
    assert run["extra"]["metadata"]["usage_metadata"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def test_a_run_without_usage_sends_none(system, runs):
    ask(system.store, system.composer, get_user("sara"), _QUESTION)
    for run in runs.created + runs.updated:
        assert not {"usage_metadata", "ls_provider", "ls_model_name"} & set(run["extra"]["metadata"])


@pytest.mark.parametrize("private", [False, True])
def test_usage_adds_no_text(runs, monkeypatch, private):
    """ids mode, and a private run even in full mode."""
    if private:
        monkeypatch.setenv("RAG_TRACE_CONTENT", "full")
        with tracing.counts_only():
            _generate()
    else:
        _generate()
    assert _texts(runs) == [] and "secret" not in runs.sent()
    assert runs.updated[0]["extra"]["metadata"]["usage_metadata"]["total_tokens"] == 1050
