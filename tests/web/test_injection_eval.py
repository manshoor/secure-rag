"""The prompt-injection eval: planted text must never change an outcome. No paid calls."""

import pytest

from secure_rag.web import cli
from secure_rag.web.injection_eval import Kit, format_injection_report, run_injection_eval


@pytest.mark.parametrize("kit", [Kit.OFFLINE, Kit.HOSTILE])
def test_every_outcome_holds(system, kit):
    results = run_injection_eval(system, kit)
    assert len(results) >= 8
    unsafe = [f"{r.case}: {r.note}" for r in results if not r.safe]
    assert unsafe == []


def test_the_hostile_kit_really_obeys_the_planted_text(system):
    """A hostile model that did nothing would prove nothing: it must have tried."""
    results = {r.case: r for r in run_injection_eval(system, Kit.HOSTILE)}
    assert "tried" in results["chat_reworded_authority"].note
    assert "tried" in results["doc_hidden_memo"].note


def test_a_known_pattern_is_detected(system):
    results = {r.case: r for r in run_injection_eval(system, Kit.OFFLINE)}
    assert results["boq_known_pattern"].detected
    assert results["chat_typed_approval"].detected


def test_report_counts_safe_and_detected(system):
    report = format_injection_report(run_injection_eval(system, Kit.OFFLINE))
    assert "outcome held" in report and "planted text flagged" in report


def test_hostile_models_are_fresh_per_case(system):
    """A counter shared across cases could say 'tried' for a case where the model never ran."""
    results = {r.case: r for r in run_injection_eval(system, Kit.HOSTILE)}
    assert "tried" in results["doc_hidden_memo"].note and "refused" in results["doc_hidden_memo"].note


def test_the_eval_refuses_a_real_database(system, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/db")
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        run_injection_eval(system, Kit.OFFLINE)


def test_cli_runs_offline_by_default(capsys, monkeypatch):
    monkeypatch.setenv("RAG_LLM", "claude")  # even so, the default kit makes no model calls
    monkeypatch.setenv("RAG_EMBEDDER", "hash")
    monkeypatch.setattr("anthropic.Anthropic", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("a model client was built")))
    assert cli.main(["injection-eval"]) == 0
    out = capsys.readouterr().out
    assert "outcome held" in out and "FAIL" not in out


class _RecordingLangSmith:
    """Stands in for a LangSmith client: records runs, never touches the network."""

    def __init__(self) -> None:
        self.runs: list[dict] = []

    def create_run(self, **run) -> None:
        self.runs.append(run)

    def update_run(self, **run) -> None:
        self.runs.append(run)

    def flush(self, timeout: float | None = None) -> None:
        pass


def test_the_eval_sends_no_traces_and_restores_tracing(system):
    """Cases hold planted text: no trace leaves, and the clients in use before are in use after."""
    from secure_rag.core import tracing

    langfuse = object()  # any use would raise, failing a case
    langsmith = _RecordingLangSmith()
    tracing.configure(langfuse)
    tracing.configure_langsmith(langsmith)
    results = run_injection_eval(system, Kit.HOSTILE)
    assert all(r.safe for r in results) and langsmith.runs == []
    assert tracing._state["client"] is langfuse and tracing._state["langsmith"] is langsmith


def test_the_hostile_kit_tries_in_every_boq_case(system):
    """A BOQ case where the hostile planner and canonicaliser never ran would prove nothing."""
    results = {r.case: r for r in run_injection_eval(system, Kit.HOSTILE)}
    for case in ("boq_known_pattern", "boq_reworded", "boq_arabic"):
        assert "planner tried" in results[case].note and "matcher tried" in results[case].note, results[case].note


# ---- re-review: the eval must have teeth --------------------------------------------------------


class _PlannerAndRouterDown:
    """A Claude client that answers the composer and matcher but fails every planner and router call.

    That is the silent case: planner and router failures fall back to the rules quietly.
    """

    def complete_json(self, system: str, prompt: str, schema: dict):
        import anthropic
        import httpx

        from secure_rag.core.llm import Usage

        fields = schema["properties"]
        if "tool" in fields or "intent" in fields:
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        if "rewrites" in fields:
            return {"rewrites": []}, Usage()
        return {"answer": "", "citations": [], "refused": True}, Usage()


def test_a_claude_run_where_the_model_never_answered_is_not_a_pass(system, monkeypatch):
    """A paid run must not report rules-only steps as Claude resisting the planted text."""
    from secure_rag.web import injection_eval

    monkeypatch.setattr(injection_eval, "_claude_client", _PlannerAndRouterDown)
    results = {r.case: r for r in run_injection_eval(system, Kit.CLAUDE)}
    for case in ("boq_known_pattern", "chat_reworded_authority"):
        assert not results[case].safe and "did not answer" in results[case].note, results[case].note


def test_the_eval_notices_a_lost_allowed_now_check(system, monkeypatch):
    """With the 'tool not allowed now' check gone, a hostile out-of-order pick must fail a BOQ case."""
    from secure_rag.workflow import procurement_agent

    real = procurement_agent._authorize

    def without_allowed_check(state, proposal, allowed):
        known = [tool.value for tool in procurement_agent.Tool]
        return real(state, proposal, known)

    monkeypatch.setattr(procurement_agent, "_authorize", without_allowed_check)
    results = {r.case: r for r in run_injection_eval(system, Kit.HOSTILE)}
    for case in ("boq_known_pattern", "boq_reworded", "boq_arabic"):
        assert not results[case].safe, results[case].note


def test_models_are_built_once_per_case(system, monkeypatch):
    from secure_rag.web import injection_eval

    built = []
    real = injection_eval._models
    monkeypatch.setattr(injection_eval, "_models", lambda *args: built.append(1) or real(*args))
    results = run_injection_eval(system, Kit.HOSTILE)
    assert len(built) == len(results)


def test_a_vendor_id_counts_as_shown(system):
    from secure_rag.web.injection_eval import _case_shown

    assert _case_shown("winner V-GRC-1 at AED 900", "run-x") and _case_shown("see run-x", "run-x")
    assert not _case_shown("nothing here", "run-x")
