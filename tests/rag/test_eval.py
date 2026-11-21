"""The golden set passes offline with zero leaks."""

from secure_rag.core.config import GOLDEN_FILE
from secure_rag.rag.evaluate import format_report, run_eval


def test_golden_set_passes_with_no_leaks(system):
    report = run_eval(system, GOLDEN_FILE)
    assert report.leaks == 0
    assert report.passed == len(report.results), format_report(report)


def test_wrong_clause_and_wrong_fact_fail_without_counting_as_leaks(system, tmp_path):
    """'10 percent' is also the delay cap: the section and wrong-fact fields catch the wrong clause."""
    import json
    from dataclasses import replace

    from secure_rag.rag.answer import Draft

    class DelayClause:
        def compose(self, question, sources):
            delay = next(s for s in sources if s.hit.chunk.section == "Delay damages")
            return Draft(f"{delay.hit.chunk.text} [{delay.source_id}]", [delay.source_id], refused=False)

    case = {
        "id": "retention", "user": "omar", "expect": "answer",
        "question": "What retention percentage applies to payments under the steel subcontract?",
        "docs": ["alpha-subcontract-steelworks"], "sections": ["Payment terms"],
        "must_contain": ["10 percent"], "must_not_contain": ["0.1 percent"],
    }
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([case]))
    report = run_eval(replace(system, composer=DelayClause()), golden)
    assert report.passed == 0 and report.leaks == 0 and report.wrong_facts == 1
    assert report.section_ok == 0 and report.section_total == 1
    assert report.context_hits == 1 and report.correct_with_context == 0


def _refused_draft_report(system, tmp_path):
    import json
    from dataclasses import replace

    from secure_rag.rag.answer import Draft

    class BadNumber:
        def compose(self, question, sources):
            return Draft(f"Up to 15 unused days may be carried over [{sources[0].source_id}].", [sources[0].source_id], refused=False)

    case = {"id": "carry", "user": "sara", "expect": "answer", "question": "How many unused leave days can I carry over?",
            "docs": ["group-hr-annual-leave"], "must_contain": ["10"]}
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([case]))
    return run_eval(replace(system, composer=BadNumber()), golden)


def test_eval_prints_refused_draft_only_with_full_trace_content(system, tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_TRACE_CONTENT", raising=False)
    assert "15 unused" not in format_report(_refused_draft_report(system, tmp_path))

    monkeypatch.setenv("RAG_TRACE_CONTENT", "full")
    assert "15 unused" in format_report(_refused_draft_report(system, tmp_path))


def test_saved_eval_run_holds_no_draft_text(system, tmp_path, monkeypatch):
    from secure_rag.rag import evaluate

    monkeypatch.setenv("RAG_TRACE_CONTENT", "full")
    report = _refused_draft_report(system, tmp_path)
    saved = {}
    monkeypatch.setattr(evaluate.db, "enabled", lambda: True)
    monkeypatch.setattr(evaluate.db, "create_schema", lambda: None)
    monkeypatch.setattr(evaluate.db, "insert_eval_run", lambda *args: saved.update(row=args[-1]))
    evaluate.save_run(report)
    assert "15 unused" not in str(saved["row"])
