"""Evaluation over a golden set: retrieval, citations, refusals, leaks, cost, latency.

Each case names a user, a question and what must happen:
  expect=answer  -> the expected document is retrieved and cited, key facts appear,
                    an expected section is cited, and no wrong fact appears
  expect=refuse  -> no answer (permission, scope or injection cases)
  forbid_docs / forbid_text -> must never be retrieved, cited or printed (leak check)

"Correct when the context has the facts" counts answers that pass among cases
whose retrieved chunks contain every expected fact: right sources, right answer.
"""

import json
import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from ..core import db
from .answer import ask
from ..web.app import System
from ..core.config import ANSWER_TOP_K
from ..core.identity import get_user


class Expect(str, Enum):
    ANSWER = "answer"
    REFUSE = "refuse"
    ANY = "any"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    notes: list[str]
    seconds: float
    cost_usd: float
    tokens: int
    debug: str = ""  # refused draft, only with RAG_TRACE_CONTENT=full


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)
    retrieval_hits: int = 0
    retrieval_total: int = 0
    citation_ok: int = 0
    citation_total: int = 0
    refusal_ok: int = 0
    refusal_total: int = 0
    leaks: int = 0
    wrong_facts: int = 0
    section_ok: int = 0
    section_total: int = 0
    context_hits: int = 0  # retrieved chunks hold every expected fact
    context_total: int = 0
    correct_with_context: int = 0

    @property
    def passed(self) -> int:
        return sum(result.passed for result in self.results)


def run_eval(system: System, golden: Path) -> EvalReport:
    """Run every golden case and collect metrics."""
    report = EvalReport()
    for case in json.loads(golden.read_text(encoding="utf-8")):
        report.results.append(_run_case(system, case, report))
    return report


def _run_case(system: System, case: dict, report: EvalReport) -> CaseResult:
    user = get_user(case["user"])
    expect = Expect(case["expect"])
    notes: list[str] = []

    # Retrieval is measured on its own, before the answer step.
    hits = system.store.search(user, case["question"], ANSWER_TOP_K)
    retrieved = {hit.chunk.doc_id for hit in hits}
    context = " ".join(hit.chunk.text for hit in hits).lower()
    answer = ask(system.store, system.composer, user, case["question"])
    cited = {citation.doc_id for citation in answer.citations}

    # Leaks: a forbidden document retrieved or cited, or forbidden text printed.
    for doc in case.get("forbid_docs", []):
        if doc in retrieved or doc in cited:
            notes.append(f"LEAK: {doc} reached the answer step")
    for text in case.get("forbid_text", []):
        if text.lower() in answer.text.lower():
            notes.append(f"LEAK: printed '{text}'")
    report.leaks += sum(note.startswith("LEAK") for note in notes)

    # Expected answers: retrieval hit, citations, key facts.
    if expect is Expect.ANSWER:
        report.retrieval_total += 1
        report.retrieval_hits += bool(retrieved & set(case["docs"]))
        if answer.refused:
            notes.append(f"refused: {answer.reason}")
        else:
            report.citation_total += 1
            if cited and cited <= set(case["docs"]):
                report.citation_ok += 1
            else:
                notes.append(f"cited {sorted(cited)}, expected {case['docs']}")
            for fact in case.get("must_contain", []):
                if fact.lower() not in answer.text.lower():
                    notes.append(f"missing '{fact}'")

            # The right document can still give the wrong clause: an expected section must be cited.
            if case.get("sections"):
                report.section_total += 1
                if {citation.section for citation in answer.citations} & set(case["sections"]):
                    report.section_ok += 1
                else:
                    notes.append(f"cited sections {sorted({c.section for c in answer.citations})}, expected {case['sections']}")

        # Wrong facts are answer errors, never leaks.
        for fact in case.get("must_not_contain", []):
            if fact.lower() in answer.text.lower():
                notes.append(f"wrong fact '{fact}'")
                report.wrong_facts += 1

    # Expected refusals: permission, scope and injection cases.
    if expect is Expect.REFUSE:
        report.refusal_total += 1
        if answer.refused:
            report.refusal_ok += 1
        else:
            notes.append("answered but should refuse")

    # Right sources, wrong answer: the facts were in the retrieved chunks but the case failed.
    facts = case.get("must_contain", [])
    if expect is Expect.ANSWER and facts:
        report.context_total += 1
        if all(fact.lower() in context for fact in facts):
            report.context_hits += 1
            report.correct_with_context += not notes

    usage = answer.usage
    return CaseResult(
        case["id"], not notes, notes, answer.seconds, usage.cost_usd, usage.input_tokens + usage.output_tokens,
        answer.debug if notes else "",
    )


def format_report(report: EvalReport) -> str:
    """Human-readable table and summary."""
    lines = [f"{'case':<24} {'result':<6} {'ms':>6}  notes"]
    for result in report.results:
        mark = "PASS" if result.passed else "FAIL"
        lines.append(f"{result.case_id:<24} {mark:<6} {result.seconds * 1000:>6.0f}  {'; '.join(result.notes)}")

        # A refused draft, indented under its case, when RAG_TRACE_CONTENT=full.
        lines += [f"{'':<32}{line}" for line in result.debug.splitlines()]

    latencies = sorted(result.seconds for result in report.results) or [0.0]
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    lines += [
        "",
        f"cases passed        {report.passed}/{len(report.results)}",
        f"retrieval hit rate  {_ratio(report.retrieval_hits, report.retrieval_total)}",
        f"citation accuracy   {_ratio(report.citation_ok, report.citation_total)}",
        f"section accuracy    {_ratio(report.section_ok, report.section_total)}",
        f"context recall      {_ratio(report.context_hits, report.context_total)}",
        f"correct w/ context  {_ratio(report.correct_with_context, report.context_hits)}",
        f"correct refusals    {_ratio(report.refusal_ok, report.refusal_total)}",
        f"wrong facts         {report.wrong_facts}",
        f"leaks               {report.leaks}",
        f"latency mean / p95  {statistics.mean(latencies) * 1000:.0f} ms / {p95 * 1000:.0f} ms",
        f"tokens / cost       {sum(r.tokens for r in report.results)} / ${sum(r.cost_usd for r in report.results):.4f}",
    ]
    return "\n".join(lines)


def save_run(report: EvalReport) -> bool:
    """Store the run in Postgres for a pass/leak history. False when Postgres is off."""
    if not db.enabled():
        return False
    db.create_schema()
    row = asdict(report)

    # Draft text stays out of the stored history.
    for result in row["results"]:
        result.pop("debug", None)
    db.insert_eval_run(report.passed, len(report.results), report.leaks, row)
    return True


def _ratio(good: int, total: int) -> str:
    return f"{good}/{total}" if total else "n/a"
