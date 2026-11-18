"""Evaluation over a golden set: retrieval, citations, refusals, leaks, cost, latency.

Each case names a user, a question and what must happen:
  expect=answer  -> the expected document is retrieved and cited, key facts appear
  expect=refuse  -> no answer (permission, scope or injection cases)
  forbid_docs / forbid_text -> must never be retrieved, cited or printed (leak check)
"""

import json
import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from . import db
from .answer import ask
from .app import System
from .config import ANSWER_TOP_K
from .identity import get_user


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
    retrieved = {hit.chunk.doc_id for hit in system.store.search(user, case["question"], ANSWER_TOP_K)}
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

    # Expected refusals: permission, scope and injection cases.
    if expect is Expect.REFUSE:
        report.refusal_total += 1
        if answer.refused:
            report.refusal_ok += 1
        else:
            notes.append("answered but should refuse")

    usage = answer.usage
    return CaseResult(
        case["id"], not notes, notes, answer.seconds, usage.cost_usd, usage.input_tokens + usage.output_tokens
    )


def format_report(report: EvalReport) -> str:
    """Human-readable table and summary."""
    lines = [f"{'case':<24} {'result':<6} {'ms':>6}  notes"]
    for result in report.results:
        mark = "PASS" if result.passed else "FAIL"
        lines.append(f"{result.case_id:<24} {mark:<6} {result.seconds * 1000:>6.0f}  {'; '.join(result.notes)}")

    latencies = sorted(result.seconds for result in report.results) or [0.0]
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    lines += [
        "",
        f"cases passed        {report.passed}/{len(report.results)}",
        f"retrieval hit rate  {_ratio(report.retrieval_hits, report.retrieval_total)}",
        f"citation accuracy   {_ratio(report.citation_ok, report.citation_total)}",
        f"correct refusals    {_ratio(report.refusal_ok, report.refusal_total)}",
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
    db.insert_eval_run(report.passed, len(report.results), report.leaks, asdict(report))
    return True


def _ratio(good: int, total: int) -> str:
    return f"{good}/{total}" if total else "n/a"
