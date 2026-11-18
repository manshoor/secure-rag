"""The golden set passes offline with zero leaks."""

from secure_rag.config import GOLDEN_FILE
from secure_rag.evaluate import format_report, run_eval


def test_golden_set_passes_with_no_leaks(system):
    report = run_eval(system, GOLDEN_FILE)
    assert report.leaks == 0
    assert report.passed == len(report.results), format_report(report)
