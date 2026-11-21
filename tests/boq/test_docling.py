"""Case 12: the sample BOQ as a PDF parses with Docling to the same lines as the xlsx.

Runs only with the docling extra installed (uv sync --extra docling). The first
run downloads Docling's layout and table models.
"""

import pytest

from secure_rag.core import config
from secure_rag.boq.boq_parser import ParserKind, parse_boq

pytest.importorskip("docling")
pytestmark = pytest.mark.docling


def _key(line):
    return (line.item_no, " ".join(line.description.split()), line.unit, line.quantity, line.section)


def test_docling_pdf_matches_the_xlsx():
    from_xlsx = parse_boq(config.SAMPLES_DIR / "boq_villa_blockwork.xlsx", ParserKind.TABULAR)
    from_pdf = parse_boq(config.SAMPLES_DIR / "boq_villa_blockwork.pdf", ParserKind.DOCLING)
    assert [_key(line) for line in from_pdf] == [_key(line) for line in from_xlsx]
