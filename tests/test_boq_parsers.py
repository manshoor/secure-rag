"""Parser adapter: backend choice, LlamaParse guard rails (fake client only), boq-check report."""

import os
import sys
import types
from decimal import Decimal

import pytest

from secure_rag import config
from secure_rag.boq import check_report
from secure_rag.boq_parser import BoqParseError, ParserKind, is_private, markdown_table_rows, parse_boq

VILLA_XLSX = config.SAMPLES_DIR / "boq_villa_blockwork.xlsx"
LABELS = config.SAMPLES_DIR / "boq_villa_blockwork.labels.json"
MARKDOWN = """# Villa BOQ

| Item | Description | Unit | Qty | Rate | Amount |
|------|-------------|------|-----|------|--------|
| A | SUBSTRUCTURE BLOCKWORK | | | | |
| A1 | Solid concrete block wall 100mm thick below DPC | m2 | 85 | 1.00 | 85.00 |
| A2 | Damp proof course membrane under blockwork | lm | 1,200 | | |
"""


@pytest.fixture
def private_dir(monkeypatch, tmp_path):
    folder = tmp_path / "private"
    folder.mkdir()
    monkeypatch.setattr(config, "PRIVATE_DIR", folder)
    return folder


@pytest.fixture
def fake_llama(monkeypatch):
    """Replaces llama_cloud.LlamaCloud; records uploads and returns MARKDOWN."""
    uploads: list[str] = []

    class Page:
        markdown = MARKDOWN

    class Parsing:
        def parse(self, **kwargs):
            uploads.append(str(kwargs["upload_file"]))
            return types.SimpleNamespace(markdown=types.SimpleNamespace(pages=[Page()]))

    class FakeClient:
        def __init__(self, api_key):
            self.parsing = Parsing()

    monkeypatch.setitem(sys.modules, "llama_cloud", types.SimpleNamespace(LlamaCloud=FakeClient))
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "llx-test")
    monkeypatch.delenv("RAG_ALLOW_CLOUD_PARSE", raising=False)
    return uploads


def test_markdown_tables_become_lines_and_rates_are_ignored(fake_llama):
    lines = parse_boq(VILLA_XLSX, ParserKind.LLAMAPARSE)
    assert [(line.item_no, line.quantity, line.section) for line in lines] == [
        ("A1", Decimal("85"), "SUBSTRUCTURE BLOCKWORK"),
        ("A2", Decimal("1200"), "SUBSTRUCTURE BLOCKWORK"),
    ]
    assert fake_llama == [str(VILLA_XLSX)]


def test_llamaparse_refuses_private_files_without_explicit_yes(fake_llama, private_dir):
    target = private_dir / "real.pdf"
    target.write_bytes(b"%PDF-1.4")
    with pytest.raises(BoqParseError, match="RAG_ALLOW_CLOUD_PARSE=yes"):
        parse_boq(target, ParserKind.LLAMAPARSE)
    assert fake_llama == []


def test_llamaparse_private_file_allowed_only_with_yes(fake_llama, private_dir, monkeypatch):
    target = private_dir / "real.pdf"
    target.write_bytes(b"%PDF-1.4")
    monkeypatch.setenv("RAG_ALLOW_CLOUD_PARSE", "yes")
    parse_boq(target, ParserKind.LLAMAPARSE)
    assert fake_llama == [str(target)]


def test_symlink_into_private_is_still_private(private_dir, tmp_path):
    target = private_dir / "real.csv"
    target.write_text("x")
    link = tmp_path / "innocent.csv"
    link.symlink_to(target)
    assert is_private(link)
    assert is_private(tmp_path / "elsewhere" / ".." / "private" / "real.csv")
    assert not is_private(VILLA_XLSX)


def test_llamaparse_needs_a_key(monkeypatch):
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)
    with pytest.raises(BoqParseError, match="LLAMA_CLOUD_API_KEY"):
        parse_boq(VILLA_XLSX, ParserKind.LLAMAPARSE)


def test_parser_choice_comes_from_env(monkeypatch):
    monkeypatch.setenv("RAG_BOQ_PARSER", "tabular")
    assert len(parse_boq(VILLA_XLSX)) == 18
    monkeypatch.setenv("RAG_BOQ_PARSER", "nonsense")
    with pytest.raises(ValueError):
        parse_boq(VILLA_XLSX)


def test_tabular_parser_refuses_pdf():
    with pytest.raises(BoqParseError, match="docling"):
        parse_boq(config.SAMPLES_DIR / "boq_villa_blockwork.pdf", ParserKind.TABULAR)


def test_separator_rows_are_skipped():
    rows = markdown_table_rows("| a | b |\n|:--|--:|\n| 1 | 2 |")
    assert rows == [["a", "b"], ["1", "2"]]


def test_boq_check_reports_counts_and_accuracy():
    report = check_report(VILLA_XLSX, ParserKind.TABULAR, LABELS)
    assert report["lines"] == 18 and report["units_recognised"] == 18
    assert report["matched"] == 18 and report["unmatched"] == 0
    assert report["accuracy"] == {"labelled": 18, "found": 1.0, "description": 1.0, "unit": 1.0, "quantity": 1.0, "section": 1.0}
    assert "description" not in str({k: v for k, v in report.items() if k != "accuracy"})
