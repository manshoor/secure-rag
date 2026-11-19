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


def test_excel_that_unzips_too_large_is_refused(tmp_path):
    """A small .xlsx that expands to 30 MB is stopped before openpyxl reads it."""
    import zipfile

    bomb = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/sharedStrings.xml", b"0" * 30_000_000)
    assert bomb.stat().st_size < 1_000_000
    with pytest.raises(BoqParseError, match="too large"):
        parse_boq(bomb, ParserKind.TABULAR)


def test_excel_xml_parsing_is_hardened():
    """openpyxl uses defusedxml when it is installed, which blocks entity expansion."""
    from openpyxl.xml import DEFUSEDXML

    assert DEFUSEDXML


def _lying_workbook(path, real_bytes: int, declared: int) -> None:
    """A zip whose central directory claims a small size for a large entry."""
    import io
    import struct
    import zipfile
    import zlib

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"\0" * real_bytes)
    raw = bytearray(buffer.getvalue())
    crc = zlib.crc32(b"\0" * declared)
    struct.pack_into("<L", raw, 14, crc)
    struct.pack_into("<L", raw, 22, declared)
    central = raw.rfind(b"PK\x01\x02")
    struct.pack_into("<L", raw, central + 16, crc)
    struct.pack_into("<L", raw, central + 24, declared)
    path.write_bytes(raw)


def test_excel_that_lies_about_its_size_is_refused_without_inflating_it(tmp_path):
    """Declares 100 bytes, holds 300 MB: the real size is counted while streaming."""
    import tracemalloc

    bomb = tmp_path / "lying.xlsx"
    _lying_workbook(bomb, 300_000_000, 100)
    tracemalloc.start()
    with pytest.raises(BoqParseError):
        parse_boq(bomb, ParserKind.TABULAR)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 50_000_000


def test_docling_backend_checks_excel_size_first(tmp_path, monkeypatch):
    import zipfile

    from secure_rag import boq_parser

    bomb = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/sharedStrings.xml", b"0" * 30_000_000)
    monkeypatch.setitem(sys.modules, "docling", None)  # the size check must come before any import
    with pytest.raises(BoqParseError, match="too large"):
        boq_parser._docling_rows(bomb)


def test_real_sheet_over_two_megabytes_still_parses(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "long.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.append(["Item", "Description", "Unit", "Qty"])
    for index in range(25_000):
        sheet.append([f"A{index}", f"Hollow concrete block 200 mm, wall type {index}", "m2", index + 1])
    book.save(path)
    assert parse_boq(path, ParserKind.TABULAR)


def test_excel_reader_over_its_memory_limit_is_stopped(tmp_path, monkeypatch):
    """XML that parses into a large tree: the child process is killed, the parent stays small."""
    import zipfile

    from secure_rag import boq_parser

    heavy = tmp_path / "heavy.xlsx"
    with zipfile.ZipFile(heavy, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types>" + b"<a/>" * 2_000_000 + b"</Types>")
    monkeypatch.setattr(boq_parser, "_CHILD_MEMORY_BYTES", 150_000_000)
    with pytest.raises(BoqParseError, match="too large to read"):
        parse_boq(heavy, ParserKind.TABULAR)


def test_excel_reader_that_runs_too_long_is_stopped(monkeypatch):
    from secure_rag import boq_parser

    monkeypatch.setattr(boq_parser, "_CHILD_SECONDS", 0.01)
    with pytest.raises(BoqParseError, match="took too long"):
        parse_boq(config.SAMPLES_DIR / "boq_villa_blockwork.xlsx", ParserKind.TABULAR)


def test_excel_reader_errors_hold_no_file_text(tmp_path):
    """A private BOQ's text never reaches an error message."""
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"PK\x03\x04 Secret Villa Client Name")
    with pytest.raises(BoqParseError) as raised:
        parse_boq(broken, ParserKind.TABULAR)
    assert "Secret" not in str(raised.value)


def test_excel_reader_output_is_capped(monkeypatch):
    """One shared string in every cell is small XML but huge output: the child stops first."""
    from secure_rag import boq_parser

    monkeypatch.setattr(boq_parser, "_MAX_OUTPUT_BYTES", 1000)
    with pytest.raises(BoqParseError, match="too large to read"):
        parse_boq(config.SAMPLES_DIR / "boq_villa_blockwork.xlsx", ParserKind.TABULAR)


def test_excel_reader_cells_are_cut_to_the_description_limit(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "long_cell.xlsx"
    book = Workbook()
    book.active.append(["Item", "Description", "Unit", "Qty"])
    book.active.append(["1", "x" * 5000, "m2", 10])
    book.save(path)
    from secure_rag.boq_parser import _xlsx_rows_in_child

    assert max(len(cell) for row in _xlsx_rows_in_child(path) for cell in row) <= 500


def test_excel_reader_does_not_load_numpy():
    """numpy's maths library starts a thread per CPU on import, which a memory rlimit can refuse."""
    import subprocess

    code = "import sys, secure_rag._xlsx_child as c; c._block_heavy_imports(); import openpyxl; print(sys.modules.get('numpy') is None)"
    assert subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True).stdout.strip() == "True"


def test_excel_readers_at_once_are_limited(monkeypatch):
    import threading

    from secure_rag import boq_parser

    monkeypatch.setattr(boq_parser, "_CHILD_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(boq_parser, "_SLOT_WAIT_SECONDS", 0.05)
    boq_parser._CHILD_SLOTS.acquire()
    with pytest.raises(BoqParseError, match="busy"):
        parse_boq(config.SAMPLES_DIR / "boq_villa_blockwork.xlsx", ParserKind.TABULAR)


def test_child_reports_too_large_when_the_reply_does_not_fit(monkeypatch, capsys):
    import json

    from secure_rag import _xlsx_child

    def no_memory(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(_xlsx_child.json, "dumps", no_memory)
    monkeypatch.setattr(_xlsx_child, "_limit_memory", lambda limit: None)
    monkeypatch.setattr(_xlsx_child, "_block_heavy_imports", lambda: None)  # in-process: keep numpy for other tests
    _xlsx_child.main([str(config.SAMPLES_DIR / "boq_villa_blockwork.xlsx"), "100", "40", "20000000", "512000000", "1000000", "500"])
    assert json.loads(capsys.readouterr().out) == {"error": "too_large"}


def test_excel_reader_output_cap_counts_encoded_bytes(tmp_path, monkeypatch):
    """Non-ASCII text grows when encoded: the cap counts bytes sent, not characters."""
    from openpyxl import Workbook

    from secure_rag import boq_parser

    path = tmp_path / "wide.xlsx"
    book = Workbook()
    book.active.append(["Item", "Description", "Unit", "Qty"])
    for index in range(20):
        book.active.append([str(index), "\U0001f9f1" * 40, "m2", 1])  # 40 characters, 160 bytes in UTF-8
    book.save(path)
    monkeypatch.setattr(boq_parser, "_MAX_OUTPUT_BYTES", 6000)  # about 4,300 characters, about 13,000 bytes as JSON
    with pytest.raises(BoqParseError, match="too large to read"):
        parse_boq(path, ParserKind.TABULAR)
