"""BOQ file -> validated line items. The only module that reads BOQ files.

    parse_boq(path) ─► backend chosen by RAG_BOQ_PARSER
                         tabular     openpyxl / csv   local, default
                         docling     Docling          local, PDF and scans (extra: docling)
                         llamaparse  LlamaParse       LlamaIndex cloud (extra: llamaparse)
                     ─► rows of cells ─► header detection ─► BoqLine (Pydantic)

Each backend only turns a file into rows of text cells. One shared step finds
the header row, reads item / description / unit / quantity, and treats rows
without a quantity as section headings. Rate and amount columns are ignored:
prices come only from the catalogue.

Files under data/private/ are never sent to a cloud parser unless
RAG_ALLOW_CLOUD_PARSE=yes is set for that run.
"""

import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from . import _xlsx_child, config

_MAX_DESCRIPTION = 500
_MAX_LINES = 5000
_MAX_ROWS = 20000  # rows read from a file before giving up on the rest
_MAX_COLUMNS = 40
# An .xlsx is a zip: a small upload can expand to gigabytes. Counted in the child before openpyxl reads it.
_MAX_UNZIPPED_BYTES = 20_000_000
# The child process that reads an .xlsx: killed past this memory or run time.
_CHILD_MODULE = "secure_rag._xlsx_child"
_CHILD_MEMORY_BYTES = 512_000_000
_CHILD_SECONDS = 10.0  # a real BOQ reads in about a second; a waiting upload holds an API thread
_CHILD_CPU_SECONDS = int(_CHILD_SECONDS) + 1  # the child's own limit, for when this process dies first
_CHILD_POLL_SECONDS = 0.05
_MAX_OUTPUT_BYTES = 10_000_000  # the child's UTF-8 reply; decoded, about 110 MB at most here (emoji text costs 4 bytes a character)
_CHILD_SLOTS = threading.BoundedSemaphore(2)  # children at once, per server process
_SLOT_WAIT_SECONDS = 5.0  # longer than one large read (about 3 s); then "busy", rather than holding an API thread
_PS_SECONDS = 2.0
_MAX_QUANTITY = Decimal("1000000000")
# Plain quantities only: 1200, 1,200, 1,200.50, 18.5. Anything else ("1.200,50", "120 m2", "TBC") is unreadable.
_QUANTITY_RE = re.compile(r"\d{1,3}(,\d{3})+(\.\d+)?|\d+(\.\d+)?")
_LLAMAPARSE_TIER = "cost_effective"  # cheapest tier that keeps table structure
_LLAMAPARSE_VERSION = "latest"
_LLAMA_KEY_ENV = "LLAMA_CLOUD_API_KEY"

# Header names seen in BOQs, lower-cased with punctuation removed.
_HEADERS = {
    "item_no": {"item", "item no", "itemno", "ref", "no", "sno", "sl no", "sr no", "item ref"},
    "description": {"description", "desc", "item description", "particulars", "description of work"},
    "unit": {"unit", "units", "uom", "unit of measure"},
    "quantity": {"qty", "quantity", "quantities"},
}
_TABULAR_SUFFIXES = {".xlsx", ".csv"}


class ParserKind(str, Enum):
    TABULAR = "tabular"
    DOCLING = "docling"
    LLAMAPARSE = "llamaparse"


class CloudParse(str, Enum):
    """Whether a private file may go to a cloud parser on this run."""

    NO = "no"
    YES = "yes"


class BoqLine(BaseModel):
    """One priced-later line as written in the BOQ. Unit is raw; boq.py normalises it.

    quantity is None when the cell is not a plain number; boq.py holds such lines for review.
    """

    item_no: str = Field(max_length=40)
    description: str = Field(min_length=1, max_length=_MAX_DESCRIPTION)
    unit: str = Field(max_length=20)
    quantity: Decimal | None = Field(gt=0)
    section: str = Field(default="", max_length=_MAX_DESCRIPTION)


class BoqParseError(ValueError):
    """The file cannot be read as a BOQ (no header row, unsupported type, refused upload)."""


def parser_kind() -> ParserKind:
    return ParserKind(os.environ.get("RAG_BOQ_PARSER", ParserKind.TABULAR.value))


def cloud_parse() -> CloudParse:
    return CloudParse(os.environ.get("RAG_ALLOW_CLOUD_PARSE", CloudParse.NO.value))


def is_private(path: Path) -> bool:
    """True for files under data/private/.

    Resolved, so '..' and symlinks cannot escape. Each parent is also compared
    by identity (samefile), so 'PRIVATE/' on a case-insensitive disk still counts.
    """
    resolved = path.resolve()
    private = config.PRIVATE_DIR.resolve()
    if resolved.is_relative_to(private):
        return True
    if not private.exists():
        return False
    return any(parent.exists() and os.path.samefile(parent, private) for parent in resolved.parents)


def parse_boq(path: Path, kind: ParserKind | None = None) -> list[BoqLine]:
    """Parse a BOQ file into validated lines with the chosen backend."""
    kind = kind or parser_kind()
    try:
        rows = _BACKENDS[kind](path)
    except BoqParseError:
        raise
    except Exception as error:  # broken zip, bad CSV, unreadable PDF: a clear stop, never a crash
        raise BoqParseError(f"the file could not be read as a BOQ ({type(error).__name__})") from error
    return _rows_to_lines(rows)


# ---- backends: file -> rows of text cells ------------------------------------------


def _tabular_rows(path: Path) -> list[list[str]]:
    suffix = path.suffix.lower()
    if suffix not in _TABULAR_SUFFIXES:
        raise BoqParseError(f"the tabular parser reads {sorted(_TABULAR_SUFFIXES)}; use RAG_BOQ_PARSER=docling for {suffix}")
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [row[:_MAX_COLUMNS] for row, _ in zip(csv.reader(handle), range(_MAX_ROWS))]

    return _xlsx_rows_in_child(path)


def _xlsx_rows_in_child(path: Path) -> list[list[str]]:
    """Read an .xlsx in a child process with memory and time limits.

    openpyxl can turn a small upload into a large XML tree, whatever the zip
    declares. The child sets a memory rlimit where the OS allows it (Linux);
    this side also watches the child's memory and run time, and kills it past
    either limit. See _xlsx_child for the child's side.
    """
    # A few children at once: each may use up to the memory limit.
    if not _CHILD_SLOTS.acquire(timeout=_SLOT_WAIT_SECONDS):
        raise BoqParseError("the server is busy reading other workbooks; try again shortly")
    try:
        return _run_child(path)
    finally:
        _CHILD_SLOTS.release()


def _run_child(path: Path) -> list[list[str]]:
    limits = (_MAX_ROWS, _MAX_COLUMNS, _MAX_UNZIPPED_BYTES, _CHILD_MEMORY_BYTES, _MAX_OUTPUT_BYTES, _MAX_DESCRIPTION, _CHILD_CPU_SECONDS)
    args = [str(value) for value in (path, *limits)]
    child = subprocess.Popen(
        [sys.executable, "-I", "-m", _CHILD_MODULE, *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    deadline = time.monotonic() + _CHILD_SECONDS
    try:
        while True:
            try:
                output, _ = child.communicate(timeout=_CHILD_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pass

            # Watch from outside: the child cannot run its own checks while openpyxl parses.
            if _rss_bytes(child.pid) > _CHILD_MEMORY_BYTES:
                raise BoqParseError("the workbook is too large to read")
            if time.monotonic() > deadline:
                raise BoqParseError("reading the workbook took too long")
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    return _child_rows(output, child.returncode)


def _child_rows(output: bytes, returncode: int) -> list[list[str]]:
    """Rows from the child's JSON reply. Errors hold a code and a type name, never file text."""
    try:
        reply = json.loads(output)
    except ValueError:
        reply = {}
    if "rows" in reply:
        return reply["rows"]
    if reply.get("error") == _xlsx_child.ERROR_TOO_LARGE:
        raise BoqParseError("the workbook is too large to read")
    if returncode == -signal.SIGXCPU:
        raise BoqParseError("reading the workbook took too long")
    if returncode < 0:  # killed by a signal: the cause is unknown
        raise BoqParseError("the file could not be read as a BOQ (the reader stopped)")
    kind = reply.get("kind", "no reply")
    raise BoqParseError(f"the file could not be read as a BOQ ({kind if kind.isidentifier() else 'error'})")


def _rss_bytes(pid: int) -> int:
    """Resident memory of a process, or 0 once it has gone. /proc on Linux, ps elsewhere."""
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        return 0
    # A stuck ps skips one check; the wall-clock deadline still applies.
    try:
        found = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=_PS_SECONDS).stdout.strip()
    except subprocess.TimeoutExpired:
        return 0
    return int(found) * 1024 if found.isdigit() else 0


def _docling_rows(path: Path) -> list[list[str]]:
    # Excel always goes through the limited child reader; Docling handles PDFs and scans.
    if path.suffix.lower() == ".xlsx":
        return _xlsx_rows_in_child(path)
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as error:
        raise BoqParseError("Docling is not installed: uv sync --extra docling") from error

    document = DocumentConverter().convert(str(path)).document
    rows: list[list[str]] = []
    for table in document.tables:
        frame = table.export_to_dataframe(doc=document)
        rows.append([_cell(name) for name in frame.columns])
        rows.extend([_cell(value) for value in record] for record in frame.itertuples(index=False))
    return rows


def _llamaparse_rows(path: Path) -> list[list[str]]:
    # Private files leave the machine only with an explicit yes for this run.
    if is_private(path) and cloud_parse() is not CloudParse.YES:
        raise BoqParseError("refusing to upload a file from data/private/ to LlamaParse; set RAG_ALLOW_CLOUD_PARSE=yes for this run")
    if not os.environ.get(_LLAMA_KEY_ENV):
        raise BoqParseError(f"LlamaParse needs {_LLAMA_KEY_ENV}")
    try:
        from llama_cloud import LlamaCloud
    except ImportError as error:
        raise BoqParseError("LlamaParse is not installed: uv sync --extra llamaparse") from error

    client = LlamaCloud(api_key=os.environ[_LLAMA_KEY_ENV])
    result = client.parsing.parse(upload_file=path, tier=_LLAMAPARSE_TIER, version=_LLAMAPARSE_VERSION, expand=["markdown"])
    pages = result.markdown.pages if result.markdown else []
    return markdown_table_rows("\n".join(getattr(page, "markdown", "") for page in pages))


def markdown_table_rows(markdown: str) -> list[list[str]]:
    """Rows of every pipe table in markdown; separator rows (|---|) are skipped."""
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
            continue
        rows.append(cells)
    return rows


_BACKENDS = {ParserKind.TABULAR: _tabular_rows, ParserKind.DOCLING: _docling_rows, ParserKind.LLAMAPARSE: _llamaparse_rows}


# ---- shared: rows -> lines --------------------------------------------------------------


def _rows_to_lines(rows: list[list[str]]) -> list[BoqLine]:
    """Find header rows, then read lines; rows with a description but no quantity are sections."""
    lines: list[BoqLine] = []
    columns: dict[str, int] | None = None
    section = ""
    for row in rows:
        # A header row (re)defines the columns; tables split across pages repeat it.
        header = _header_columns(row)
        if header:
            columns = header
            continue
        if columns is None:
            continue

        item_no, description, unit, quantity = (_at(row, columns[key]) for key in ("item_no", "description", "unit", "quantity"))
        if not description and not item_no:
            continue

        # No quantity and no unit: a section heading (e.g. "B  GROUND FLOOR BLOCKWORK").
        # A unit with an unreadable quantity is a line to check by hand, not a heading.
        amount = _decimal(quantity)
        if amount is None and not unit:
            if not quantity:
                section = (description or item_no)[:_MAX_DESCRIPTION]
            continue
        try:
            lines.append(BoqLine(item_no=item_no[:40], description=description[:_MAX_DESCRIPTION], unit=unit[:20], quantity=amount, section=section))
        except ValidationError:
            continue
        if len(lines) >= _MAX_LINES:
            break
    if columns is None:
        raise BoqParseError("no header row with description, unit and quantity columns")
    return lines


def _header_columns(row: list[str]) -> dict[str, int] | None:
    found: dict[str, int] = {}
    for index, cell in enumerate(row):
        name = re.sub(r"[^a-z ]", "", cell.lower()).strip()
        for key, names in _HEADERS.items():
            if name in names and key not in found:
                found[key] = index
    needed = {"description", "unit", "quantity"}
    if not needed <= set(found):
        return None
    found.setdefault("item_no", -1)
    return found


def _at(row: list[str], index: int) -> str:
    return row[index].strip() if 0 <= index < len(row) else ""


def _decimal(text: str) -> Decimal | None:
    stripped = text.strip()
    if not _QUANTITY_RE.fullmatch(stripped):
        return None
    try:
        value = Decimal(stripped.replace(",", ""))
    except InvalidOperation:
        return None
    return value if 0 < value < _MAX_QUANTITY else None


def _cell(value: object) -> str:
    """Spreadsheet values as text; 85.0 becomes '85' so quantities read naturally."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()
