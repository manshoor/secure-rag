"""Reads one .xlsx in a child process, so a hostile workbook cannot exhaust the server.

    boq_parser ──spawn──► python -I -m secure_rag._xlsx_child <path> <rows> <cols> <bytes> <memory> <output> <cell>
                              1. block numpy, then a memory rlimit where the OS allows it (Linux)
                              2. stream every zip entry, count real sizes (declared sizes can lie)
                              3. openpyxl read-only ─► rows of text cells, each cell and the reply capped
                          ◄── stdout: {"rows": [...]} or {"error": "<code>", "kind": "<type>"}
    boq_parser also watches this process's memory and run time, and kills it past its limits.

Errors carry a code and an exception type name only, never text from the file.
"""

import copy
import json
import resource
import sys
import zipfile

ERROR_TOO_LARGE = "too_large"
ERROR_UNREADABLE = "unreadable"

_UNZIP_CHUNK = 65536
_CELL_SEPARATOR = 2  # ", " between cells in the JSON reply
_TOO_LARGE_REPLY = '{"error": "too_large"}'  # a literal: building it must not need memory
_NO_SIZE_LIMIT = 2**62  # zipfile then reads to the real end of the data; the counter enforces the limit


class _TooLarge(Exception):
    """The workbook unzips past the limit."""


def main(argv: list[str]) -> None:
    path = argv[0]
    max_rows, max_columns, max_unzipped, memory, max_output, max_cell = (int(value) for value in argv[1:7])
    _block_heavy_imports()
    _limit_memory(memory)

    # The reply is built inside the try: running out of memory there is still "too large".
    try:
        _check_unzipped_size(path, max_unzipped)
        reply = json.dumps({"rows": _rows(path, max_rows, max_columns, max_cell, max_output)})
    except (_TooLarge, MemoryError):
        reply = _TOO_LARGE_REPLY
    except Exception as error:  # broken zip, bad XML: a code and a type name, no file text
        reply = '{"error": "%s", "kind": "%s"}' % (ERROR_UNREADABLE, type(error).__name__)
    sys.stdout.write(reply)


def _block_heavy_imports() -> None:
    """openpyxl imports numpy when it can; numpy's maths library starts a thread per CPU,
    which can pass the memory rlimit on a large host. openpyxl works without it."""
    sys.modules["numpy"] = None


def _limit_memory(limit: int) -> None:
    """Hard address-space limit. Linux enforces it; macOS refuses it, and the parent's watch applies."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        pass


def _check_unzipped_size(path: str, limit: int) -> None:
    """Stream every entry in small chunks and count its real size; memory stays at one chunk."""
    total = 0
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            unbounded = copy.copy(entry)
            unbounded.file_size = _NO_SIZE_LIMIT
            size = 0
            with archive.open(unbounded) as stream:
                while chunk := stream.read(_UNZIP_CHUNK):
                    size += len(chunk)
                    if total + size > limit:
                        raise _TooLarge
            if size != entry.file_size:
                raise ValueError("declared size does not match")
            total += size


def _rows(path: str, max_rows: int, max_columns: int, max_cell: int, max_output: int) -> list[list[str]]:
    """Rows of text cells. One shared string can fill every cell: the reply size is counted, not the XML.

    Each cell is counted as encoded in the reply: non-ASCII text grows up to 12 bytes per character.
    """
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True)
    rows: list[list[str]] = []
    output = 0
    for sheet in book.worksheets:
        for row in sheet.iter_rows(max_col=max_columns, values_only=True):
            if len(rows) >= max_rows:
                break
            cells = [_cell(value)[:max_cell] for value in row]
            output += sum(len(json.dumps(cell)) + _CELL_SEPARATOR for cell in cells)
            if output > max_output:
                raise _TooLarge
            rows.append(cells)
    book.close()
    return rows


def _cell(value: object) -> str:
    """Spreadsheet values as text; 85.0 becomes '85' so quantities read naturally."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


if __name__ == "__main__":
    main(sys.argv[1:])
