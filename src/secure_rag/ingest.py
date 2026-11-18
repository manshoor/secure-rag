"""Document ingestion: parse, split into cited chunks, screen for injection.

    file (.md / .txt / .pdf)
      -> front matter (id, title, company, roles, canary)
      -> sections by heading (PDF: by page)
      -> chunks of <= CHUNK_MAX_WORDS words with overlap
      -> injection screen: suspicious chunks are quarantined
"""

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from .config import CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS
from .guard import injection_findings
from .identity import DocAccess

_SUPPORTED = {".md", ".txt", ".pdf"}
_FRONT_MATTER = "---"
_REQUIRED_KEYS = ("id", "title", "company", "roles")


@dataclass(frozen=True)
class Document:
    """One source file with its access tags."""

    doc_id: str
    title: str
    access: DocAccess
    canary: str
    sections: list[tuple[str, str]]  # (section label, text)


@dataclass(frozen=True)
class Chunk:
    """The unit that is embedded, searched and cited."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    access: DocAccess


@dataclass
class IngestReport:
    """What ingestion kept, and what it quarantined and why."""

    chunks: list[Chunk] = field(default_factory=list)
    quarantined: list[tuple[Chunk, list[str]]] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)


def load_folder(folder: Path) -> IngestReport:
    """Ingest every supported file in folder, sorted by name for stable ids."""
    report = IngestReport()
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in _SUPPORTED:
            continue
        document = _load_document(path)

        # Ids must be unique: chunk ids derive from them, and search resolves hits by chunk id.
        if any(existing.doc_id == document.doc_id for existing in report.documents):
            raise ValueError(f"{path.name}: duplicate document id '{document.doc_id}'")
        report.documents.append(document)
        _chunk_document(document, report)
    return report


def _load_document(path: Path) -> Document:
    """Read metadata and sections from a text or PDF file."""

    # PDFs carry metadata in a sidecar <name>.meta file with the same front matter.
    if path.suffix.lower() == ".pdf":
        meta, _ = _split_front_matter(path.with_suffix(".meta").read_text(encoding="utf-8"))
        reader = PdfReader(str(path))
        sections = [(f"page {number}", page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
        return _make_document(meta, sections, path)

    # Markdown / text: front matter, then one section per heading.
    meta, body = _split_front_matter(path.read_text(encoding="utf-8"))
    return _make_document(meta, _sections_by_heading(body), path)


def _make_document(meta: dict[str, str], sections: list[tuple[str, str]], path: Path) -> Document:
    """Validate front matter and build the Document."""
    missing = [key for key in _REQUIRED_KEYS if not meta.get(key)]
    if missing:
        raise ValueError(f"{path.name}: front matter missing {missing}")
    roles = frozenset(role.strip() for role in meta["roles"].split(",") if role.strip())
    access = DocAccess(company=meta["company"].strip(), roles=roles)
    return Document(meta["id"], meta["title"], access, meta.get("canary", ""), sections)


def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Parse a simple 'key: value' block between --- lines."""
    lines = raw.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER:
        return {}, raw
    meta: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONT_MATTER:
            return meta, "\n".join(lines[index + 1 :])
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, ""


def _sections_by_heading(body: str) -> list[tuple[str, str]]:
    """Split markdown on '#' headings; text before the first heading is 'intro'."""
    sections: list[tuple[str, str]] = []
    label, buffer = "intro", []
    for line in body.splitlines():
        if line.startswith("#"):
            _append_section(sections, label, buffer)
            label, buffer = line.lstrip("#").strip(), []
            continue
        buffer.append(line)
    _append_section(sections, label, buffer)
    return sections


def _append_section(sections: list[tuple[str, str]], label: str, buffer: list[str]) -> None:
    """Keep a section only if it has text."""
    text = " ".join(" ".join(buffer).split())
    if text:
        sections.append((label, text))


def _chunk_document(document: Document, report: IngestReport) -> None:
    """Split sections into word windows and screen each one."""
    for section, text in document.sections:
        for part_index, part in enumerate(_word_windows(text)):
            chunk_id = f"{document.doc_id}#{_slug(section)}-{part_index}"
            chunk = Chunk(chunk_id, document.doc_id, document.title, section, part, document.access)

            # Quarantine chunks that address the model; they never reach the index.
            # Title and heading are screened too: they are shown to the model with the text.
            findings = injection_findings(part) + injection_findings(section) + injection_findings(document.title)
            if findings:
                report.quarantined.append((chunk, findings))
                continue
            report.chunks.append(chunk)


def _word_windows(text: str) -> list[str]:
    """Cut text into windows of CHUNK_MAX_WORDS with CHUNK_OVERLAP_WORDS overlap."""
    words = text.split()
    if len(words) <= CHUNK_MAX_WORDS:
        return [text]
    step = CHUNK_MAX_WORDS - CHUNK_OVERLAP_WORDS
    return [" ".join(words[start : start + CHUNK_MAX_WORDS]) for start in range(0, len(words), step)]


def _slug(label: str) -> str:
    """Lowercase, hyphenated section label for chunk ids."""
    return "-".join("".join(ch if ch.isalnum() else " " for ch in label.lower()).split())
