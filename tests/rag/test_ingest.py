"""Ingestion: chunking, metadata and quarantine of planted instructions."""

import pytest

from secure_rag.core.config import DOCS_DIR
from secure_rag.rag.ingest import load_folder


def test_every_chunk_carries_access_tags_and_a_section():
    report = load_folder(DOCS_DIR)
    assert report.chunks
    for chunk in report.chunks:
        assert chunk.access.company and chunk.access.roles
        assert chunk.section and chunk.text


def test_planted_instruction_is_quarantined_but_rest_of_doc_is_kept():
    report = load_folder(DOCS_DIR)
    quarantined = {chunk.chunk_id for chunk, _ in report.quarantined}
    indexed = {chunk.chunk_id for chunk in report.chunks}
    assert "alpha-site-safety-notice#contractor-note-0" in quarantined
    assert "alpha-site-safety-notice#heat-stress-0" in indexed
    assert not any("IGNORE ALL PREVIOUS" in chunk.text for chunk in report.chunks)


def test_missing_front_matter_is_an_error(tmp_path):
    (tmp_path / "bad.md").write_text("---\nid: x\ntitle: X\n---\n# A\ntext\n", encoding="utf-8")
    with pytest.raises(ValueError, match="company"):
        load_folder(tmp_path)


def test_long_sections_are_split_with_overlap(tmp_path):
    words = " ".join(f"word{i}" for i in range(300))
    (tmp_path / "long.md").write_text(
        f"---\nid: long\ntitle: Long\ncompany: group\nroles: employee\n---\n# Body\n{words}\n", encoding="utf-8"
    )
    chunks = load_folder(tmp_path).chunks
    assert len(chunks) >= 3
    assert chunks[0].text.split()[-1] in chunks[1].text.split()  # overlap keeps context


def test_duplicate_document_id_is_an_error(tmp_path):
    # Two files with one id would share chunk ids, so a hit on one could resolve to the other.
    for name, roles in (("a.md", "employee"), ("b.md", "executive")):
        (tmp_path / name).write_text(
            f"---\nid: same\ntitle: T\ncompany: group\nroles: {roles}\n---\n# Notes\ntext\n", encoding="utf-8"
        )
    with pytest.raises(ValueError, match="duplicate document id 'same'"):
        load_folder(tmp_path)
