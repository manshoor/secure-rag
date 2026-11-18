"""Wires the pieces together from settings, so CLI, eval and tests build the same system."""

from dataclasses import dataclass
from pathlib import Path

from . import db
from .agent import ClaudePlanner, KeywordPlanner, Planner
from .answer import ClaudeComposer, Composer, ExtractiveComposer
from .config import DOCS_DIR, EmbedderKind, LlmKind, embedder_kind, llm_kind
from .embed import make_embedder
from .identity import register_documents
from .ingest import IngestReport, load_folder
from .store import KnowledgeStore


@dataclass
class System:
    """Everything a request needs."""

    report: IngestReport
    store: KnowledgeStore
    composer: Composer
    planner: Planner


def build_system(
    docs: Path = DOCS_DIR, embedder: EmbedderKind | None = None, llm: LlmKind | None = None
) -> System:
    """Ingest the documents and choose embedder and model from settings."""
    report = load_folder(docs)

    # Postgres on: make sure tables exist and every document has an access row.
    # Existing rows are kept, so a restart never undoes a live revocation.
    if db.enabled():
        db.create_schema()
        register_documents(db.Overwrite.KEEP, {d.doc_id: (d.title, d.access) for d in report.documents})

    store = KnowledgeStore(make_embedder(embedder or embedder_kind()), report.chunks, report.documents)
    if (llm or llm_kind()) is LlmKind.CLAUDE:
        return System(report, store, ClaudeComposer(), ClaudePlanner())
    return System(report, store, ExtractiveComposer(), KeywordPlanner())
