"""Wires the pieces together from settings, so CLI, eval and tests build the same system."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import db
from ..workflow.agent import ClaudePlanner, KeywordPlanner, Planner
from ..rag.answer import ClaudeComposer, Composer, ExtractiveComposer
from ..core.config import DOCS_DIR, EmbedderKind, LlmKind, embedder_kind, llm_kind
from ..rag.embed import make_embedder
from ..core.identity import register_documents
from ..rag.ingest import IngestReport, load_folder
from ..rag.store import KnowledgeStore


@dataclass
class System:
    """Everything a request needs."""

    report: IngestReport
    store: KnowledgeStore
    composer: Composer
    planner: Planner
    router: Any = None  # chat.ClaudeRouter in Claude mode; None means keyword rules only


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
        from .chat import ClaudeRouter  # chat imports this module

        return System(report, store, ClaudeComposer(), ClaudePlanner(), ClaudeRouter())
    return System(report, store, ExtractiveComposer(), KeywordPlanner())


def build_workflows(system: System, checkpointer=None):
    """Workflow runner for this system: offline intake and notes, or Claude when RAG_LLM=claude."""
    from ..boq.boq import ClaudeCanonicalizer
    from ..workflow.intake import ClaudeIntake, ClaudeSummarizer, OfflineIntake, TemplateSummarizer
    from ..workflow.procurement_agent import ClaudeStepPlanner
    from ..workflow.workflow import WorkflowRunner

    if isinstance(system.planner, ClaudePlanner):
        return WorkflowRunner(
            system.store, system.composer, ClaudeIntake(), checkpointer, ClaudeSummarizer(),
            canonicalizer=ClaudeCanonicalizer(), step_planner=ClaudeStepPlanner(),
        )
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), checkpointer, TemplateSummarizer())
