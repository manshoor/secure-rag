"""Shared fixtures: an offline system and a temp data folder per test."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag import config, tracing
from secure_rag.agent import AgentRunner, KeywordPlanner
from secure_rag.app import System, build_system
from secure_rag.config import EmbedderKind, LlmKind


@pytest.fixture(autouse=True)
def temp_data_dir(tmp_path, monkeypatch):
    """Audit log, outbox and checkpoints go to a temp folder, never data/runtime."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def header_auth(monkeypatch):
    """API tests sign in with the X-User header, which needs RAG_AUTH=header."""
    monkeypatch.setenv("RAG_AUTH", "header")


@pytest.fixture(autouse=True)
def tracing_off():
    """No test sends traces, even if LANGFUSE_* or LANGSMITH_KEY are set in the shell."""
    tracing.configure(None)
    tracing.configure_langsmith(None)
    yield
    tracing.configure(None)
    tracing.configure_langsmith(None)


@pytest.fixture(scope="session")
def system() -> System:
    """Offline system: hash embeddings, extractive answers, keyword planner. No network."""
    return build_system(embedder=EmbedderKind.HASH, llm=LlmKind.OFFLINE)


@pytest.fixture
def runner(system) -> AgentRunner:
    return AgentRunner(system.store, system.composer, KeywordPlanner(), InMemorySaver())


@pytest.fixture(scope="session", autouse=True)
def offline_services():
    """Unit tests never touch a real Postgres or Qdrant, even if the shell exports them.

    Session scope so it runs before the session-wide system fixture builds the index.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("DATABASE_URL", raising=False)
        patch.delenv("QDRANT_URL", raising=False)
        yield
