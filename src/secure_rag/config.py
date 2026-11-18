"""Settings and named constants shared by every module."""

import os
from enum import Enum
from pathlib import Path

# Folders: documents ship with the repo; runtime state (audit log, agent
# checkpoints, outbox) goes to RAG_DATA_DIR so tests can point it at a temp dir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = Path(os.environ.get("RAG_DOCS_DIR", PROJECT_ROOT / "data" / "docs"))
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", PROJECT_ROOT / "data" / "runtime"))
GOLDEN_FILE = PROJECT_ROOT / "data" / "golden.json"

# Chunking: split on headings first, then cap long sections by words.
CHUNK_MAX_WORDS = 120
CHUNK_OVERLAP_WORDS = 20

# Retrieval: each search returns this many candidates before fusion,
# and the answer step sees at most ANSWER_TOP_K chunks.
SEARCH_CANDIDATES = 8
ANSWER_TOP_K = 4
RRF_K = 60  # Reciprocal rank fusion constant from the original RRF paper.

# A chunk counts as relevant only if it shares at least this many
# distinct query terms (stop words removed) or passes the vector floor.
MIN_TERM_OVERLAP = 2

# Every company sees documents tagged with this value.
GROUP_COMPANY = "group"

# Claude: model and per-million-token prices (USD) used for cost reports.
CLAUDE_MODEL = os.environ.get("RAG_MODEL", "claude-opus-5-5")
PRICE_INPUT_PER_MTOK = 4.00
PRICE_OUTPUT_PER_MTOK = 20.00
TOKENS_PER_MTOK = 1_000_000
ANSWER_MAX_TOKENS = 4000
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Ollama embeddings (optional): used only when RAG_EMBEDDER=ollama.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
HTTP_TIMEOUT_SECONDS = 30

# Postgres (users, roles, document access, audit, agent checkpoints, eval runs)
# and a Qdrant server (persistent vectors). Both optional: unset means the
# offline defaults (built-in users, JSONL audit, SQLite checkpoints, in-memory
# Qdrant). Read at call time so tests can clear them.
_DATABASE_URL_ENV = "DATABASE_URL"
_QDRANT_URL_ENV = "QDRANT_URL"


class AuthMode(str, Enum):
    """How the API learns who is calling."""

    OFF = "off"  # default: no sign-in method, every request is 401
    HEADER = "header"  # trust X-User; anyone can claim any user, local use only


_AUTH_ENV = "RAG_AUTH"


def auth_mode() -> AuthMode:
    """Read RAG_AUTH at call time (default: off)."""
    return AuthMode(os.environ.get(_AUTH_ENV, AuthMode.OFF.value))


def database_url() -> str:
    """Postgres connection string, or '' when Postgres is off."""
    return os.environ.get(_DATABASE_URL_ENV, "")


def qdrant_url() -> str:
    """Qdrant server URL, or '' for an in-memory index."""
    return os.environ.get(_QDRANT_URL_ENV, "")


class EmbedderKind(str, Enum):
    """Which embedding backend builds the vectors."""

    HASH = "hash"
    OLLAMA = "ollama"


class LlmKind(str, Enum):
    """Which model writes answers and plans tool calls."""

    OFFLINE = "offline"
    CLAUDE = "claude"


def embedder_kind() -> EmbedderKind:
    """Read the embedder choice from RAG_EMBEDDER (default: hash)."""
    return EmbedderKind(os.environ.get("RAG_EMBEDDER", EmbedderKind.HASH.value))


def llm_kind() -> LlmKind:
    """Read the model choice from RAG_LLM (default: offline)."""
    return LlmKind(os.environ.get("RAG_LLM", LlmKind.OFFLINE.value))
