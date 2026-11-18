"""Hybrid search with permissions enforced inside each retriever.

    query, user
      |-- vector search (Qdrant, payload filter: company + roles) --|
      |-- keyword search (BM25 over chunks the user may read)  ----|--> RRF fusion
                                                                       -> can_read() re-check
                                                                          against CURRENT access
                                                                          (Postgres when configured)
                                                                       -> relevance floor

Vectors live in a Qdrant server when QDRANT_URL is set. The collection name is a
fingerprint of the chunks and the embedder, so a restart with the same documents
reuses it without re-embedding, and changed documents get a new collection.
"""

import hashlib

import math
from collections import Counter
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from . import config
from .config import MIN_TERM_OVERLAP, RRF_K, SEARCH_CANDIDATES
from .embed import Embedder, tokens
from .identity import (
    DocAccess,
    User,
    allowed_companies,
    can_read,
    companies_named,
    current_access,
    without_company_names,
)
from .ingest import Chunk, Document

_COLLECTION_PREFIX = "chunks_"
_FINGERPRINT_CHARS = 16
_IN_MEMORY = ":memory:"
_BM25_K1 = 1.5  # Standard BM25 term-frequency saturation.
_BM25_B = 0.75  # Standard BM25 length normalisation.


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk with the scores that ranked it."""

    chunk: Chunk
    fused: float
    vector: float
    keyword: float
    overlap: int  # distinct query terms found in the chunk


class KnowledgeStore:
    """Holds the chunks in Qdrant (vectors) and an in-memory BM25 index (keywords)."""

    def __init__(self, embedder: Embedder, chunks: list[Chunk], documents: list[Document]) -> None:
        self._embedder = embedder
        self._chunks = chunks
        self._documents = documents
        self._index_of = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
        self._stamped = {doc.doc_id: doc.access for doc in documents}
        self._qdrant = QdrantClient(url=config.qdrant_url()) if config.qdrant_url() else QdrantClient(_IN_MEMORY)
        self._collection = _COLLECTION_PREFIX + self._fingerprint()
        self._index_vectors()
        self._index_keywords()

    # ---- indexing ---------------------------------------------------------

    def _fingerprint(self) -> str:
        """Hash of everything stored in the collection: embedder, chunk text and access tags."""
        digest = hashlib.sha256()
        digest.update(f"{type(self._embedder).__name__}|{self._embedder.dimensions}|{config.OLLAMA_EMBED_MODEL}".encode())
        for chunk in self._chunks:
            parts = (chunk.chunk_id, chunk.title, chunk.section, chunk.text, chunk.access.company, *sorted(chunk.access.roles))
            digest.update("\x1f".join(parts).encode())
            digest.update(b"\x1e")
        return digest.hexdigest()[:_FINGERPRINT_CHARS]

    def _index_vectors(self) -> None:
        """Embed every chunk and store access tags as Qdrant payload for filtering.

        A complete collection with this fingerprint is reused as is. A partial
        one (an earlier run stopped mid-way) is rebuilt.
        """
        if self._qdrant.collection_exists(self._collection):
            if self._qdrant.count(self._collection, exact=True).count == len(self._chunks):
                return
            self._qdrant.delete_collection(self._collection)
        self._qdrant.create_collection(
            self._collection, vectors_config=VectorParams(size=self._embedder.dimensions, distance=Distance.COSINE)
        )

        # A server filters faster with indexes on the permission fields (in-memory mode ignores them).
        if config.qdrant_url():
            for field_name in ("company", "roles"):
                self._qdrant.create_payload_index(self._collection, field_name, PayloadSchemaType.KEYWORD)

        vectors = self._embedder.embed([f"{c.title}. {c.section}. {c.text}" for c in self._chunks])
        points = [
            PointStruct(
                id=index,
                vector=vector,
                payload={"chunk_id": chunk.chunk_id, "company": chunk.access.company, "roles": sorted(chunk.access.roles)},
            )
            for index, (chunk, vector) in enumerate(zip(self._chunks, vectors))
        ]
        self._qdrant.upsert(self._collection, points=points)

    def _index_keywords(self) -> None:
        """Precompute BM25 term counts per chunk."""
        self._terms = [Counter(tokens(f"{c.title} {c.section} {c.text}")) for c in self._chunks]
        self._lengths = [sum(counts.values()) for counts in self._terms]
        self._avg_length = sum(self._lengths) / max(len(self._lengths), 1)
        document_frequency: Counter[str] = Counter()
        for counts in self._terms:
            document_frequency.update(counts.keys())
        total = len(self._chunks)
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in document_frequency.items()
        }

    # ---- search -----------------------------------------------------------

    def search(self, user: User, query: str, limit: int) -> list[Hit]:
        """Return up to limit relevant chunks the user may read, best first.

        A question that names companies is about those companies: other
        companies' chunks are dropped, group-wide ones stay. Naming only a
        company the user cannot read leaves group-wide chunks only.
        """
        named = companies_named(query)
        query = without_company_names(query)  # the name scopes the search; it is not a topic
        vector_ranked = self._vector_search(user, query)
        keyword_ranked = self._keyword_search(user, query)
        query_terms = set(tokens(query))

        # Reciprocal rank fusion: robust to the two scores having different scales.
        fused: dict[int, float] = {}
        for ranked in (vector_ranked, keyword_ranked):
            for rank, (index, _) in enumerate(ranked):
                fused[index] = fused.get(index, 0.0) + 1.0 / (RRF_K + rank + 1)

        vector_scores = dict(vector_ranked)
        keyword_scores = dict(keyword_ranked)

        # Access as it is now (one Postgres query), not as it was at indexing.
        # A document without a current access row is unreadable.
        candidate_docs = {self._chunks[index].doc_id for index in fused}
        access_now = current_access({doc_id: self._stamped[doc_id] for doc_id in candidate_docs})

        hits: list[Hit] = []
        for index in sorted(fused, key=fused.get, reverse=True):
            chunk = self._chunks[index]

            # Defence in depth: re-check access even though both retrievers filtered.
            if not _readable(user, chunk.access, access_now.get(chunk.doc_id)):
                continue

            # Scope: the right rule of the wrong company is a wrong answer.
            if named and chunk.access.company not in named | {config.GROUP_COMPANY}:
                continue

            # Relevance floor: drop weak matches so off-topic questions get a refusal.
            overlap = len(query_terms & set(self._terms[index]))
            vector_score = vector_scores.get(index, 0.0)
            if overlap < MIN_TERM_OVERLAP and vector_score < self._embedder.min_score:
                continue
            hits.append(Hit(chunk, fused[index], vector_score, keyword_scores.get(index, 0.0), overlap))
        return hits[:limit]

    def _vector_search(self, user: User, query: str) -> list[tuple[int, float]]:
        """Nearest chunks by cosine, filtered by company and roles inside Qdrant."""
        permission = Filter(
            must=[
                FieldCondition(key="company", match=MatchAny(any=allowed_companies(user))),
                FieldCondition(key="roles", match=MatchAny(any=sorted(user.roles))),
            ]
        )
        query_vector = self._embedder.embed([query])[0]
        result = self._qdrant.query_points(
            self._collection, query=query_vector, query_filter=permission, limit=SEARCH_CANDIDATES,
            with_payload=["chunk_id"],
        )

        # Resolve hits by chunk id, not point id, so a stored index can never point at the wrong chunk.
        ranked: list[tuple[int, float]] = []
        for point in result.points:
            index = self._index_of.get((point.payload or {}).get("chunk_id", ""))
            if index is not None:
                ranked.append((index, float(point.score)))
        return ranked

    def _keyword_search(self, user: User, query: str) -> list[tuple[int, float]]:
        """BM25 over the chunks the user may read."""
        query_terms = tokens(query)
        scored: list[tuple[int, float]] = []
        for index, counts in enumerate(self._terms):
            if not can_read(user, self._chunks[index].access):
                continue
            score = self._bm25(counts, self._lengths[index], query_terms)
            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:SEARCH_CANDIDATES]

    def _bm25(self, counts: Counter[str], length: int, query_terms: list[str]) -> float:
        """Okapi BM25 score of one chunk."""
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            norm = frequency + _BM25_K1 * (1 - _BM25_B + _BM25_B * length / self._avg_length)
            score += self._idf.get(term, 0.0) * frequency * (_BM25_K1 + 1) / norm
        return score

    # ---- output-check support ---------------------------------------------

    def hidden_canaries(self, user: User) -> list[str]:
        """Canary strings of documents the user may NOT read right now."""
        access_now = current_access({doc.doc_id: doc.access for doc in self._documents if doc.canary})
        return [
            doc.canary
            for doc in self._documents
            if doc.canary and not _readable(user, doc.access, access_now.get(doc.doc_id))
        ]


def _readable(user: User, stamped: DocAccess, current: DocAccess | None) -> bool:
    """Readable under both the indexed tags and the current row; a missing row means no.

    Requiring both means a narrowed row applies at once, and a row widened by
    hand cannot open what the indexed tags (and the output check) still block.
    """
    return current is not None and can_read(user, stamped) and can_read(user, current)
