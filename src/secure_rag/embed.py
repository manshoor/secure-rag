"""Embedding backends behind one small interface.

HashEmbedder needs no model or network, so tests and offline runs are
deterministic. OllamaEmbedder gives real semantic vectors from a local model.
"""

import hashlib
import json
import math
import re
import urllib.request
from typing import Protocol

from .config import HTTP_TIMEOUT_SECONDS, OLLAMA_EMBED_MODEL, OLLAMA_URL, EmbedderKind

_HASH_DIMENSIONS = 512
# Words, and numbers kept whole with their separators ("50,000", "0.1", "12:30").
_TOKEN_RE = re.compile(r"[0-9]+(?:[,.:][0-9]+)*|[a-z]+")

# Light suffix stripping so "carried" matches "carry" and "requests" matches "request".
_SUFFIXES = (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""))
_MIN_STEM_LENGTH = 3

# Words too common to carry meaning in a query.
STOP_WORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it many may much of on or our "
    "the this to what when where which who why will with you your me my we us".split()
)


class Embedder(Protocol):
    """Turns text into a fixed-size vector."""

    dimensions: int
    min_score: float  # cosine floor below which a vector hit is noise

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def tokens(text: str) -> list[str]:
    """Lowercase, stemmed tokens without stop words. Shared with the keyword index."""
    return [_stem(token) for token in _TOKEN_RE.findall(text.lower()) if token not in STOP_WORDS]


def _stem(token: str) -> str:
    """Strip one common English suffix, keeping at least _MIN_STEM_LENGTH letters."""
    if not token.isalpha():
        return token
    for suffix, replacement in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM_LENGTH:
            return token[: -len(suffix)] + replacement
    return token


class HashEmbedder:
    """Feature hashing of words and word pairs. Lexical, but stable and free."""

    dimensions = _HASH_DIMENSIONS
    min_score = 0.25

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        words = tokens(text)
        features = words + [f"{left}_{right}" for left, right in zip(words, words[1:])]

        # Each feature adds +1 or -1 to one bucket; the sign halves collisions.
        for feature in features:
            digest = hashlib.sha256(feature.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[bucket] += 1.0 if digest[4] % 2 else -1.0

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OllamaEmbedder:
    """Calls a local Ollama server's /api/embed endpoint."""

    min_score = 0.45

    def __init__(self) -> None:
        self.dimensions = len(self.embed(["probe"])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": OLLAMA_EMBED_MODEL, "input": texts}).encode()
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/embed", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return json.loads(response.read())["embeddings"]


def make_embedder(kind: EmbedderKind) -> Embedder:
    """Build the embedder named by kind."""
    if kind is EmbedderKind.OLLAMA:
        return OllamaEmbedder()
    return HashEmbedder()
