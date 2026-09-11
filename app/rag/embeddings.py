"""Text embeddings.

Two interchangeable backends:

* ``sentence-transformers`` - the real thing, used whenever the package is
  installed and the model can be loaded.
* ``hashing`` - a deterministic hashed bag-of-words/character-ngram vector. It
  needs no download and no network, which keeps the prototype runnable and the
  tests fast. It is a genuine lexical vector space, not a stub that fakes
  scores, but it is weaker at synonyms than a trained model.

Both produce L2-normalised vectors of ``settings.embedding_dim`` dimensions, so
cosine similarity is a plain dot product.
"""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Hashed word + character-trigram features projected into `dim` buckets.

    Words are weighted above character trigrams so that exact term overlap
    ("sony", "headphones") dominates, while trigrams still give partial credit
    for typos and morphological variants ("headphone" vs "headphones").
    """

    name = "hashing"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        for token in tokens:
            vector[self._bucket(f"w:{token}")] += 1.0
            padded = f"#{token}#"
            for index in range(len(padded) - 2):
                vector[self._bucket(f"c:{padded[index : index + 3]}")] += 0.30
        for left, right in zip(tokens, tokens[1:]):
            vector[self._bucket(f"b:{left}_{right}")] += 0.50
        return vector

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.vstack([self._encode_one(text) for text in texts])
        return _normalize_rows(matrix)


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model."""

    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        # The accessor was renamed in newer sentence-transformers releases.
        dimension_getter = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(dimension_getter())
        self.name = f"sentence-transformers:{model_name}"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Build the configured embedder once."""
    backend = settings.embedding_backend

    if backend in {"auto", "sentence-transformers"}:
        try:
            embedder = SentenceTransformerEmbedder(settings.embedding_model)
        except Exception as exc:  # noqa: BLE001 - any load failure falls back
            if backend == "sentence-transformers":
                raise RuntimeError(
                    f"EMBEDDING_BACKEND=sentence-transformers but the model could not be "
                    f"loaded: {exc}"
                ) from exc
            logger.warning(
                "sentence-transformers unavailable (%s). Using the hashing embedder.", exc
            )
        else:
            if embedder.dim != settings.embedding_dim:
                raise RuntimeError(
                    f"EMBEDDING_DIM is {settings.embedding_dim} but "
                    f"{settings.embedding_model} produces {embedder.dim} dimensions. "
                    "Re-seed the database after changing either."
                )
            logger.info("Embedding backend: %s (%d dims).", embedder.name, embedder.dim)
            return embedder

    logger.info("Embedding backend: hashing (%d dims).", settings.embedding_dim)
    return HashingEmbedder(settings.embedding_dim)


def embed_many(texts: list[str]) -> list[list[float]]:
    return [row.tolist() for row in get_embedder().encode(texts)]


def embed(text: str) -> list[float]:
    return embed_many([text])[0]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity clipped to [0, 1] so it can be blended with other signals."""
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.clip(float(np.dot(a, b)) / denominator, 0.0, 1.0))
