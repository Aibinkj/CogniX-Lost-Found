"""Decides how embeddings are stored and compared.

The prototype targets PostgreSQL + pgvector. Not every PostgreSQL install has
the `vector` extension available (notably stock Windows builds), so the storage
layer degrades to a `REAL[]` column ranked with NumPy instead of failing. Both
paths expose the same interface to `app.rag.vector_search`; only the ranking
site differs.
"""

from __future__ import annotations

import logging
from typing import Literal

from sqlalchemy import Engine, text
from sqlalchemy.dialects.postgresql import ARRAY, REAL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import TypeDecorator, TypeEngine

from app.config import settings

logger = logging.getLogger(__name__)

Backend = Literal["pgvector", "numpy"]

_resolved_backend: Backend | None = None


class VectorExtensionUnavailable(RuntimeError):
    """Raised when pgvector is explicitly required but cannot be enabled."""


def resolve_backend(engine: Engine) -> Backend:
    """Detect (once) whether pgvector is usable on the connected server."""
    global _resolved_backend
    if _resolved_backend is not None:
        return _resolved_backend

    requested = settings.vector_backend
    if requested == "numpy":
        _resolved_backend = "numpy"
        logger.info("Vector backend: numpy (forced by VECTOR_BACKEND).")
        return _resolved_backend

    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        _resolved_backend = "pgvector"
        logger.info("Vector backend: pgvector.")
    except SQLAlchemyError as exc:
        if requested == "pgvector":
            raise VectorExtensionUnavailable(
                "VECTOR_BACKEND=pgvector but the `vector` extension could not be created. "
                "Start the bundled database with `docker compose up -d` or install pgvector."
            ) from exc
        _resolved_backend = "numpy"
        logger.warning(
            "pgvector unavailable (%s). Falling back to REAL[] storage with NumPy ranking.",
            exc.__class__.__name__,
        )

    return _resolved_backend


def current_backend() -> Backend:
    """The backend resolved so far. Defaults to numpy before any connection."""
    return _resolved_backend or "numpy"


def reset_backend_cache() -> None:
    """Test hook: forget the detected backend."""
    global _resolved_backend
    _resolved_backend = None


class EmbeddingVector(TypeDecorator):
    """A fixed-width float vector column backed by pgvector or `REAL[]`.

    The concrete type is chosen at SQL-compile time, which always happens after
    the engine (and therefore backend detection) exists.
    """

    cache_ok = True
    impl = ARRAY(REAL)

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect) -> TypeEngine:
        if current_backend() == "pgvector":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(ARRAY(REAL))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return [float(component) for component in value]

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return [float(component) for component in value]
