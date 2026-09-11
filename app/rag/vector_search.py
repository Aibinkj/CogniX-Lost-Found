"""Semantic retrieval over the found-items and help-article tables.

    query text -> embedding -> pgvector (or NumPy) -> top-K found items
                                                   -> top-K help articles

Results are built from ``public_dict()``, so ``hidden_features`` cannot leak
through the retrieval path.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.database.connection import session_scope
from app.database.models import FoundItem, HelpArticle
from app.database.vector_support import current_backend
from app.rag.embeddings import embed

logger = logging.getLogger(__name__)


def _vector_literal(query_vector: list[float]) -> str:
    return "[" + ",".join(f"{component:.6f}" for component in query_vector) + "]"


def _top_k_cosine(
    ids: list[int], vectors: list[list[float]], query_vector: list[float], top_k: int
) -> list[tuple[int, float]]:
    """In-process cosine ranking shared by the NumPy fallbacks."""
    matrix = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    query_norm = float(np.linalg.norm(query)) or 1.0
    similarities = (matrix @ query) / (norms * query_norm)

    order = np.argsort(-similarities)[:top_k]
    return [(ids[index], float(np.clip(similarities[index], 0.0, 1.0))) for index in order]


def _rank_with_pgvector(
    session: Session, query_vector: list[float], top_k: int, statuses: tuple[str, ...]
) -> list[tuple[int, float]]:
    """Let PostgreSQL do the ranking with the cosine-distance operator."""
    statement = text(
        """
        SELECT id, 1 - (embedding <=> CAST(:query AS vector)) AS similarity
        FROM found_items
        WHERE embedding IS NOT NULL AND status = ANY(:statuses)
        ORDER BY embedding <=> CAST(:query AS vector)
        LIMIT :top_k
        """
    )
    rows = session.execute(
        statement,
        {
            "query": _vector_literal(query_vector),
            "statuses": list(statuses),
            "top_k": top_k,
        },
    ).all()
    return [(int(row.id), float(row.similarity)) for row in rows]


def _rank_with_numpy(
    session: Session, query_vector: list[float], top_k: int, statuses: tuple[str, ...]
) -> list[tuple[int, float]]:
    """Fallback ranking: load stored vectors and score them in process.

    Fine for a prototype-sized table; an ANN index is what pgvector buys you.
    """
    rows = session.execute(
        select(FoundItem.id, FoundItem.embedding).where(
            FoundItem.embedding.is_not(None), FoundItem.status.in_(statuses)
        )
    ).all()
    if not rows:
        return []
    return _top_k_cosine([int(row[0]) for row in rows], [row[1] for row in rows], query_vector, top_k)


def search_matches(
    query: str,
    top_k: int | None = None,
    statuses: tuple[str, ...] = ("available",),
) -> list[dict[str, Any]]:
    """Return the top-K semantically closest available found items.

    Each result carries the public item fields plus a ``similarity`` in [0, 1].
    """
    query = (query or "").strip()
    if not query:
        return []

    top_k = top_k or settings.search_top_k
    query_vector = embed(query)

    with session_scope() as session:
        backend = current_backend()
        if backend == "pgvector":
            ranked = _rank_with_pgvector(session, query_vector, top_k, statuses)
        else:
            ranked = _rank_with_numpy(session, query_vector, top_k, statuses)

        if not ranked:
            return []

        scores = dict(ranked)
        items = session.scalars(
            select(FoundItem).where(FoundItem.id.in_(list(scores)))
        ).all()
        by_id = {item.id: item for item in items}

        results: list[dict[str, Any]] = []
        for item_id, similarity in ranked:
            item = by_id.get(item_id)
            if item is None:
                continue
            payload = item.public_dict()
            payload["similarity"] = round(similarity, 4)
            results.append(payload)

    logger.debug("search_matches(%r) -> %d results via %s", query, len(results), backend)
    return results


def search_help_articles(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Return the top-K help articles (desks and FAQs) closest to the question.

    Each result carries the article fields plus a ``similarity`` in [0, 1].
    """
    query = (query or "").strip()
    if not query:
        return []

    query_vector = embed(query)

    with session_scope() as session:
        backend = current_backend()
        if backend == "pgvector":
            rows = session.execute(
                text(
                    """
                    SELECT id, 1 - (embedding <=> CAST(:query AS vector)) AS similarity
                    FROM help_articles
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:query AS vector)
                    LIMIT :top_k
                    """
                ),
                {"query": _vector_literal(query_vector), "top_k": top_k},
            ).all()
            ranked = [(int(row.id), float(row.similarity)) for row in rows]
        else:
            rows = session.execute(
                select(HelpArticle.id, HelpArticle.embedding).where(HelpArticle.embedding.is_not(None))
            ).all()
            ranked = (
                _top_k_cosine([int(row[0]) for row in rows], [row[1] for row in rows], query_vector, top_k)
                if rows
                else []
            )

        if not ranked:
            return []

        by_id = {
            article.id: article
            for article in session.scalars(
                select(HelpArticle).where(HelpArticle.id.in_([article_id for article_id, _ in ranked]))
            ).all()
        }
        results: list[dict[str, Any]] = []
        for article_id, similarity in ranked:
            article = by_id.get(article_id)
            if article is None:
                continue
            payload = article.public_dict()
            payload["similarity"] = round(similarity, 4)
            results.append(payload)

    logger.debug("search_help_articles(%r) -> %d results via %s", query, len(results), backend)
    return results
