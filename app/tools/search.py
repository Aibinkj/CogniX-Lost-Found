"""Search tool.

The retrieval implementation lives in :mod:`app.rag.vector_search`; this module
is the tool-layer entry point the agent calls, and the place to add filtering or
re-ranking without touching the RAG internals.
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.rag.vector_search import search_matches as _semantic_search


def search_matches(
    query: str,
    top_k: int | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Return the top-K available found items closest to `query`.

    Results never include `hidden_features`. `category`, when given, filters the
    retrieved set rather than the search itself, so a mis-extracted category
    cannot silently hide the right item - it is a post-filter over a wider pull.
    """
    top_k = top_k or settings.search_top_k
    pull = top_k * 3 if category else top_k
    results = _semantic_search(query, top_k=pull)

    if category:
        filtered = [row for row in results if row.get("category") == category]
        results = filtered or results

    return results[:top_k]
