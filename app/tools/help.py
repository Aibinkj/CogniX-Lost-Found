"""Help-desk knowledge tools.

`search_help` is the RAG entry point for questions about the service itself -
contacting staff, desk hours, how collection works. `get_desk_contact` is an
exact lookup the agent uses to quote a desk's contact details in pickup and
escalation messages. Both read the `help_articles` table.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from app.database.connection import session_scope
from app.database.models import HelpArticle
from app.rag.vector_search import search_help_articles

# The desk that owns escalated cases and anything a local desk cannot resolve.
CENTRAL_DESK_LOCATION = "Main Reception"


def search_help(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Return the top-K help articles closest to `query`, with similarity."""
    return search_help_articles(query, top_k=top_k)


def get_desk_contact(location: str | None) -> dict[str, Any] | None:
    """The desk serving `location` (a found-item or pickup location), if known."""
    if not location:
        return None
    with session_scope() as session:
        article = session.scalar(
            select(HelpArticle).where(
                HelpArticle.kind == "desk",
                func.lower(HelpArticle.desk_location) == location.strip().lower(),
            )
        )
        return article.public_dict() if article else None


def format_contact(desk: dict[str, Any] | None) -> str:
    """One-line contact summary: "Title (phone, email; hours)"."""
    if not desk:
        return ""
    reach = ", ".join(part for part in (desk.get("phone"), desk.get("email")) if part)
    details = "; ".join(part for part in (reach, desk.get("hours")) if part)
    return f"{desk['title']} ({details})" if details else desk["title"]
