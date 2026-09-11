"""Tool for recording a user's lost-property report."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.database.connection import session_scope
from app.database.models import LostItem, User
from app.rag.embeddings import embed

logger = logging.getLogger(__name__)


def _resolve_user_id(session, user_email: str | None, user_name: str | None) -> int | None:
    """Find or create the reporting user. No auth in this prototype."""
    if not user_email:
        return None
    user = session.scalar(select(User).where(User.email == user_email))
    if user is None:
        user = User(name=user_name or user_email.split("@")[0], email=user_email)
        session.add(user)
        session.flush()
    return user.id


def report_lost_item(
    description: str,
    category: str | None = None,
    brand: str | None = None,
    color: str | None = None,
    location: str | None = None,
    lost_time: datetime | None = None,
    raw_query: str = "",
    user_email: str | None = None,
    user_name: str | None = None,
) -> dict[str, Any]:
    """Persist a lost-item report and index it for reverse matching.

    Storing the report means a later-registered found item can be matched back
    to this user, rather than the search being a one-shot query.
    """
    if not description.strip():
        raise ValueError("description is required.")

    search_text = " ".join(
        part for part in (color, brand, category, description, location) if part
    )

    with session_scope() as session:
        record = LostItem(
            user_id=_resolve_user_id(session, user_email, user_name),
            category=(category or "").strip().lower() or None,
            brand=(brand or "").strip() or None,
            color=(color or "").strip().lower() or None,
            description=description.strip(),
            location=(location or "").strip() or None,
            lost_time=lost_time,
            raw_query=raw_query or description,
            embedding=embed(search_text),
        )
        session.add(record)
        session.flush()
        payload = {
            "id": record.id,
            "category": record.category,
            "brand": record.brand,
            "color": record.color,
            "description": record.description,
            "location": record.location,
            "lost_time": record.lost_time.isoformat() if record.lost_time else None,
            "status": record.status,
        }

    logger.info("Recorded lost item report %s.", payload["id"])
    return payload
