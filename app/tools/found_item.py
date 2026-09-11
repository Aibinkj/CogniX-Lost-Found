"""Tools for the staff side: registering and reading found items."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.database.connection import session_scope
from app.database.models import FoundItem
from app.rag.embeddings import embed

logger = logging.getLogger(__name__)


def register_found_item(
    category: str,
    description: str,
    location: str,
    found_time: datetime | None = None,
    brand: str | None = None,
    color: str | None = None,
    hidden_features: str = "",
    status: str = "available",
) -> dict[str, Any]:
    """Store an item handed in at a desk and index it for semantic search.

    `hidden_features` is ownership evidence recorded by staff. It is stored but
    never returned here.
    """
    if not category.strip() or not description.strip() or not location.strip():
        raise ValueError("category, description and location are required.")

    item = FoundItem(
        category=category.strip().lower(),
        brand=(brand or "").strip() or None,
        color=(color or "").strip().lower() or None,
        description=description.strip(),
        hidden_features=hidden_features.strip(),
        location=location.strip(),
        found_time=found_time or datetime.now(timezone.utc),
        status=status,
    )
    item.embedding = embed(item.embedding_text())

    with session_scope() as session:
        session.add(item)
        session.flush()
        payload = item.public_dict()

    logger.info("Registered found item %s (%s at %s).", payload["id"], category, location)
    return payload


def get_found_item(item_id: int, include_hidden: bool = False) -> dict[str, Any] | None:
    """Fetch one item. `include_hidden` is for staff/debug paths only."""
    with session_scope() as session:
        item = session.get(FoundItem, item_id)
        if item is None:
            return None
        payload = item.public_dict()
        if include_hidden:
            payload["hidden_features"] = item.hidden_features
        return payload


def list_found_items(status: str | None = "available", limit: int = 100) -> list[dict[str, Any]]:
    """List items for a staff view. Never includes hidden features."""
    with session_scope() as session:
        statement = select(FoundItem).order_by(FoundItem.found_time.desc()).limit(limit)
        if status:
            statement = statement.where(FoundItem.status == status)
        return [item.public_dict() for item in session.scalars(statement).all()]


def set_item_status(item_id: int, status: str) -> dict[str, Any] | None:
    with session_scope() as session:
        item = session.get(FoundItem, item_id)
        if item is None:
            return None
        item.status = status
        session.flush()
        return item.public_dict()
