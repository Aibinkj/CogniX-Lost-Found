"""Pickup request creation."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select

from app.database.connection import session_scope
from app.database.models import Claim, FoundItem, PickupRequest

logger = logging.getLogger(__name__)

REFERENCE_PREFIX = "LF"
REFERENCE_START = 1024


def _next_reference(session) -> str:
    """Human-friendly sequential reference: LF1024, LF1025, ..."""
    count = session.scalar(select(func.count()).select_from(PickupRequest)) or 0
    return f"{REFERENCE_PREFIX}{REFERENCE_START + count}"


def create_pickup_request(
    item_id: int,
    claim_id: int | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    """Reserve a verified item for collection and mark it as claimed.

    Only call after ``verify_ownership`` has passed; the reservation flips the
    item out of ``available`` so it can no longer be matched by search.
    """
    with session_scope() as session:
        item = session.get(FoundItem, item_id)
        if item is None:
            raise ValueError(f"No found item with id {item_id}.")
        if item.status != "available":
            existing = session.scalar(
                select(PickupRequest).where(PickupRequest.found_item_id == item_id)
            )
            if existing:
                return {
                    "pickup_request_id": existing.reference,
                    "item_id": item_id,
                    "status": existing.status,
                    "location": existing.location,
                    "already_existed": True,
                }
            raise ValueError(f"Item {item_id} is not available (status: {item.status}).")

        if claim_id is not None and session.get(Claim, claim_id) is None:
            claim_id = None

        request = PickupRequest(
            reference=_next_reference(session),
            claim_id=claim_id,
            found_item_id=item_id,
            user_id=user_id,
            location=item.location,
            status="pending",
        )
        session.add(request)
        item.status = "claimed"
        session.flush()

        payload = {
            "pickup_request_id": request.reference,
            "item_id": item_id,
            "status": request.status,
            "location": request.location,
            "already_existed": False,
        }

    logger.info("Created pickup request %s for item %s.", payload["pickup_request_id"], item_id)
    return payload


def get_pickup_request(reference: str) -> dict[str, Any] | None:
    with session_scope() as session:
        request = session.scalar(
            select(PickupRequest).where(PickupRequest.reference == reference)
        )
        if request is None:
            return None
        return {
            "pickup_request_id": request.reference,
            "item_id": request.found_item_id,
            "status": request.status,
            "location": request.location,
            "created_at": request.created_at.isoformat() if request.created_at else None,
        }
