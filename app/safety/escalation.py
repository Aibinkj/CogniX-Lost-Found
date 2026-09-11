"""Human escalation.

The agent hands a case to staff rather than guessing when any of these hold:

* the best confidence is below ``CONFIDENCE_LOW_THRESHOLD``
* ownership verification failed
* several candidates are within ``AMBIGUITY_MARGIN`` and clarification did not
  separate them

An escalation is a durable record, not just a message: staff can review the
request, the candidates the agent considered, and the scores behind the call.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select

from app.database.connection import session_scope
from app.database.models import Escalation

logger = logging.getLogger(__name__)

REFERENCE_PREFIX = "ESC"
REFERENCE_START = 500

ESCALATION_MESSAGE = (
    "Your item could not be confidently verified. I've escalated this case to staff "
    "for review, and they'll follow up with you."
)


def _next_reference(session) -> str:
    count = session.scalar(select(func.count()).select_from(Escalation)) or 0
    return f"{REFERENCE_PREFIX}{REFERENCE_START + count}"


def escalate_to_staff(
    user_request: str,
    reason: str,
    candidate_ids: list[int] | None = None,
    confidence_scores: dict[str, Any] | None = None,
    verification_result: str | None = None,
) -> dict[str, Any]:
    """Open a staff review case and return its reference."""
    with session_scope() as session:
        escalation = Escalation(
            reference=_next_reference(session),
            user_request=user_request or "",
            candidate_ids=candidate_ids or [],
            confidence_scores=confidence_scores or {},
            verification_result=verification_result,
            reason=reason,
        )
        session.add(escalation)
        session.flush()
        payload = {
            "escalation_id": escalation.reference,
            "status": escalation.status,
            "reason": escalation.reason,
            "candidate_ids": escalation.candidate_ids,
            "verification_result": escalation.verification_result,
        }

    logger.info("Escalated to staff as %s (%s).", payload["escalation_id"], reason)
    return payload


def get_escalation(reference: str) -> dict[str, Any] | None:
    """Public status of one case. Omits the request text, which may be personal."""
    with session_scope() as session:
        row = session.scalar(select(Escalation).where(Escalation.reference == reference))
        if row is None:
            return None
        return {
            "escalation_id": row.reference,
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }


def list_escalations(status: str | None = "open", limit: int = 50) -> list[dict[str, Any]]:
    with session_scope() as session:
        statement = select(Escalation).order_by(Escalation.created_at.desc()).limit(limit)
        if status:
            statement = statement.where(Escalation.status == status)
        return [
            {
                "escalation_id": row.reference,
                "user_request": row.user_request,
                "candidate_ids": row.candidate_ids,
                "confidence_scores": row.confidence_scores,
                "verification_result": row.verification_result,
                "reason": row.reason,
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in session.scalars(statement).all()
        ]
