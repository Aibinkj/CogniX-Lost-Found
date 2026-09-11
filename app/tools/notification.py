"""User notification.

The prototype has no mail server, so notifications are persisted to the
`notifications` table and rendered in the UI. Swapping in SMTP or a push
provider means changing only this module.
"""

from __future__ import annotations

import logging
from typing import Any

from app.database.connection import session_scope
from app.database.models import Notification

logger = logging.getLogger(__name__)


def notify_user(
    recipient: str,
    subject: str,
    body: str,
    channel: str = "in_app",
) -> dict[str, Any]:
    """Record a message for the user and return the delivered payload."""
    if not body.strip():
        raise ValueError("body is required.")

    with session_scope() as session:
        notification = Notification(
            channel=channel,
            recipient=recipient or "anonymous",
            subject=subject[:200],
            body=body.strip(),
        )
        session.add(notification)
        session.flush()
        payload = {
            "id": notification.id,
            "channel": notification.channel,
            "recipient": notification.recipient,
            "subject": notification.subject,
            "body": notification.body,
        }

    logger.info("Notification %s queued for %s.", payload["id"], payload["recipient"])
    return payload
