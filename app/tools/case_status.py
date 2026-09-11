"""Reference lookup: the status of a pickup request (LF####) or staff case (ESC####)."""

from __future__ import annotations

import re
from typing import Any

from app.safety.escalation import REFERENCE_PREFIX as ESCALATION_PREFIX
from app.safety.escalation import get_escalation
from app.tools.pickup import REFERENCE_PREFIX as PICKUP_PREFIX
from app.tools.pickup import get_pickup_request

REFERENCE_RE = re.compile(rf"\b({PICKUP_PREFIX}|{ESCALATION_PREFIX})[\s-]?(\d{{3,6}})\b", re.IGNORECASE)


def find_reference(message: str) -> str | None:
    """Normalise the first reference in `message`, e.g. "lf 1024" -> "LF1024"."""
    match = REFERENCE_RE.search(message or "")
    return f"{match.group(1).upper()}{match.group(2)}" if match else None


def lookup_case(reference: str) -> dict[str, Any]:
    """Look up a reference. Returns only status-level fields, never case contents."""
    reference = (reference or "").strip().upper()
    if reference.startswith(ESCALATION_PREFIX):
        case = get_escalation(reference)
        return {"reference": reference, "type": "escalation", "found": case is not None, "record": case}
    if reference.startswith(PICKUP_PREFIX):
        pickup = get_pickup_request(reference)
        return {"reference": reference, "type": "pickup", "found": pickup is not None, "record": pickup}
    return {"reference": reference, "type": "unknown", "found": False, "record": None}
