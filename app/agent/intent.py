"""Decide what kind of message the user sent before doing anything with it.

    report  - describes a lost item, or answers one of the agent's questions about it
    answer  - the reply to a pending ownership question
    help    - a question about the service (staff contact, desks, collection, policy)
    status  - asks about a pickup request or staff case (LF#### / ESC####)
    other   - greetings, thanks, anything unrelated

Deterministic rules decide every clear case. The LLM, when configured, is only
asked about messages the rules cannot place - and never while an ownership
answer is pending, so whether a message counts as a verification attempt does
not depend on a model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

from app.agent.extraction import extract_with_rules
from app.agent.prompts import INTENT_SYSTEM, INTENT_USER_TEMPLATE
from app.agent.state import AWAITING_NOTHING, AWAITING_OWNERSHIP_PROOF
from app.llm import LLMError, get_llm
from app.tools.case_status import find_reference

logger = logging.getLogger(__name__)

INTENT_REPORT = "report"
INTENT_ANSWER = "answer"
INTENT_HELP = "help"
INTENT_STATUS = "status"
INTENT_OTHER = "other"

_STAFF = r"(?:staff|desk|office|someone|somebody|anyone|person|human|team|reception|security|you|them)"

HELP_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"\b(?:contact|reach|call|ring|email|e-mail|speak (?:to|with)|talk (?:to|with)|get in touch with|get hold of)\b.*\b{_STAFF}\b",
        r"\b(?:staff|desk|office|reception)\b.*\b(?:contact|number|phone|email|hours|open|opening|close|closed|located|location|where)\b",
        r"\b(?:opening|office|desk|working) hours\b",
        r"\bwhen (?:is|are|does|do)\b.*\b(?:open|close|closed)\b",
        r"\bwhere (?:is|are|do i|can i|should i)\b.*\b(?:desk|office|reception|collect|pick|hand|drop|take)\b",
        r"\bhow (?:do|can|should) i\b.*\b(?:collect|pick (?:it |them )?up|claim|get (?:it|them) back|contact|reach|hand (?:it |them )?in|report)\b",
        r"\bwhat (?:do|should) i (?:need|bring)\b",
        # "Do I need ID?" is a question; "I lost my ID card" is not, so anchor on the verb.
        r"\b(?:need|bring|show|required)\b.*\b(?:photo id|id card|identification|id)\b",
        r"\bhow long\b.*\b(?:keep|kept|hold|held|store|stored)\b",
        r"\b(?:someone else|a friend|on my behalf)\b.*\b(?:collect|pick)",
    )
)

# Questions about the assistant itself get the capability intro, not a help search.
ABOUT_ASSISTANT_RE = re.compile(
    r"\bwhat can you (?:do|help with)\b|\bhow does (?:this|it) work\b|\bwho are you\b|\bwhat are you\b",
    re.IGNORECASE,
)

# Someone who found an item is not a claimant, even though they describe one.
FINDER_RE = re.compile(r"\bi(?: have|'ve| just)? found\b", re.IGNORECASE)

STATUS_RE = re.compile(
    r"\b(?:status|update|progress|any news)\b.*\b(?:case|claim|pickup|request|report|reference)\b"
    r"|\b(?:my|the) (?:case|claim|pickup request)\b.*\b(?:status|update|progress)\b",
    re.IGNORECASE,
)

SMALL_TALK_RE = re.compile(
    r"^\s*(?:hi|hello|hey|hiya|good (?:morning|afternoon|evening)|thanks|thank you|thx|cheers|"
    r"ok|okay|cool|great|bye|goodbye)\b[\s!.,]*(?:there|so much|a lot)?[\s!.,]*$",
    re.IGNORECASE,
)

LOST_VERB_RE = re.compile(
    r"\b(?:lost|left|missing|dropped|forgot|misplaced|can't find|cannot find|lose)\b", re.IGNORECASE
)


@dataclass
class IntentDecision:
    intent: str
    method: str  # "rules" or "llm:<provider>:<model>"
    reason: str
    reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _describes_item(message: str) -> bool:
    """A concrete item description: a category plus at least one other attribute."""
    details = extract_with_rules(message)
    return bool(details.category) and bool(details.color or details.brand or details.location)


def _asks_for_help(message: str) -> bool:
    return any(pattern.search(message) for pattern in HELP_PATTERNS)


def _classify_with_llm(message: str) -> IntentDecision | None:
    client = get_llm()
    if client is None:
        return None
    try:
        payload = client.complete_json(
            INTENT_SYSTEM, INTENT_USER_TEMPLATE.format(message=message), max_tokens=60
        )
    except (LLMError, KeyError, ValueError) as exc:
        logger.warning("LLM intent classification failed (%s); using rules.", exc)
        return None
    intent = str(payload.get("intent", "")).strip().lower()
    if intent not in {INTENT_REPORT, INTENT_HELP, INTENT_OTHER}:
        return None
    return IntentDecision(intent, f"llm:{client.provider}:{client.model}", "model classification")


def classify_message(message: str, awaiting: str = AWAITING_NOTHING) -> IntentDecision:
    """Route one message. Never raises; the rules always produce a decision."""
    text = (message or "").strip()

    if reference := find_reference(text):
        return IntentDecision(INTENT_STATUS, "rules", "contains a case reference", reference)
    if STATUS_RE.search(text):
        return IntentDecision(INTENT_STATUS, "rules", "asks about an existing case")
    if SMALL_TALK_RE.match(text):
        return IntentDecision(INTENT_OTHER, "rules", "greeting or acknowledgement")
    if ABOUT_ASSISTANT_RE.search(text):
        return IntentDecision(INTENT_OTHER, "rules", "asks what the assistant does")
    if FINDER_RE.search(text):
        return IntentDecision(INTENT_HELP, "rules", "reports finding an item, not losing one")
    if _asks_for_help(text) and not _describes_item(text):
        return IntentDecision(INTENT_HELP, "rules", "question about the service")

    if awaiting == AWAITING_OWNERSHIP_PROOF:
        return IntentDecision(INTENT_ANSWER, "rules", "reply to the pending ownership question")
    if awaiting != AWAITING_NOTHING:
        return IntentDecision(INTENT_REPORT, "rules", "reply to the agent's question about the item")

    if LOST_VERB_RE.search(text) or extract_with_rules(text).category:
        return IntentDecision(INTENT_REPORT, "rules", "describes a lost item")

    if decision := _classify_with_llm(text):
        return decision

    if text.endswith("?"):
        return IntentDecision(INTENT_HELP, "rules", "unrecognised question; answer from the help base or say so")
    return IntentDecision(INTENT_REPORT, "rules", "default: treat as an item description")
