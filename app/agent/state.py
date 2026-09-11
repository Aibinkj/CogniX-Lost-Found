"""The agent's working state."""

from __future__ import annotations

from typing import Any, TypedDict

# `awaiting` values - what the agent is blocked on when a turn ends.
AWAITING_NOTHING = ""
AWAITING_MORE_INFO = "more_info"
AWAITING_CLARIFICATION = "clarification"
AWAITING_OWNERSHIP_PROOF = "ownership_proof"

# `next_action` values - the terminal decision of a turn, shown in the trace.
ACTION_ASK_USER = "ASK_USER"
ACTION_ASK_CLARIFICATION = "ASK_CLARIFICATION"
ACTION_REQUEST_VERIFICATION = "REQUEST_VERIFICATION"
ACTION_CREATE_PICKUP = "CREATE_PICKUP"
ACTION_ESCALATE = "HUMAN_ESCALATION"
ACTION_ANSWER_HELP = "ANSWER_HELP"
ACTION_CASE_STATUS = "CASE_STATUS"
ACTION_OUT_OF_SCOPE = "OUT_OF_SCOPE"

MAX_CLARIFICATION_ROUNDS = 2


class AgentState(TypedDict, total=False):
    """Carried between turns. Streamlit persists this per session."""

    session_id: str
    user_email: str | None

    user_message: str
    conversation_history: list[dict[str, str]]

    # Per turn: what kind of message this is, and the non-search outcomes.
    intent: str
    intent_reference: str | None
    help_result: dict[str, Any] | None
    case_lookup: dict[str, Any] | None
    # LF/ESC references this session produced; survives the reset after a case closes.
    case_references: list[str]

    item_details: dict[str, Any]
    lost_item_id: int | None

    candidate_matches: list[dict[str, Any]]
    selected_candidate: dict[str, Any] | None
    confidence: dict[str, Any] | None

    verification_status: str
    verification_result: dict[str, Any] | None

    pickup_request: dict[str, Any] | None
    escalation: dict[str, Any] | None
    notification: dict[str, Any] | None

    next_action: str
    awaiting: str
    agent_response: str
    clarification_rounds: int

    trace: list[dict[str, Any]]
    errors: list[str]


def new_state(session_id: str, user_email: str | None = None) -> AgentState:
    return AgentState(
        session_id=session_id,
        user_email=user_email,
        user_message="",
        conversation_history=[],
        intent="",
        intent_reference=None,
        help_result=None,
        case_lookup=None,
        case_references=[],
        item_details={},
        lost_item_id=None,
        candidate_matches=[],
        selected_candidate=None,
        confidence=None,
        verification_status="pending",
        verification_result=None,
        pickup_request=None,
        escalation=None,
        notification=None,
        next_action="",
        awaiting=AWAITING_NOTHING,
        agent_response="",
        clarification_rounds=0,
        trace=[],
        errors=[],
    )


def trace_step(state: AgentState, step: str, detail: dict[str, Any]) -> list[dict[str, Any]]:
    """Append one entry to the workflow trace and return the new list."""
    return [*state.get("trace", []), {"step": step, "detail": detail}]
