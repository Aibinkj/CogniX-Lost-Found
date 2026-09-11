"""Intent routing, grounded help answers and reference lookups."""

from __future__ import annotations

import pytest

from app.agent.graph import run_turn
from app.agent.intent import (
    INTENT_ANSWER,
    INTENT_HELP,
    INTENT_OTHER,
    INTENT_REPORT,
    INTENT_STATUS,
    classify_message,
)
from app.agent.state import (
    ACTION_ANSWER_HELP,
    ACTION_CASE_STATUS,
    ACTION_CREATE_PICKUP,
    ACTION_ESCALATE,
    ACTION_OUT_OF_SCOPE,
    AWAITING_CLARIFICATION,
    AWAITING_OWNERSHIP_PROOF,
)
from app.tools.case_status import find_reference, lookup_case
from app.tools.help import CENTRAL_DESK_LOCATION, get_desk_contact, search_help


def steps(state) -> list[str]:
    return [entry["step"] for entry in state.get("trace", [])]


# --- intent rules ------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("How can I reach this staff?", INTENT_HELP),
        ("When is the library security desk open?", INTENT_HELP),
        ("Do I need to bring ID to collect it?", INTENT_HELP),
        ("How long do you keep lost items?", INTENT_HELP),
        ("I found a wallet outside the gym, where do I hand it in?", INTENT_HELP),
        ("What's the status of LF1024?", INTENT_STATUS),
        ("any update on my case?", INTENT_STATUS),
        ("hello", INTENT_OTHER),
        ("thanks!", INTENT_OTHER),
        ("what can you do?", INTENT_OTHER),
        ("I lost my black Sony headphones near the library yesterday around 4 PM.", INTENT_REPORT),
        ("I lost my ID card in the library", INTENT_REPORT),
        ("I lost something.", INTENT_REPORT),
        # A concrete item description wins over a trailing service question.
        ("I left my blue backpack at the gym reception, where can I get it?", INTENT_REPORT),
    ],
)
def test_intent_rules(message, intent):
    assert classify_message(message).intent == intent


def test_reference_is_normalised():
    assert find_reference("can you check lf 1024 please") == "LF1024"
    assert find_reference("case esc-500?") == "ESC500"
    assert find_reference("no reference here") is None


def test_pending_ownership_question_takes_answers_but_not_side_questions():
    assert classify_message("There is a scratch on the left earcup.", AWAITING_OWNERSHIP_PROOF).intent == INTENT_ANSWER
    assert classify_message("It's black.", AWAITING_OWNERSHIP_PROOF).intent == INTENT_ANSWER
    assert classify_message("How can I contact the staff?", AWAITING_OWNERSHIP_PROOF).intent == INTENT_HELP


def test_clarification_replies_stay_on_the_item():
    decision = classify_message("I really can't remember any more than that.", AWAITING_CLARIFICATION)
    assert decision.intent == INTENT_REPORT


# --- help knowledge base -------------------------------------------------------


def test_help_search_finds_the_right_article():
    assert search_help("how long do you keep items")[0]["slug"] == "faq-retention"
    assert search_help("When is the library desk open?")[0]["desk_location"] == "Library Security Desk"


def test_every_found_item_location_has_a_desk():
    from sqlalchemy import select

    from app.database.connection import session_scope
    from app.database.models import FoundItem

    with session_scope() as session:
        locations = set(session.scalars(select(FoundItem.location)).all())
    assert all(get_desk_contact(location) for location in locations)


def test_staff_question_gets_a_grounded_contact_not_a_search():
    state = run_turn(None, "How can I reach this staff?")

    assert state["next_action"] == ACTION_ANSWER_HELP
    assert steps(state) == ["CLASSIFY_INTENT", "SEARCH_HELP"]
    assert state["help_result"]["grounded"] is True
    assert state["help_result"]["sources"][0]["slug"] == "faq-contact-staff"
    assert get_desk_contact(CENTRAL_DESK_LOCATION)["phone"] in state["agent_response"]
    assert "category" not in state["agent_response"]
    assert state["lost_item_id"] is None
    assert state["candidate_matches"] == []


def test_unanswerable_question_is_not_guessed():
    state = run_turn(None, "What's the capital of France?")

    assert state["next_action"] == ACTION_ANSWER_HELP
    assert state["help_result"]["grounded"] is False
    assert "won't guess" in state["agent_response"]
    assert "Paris" not in state["agent_response"]
    assert state["escalation"] is None


def test_greeting_explains_scope_without_searching():
    state = run_turn(None, "hello")
    assert state["next_action"] == ACTION_OUT_OF_SCOPE
    assert "SEARCH_MATCHES" not in steps(state)


# --- side questions during a pending ownership check ---------------------------


def test_question_during_verification_does_not_count_as_a_failed_attempt(demo_query):
    state = run_turn(None, demo_query)
    assert state["awaiting"] == AWAITING_OWNERSHIP_PROOF
    candidate_id = state["selected_candidate"]["id"]

    state = run_turn(state, "How can I contact the staff?")
    assert state["next_action"] == ACTION_ANSWER_HELP
    assert state["awaiting"] == AWAITING_OWNERSHIP_PROOF
    assert state["verification_status"] == "pending"
    assert state["escalation"] is None
    assert "VERIFY_OWNERSHIP" not in steps(state)
    assert "distinctive feature" in state["agent_response"]
    # The desk holding the matched item is named, but no ownership evidence is.
    assert "Library Security Desk" in state["agent_response"]
    assert "scratch" not in state["agent_response"].lower()

    state = run_turn(state, "There is a scratch on the left earcup.")
    assert state["next_action"] == ACTION_CREATE_PICKUP
    assert state["selected_candidate"]["id"] == candidate_id


# --- escalation contact and reference lookups ----------------------------------


def test_escalation_tells_the_user_how_to_reach_staff(demo_query):
    state = run_turn(None, demo_query)
    state = run_turn(state, "It has a big yellow smiley face painted on it.")

    assert state["next_action"] == ACTION_ESCALATE
    phone = get_desk_contact(CENTRAL_DESK_LOCATION)["phone"]
    assert phone in state["agent_response"]
    assert state["escalation"]["contact"]

    reference = state["escalation"]["escalation_id"]
    follow_up = run_turn(state, "How can I reach this staff?")
    assert follow_up["next_action"] == ACTION_ANSWER_HELP
    assert reference in follow_up["agent_response"]


def test_pickup_reference_lookup(demo_query):
    state = run_turn(None, demo_query)
    state = run_turn(state, "There is a scratch on the left earcup.")
    reference = state["pickup_request"]["pickup_request_id"]

    lookup = run_turn(None, f"What's the status of {reference}?")
    assert lookup["next_action"] == ACTION_CASE_STATUS
    assert lookup["case_lookup"]["found"] is True
    assert "Library Security Desk" in lookup["agent_response"]

    # "My case" without a reference falls back to this session's latest one.
    latest = run_turn(state, "any update on my pickup request?")
    assert latest["case_lookup"]["reference"] == reference


def test_unknown_reference_is_reported_without_details():
    result = lookup_case("ESC999999")
    assert result == {"reference": "ESC999999", "type": "escalation", "found": False, "record": None}

    state = run_turn(None, "status of ESC999999")
    assert "couldn't find" in state["agent_response"]
