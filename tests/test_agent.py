"""End-to-end agent workflow tests."""

from __future__ import annotations

from app.agent.extraction import extract_with_rules
from app.agent.graph import render_trace, run_turn
from app.agent.state import (
    ACTION_ASK_CLARIFICATION,
    ACTION_ASK_USER,
    ACTION_CREATE_PICKUP,
    ACTION_ESCALATE,
    ACTION_REQUEST_VERIFICATION,
    AWAITING_OWNERSHIP_PROOF,
)


def steps(state) -> list[str]:
    return [entry["step"] for entry in state.get("trace", [])]


# --- the demo scenario -----------------------------------------------------


def test_demo_scenario_end_to_end(demo_query, sony_headphones_id):
    state = run_turn(None, demo_query)

    assert state["next_action"] == ACTION_REQUEST_VERIFICATION
    assert state["awaiting"] == AWAITING_OWNERSHIP_PROOF
    assert state["selected_candidate"]["id"] == sony_headphones_id
    assert state["confidence"]["level"] == "HIGH"
    assert "distinctive feature" in state["agent_response"].lower()

    state = run_turn(state, "There is a scratch on the left earcup.")

    assert state["next_action"] == ACTION_CREATE_PICKUP
    assert state["verification_status"] == "passed"
    assert state["pickup_request"]["pickup_request_id"].startswith("LF")
    assert state["pickup_request"]["location"] == "Library Security Desk"
    assert "Match verified" in state["agent_response"]


def test_demo_scenario_extracts_every_field(demo_query):
    details = extract_with_rules(demo_query)
    assert details.category == "headphones"
    assert details.brand == "Sony"
    assert details.color == "black"
    assert details.location == "library"
    assert details.lost_time is not None


def test_agent_never_reveals_hidden_features_before_verification(demo_query):
    state = run_turn(None, demo_query)
    response = state["agent_response"].lower()
    assert "scratch" not in response
    assert "sticker" not in response
    assert "hidden_features" not in str(state["candidate_matches"])


# --- failure and safety paths ---------------------------------------------


def test_wrong_ownership_answer_escalates(demo_query):
    state = run_turn(None, demo_query)
    state = run_turn(state, "It has a big yellow smiley face painted on it.")

    assert state["next_action"] == ACTION_ESCALATE
    assert state["verification_status"] == "failed"
    assert state["escalation"]["escalation_id"].startswith("ESC")
    assert state["escalation"]["verification_result"] == "FAIL"
    assert state["pickup_request"] is None


def test_no_match_escalates_rather_than_guessing():
    state = run_turn(None, "I lost my green pet iguana near the botanical greenhouse this morning.")
    assert state["next_action"] == ACTION_ESCALATE
    assert state["escalation"] is not None
    assert state["pickup_request"] is None


def test_nothing_close_is_worded_differently_to_a_failed_verification():
    """"We hold nothing like that" is a different message to "we can't be sure"."""
    nothing = run_turn(None, "I lost my green pet iguana near the botanical greenhouse.")
    assert "couldn't find anything matching" in nothing["agent_response"]
    assert "similarity" in nothing["escalation"]["reason"]


def test_terse_report_asks_for_detail_instead_of_escalating():
    """"I lost my keys" is thin, not unmatched: ask, then match once details arrive."""
    state = run_turn(None, "I lost my keys")
    assert state["next_action"] == ACTION_ASK_CLARIFICATION
    assert state["escalation"] is None
    assert "where you lost them" in state["agent_response"]
    assert state["agent_response"].endswith("?")

    state = run_turn(state, "They're silver, I think I left them at main reception this morning")
    assert state["next_action"] == ACTION_REQUEST_VERIFICATION
    assert state["selected_candidate"]["category"] == "keys"


def test_terse_report_escalates_after_the_clarification_cap():
    state = run_turn(None, "I lost my keys")
    for _ in range(3):
        if state["next_action"] == ACTION_ESCALATE:
            break
        state = run_turn(state, "I don't know")
    assert state["next_action"] == ACTION_ESCALATE
    # Keys are held, just not confidently these ones - so don't claim nothing matched.
    assert "couldn't find anything matching" not in state["agent_response"]
    assert "possible matches" in state["agent_response"]


def test_new_details_after_the_cap_earn_another_question():
    """Two unhelpful replies, then real details: ask about what's missing, don't escalate."""
    state = run_turn(None, "I lost my phone")
    state = run_turn(state, "not sure")
    assert state["clarification_rounds"] == 2

    state = run_turn(state, "I lost a green samsung phone")
    assert state["next_action"] == ACTION_ASK_CLARIFICATION
    assert state["escalation"] is None


def test_contradicted_brand_is_not_offered_as_a_likely_match():
    """No green Samsung is held; the green Google at the right desk must not be HIGH."""
    state = run_turn(None, "I lost a green samsung phone")
    state = run_turn(state, "at the bus stop")
    assert state["next_action"] != ACTION_REQUEST_VERIFICATION
    assert state["pickup_request"] is None


def test_vague_input_asks_for_more_information():
    state = run_turn(None, "I lost something.")
    assert state["next_action"] == ACTION_ASK_USER
    assert "NEED_MORE_INFORMATION" in steps(state)
    assert state["candidate_matches"] == []


def test_follow_up_details_are_merged_and_searched():
    state = run_turn(None, "I lost something.")
    assert state["next_action"] == ACTION_ASK_USER

    state = run_turn(state, "It was a teal Hydro Flask water bottle at the sports centre.")
    assert state["next_action"] != ACTION_ASK_USER
    assert state["item_details"]["category"] == "water_bottle"
    assert state["candidate_matches"]


def test_ambiguous_candidates_trigger_clarification_not_a_claim():
    """A colourless, brandless report over near-identical items must not claim."""
    state = run_turn(None, "I lost my headphones somewhere near the library yesterday.")
    assert state["next_action"] in {ACTION_ASK_CLARIFICATION, ACTION_ESCALATE}
    assert state["pickup_request"] is None
    if state["next_action"] == ACTION_ASK_CLARIFICATION:
        assert state["agent_response"].strip().endswith("?")


def test_repeated_ambiguity_eventually_escalates():
    state = run_turn(None, "I lost my headphones somewhere near the library yesterday.")
    for _ in range(3):
        if state["next_action"] == ACTION_ESCALATE:
            break
        state = run_turn(state, "I really can't remember any more than that.")
    assert state["next_action"] in {ACTION_ESCALATE, ACTION_REQUEST_VERIFICATION}
    assert state["pickup_request"] is None


def test_generic_ownership_answer_does_not_release_the_item(demo_query):
    state = run_turn(None, demo_query)
    state = run_turn(state, "It's black.")
    assert state["verification_status"] == "failed"
    assert state["pickup_request"] is None


# --- observability ---------------------------------------------------------


def test_trace_records_the_whole_workflow(demo_query):
    state = run_turn(None, demo_query)
    assert steps(state) == [
        "CLASSIFY_INTENT",
        "UNDERSTAND_REQUEST",
        "SEARCH_MATCHES",
        "CALCULATE_CONFIDENCE",
        "CONFIDENCE_ROUTER",
    ]

    state = run_turn(state, "There is a scratch on the left earcup.")
    assert steps(state) == ["CLASSIFY_INTENT", "VERIFY_OWNERSHIP", "CREATE_PICKUP", "NOTIFY_USER"]


def test_rendered_trace_is_human_readable(demo_query):
    state = run_turn(None, demo_query)
    rendered = render_trace(state)
    assert "USER INPUT" in rendered
    assert "search_matches()" in rendered
    assert "Confidence" in rendered


def test_a_completed_case_starts_a_fresh_search(demo_query):
    state = run_turn(None, demo_query)
    state = run_turn(state, "There is a scratch on the left earcup.")
    assert state["pickup_request"] is not None

    state = run_turn(state, "I also lost a teal Hydro Flask at the sports centre.")
    assert state["pickup_request"] is None
    assert state["item_details"]["category"] == "water_bottle"


def test_the_same_item_cannot_be_claimed_twice(demo_query):
    state = run_turn(None, demo_query)
    state = run_turn(state, "There is a scratch on the left earcup.")
    reference = state["pickup_request"]["pickup_request_id"]

    second = run_turn(None, demo_query)
    if second["next_action"] == ACTION_REQUEST_VERIFICATION:
        assert second["selected_candidate"]["id"] != state["selected_candidate"]["id"]
    assert reference.startswith("LF")


def test_lost_report_is_persisted(demo_query):
    state = run_turn(None, demo_query)
    assert state["lost_item_id"] is not None
    assert state["errors"] == []
