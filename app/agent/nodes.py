"""LangGraph nodes.

Each node does one job, calls real tools, records a trace entry, and returns a
state delta. Routing between them is driven by the state (confidence level,
ambiguity, verification outcome) - that is what makes this an agent rather than
a scripted chat flow.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable

from app.agent.extraction import ItemDetails, extract_item_details
from app.agent.help_answer import answer_help_question
from app.agent.intent import INTENT_HELP, INTENT_OTHER, INTENT_STATUS, classify_message
from app.agent.state import (
    ACTION_ANSWER_HELP,
    ACTION_ASK_CLARIFICATION,
    ACTION_ASK_USER,
    ACTION_CASE_STATUS,
    ACTION_CREATE_PICKUP,
    ACTION_ESCALATE,
    ACTION_OUT_OF_SCOPE,
    ACTION_REQUEST_VERIFICATION,
    AWAITING_CLARIFICATION,
    AWAITING_MORE_INFO,
    AWAITING_NOTHING,
    AWAITING_OWNERSHIP_PROOF,
    MAX_CLARIFICATION_ROUNDS,
    AgentState,
    trace_step,
)
from app.config import settings
from app.observability import get_tracer
from app.safety.confidence import LEVEL_HIGH, LEVEL_LOW, is_ambiguous, score_candidates
from app.safety.escalation import ESCALATION_MESSAGE, escalate_to_staff
from app.tools.case_status import lookup_case
from app.tools.help import CENTRAL_DESK_LOCATION, format_contact, get_desk_contact
from app.tools.lost_item import report_lost_item
from app.tools.notification import notify_user
from app.tools.pickup import create_pickup_request, get_pickup_request
from app.tools.search import search_matches
from app.tools.verification import build_verification_question, verify_ownership

logger = logging.getLogger(__name__)

# Below this similarity, the nearest neighbour is not a plausible match at all,
# which is a different message to the user than "found something, not sure".
NO_MATCH_SIMILARITY_FLOOR = 0.45

# Appended when a side question interrupts a pending ownership check, which stays open.
PENDING_PROOF_REMINDER = (
    "Whenever you're ready, describe one distinctive feature of your item so I can verify it's yours."
)


def traced(name: str) -> Callable:
    """Wrap a node in a Langfuse span and turn crashes into escalations."""

    def decorator(func: Callable[[AgentState], dict[str, Any]]):
        @wraps(func)
        def wrapper(state: AgentState) -> dict[str, Any]:
            tracer = get_tracer()
            with tracer.span(
                f"node:{name}",
                input={"user_message": state.get("user_message", "")},
                session_id=state.get("session_id"),
            ) as span:
                try:
                    result = func(state)
                except Exception as exc:  # noqa: BLE001 - degrade to a safe answer
                    logger.exception("Node %s failed.", name)
                    return _fail_safe(state, name, exc)
                span.update(output={"next_action": result.get("next_action", "")})
                return result

        return wrapper

    return decorator


def _fail_safe(state: AgentState, node: str, exc: Exception) -> dict[str, Any]:
    """A node crashed: never guess, hand the case to a human."""
    return {
        "next_action": ACTION_ESCALATE,
        "awaiting": AWAITING_NOTHING,
        "agent_response": (
            "Something went wrong while processing your request, so I've passed it to "
            "staff rather than risk a wrong match."
        ),
        "errors": [*state.get("errors", []), f"{node}: {exc}"],
        "trace": trace_step(state, node, {"error": str(exc)}),
    }


def _details(state: AgentState) -> ItemDetails:
    return ItemDetails.from_dict(state.get("item_details")) or ItemDetails()


def _summarize(candidate: dict[str, Any]) -> str:
    parts = [candidate.get("color"), candidate.get("brand"), candidate.get("category")]
    label = " ".join(str(p) for p in parts if p) or candidate.get("description", "item")
    return f"{label} (handed in at {candidate.get('location')})"


def _is_thin_report(details: ItemDetails) -> bool:
    """Too little said to judge a match: no location, or nothing about how it looks."""
    return details.location is None or not (details.color or details.brand)


def _has_plausible_candidate(scored: list[dict[str, Any]], details: ItemDetails) -> bool:
    """Something held is at least the same kind of item, or semantically close."""
    if not scored:
        return False
    if details.category and any(row.get("category") == details.category for row in scored):
        return True
    return scored[0]["similarity"] >= NO_MATCH_SIMILARITY_FLOOR


def _remember_reference(state: AgentState, reference: str) -> list[str]:
    references = [ref for ref in state.get("case_references", []) if ref != reference]
    return [*references, reference][-5:]


def _with_pending_proof_reminder(state: AgentState, response: str) -> str:
    if state.get("awaiting") == AWAITING_OWNERSHIP_PROOF and state.get("selected_candidate"):
        return f"{response}\n\n{PENDING_PROOF_REMINDER}"
    return response


def _context_location(state: AgentState) -> str | None:
    """The desk the user's current match or latest pickup is held at, if any."""
    if state.get("awaiting") == AWAITING_OWNERSHIP_PROOF and state.get("selected_candidate"):
        return state["selected_candidate"].get("location")
    if state.get("pickup_request"):
        return state["pickup_request"].get("location")
    for reference in reversed(state.get("case_references", [])):
        if reference.startswith("LF") and (pickup := get_pickup_request(reference)):
            return pickup["location"]
    return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@traced("classify_intent")
def classify_intent(state: AgentState) -> dict[str, Any]:
    """Decide what the message is before treating it as an item description."""
    decision = classify_message(state.get("user_message", ""), awaiting=state.get("awaiting", AWAITING_NOTHING))
    return {
        "intent": decision.intent,
        "intent_reference": decision.reference,
        "trace": trace_step(state, "CLASSIFY_INTENT", decision.to_dict()),
    }


@traced("answer_help")
def answer_help(state: AgentState) -> dict[str, Any]:
    """Answer a question about the service from the help knowledge base only."""
    result = answer_help_question(
        state.get("user_message", ""),
        references=state.get("case_references", []),
        context_location=_context_location(state),
    )
    return {
        "help_result": result,
        "next_action": ACTION_ANSWER_HELP,
        "agent_response": _with_pending_proof_reminder(state, result["answer"]),
        "trace": trace_step(
            state,
            "SEARCH_HELP",
            {
                "tool": "search_help",
                "method": result["method"],
                "grounded": result["grounded"],
                "top_similarity": result["top_similarity"],
                "threshold": settings.help_min_similarity,
                "retrieved": result["retrieved"],
                "cited": [source["slug"] for source in result["sources"]],
            },
        ),
    }


@traced("lookup_case")
def lookup_case_node(state: AgentState) -> dict[str, Any]:
    """Report the status of a pickup request or staff case by reference."""
    references = state.get("case_references", [])
    reference = state.get("intent_reference") or (references[-1] if references else None)
    central = format_contact(get_desk_contact(CENTRAL_DESK_LOCATION))

    if reference is None:
        return {
            "next_action": ACTION_CASE_STATUS,
            "agent_response": _with_pending_proof_reminder(
                state,
                "Which reference should I check? It looks like LF1024 for a pickup request "
                "or ESC500 for a staff case.",
            ),
            "trace": trace_step(state, "LOOKUP_CASE", {"tool": "lookup_case", "reference": None}),
        }

    result = lookup_case(reference)
    record = result["record"] or {}
    if not result["found"]:
        response = f"I couldn't find a pickup request or case with reference {reference}. Please check it"
        response += f", or contact {central}." if central else "."
    elif result["type"] == "pickup":
        desk = get_desk_contact(record.get("location"))
        response = (
            f"Pickup request {reference} is {record['status']}. Collect it from "
            f"{format_contact(desk) if desk else record['location']} and quote the reference."
        )
    else:
        response = f"Case {reference} is {record['status']} with staff, and they'll follow up with you."
        if central:
            response += f" To chase it, contact {central} and quote {reference}."

    return {
        "case_lookup": result,
        "next_action": ACTION_CASE_STATUS,
        "agent_response": _with_pending_proof_reminder(state, response),
        "trace": trace_step(
            state,
            "LOOKUP_CASE",
            {"tool": "lookup_case", "reference": reference, "found": result["found"], "type": result["type"]},
        ),
    }


@traced("out_of_scope")
def out_of_scope(state: AgentState) -> dict[str, Any]:
    """Greetings and unrelated messages: say what the agent can do, don't search."""
    return {
        "next_action": ACTION_OUT_OF_SCOPE,
        "agent_response": _with_pending_proof_reminder(
            state,
            "I'm the campus Lost & Found assistant. I can search for something you've lost, "
            "answer questions about desks, staff contacts and collection, or check a reference "
            "like LF1024. What can I help you with?",
        ),
        "trace": trace_step(state, "OUT_OF_SCOPE", {"intent": state.get("intent")}),
    }


@traced("understand_request")
def understand_request(state: AgentState) -> dict[str, Any]:
    """Extract structured item details and persist the lost-item report."""
    message = state.get("user_message", "")
    extracted = extract_item_details(message)

    previous = ItemDetails.from_dict(state.get("item_details"))
    details = previous.merge(extracted) if previous else extracted

    updates: dict[str, Any] = {"item_details": details.to_dict()}
    errors = list(state.get("errors", []))

    if state.get("lost_item_id") is None:
        try:
            record = report_lost_item(
                description=details.description or message,
                category=details.category,
                brand=details.brand,
                color=details.color,
                location=details.location,
                lost_time=details.lost_time,
                raw_query=message,
                user_email=state.get("user_email"),
            )
            updates["lost_item_id"] = record["id"]
        except Exception as exc:  # noqa: BLE001 - matching still works unrecorded
            logger.warning("Could not persist the lost-item report: %s", exc)
            errors.append(f"report_lost_item: {exc}")

    updates["errors"] = errors
    updates["trace"] = trace_step(
        state,
        "UNDERSTAND_REQUEST",
        {
            "tool": "report_lost_item",
            "extracted_by": details.extracted_by,
            "extracted": {
                "category": details.category,
                "brand": details.brand,
                "color": details.color,
                "location": details.location,
                "time": details.time_phrase,
            },
            "lost_item_id": updates.get("lost_item_id", state.get("lost_item_id")),
        },
    )
    return updates


@traced("ask_user")
def ask_user(state: AgentState) -> dict[str, Any]:
    """Too little to search on - ask for the missing attributes."""
    details = _details(state)
    missing = details.missing_fields()
    wanted = " and ".join(missing[:2]) if missing else "a bit more detail"

    return {
        "next_action": ACTION_ASK_USER,
        "awaiting": AWAITING_MORE_INFO,
        "agent_response": (
            f"I can help you find that. Could you tell me the {wanted}? "
            "For example: \"a black Sony headset, lost near the library yesterday afternoon\"."
        ),
        "trace": trace_step(state, "NEED_MORE_INFORMATION", {"missing": missing, "decision": "ASK_USER"}),
    }


@traced("search_matches")
def search_matches_node(state: AgentState) -> dict[str, Any]:
    """Semantic retrieval over available found items."""
    details = _details(state)
    query = details.search_text() or state.get("user_message", "")
    candidates = search_matches(query, top_k=settings.search_top_k)

    return {
        "candidate_matches": candidates,
        "trace": trace_step(
            state,
            "SEARCH_MATCHES",
            {
                "tool": "search_matches",
                "query": query,
                "top_k": settings.search_top_k,
                "results": [
                    {"id": c["id"], "similarity": c["similarity"], "description": c["description"]}
                    for c in candidates
                ],
            },
        ),
    }


@traced("calculate_confidence")
def calculate_confidence(state: AgentState) -> dict[str, Any]:
    """Blend the retrieval score with the structured signals."""
    details = _details(state)
    scored = score_candidates(state.get("candidate_matches", []), details)

    top = scored[0]["confidence"] if scored else None
    return {
        "candidate_matches": scored,
        "confidence": top,
        "trace": trace_step(
            state,
            "CALCULATE_CONFIDENCE",
            {
                "weights": settings.weights,
                "scores": [
                    {
                        "id": row["id"],
                        "score": row["confidence"]["score"],
                        "level": row["confidence"]["level"],
                        "signals": row["confidence"]["signals"],
                    }
                    for row in scored
                ],
            },
        ),
    }


@traced("ask_clarification")
def ask_clarification(state: AgentState) -> dict[str, Any]:
    """Several plausible items, or one only moderately likely - ask, don't guess."""
    scored = state.get("candidate_matches", [])
    details = _details(state)
    rounds = state.get("clarification_rounds", 0) + 1

    # Low confidence on a thin report means "not enough said", not "no match":
    # ask for what's missing instead of naming candidates.
    if scored and scored[0]["confidence"]["level"] == LEVEL_LOW:
        plural = (details.category or "").endswith("s")
        wanted = []
        if not details.location:
            wanted.append(f"where you lost {'them' if plural else 'it'}")
        if details.lost_time is None:
            wanted.append("roughly when")
        if not (details.color or details.brand):
            look = "they look" if plural else "it looks"
            wanted.append(f"what {look} like - colour, brand or anything attached")
        listed = ", ".join(wanted[:-1]) + f" and {wanted[-1]}" if len(wanted) > 1 else wanted[0]
        return {
            "next_action": ACTION_ASK_CLARIFICATION,
            "awaiting": AWAITING_CLARIFICATION,
            "clarification_rounds": rounds,
            "agent_response": (
                "Nothing I'm holding is a confident match yet, so I need a little more detail. "
                f"Could you tell me {listed}?"
            ),
            "trace": trace_step(
                state,
                "CONFIDENCE_ROUTER",
                {
                    "decision": "ASK_CLARIFICATION",
                    "reason": "low confidence on a thin report",
                    "missing": wanted,
                    "round": rounds,
                },
            ),
        }

    differing: list[str] = []
    for field in ("color", "brand", "location", "category"):
        values = {str(row.get(field)) for row in scored[:3] if row.get(field)}
        if len(values) > 1:
            differing.append(field)

    # Only worth asking about an attribute that both separates the candidates
    # and the user has not already told us.
    questions: dict[str, str] = {
        "color": "What colour was it, exactly?",
        "brand": "Do you remember the brand?",
        "location": "Whereabouts did you last have it?",
        "category": "Could you describe the item itself a little more?",
    }
    unknown = [field for field in differing if not getattr(details, field, None)]

    if unknown:
        question = questions[unknown[0]]
    elif details.lost_time is None:
        question = "Roughly when did you lose it?"
    else:
        question = (
            "Could you add any other detail - anything unusual about it, or "
            "exactly where you last had it?"
        )

    count = len(scored)
    return {
        "next_action": ACTION_ASK_CLARIFICATION,
        "awaiting": AWAITING_CLARIFICATION,
        "clarification_rounds": rounds,
        "agent_response": (
            f"I found {count} possible {'match' if count == 1 else 'matches'}, but none I'm "
            f"confident enough about to hand over. {question}"
        ),
        "trace": trace_step(
            state,
            "CONFIDENCE_ROUTER",
            {
                "decision": "ASK_CLARIFICATION",
                "reason": "ambiguous or medium confidence",
                "differing_attributes": differing,
                "round": rounds,
            },
        ),
    }


@traced("request_verification")
def request_verification(state: AgentState) -> dict[str, Any]:
    """High confidence: name the match, then demand ownership evidence."""
    scored = state.get("candidate_matches", [])
    top = scored[0]
    confidence = top["confidence"]

    question = build_verification_question()
    percent = round(confidence["score"] * 100)

    return {
        "selected_candidate": top,
        "confidence": confidence,
        "next_action": ACTION_REQUEST_VERIFICATION,
        "awaiting": AWAITING_OWNERSHIP_PROOF,
        "verification_status": "pending",
        "agent_response": (
            f"I found a likely match: {_summarize(top)}. "
            f"Confidence: {percent}%.\n\n{question}"
        ),
        "trace": trace_step(
            state,
            "CONFIDENCE_ROUTER",
            {
                "decision": "VERIFY_OWNERSHIP",
                "candidate_id": top["id"],
                "score": confidence["score"],
                "level": confidence["level"],
                "reasons": confidence["reasons"],
            },
        ),
    }


@traced("verify_ownership")
def verify_ownership_node(state: AgentState) -> dict[str, Any]:
    """Check the claimant's answer against the stored ownership evidence."""
    candidate = state.get("selected_candidate") or {}
    confidence = state.get("confidence") or {}

    result = verify_ownership(
        candidate_id=int(candidate["id"]),
        user_answer=state.get("user_message", ""),
        lost_item_id=state.get("lost_item_id"),
        confidence_score=confidence.get("score"),
        confidence_level=confidence.get("level"),
    )

    return {
        "verification_result": result,
        "verification_status": "passed" if result["verified"] else "failed",
        "trace": trace_step(
            state,
            "VERIFY_OWNERSHIP",
            {
                "tool": "verify_ownership",
                "candidate_id": candidate.get("id"),
                "method": result["method"],
                "match_score": result["confidence"],
                "result": "PASS" if result["verified"] else "FAIL",
            },
        ),
    }


@traced("create_pickup")
def create_pickup(state: AgentState) -> dict[str, Any]:
    """Reserve the verified item for collection."""
    candidate = state.get("selected_candidate") or {}
    request = create_pickup_request(item_id=int(candidate["id"]))

    return {
        "pickup_request": request,
        "case_references": _remember_reference(state, request["pickup_request_id"]),
        "next_action": ACTION_CREATE_PICKUP,
        "trace": trace_step(
            state,
            "CREATE_PICKUP",
            {
                "tool": "create_pickup_request",
                "pickup_request_id": request["pickup_request_id"],
                "item_id": request["item_id"],
                "location": request["location"],
            },
        ),
    }


@traced("notify_user")
def notify_user_node(state: AgentState) -> dict[str, Any]:
    """Tell the user where and under what reference to collect the item."""
    candidate = state.get("selected_candidate") or {}
    request = state.get("pickup_request") or {}

    label = str(candidate.get("category") or "item").replace("_", " ")
    verb = "are" if label.endswith("s") else "is"
    message = (
        f"Match verified. Your {label} {verb} available at {request.get('location')}. "
        f"Pickup request {request.get('pickup_request_id')} has been created."
    )
    desk = get_desk_contact(request.get("location"))
    if desk and desk.get("hours"):
        message += f" Opening hours: {desk['hours']}."

    notification = notify_user(
        recipient=state.get("user_email") or "anonymous",
        subject=f"Pickup request {request.get('pickup_request_id')}",
        body=message,
    )

    return {
        "notification": notification,
        "awaiting": AWAITING_NOTHING,
        "agent_response": message,
        "trace": trace_step(
            state,
            "NOTIFY_USER",
            {"tool": "notify_user", "notification_id": notification["id"]},
        ),
    }


@traced("human_escalation")
def human_escalation(state: AgentState) -> dict[str, Any]:
    """No confident claim: open a staff review case instead of guessing."""
    scored = state.get("candidate_matches", [])
    verification = state.get("verification_result")

    # Retrieval always returns its nearest neighbours, so "nothing matched"
    # means nothing held is the same kind of item or semantically close.
    best_similarity = scored[0]["similarity"] if scored else 0.0
    nothing_close = not _has_plausible_candidate(scored, _details(state))

    if verification is not None and not verification.get("verified"):
        reason = "Ownership verification failed."
    elif nothing_close:
        reason = f"No semantically similar item is currently held (best similarity {best_similarity:.2f})."
    elif state.get("clarification_rounds", 0) >= MAX_CLARIFICATION_ROUNDS:
        reason = "Candidates remained ambiguous after clarification."
    else:
        reason = (
            f"Best confidence {scored[0]['confidence']['score']:.2f} is below the "
            f"{settings.confidence_low_threshold:.2f} threshold."
        )

    escalation = escalate_to_staff(
        user_request=state.get("user_message", ""),
        reason=reason,
        candidate_ids=[int(row["id"]) for row in scored[:5]],
        confidence_scores={str(row["id"]): row["confidence"]["score"] for row in scored[:5]},
        verification_result=(
            None if verification is None else ("PASS" if verification["verified"] else "FAIL")
        ),
    )

    if verification is not None:
        response = f"{ESCALATION_MESSAGE} Reference: {escalation['escalation_id']}."
    elif nothing_close:
        response = (
            "I couldn't find anything matching that description in what's currently held. "
            "I've logged your report so staff can check it against new items. "
            f"Reference: {escalation['escalation_id']}."
        )
    else:
        response = (
            "I found possible matches, but none close enough to be sure it's yours, so I've "
            "passed your report to staff to check by hand. "
            f"Reference: {escalation['escalation_id']}."
        )

    # The next question after an escalation is usually "how do I reach them?"
    central = get_desk_contact(CENTRAL_DESK_LOCATION)
    if central:
        escalation = {**escalation, "contact": format_contact(central), "contact_is_sample": central["is_sample"]}
        response += f" To reach staff directly: {escalation['contact']}."

    return {
        "escalation": escalation,
        "case_references": _remember_reference(state, escalation["escalation_id"]),
        "next_action": ACTION_ESCALATE,
        "awaiting": AWAITING_NOTHING,
        "agent_response": response,
        "trace": trace_step(
            state,
            "HUMAN_ESCALATION",
            {
                "tool": "escalate_to_staff",
                "escalation_id": escalation["escalation_id"],
                "reason": reason,
                "candidate_ids": escalation["candidate_ids"],
            },
        ),
    }


# ---------------------------------------------------------------------------
# Routers - the decision points
# ---------------------------------------------------------------------------


def intent_router(state: AgentState) -> str:
    """Side questions are answered without touching the case; everything else
    resumes it - a pending ownership question, or the item search."""
    intent = state.get("intent")
    if intent == INTENT_STATUS:
        return "lookup_case"
    if intent == INTENT_HELP:
        return "answer_help"
    if intent == INTENT_OTHER:
        return "out_of_scope"
    if state.get("awaiting") == AWAITING_OWNERSHIP_PROOF and state.get("selected_candidate"):
        return "verify_ownership"
    return "understand_request"


def need_more_information(state: AgentState) -> str:
    """Refuse to search on a description too thin to rank meaningfully."""
    details = _details(state)
    words = len(details.description.split())
    if details.known_field_count() == 0 and words < 6:
        return "ask_user"
    if details.category is None and details.known_field_count() < 2:
        return "ask_user"
    return "search_matches"


def confidence_router(state: AgentState) -> str:
    """Route on the blended score, not on similarity alone."""
    scored = state.get("candidate_matches", [])
    if not scored:
        return "human_escalation"

    top = scored[0]["confidence"]
    rounds = state.get("clarification_rounds", 0)

    if top["level"] == LEVEL_LOW:
        # Low because the user said little, not because nothing fits: ask first.
        details = _details(state)
        if (
            rounds < MAX_CLARIFICATION_ROUNDS
            and _is_thin_report(details)
            and _has_plausible_candidate(scored, details)
        ):
            return "ask_clarification"
        return "human_escalation"

    if is_ambiguous(scored):
        return "ask_clarification" if rounds < MAX_CLARIFICATION_ROUNDS else "human_escalation"

    if top["level"] == LEVEL_HIGH:
        return "request_verification"

    # MEDIUM: worth a question, but not two.
    return "ask_clarification" if rounds < MAX_CLARIFICATION_ROUNDS else "human_escalation"


def verification_router(state: AgentState) -> str:
    return "create_pickup" if state.get("verification_status") == "passed" else "human_escalation"
