"""The LangGraph workflow.

                                START
                                  |
                           CLASSIFY_INTENT
                                  |
               +------------------+-----------------+------------+-------------+
               |                  |                 |            |             |
            answer             report            status        help          other
        (proof pending)           |                 |            |             |
               |                  |            LOOKUP_CASE  SEARCH_HELP  OUT_OF_SCOPE
               |                  |               (end)        (end)         (end)
       VERIFY_OWNERSHIP   UNDERSTAND_REQUEST
               |                  |
               |        NEED_MORE_INFORMATION?
               |          yes            no
               |           |              |
               |        ASK_USER    SEARCH_MATCHES
               |          (end)           |
               |                 CALCULATE_CONFIDENCE
               |                          |
               |                  CONFIDENCE_ROUTER
               |             LOW      MEDIUM/AMBIGUOUS     HIGH
               |              |             |               |
               |     HUMAN_ESCALATION  ASK_CLARIFICATION  REQUEST_VERIFICATION
               |              |            (end)            (end)
        VERIFICATION_ROUTER   |
         PASS        FAIL ----+
          |
      CREATE_PICKUP
          |
      NOTIFY_USER
          |
         END

A turn ends whenever the agent needs the user: ASK_USER, ASK_CLARIFICATION and
REQUEST_VERIFICATION all set ``awaiting`` and stop. The next message re-enters
the graph with the same state, which is how the multi-turn verification handshake
works without holding a thread open.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.state import AWAITING_NOTHING, AgentState, new_state
from app.observability import get_tracer

logger = logging.getLogger(__name__)


def build_graph():
    """Wire the nodes and routers into a compiled LangGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", nodes.classify_intent)
    graph.add_node("answer_help", nodes.answer_help)
    graph.add_node("lookup_case", nodes.lookup_case_node)
    graph.add_node("out_of_scope", nodes.out_of_scope)
    graph.add_node("understand_request", nodes.understand_request)
    graph.add_node("ask_user", nodes.ask_user)
    graph.add_node("search_matches", nodes.search_matches_node)
    graph.add_node("calculate_confidence", nodes.calculate_confidence)
    graph.add_node("ask_clarification", nodes.ask_clarification)
    graph.add_node("request_verification", nodes.request_verification)
    graph.add_node("verify_ownership", nodes.verify_ownership_node)
    graph.add_node("create_pickup", nodes.create_pickup)
    graph.add_node("notify_user", nodes.notify_user_node)
    graph.add_node("human_escalation", nodes.human_escalation)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        nodes.intent_router,
        {
            "lookup_case": "lookup_case",
            "answer_help": "answer_help",
            "out_of_scope": "out_of_scope",
            "understand_request": "understand_request",
            "verify_ownership": "verify_ownership",
        },
    )
    graph.add_edge("lookup_case", END)
    graph.add_edge("answer_help", END)
    graph.add_edge("out_of_scope", END)

    graph.add_conditional_edges(
        "understand_request",
        nodes.need_more_information,
        {"ask_user": "ask_user", "search_matches": "search_matches"},
    )
    graph.add_edge("ask_user", END)

    graph.add_edge("search_matches", "calculate_confidence")
    graph.add_conditional_edges(
        "calculate_confidence",
        nodes.confidence_router,
        {
            "human_escalation": "human_escalation",
            "ask_clarification": "ask_clarification",
            "request_verification": "request_verification",
        },
    )
    graph.add_edge("ask_clarification", END)
    graph.add_edge("request_verification", END)

    graph.add_conditional_edges(
        "verify_ownership",
        nodes.verification_router,
        {"create_pickup": "create_pickup", "human_escalation": "human_escalation"},
    )
    graph.add_edge("create_pickup", "notify_user")
    graph.add_edge("notify_user", END)
    graph.add_edge("human_escalation", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def run_turn(state: AgentState | None, message: str, user_email: str | None = None) -> AgentState:
    """Run one conversational turn and return the updated state.

    Pass the state returned by the previous turn to continue a conversation;
    pass ``None`` to start a new one.
    """
    if state is None:
        state = new_state(session_id=str(uuid.uuid4()), user_email=user_email)

    turn_state: AgentState = {
        **state,
        "user_message": message,
        "agent_response": "",
        "next_action": "",
        "intent": "",
        "intent_reference": None,
        "help_result": None,
        "case_lookup": None,
        "trace": [],
        "conversation_history": [*state.get("conversation_history", []), {"role": "user", "content": message}],
    }

    # A finished case (collected or escalated) starts a fresh search next time,
    # but its reference is kept so "how do I reach staff?" can still quote it.
    if state.get("awaiting", AWAITING_NOTHING) == AWAITING_NOTHING and (
        state.get("pickup_request") or state.get("escalation")
    ):
        fresh = new_state(session_id=state.get("session_id", str(uuid.uuid4())), user_email=user_email)
        turn_state = {
            **fresh,
            "user_message": message,
            "case_references": state.get("case_references", []),
            "conversation_history": turn_state["conversation_history"],
        }

    tracer = get_tracer()
    with tracer.span(
        "lost_found_agent_turn",
        input={"message": message},
        session_id=turn_state.get("session_id"),
    ) as span:
        result: dict[str, Any] = get_graph().invoke(turn_state)
        span.update(
            output={
                "next_action": result.get("next_action"),
                "awaiting": result.get("awaiting"),
                "response": result.get("agent_response"),
            }
        )
    tracer.flush()

    final: AgentState = {**turn_state, **result}
    final["conversation_history"] = [
        *turn_state["conversation_history"],
        {"role": "assistant", "content": final.get("agent_response", "")},
    ]
    return final


def render_trace(state: AgentState) -> str:
    """Flatten the trace into the arrow diagram shown in the debug panel."""
    lines: list[str] = ["USER INPUT"]
    for entry in state.get("trace", []):
        detail = entry["detail"]
        step = entry["step"]
        if step == "CLASSIFY_INTENT":
            lines.append(f"CLASSIFY_INTENT [{detail.get('method')}] -> {detail.get('intent')}")
        elif step == "SEARCH_HELP":
            outcome = "grounded" if detail.get("grounded") else "no reliable source"
            cited = ", ".join(detail.get("cited") or []) or "none"
            lines.append(
                f"search_help() [{detail.get('method')}] -> top similarity = "
                f"{detail.get('top_similarity', 0):.2f}, {outcome}; cited: {cited}"
            )
        elif step == "LOOKUP_CASE":
            lines.append(f"lookup_case({detail.get('reference')}) -> found={detail.get('found')}")
        elif step == "OUT_OF_SCOPE":
            lines.append("OUT_OF_SCOPE -> explain what the agent can do")
        elif step == "UNDERSTAND_REQUEST":
            extracted = detail.get("extracted", {})
            known = ", ".join(f"{k}={v}" for k, v in extracted.items() if v) or "nothing extracted"
            lines.append(f"Extract item information [{detail.get('extracted_by')}] -> {known}")
        elif step == "SEARCH_MATCHES":
            results = detail.get("results", [])
            best = f"{results[0]['similarity']:.2f}" if results else "n/a"
            lines.append(f"search_matches() -> {len(results)} hits, top similarity = {best}")
        elif step == "CALCULATE_CONFIDENCE":
            scores = detail.get("scores", [])
            best = f"{scores[0]['score']:.2f} ({scores[0]['level']})" if scores else "n/a"
            lines.append(f"Confidence = {best}")
        elif step == "CONFIDENCE_ROUTER":
            lines.append(f"CONFIDENCE_ROUTER -> {detail.get('decision')}")
        elif step == "VERIFY_OWNERSHIP":
            lines.append(
                f"verify_ownership() [{detail.get('method')}] -> {detail.get('result')} "
                f"({detail.get('match_score'):.2f})"
            )
        elif step == "CREATE_PICKUP":
            lines.append(f"create_pickup_request() -> {detail.get('pickup_request_id')}")
        elif step == "NOTIFY_USER":
            lines.append("notify_user()")
        elif step == "HUMAN_ESCALATION":
            lines.append(f"escalate_to_staff() -> {detail.get('escalation_id')}")
        elif step == "NEED_MORE_INFORMATION":
            lines.append(f"NEED_MORE_INFORMATION -> {detail.get('decision')}")
        else:
            lines.append(step)
    return "\n  ↓\n".join(lines)
