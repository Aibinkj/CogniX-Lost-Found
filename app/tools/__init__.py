"""Callable agent tools.

Every tool is a plain Python function with a typed signature that reads or
writes the real database. The LangGraph nodes call these directly; an MCP
server (FastMCP) or an LLM tool-calling loop can wrap the same functions
without changes.
"""

from app.safety.escalation import escalate_to_staff
from app.tools.case_status import lookup_case
from app.tools.found_item import get_found_item, list_found_items, register_found_item
from app.tools.help import get_desk_contact, search_help
from app.tools.lost_item import report_lost_item
from app.tools.notification import notify_user
from app.tools.pickup import create_pickup_request, get_pickup_request
from app.tools.search import search_matches
from app.tools.verification import verify_ownership

TOOLS = {
    "report_lost_item": report_lost_item,
    "register_found_item": register_found_item,
    "search_matches": search_matches,
    "verify_ownership": verify_ownership,
    "create_pickup_request": create_pickup_request,
    "notify_user": notify_user,
    "escalate_to_staff": escalate_to_staff,
    "search_help": search_help,
    "get_desk_contact": get_desk_contact,
    "lookup_case": lookup_case,
}

__all__ = [
    "TOOLS",
    "create_pickup_request",
    "escalate_to_staff",
    "get_desk_contact",
    "get_found_item",
    "get_pickup_request",
    "list_found_items",
    "lookup_case",
    "notify_user",
    "register_found_item",
    "report_lost_item",
    "search_help",
    "search_matches",
    "verify_ownership",
]
