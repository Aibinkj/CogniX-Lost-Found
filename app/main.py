"""FastAPI backend.

Exposes the agent and the tools over HTTP. The Streamlit UI talks to the agent
in-process by default (one less thing to start for a demo) and can be pointed at
this API instead - see `streamlit_app.py`.

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import render_trace, run_turn
from app.agent.state import AgentState
from app.config import configure_logging, settings
from app.database.connection import database_is_reachable
from app.database.vector_support import current_backend
from app.observability import get_tracer
from app.rag.embeddings import get_embedder
from app.safety.escalation import list_escalations
from app.tools.found_item import list_found_items, register_found_item
from app.tools.pickup import get_pickup_request
from app.tools.search import search_matches
from app.tools.verification import verify_ownership

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lost & Found AI Agent",
    version="0.1.0",
    description="Semantic matching, confidence scoring and ownership verification for lost property.",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    state: dict[str, Any] | None = None
    user_email: str | None = None


class ChatResponse(BaseModel):
    response: str
    next_action: str
    awaiting: str
    state: dict[str, Any]
    trace: list[dict[str, Any]]
    trace_text: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = None
    category: str | None = None


class VerifyRequest(BaseModel):
    candidate_id: int
    user_answer: str = Field(min_length=1)


class FoundItemRequest(BaseModel):
    category: str
    description: str
    location: str
    brand: str | None = None
    color: str | None = None
    hidden_features: str = ""
    found_time: datetime | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    """Startup diagnostics: which backends actually came up."""
    reachable = database_is_reachable()
    return {
        "status": "ok" if reachable else "degraded",
        "database": "reachable" if reachable else "unreachable",
        "vector_backend": current_backend(),
        "embedding_backend": get_embedder().name,
        "llm": f"{settings.llm_provider}:{settings.llm_model}" if settings.llm_enabled else "disabled",
        "langfuse": "tracing" if get_tracer().enabled else "disabled",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run one agent turn. Echo `state` back on the next call to continue."""
    previous: AgentState | None = request.state  # type: ignore[assignment]
    try:
        result = run_turn(previous, request.message, user_email=request.user_email)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent turn failed.")
        raise HTTPException(status_code=500, detail=f"Agent failure: {exc}") from exc

    return ChatResponse(
        response=result.get("agent_response", ""),
        next_action=result.get("next_action", ""),
        awaiting=result.get("awaiting", ""),
        state=dict(result),
        trace=result.get("trace", []),
        trace_text=render_trace(result),
    )


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    """Semantic search. Never returns hidden ownership features."""
    results = search_matches(request.query, top_k=request.top_k, category=request.category)
    return {"query": request.query, "count": len(results), "results": results}


@app.post("/verify")
def verify(request: VerifyRequest) -> dict[str, Any]:
    return verify_ownership(request.candidate_id, request.user_answer)


@app.get("/found-items")
def found_items(status: str | None = "available", limit: int = 100) -> dict[str, Any]:
    items = list_found_items(status=status, limit=limit)
    return {"count": len(items), "items": items}


@app.post("/found-items", status_code=201)
def create_found_item(request: FoundItemRequest) -> dict[str, Any]:
    try:
        return register_found_item(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/pickup-requests/{reference}")
def pickup_request(reference: str) -> dict[str, Any]:
    result = get_pickup_request(reference)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No pickup request {reference}.")
    return result


@app.get("/escalations")
def escalations(status: str | None = "open", limit: int = 50) -> dict[str, Any]:
    rows = list_escalations(status=status, limit=limit)
    return {"count": len(rows), "escalations": rows}
