"""Answer questions about the service from the help knowledge base - and only from it.

    question -> search_help() -> similarity gate -> LLM answer with [n] citations
                                                 -> or the top article verbatim

Below `HELP_MIN_SIMILARITY` nothing is answered: the agent says it does not know
and points to staff. An LLM reply that cites nothing, or declares the sources
insufficient, is discarded in favour of the retrieved article itself, so every
sentence the user reads traces back to a stored article.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.agent.prompts import HELP_ANSWER_SYSTEM, HELP_ANSWER_USER_TEMPLATE
from app.config import settings
from app.llm import LLMError, get_llm
from app.tools.help import CENTRAL_DESK_LOCATION, format_contact, get_desk_contact, search_help

logger = logging.getLogger(__name__)

NOT_IN_SOURCES = "NOT_IN_SOURCES"
CONTACT_ARTICLE = "faq-contact-staff"
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _source(number: int, article: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": number,
        "slug": article["slug"],
        "title": article["title"],
        "similarity": article.get("similarity"),
        "is_sample": article.get("is_sample", False),
    }


def _answer_with_llm(
    question: str, articles: list[dict[str, Any]], context: list[str]
) -> tuple[str, list[int], str] | None:
    client = get_llm()
    if client is None:
        return None

    sources = "\n\n".join(f"[{n}] {a['title']}\n{a['body']}" for n, a in enumerate(articles, start=1))
    context_block = "Context:\n" + "\n".join(f"- {line}" for line in context) + "\n\n" if context else ""
    try:
        reply = client.complete(
            HELP_ANSWER_SYSTEM,
            HELP_ANSWER_USER_TEMPLATE.format(context=context_block, sources=sources, question=question),
            max_tokens=300,
        ).text.strip()
    except LLMError as exc:
        logger.warning("LLM help answer failed (%s); answering from the top article.", exc)
        return None

    cited = sorted({int(n) for n in _CITATION_RE.findall(reply) if 1 <= int(n) <= len(articles)})
    if NOT_IN_SOURCES in reply or not cited:
        logger.info("LLM help answer was not grounded (%r); answering from the top article.", reply[:120])
        return None
    return reply, cited, f"llm:{client.provider}:{client.model}"


def answer_help_question(
    question: str,
    references: list[str] | None = None,
    context_location: str | None = None,
) -> dict[str, Any]:
    """Grounded answer plus the evidence behind it.

    `references` are the user's recent case references (LF/ESC) and
    `context_location` the desk tied to their current match or pickup, so "how
    do I reach this staff?" can resolve "this".
    """
    hits = search_help(question, top_k=3)
    grounded = [hit for hit in hits if hit["similarity"] >= settings.help_min_similarity]
    retrieved = [_source(n, hit) for n, hit in enumerate(hits, start=1)]
    top_similarity = hits[0]["similarity"] if hits else 0.0

    if not grounded:
        central = get_desk_contact(CENTRAL_DESK_LOCATION)
        answer = "I couldn't find that in my help information, so I won't guess."
        if central:
            answer += f" Lost & Found staff can help with anything I can't: {format_contact(central)}."
        return {
            "answer": answer,
            "grounded": False,
            "method": "no-source",
            "sources": [_source(1, central)] if central else [],
            "retrieved": retrieved,
            "top_similarity": top_similarity,
        }

    articles = list(grounded)
    context_desk = get_desk_contact(context_location)
    if context_desk and all(a["slug"] != context_desk["slug"] for a in articles):
        articles.append(context_desk)

    context: list[str] = []
    if references:
        context.append(f"The user's recent case references: {', '.join(references)}.")
    if context_desk:
        context.append(f"The user's item is held at {context_desk['title']}.")

    if llm := _answer_with_llm(question, articles, context):
        answer, cited, method = llm
        sources = [_source(n, articles[n - 1]) for n in cited]
    else:
        top = articles[0]
        answer, method = f"{top['body']} [1]", "extractive"
        sources = [_source(1, top)]
        # "This staff" right after a match or escalation means the desk involved.
        if top["slug"] == CONTACT_ARTICLE and context_desk:
            answer += f" Your item is held at {format_contact(context_desk)} [2]."
            sources.append(_source(2, context_desk))
        if references and top["slug"] == CONTACT_ARTICLE:
            answer += f" Quote your reference {references[-1]} when you get in touch."

    return {
        "answer": answer,
        "grounded": True,
        "method": method,
        "sources": sources,
        "retrieved": retrieved,
        "top_similarity": top_similarity,
    }
