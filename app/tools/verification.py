"""Ownership verification.

The desk knows things about an item that only its owner should know - a scratch
in a particular place, a sticker inside the case. Those live in
``FoundItem.hidden_features`` and are never returned by search, never rendered
in the UI, and never put in a prompt unless the user has already been matched to
that one item and ``VERIFICATION_USE_LLM`` is on.

Matching is deterministic by default: the claimant's answer is scored against
each recorded mark by weighted token coverage, with synonym and stem handling so
that "a scuff on the left ear cup" matches "scratch on left earcup". Answers made
only of publicly visible attributes ("it's black", "it's a Sony") are rejected
outright, because the search result already told the user those.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import settings
from app.database.connection import session_scope
from app.database.models import Claim, FoundItem
from app.rag.embeddings import tokenize

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "is", "was", "are", "were", "be", "been", "it", "its",
    "there", "here", "this", "that", "these", "those", "i", "my", "mine", "me",
    "has", "have", "had", "with", "and", "or", "of", "to", "in", "on", "at",
    "for", "from", "by", "as", "but", "also", "some", "kind", "sort", "bit",
    "very", "really", "quite", "think", "remember", "believe", "pretty", "sure",
    "one", "you", "can", "see", "look", "looks", "like", "got", "get", "put",
    "if", "so", "just", "about",
    "they", "them", "their", "theirs", "he", "she", "him", "his", "her", "hers",
    "we", "us", "our", "ours", "your", "yours",
}

# Words a claimant could read off the search result, so they prove nothing.
PUBLIC_ATTRIBUTE_WORDS = {
    "black", "white", "silver", "grey", "gray", "blue", "navy", "red", "green",
    "yellow", "orange", "purple", "pink", "brown", "beige", "gold", "teal",
    "maroon", "charcoal", "cream", "clear", "transparent",
    "headphones", "headphone", "earbuds", "earbud", "phone", "wallet",
    "backpack", "laptop", "watch", "bottle", "book", "charger", "keys",
    "umbrella", "glasses", "tablet", "jacket", "case", "bag",
    "sony", "bose", "apple", "samsung", "jbl", "anker", "dell", "hp", "lenovo",
    "nike", "adidas", "fitbit", "garmin", "beats", "kindle", "logitech",
}

# Low-information modifiers: they refine a mark but do not identify one.
WEAK_TOKENS = {
    "small", "little", "tiny", "big", "large", "slight", "minor", "faint",
    "thin", "thick", "long", "short", "old", "new", "inside", "outside",
    "near", "side", "top", "bottom", "front", "back", "edge", "corner", "part",
}

SYNONYM_GROUPS: tuple[set[str], ...] = (
    {"scratch", "scratched", "scuff", "scuffed", "mark", "marking", "nick", "graze", "scrape"},
    {"crack", "cracked", "chip", "chipped", "split", "broken"},
    {"dent", "dented", "ding", "dinged"},
    {"sticker", "decal", "label", "badge", "patch"},
    {"engraved", "engraving", "etched", "etching", "inscribed", "inscription", "initials", "monogram"},
    {"earcup", "earpiece", "earpad", "cup", "pad", "cushion", "ear"},
    {"case", "pouch", "box", "cover", "sleeve", "shell"},
    {"strap", "band", "wristband", "lanyard"},
    {"screen", "display", "glass"},
    {"lid", "cap", "top"},
    {"photo", "picture", "image", "portrait"},
    {"torn", "tear", "ripped", "rip", "frayed", "fray"},
    {"stain", "stained", "spot", "smudge", "discoloured", "discolored"},
    {"left", "lefthand"},
    {"right", "righthand"},
    {"zip", "zipper", "zippered"},
    {"pocket", "compartment", "slot"},
    {"card", "receipt", "ticket", "note", "slip"},
    {"logo", "emblem", "sign"},
    {"key", "keys", "keychain", "keyring", "fob"},
    {"screw", "bolt", "hinge"},
    {"missing", "lost", "absent", "gone"},
)

_SYNONYM_LOOKUP: dict[str, int] = {}
for _index, _group in enumerate(SYNONYM_GROUPS):
    for _word in _group:
        _SYNONYM_LOOKUP[_word] = _index


def _stem(token: str) -> str:
    """Crude suffix stripping - enough to tie plurals and simple inflections."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _variants(token: str) -> set[str]:
    """All forms a token should be considered equal to."""
    forms = {token, _stem(token)}
    for form in list(forms):
        group_index = _SYNONYM_LOOKUP.get(form)
        if group_index is not None:
            forms.add(f"@group{group_index}")
    return forms


def _content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1]


def _answer_vocabulary(answer: str) -> set[str]:
    """Every form present in the answer, including glued adjacent-word compounds."""
    tokens = _content_tokens(answer)
    vocabulary: set[str] = set()
    for token in tokens:
        vocabulary |= _variants(token)
    # "ear cup" should also satisfy "earcup".
    for left, right in zip(tokens, tokens[1:]):
        vocabulary |= _variants(f"{left}{right}")
    return vocabulary


def split_features(hidden_features: str) -> list[str]:
    """Split the recorded evidence into individual marks."""
    return [part.strip() for part in re.split(r"[;\n]+", hidden_features or "") if part.strip()]


def _clause_score(clause: str, vocabulary: set[str]) -> float:
    """Weighted fraction of a recorded mark that the answer covers."""
    tokens = _content_tokens(clause)
    if not tokens:
        return 0.0

    total = 0.0
    matched = 0.0
    for token in tokens:
        weight = 0.4 if token in WEAK_TOKENS else 1.0
        total += weight
        if _variants(token) & vocabulary:
            matched += weight

    return matched / total if total else 0.0


def _is_generic_answer(answer: str) -> bool:
    """True when the answer only restates attributes the user was already shown."""
    tokens = _content_tokens(answer)
    if not tokens:
        return True
    informative = [
        token for token in tokens
        if token not in PUBLIC_ATTRIBUTE_WORDS and _stem(token) not in PUBLIC_ATTRIBUTE_WORDS
    ]
    return not informative


def match_hidden_features(hidden_features: str, user_answer: str) -> tuple[float, int]:
    """Best coverage score across the recorded marks, and which one matched."""
    clauses = split_features(hidden_features)
    if not clauses:
        return 0.0, -1

    vocabulary = _answer_vocabulary(user_answer)
    scores = [_clause_score(clause, vocabulary) for clause in clauses]
    best_index = max(range(len(scores)), key=lambda index: scores[index])
    return scores[best_index], best_index


def _verify_with_llm(hidden_features: str, user_answer: str) -> dict[str, Any] | None:
    """Optional second opinion. Sends only this one item's marks."""
    from app.agent.prompts import VERIFICATION_SYSTEM, VERIFICATION_USER_TEMPLATE
    from app.llm import LLMError, get_llm

    client = get_llm()
    if client is None:
        return None
    try:
        payload = client.complete_json(
            VERIFICATION_SYSTEM,
            VERIFICATION_USER_TEMPLATE.format(
                hidden_features="\n".join(f"- {c}" for c in split_features(hidden_features)),
                user_answer=user_answer,
            ),
            max_tokens=250,
        )
    except (LLMError, ValueError) as exc:
        logger.warning("LLM verification failed (%s); using the deterministic result.", exc)
        return None

    return {
        "match": bool(payload.get("match")),
        "score": float(payload.get("score", 0.0) or 0.0),
        "reason": str(payload.get("reason", ""))[:300],
    }


def verify_ownership(
    candidate_id: int,
    user_answer: str,
    lost_item_id: int | None = None,
    confidence_score: float | None = None,
    confidence_level: str | None = None,
    record_claim: bool = True,
) -> dict[str, Any]:
    """Check a claimant's distinctive-feature answer against stored evidence.

    Returns ``{verified, confidence, reason, method}``. The reason never quotes
    the stored evidence.
    """
    user_answer = (user_answer or "").strip()

    with session_scope() as session:
        item = session.get(FoundItem, candidate_id)
        if item is None:
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": f"No found item with id {candidate_id}.",
                "method": "lookup",
            }
        hidden_features = item.hidden_features
        item_status = item.status

    if item_status != "available":
        return {
            "verified": False,
            "confidence": 0.0,
            "reason": "This item is no longer available to claim.",
            "method": "status",
        }

    if not user_answer:
        return {
            "verified": False,
            "confidence": 0.0,
            "reason": "No distinctive feature was provided.",
            "method": "empty",
        }

    if _is_generic_answer(user_answer):
        result = {
            "verified": False,
            "confidence": 0.0,
            "reason": (
                "The description only repeats details already shown in the match "
                "(such as colour or brand), so it cannot confirm ownership."
            ),
            "method": "generic-answer-guard",
        }
    else:
        score, _ = match_hidden_features(hidden_features, user_answer)
        verified = score >= settings.verification_threshold
        method = "token-overlap"

        if settings.verification_use_llm:
            judged = _verify_with_llm(hidden_features, user_answer)
            if judged is not None:
                method = "token-overlap+llm"
                # Agreement raises certainty; disagreement is resolved
                # conservatively - both signals must accept.
                score = (score + judged["score"]) / 2
                verified = verified and judged["match"]

        result = {
            "verified": bool(verified),
            "confidence": round(float(min(1.0, max(0.0, score))), 4),
            "reason": (
                "User-provided distinctive feature matches stored ownership evidence."
                if verified
                else "The described feature does not match the ownership evidence on record."
            ),
            "method": method,
        }

    if record_claim:
        try:
            with session_scope() as session:
                session.add(
                    Claim(
                        lost_item_id=lost_item_id,
                        found_item_id=candidate_id,
                        confidence_score=confidence_score,
                        confidence_level=confidence_level,
                        verification_status="passed" if result["verified"] else "failed",
                        verification_score=result["confidence"],
                        user_answer=user_answer,
                        reasons={"method": result["method"], "reason": result["reason"]},
                    )
                )
        except Exception as exc:  # noqa: BLE001 - never fail a check on bookkeeping
            logger.error("Could not record claim for item %s: %s", candidate_id, exc)

    logger.info(
        "verify_ownership(item=%s) -> %s (%.2f)",
        candidate_id,
        result["verified"],
        result["confidence"],
    )
    return result


def build_verification_question() -> str:
    """Ask for evidence without hinting at what is on file.

    Deliberately gives no examples. Suggesting the kinds of mark we record
    ("a sticker? an engraving?") narrows a guesser's search space, and any
    category-tailored hint leaks something about the matched item.
    """
    return (
        "Before I can release it, please describe one distinctive feature of your item - "
        "something specific to yours that isn't in the description above. "
        "Only the owner would know it."
    )
