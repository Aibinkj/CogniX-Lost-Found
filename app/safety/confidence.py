"""Explainable confidence scoring.

Vector similarity alone is not enough to act on: "black Sony headphones" and
"black Bose headphones" embed close together but are different items. The final
score blends five independent signals with explicit, configurable weights
(`WEIGHT_*` in `.env`):

    semantic 0.50 | location 0.20 | time 0.15 | category 0.10 | brand 0.05

Every signal contributes a sub-score in [0, 1] and a human-readable reason, so
the total can always be explained back to the user.

Unknown signals (the user never said where, or when) are neutral: their weight
is redistributed across the signals that *are* known rather than scored as zero,
otherwise a terse-but-correct report would be punished for being terse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from app.config import settings

if TYPE_CHECKING:
    # Annotation only: a runtime import loops app.tools -> app.safety -> app.agent -> app.tools.
    from app.agent.extraction import ItemDetails

# Terms that mean the same place. Keeps "library" ~ "Library Security Desk".
LOCATION_SYNONYMS: dict[str, set[str]] = {
    "library": {"library", "reading room", "study room", "stacks"},
    "cafeteria": {"cafeteria", "canteen", "dining hall", "food court", "cafe"},
    "gym": {"gym", "sports centre", "sports center", "fitness", "stadium"},
    "union": {"union", "student union", "students union"},
    "parking": {"parking", "car park", "garage", "parking lot"},
    "bus": {"bus", "bus stop", "shuttle", "transport"},
    "lab": {"lab", "laboratory", "computer lab"},
    "lecture": {"lecture hall", "lecture theatre", "lecture theater", "classroom", "auditorium"},
    "dorm": {"dorm", "dormitory", "residence hall", "halls"},
    "pool": {"pool", "swimming pool", "aquatic"},
    "bookstore": {"bookstore", "book shop", "bookshop"},
}

# Categories that people reasonably confuse with one another.
CATEGORY_NEIGHBOURS: dict[str, set[str]] = {
    "headphones": {"earbuds"},
    "earbuds": {"headphones"},
    "phone": {"tablet"},
    "tablet": {"phone", "laptop"},
    "laptop": {"tablet"},
    "backpack": {"wallet"},
    "book": {"tablet"},
}

# Hours of separation at which the time signal decays to ~0.
TIME_DECAY_HOURS = 48.0

LEVEL_HIGH = "HIGH"
LEVEL_MEDIUM = "MEDIUM"
LEVEL_LOW = "LOW"


@dataclass
class ConfidenceResult:
    item_id: int
    score: float
    level: str
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, float | None] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": round(self.score, 4),
            "level": self.level,
            "reasons": self.reasons,
            "signals": {k: (round(v, 4) if v is not None else None) for k, v in self.signals.items()},
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
        }


def _normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def _location_group(text: str) -> set[str]:
    """All synonym groups a location string belongs to."""
    groups = set()
    for key, terms in LOCATION_SYNONYMS.items():
        if any(term in text for term in terms):
            groups.add(key)
    return groups


def score_location(user_location: str | None, item_location: str | None) -> tuple[float | None, str]:
    user, item = _normalize(user_location), _normalize(item_location)
    if not user or not item:
        return None, "Location not stated by the user"

    if user in item or item in user:
        return 1.0, f"Location matches ({item_location})"

    shared = _location_group(user) & _location_group(item)
    if shared:
        return 0.9, f"Location matches ({item_location})"

    user_words = set(user.split()) - {"the", "near", "at", "in", "desk", "security"}
    item_words = set(item.split()) - {"the", "near", "at", "in", "desk", "security"}
    if user_words & item_words:
        return 0.6, f"Partial location overlap with {item_location}"

    return 0.15, f"Location differs (user said {user_location}, item was handed in at {item_location})"


def score_time(lost_time: datetime | None, found_time: datetime | None) -> tuple[float | None, str]:
    if lost_time is None or found_time is None:
        return None, "Time not stated by the user"

    if lost_time.tzinfo is None:
        lost_time = lost_time.replace(tzinfo=timezone.utc)
    if found_time.tzinfo is None:
        found_time = found_time.replace(tzinfo=timezone.utc)

    gap_hours = abs((found_time - lost_time).total_seconds()) / 3600.0

    # An item cannot be handed in long before it was lost; penalise that hard.
    if found_time < lost_time - timedelta(hours=2):
        return 0.1, f"Item was handed in {gap_hours:.0f}h before the reported loss"

    score = max(0.0, 1.0 - gap_hours / TIME_DECAY_HOURS)
    if gap_hours <= 3:
        return score, f"Found within {gap_hours:.0f}h of the reported loss"
    if gap_hours <= 12:
        return score, f"Found about {gap_hours:.0f}h after the reported loss"
    return score, f"Found {gap_hours:.0f}h from the reported time"


def score_category(user_category: str | None, item_category: str | None) -> tuple[float | None, str]:
    user, item = _normalize(user_category), _normalize(item_category)
    if not user or not item:
        return None, "Category not stated by the user"
    if user == item:
        return 1.0, f"Category matches ({item_category})"
    if item in CATEGORY_NEIGHBOURS.get(user, set()):
        return 0.5, f"Related category ({item_category} vs {user_category})"
    return 0.0, f"Category differs ({item_category} vs {user_category})"


def score_brand(user_brand: str | None, item_brand: str | None) -> tuple[float | None, str]:
    user, item = _normalize(user_brand), _normalize(item_brand)
    if not user:
        return None, "Brand not stated by the user"
    if not item:
        return 0.5, "Item has no recorded brand"
    if user == item or user in item or item in user:
        return 1.0, f"Brand matches ({item_brand})"
    return 0.0, f"Brand differs ({item_brand} vs {user_brand})"


def score_color(user_color: str | None, item_color: str | None) -> tuple[float | None, str]:
    """Colour is folded into the semantic signal rather than weighted separately."""
    user, item = _normalize(user_color), _normalize(item_color)
    if not user or not item:
        return None, "Colour not stated by the user"
    if user == item or user in item or item in user:
        return 1.0, f"Colour matches ({item_color})"
    if {user, item} <= {"grey", "gray", "silver", "charcoal"}:
        return 0.8, f"Similar colour ({item_color})"
    return 0.0, f"Colour differs ({item_color} vs {user_color})"


def _level_for(score: float) -> str:
    if score >= settings.confidence_high_threshold:
        return LEVEL_HIGH
    if score >= settings.confidence_low_threshold:
        return LEVEL_MEDIUM
    return LEVEL_LOW


def score_candidate(
    candidate: dict[str, Any],
    details: ItemDetails,
    weights: dict[str, float] | None = None,
) -> ConfidenceResult:
    """Blend the signals for one search result into an explainable score."""
    weights = weights or settings.weights

    semantic = float(candidate.get("similarity", 0.0))
    found_time = candidate.get("found_time")
    if isinstance(found_time, str):
        try:
            found_time = datetime.fromisoformat(found_time)
        except ValueError:
            found_time = None

    location_score, location_reason = score_location(details.location, candidate.get("location"))
    time_score, time_reason = score_time(details.lost_time, found_time)
    category_score, category_reason = score_category(details.category, candidate.get("category"))
    brand_score, brand_reason = score_brand(details.brand, candidate.get("brand"))
    color_score, color_reason = score_color(details.color, candidate.get("color"))

    signals: dict[str, float | None] = {
        "semantic": semantic,
        "location": location_score,
        "time": time_score,
        "category": category_score,
        "brand": brand_score,
    }

    # Redistribute the weight of unknown signals over the known ones.
    active = {name: weights[name] for name, value in signals.items() if value is not None}
    total_weight = sum(active.values()) or 1.0
    score = sum(signals[name] * weight for name, weight in active.items()) / total_weight

    # Colour is a strong disambiguator between otherwise identical items, so a
    # stated-and-contradicted colour applies a direct penalty.
    if color_score == 0.0:
        score *= 0.80
    elif color_score is not None and color_score >= 0.8:
        score = min(1.0, score * 1.03)

    # A different kind of item is not the user's item, however well its desk or
    # wording lines up. As a 10% weight alone, a wallet at the right desk could
    # outrank the only set of keys for "I lost my keys near the library".
    if category_score == 0.0:
        score *= 0.70

    # Same for a stated-and-contradicted brand: at 5% alone, "green Samsung phone,
    # at the bus stop" scored a green Google phone at the bus stop as HIGH.
    if brand_score == 0.0:
        score *= 0.80

    score = float(max(0.0, min(1.0, score)))

    reasons: list[str] = []
    if semantic >= 0.75:
        reasons.append(f"Strong semantic similarity ({semantic:.2f})")
    elif semantic >= 0.5:
        reasons.append(f"Moderate semantic similarity ({semantic:.2f})")
    else:
        reasons.append(f"Weak semantic similarity ({semantic:.2f})")

    for value, reason in (
        (location_score, location_reason),
        (time_score, time_reason),
        (category_score, category_reason),
        (brand_score, brand_reason),
        (color_score, color_reason),
    ):
        if value is not None:
            reasons.append(reason)

    return ConfidenceResult(
        item_id=int(candidate["id"]),
        score=score,
        level=_level_for(score),
        reasons=reasons,
        signals={**signals, "color": color_score},
        weights={name: round(weight, 4) for name, weight in weights.items()},
    )


def score_candidates(
    candidates: list[dict[str, Any]],
    details: ItemDetails,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Score every candidate and return them sorted best-first."""
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        result = score_candidate(candidate, details, weights)
        scored.append({**candidate, "confidence": result.to_dict()})
    scored.sort(key=lambda row: row["confidence"]["score"], reverse=True)
    return scored


def is_ambiguous(scored: list[dict[str, Any]], margin: float | None = None) -> bool:
    """True when the top two candidates are too close to choose between."""
    margin = settings.ambiguity_margin if margin is None else margin
    if len(scored) < 2:
        return False
    top, runner_up = scored[0]["confidence"], scored[1]["confidence"]
    if top["level"] == LEVEL_LOW:
        return False
    return (top["score"] - runner_up["score"]) < margin and runner_up["level"] != LEVEL_LOW
