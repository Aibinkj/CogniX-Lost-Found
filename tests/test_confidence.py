"""Confidence scoring tests.

These are pure functions, so the expectations are about relative ordering and
explainability rather than exact numbers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.agent.extraction import ItemDetails
from app.config import settings
from app.safety.confidence import (
    LEVEL_HIGH,
    LEVEL_LOW,
    is_ambiguous,
    score_candidate,
    score_candidates,
)

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
LOST_AT = NOW - timedelta(hours=20)


def candidate(
    item_id: int = 1,
    similarity: float = 0.80,
    category: str = "headphones",
    brand: str | None = "Sony",
    color: str | None = "black",
    location: str = "Library Security Desk",
    found_offset_hours: float = 19.0,
) -> dict:
    return {
        "id": item_id,
        "similarity": similarity,
        "category": category,
        "brand": brand,
        "color": color,
        "location": location,
        "found_time": (NOW - timedelta(hours=found_offset_hours)).isoformat(),
        "status": "available",
    }


def details(
    category: str | None = "headphones",
    brand: str | None = "Sony",
    color: str | None = "black",
    location: str | None = "library",
    lost_time: datetime | None = LOST_AT,
) -> ItemDetails:
    return ItemDetails(
        category=category,
        brand=brand,
        color=color,
        location=location,
        lost_time=lost_time,
        description="black Sony headphones",
    )


def test_exact_match_scores_high():
    result = score_candidate(candidate(), details())
    assert result.level == LEVEL_HIGH
    assert result.score > settings.confidence_high_threshold


def test_wrong_brand_lowers_the_score():
    exact = score_candidate(candidate(), details()).score
    wrong = score_candidate(candidate(brand="Bose"), details()).score
    assert wrong < exact


def test_wrong_location_lowers_the_score():
    exact = score_candidate(candidate(), details()).score
    wrong = score_candidate(candidate(location="Swimming Pool Reception"), details()).score
    assert wrong < exact


def test_wrong_category_lowers_the_score_more_than_wrong_brand():
    base = score_candidate(candidate(), details()).score
    brand_penalty = base - score_candidate(candidate(brand="Bose"), details()).score
    category_penalty = base - score_candidate(candidate(category="wallet"), details()).score
    assert category_penalty > brand_penalty


def test_right_category_outranks_wrong_category_at_the_right_desk():
    """"Keys near the library": the only keys beat a wallet handed in at the library."""
    report = details(category="keys", brand=None, color=None, location="library", lost_time=None)
    scored = score_candidates(
        [
            candidate(item_id=1, similarity=0.49, category="wallet", brand=None, color="brown"),
            candidate(item_id=2, similarity=0.40, category="keys", brand=None, color="silver",
                      location="Main Reception"),
        ],
        report,
    )
    assert scored[0]["id"] == 2


def test_time_proximity_matters():
    close = score_candidate(candidate(found_offset_hours=19.0), details()).score
    distant = score_candidate(candidate(found_offset_hours=200.0), details()).score
    assert close > distant


def test_item_handed_in_before_the_loss_is_penalised():
    # Reported lost 20h ago, but handed in 40h ago - impossible ordering.
    result = score_candidate(candidate(found_offset_hours=40.0), details())
    assert result.signals["time"] is not None
    assert result.signals["time"] <= 0.2


def test_wrong_colour_lowers_the_score():
    exact = score_candidate(candidate(), details()).score
    wrong = score_candidate(candidate(color="red"), details()).score
    assert wrong < exact


def test_unstated_signals_are_neutral_not_zero():
    """A terse report should not be punished for omitting the location."""
    stated = score_candidate(candidate(), details())
    terse = score_candidate(candidate(), details(location=None, lost_time=None))

    assert terse.signals["location"] is None
    assert terse.signals["time"] is None
    # Both signals matched in `stated`, so terse can only be slightly lower.
    assert terse.score > stated.score - 0.15


def test_weights_are_explicit_and_normalised():
    result = score_candidate(candidate(), details())
    assert set(result.weights) == {"semantic", "location", "time", "category", "brand"}
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6


def test_reasons_explain_every_known_signal():
    result = score_candidate(candidate(), details())
    joined = " ".join(result.reasons).lower()
    assert "semantic" in joined
    assert "location" in joined
    assert "brand" in joined
    assert "category" in joined


def test_low_similarity_and_mismatches_produce_a_low_level():
    result = score_candidate(
        candidate(similarity=0.20, category="wallet", brand="Herschel", color="brown",
                  location="Gym Reception", found_offset_hours=300.0),
        details(),
    )
    assert result.level == LEVEL_LOW
    assert result.score < settings.confidence_low_threshold


def test_candidates_are_returned_best_first():
    scored = score_candidates(
        [
            candidate(item_id=1, similarity=0.40, brand="JBL", color="red"),
            candidate(item_id=2, similarity=0.85),
            candidate(item_id=3, similarity=0.60, brand="Bose"),
        ],
        details(),
    )
    scores = [row["confidence"]["score"] for row in scored]
    assert scores == sorted(scores, reverse=True)
    assert scored[0]["id"] == 2


def test_near_identical_candidates_are_flagged_ambiguous():
    scored = score_candidates(
        [candidate(item_id=1, similarity=0.82), candidate(item_id=2, similarity=0.82)],
        details(),
    )
    assert is_ambiguous(scored)


def test_a_clear_winner_is_not_ambiguous():
    scored = score_candidates(
        [
            candidate(item_id=1, similarity=0.88),
            candidate(item_id=2, similarity=0.30, category="wallet", brand=None, color="brown"),
        ],
        details(),
    )
    assert not is_ambiguous(scored)


def test_single_candidate_is_never_ambiguous():
    scored = score_candidates([candidate()], details())
    assert not is_ambiguous(scored)
