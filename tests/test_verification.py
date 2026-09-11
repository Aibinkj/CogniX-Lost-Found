"""Ownership verification tests.

The stored evidence for the demo item is:
    "scratch on left earcup; small sticker inside case"
"""

from __future__ import annotations

from app.tools.pickup import create_pickup_request
from app.tools.verification import (
    match_hidden_features,
    split_features,
    verify_ownership,
)

HIDDEN = "scratch on left earcup; small sticker inside case"


def test_correct_feature_passes(sony_headphones_id):
    result = verify_ownership(sony_headphones_id, "There is a scratch on the left earcup.")
    assert result["verified"] is True
    assert result["confidence"] >= 0.55


def test_incorrect_feature_fails(sony_headphones_id):
    result = verify_ownership(sony_headphones_id, "It has a bright red logo across the top.")
    assert result["verified"] is False


def test_paraphrase_with_synonyms_passes(sony_headphones_id):
    result = verify_ownership(sony_headphones_id, "the left ear cup is scuffed")
    assert result["verified"] is True


def test_second_recorded_feature_also_passes(sony_headphones_id):
    result = verify_ownership(sony_headphones_id, "there's a little sticker inside the case")
    assert result["verified"] is True


def test_generic_colour_answer_is_rejected(sony_headphones_id):
    result = verify_ownership(sony_headphones_id, "it is black")
    assert result["verified"] is False
    assert result["method"] == "generic-answer-guard"


def test_generic_brand_answer_is_rejected(sony_headphones_id):
    result = verify_ownership(sony_headphones_id, "they are Sony headphones")
    assert result["verified"] is False
    assert result["method"] == "generic-answer-guard"


def test_empty_answer_is_rejected(sony_headphones_id):
    result = verify_ownership(sony_headphones_id, "   ")
    assert result["verified"] is False
    assert result["confidence"] == 0.0


def test_unknown_item_is_handled_without_crashing():
    result = verify_ownership(999_999, "a scratch on the left earcup")
    assert result["verified"] is False
    assert "999999" in result["reason"] or "no found item" in result["reason"].lower()


def test_reason_never_leaks_the_stored_evidence(sony_headphones_id):
    for answer in ("it is black", "a red logo", "scratch on the left earcup"):
        result = verify_ownership(sony_headphones_id, answer)
        assert "earcup" not in result["reason"].lower()
        assert "sticker" not in result["reason"].lower()


def test_already_claimed_item_cannot_be_verified_again(sony_headphones_id):
    create_pickup_request(item_id=sony_headphones_id)
    result = verify_ownership(sony_headphones_id, "a scratch on the left earcup")
    assert result["verified"] is False
    assert result["method"] == "status"


def test_partial_recall_of_one_mark_is_not_enough():
    """Naming only the object, not the mark, must not pass."""
    score, _ = match_hidden_features(HIDDEN, "it has a case")
    assert score < 0.55


def test_full_recall_scores_near_one():
    score, index = match_hidden_features(HIDDEN, "scratch on the left earcup")
    assert score > 0.9
    assert index == 0


def test_matching_picks_the_best_clause():
    _, index = match_hidden_features(HIDDEN, "a small sticker inside the case")
    assert index == 1


def test_plural_and_inflected_forms_match():
    score, _ = match_hidden_features(HIDDEN, "the left earcups are scratched")
    assert score > 0.9


def test_features_split_on_semicolons():
    assert split_features(HIDDEN) == ["scratch on left earcup", "small sticker inside case"]


def test_item_with_no_recorded_evidence_scores_zero():
    score, index = match_hidden_features("", "anything at all")
    assert score == 0.0
    assert index == -1


def test_verification_records_a_claim(sony_headphones_id):
    from sqlalchemy import select

    from app.database.connection import session_scope
    from app.database.models import Claim

    verify_ownership(sony_headphones_id, "scratch on the left earcup", confidence_score=0.9)

    with session_scope() as session:
        claim = session.scalar(select(Claim).where(Claim.found_item_id == sony_headphones_id))
        assert claim is not None
        assert claim.verification_status == "passed"
