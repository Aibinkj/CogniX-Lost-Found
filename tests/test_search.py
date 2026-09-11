"""Semantic retrieval tests."""

from __future__ import annotations

from app.tools.found_item import list_found_items
from app.tools.pickup import create_pickup_request
from app.tools.search import search_matches


def test_demo_query_ranks_the_black_sony_headphones_first(demo_query, sony_headphones_id):
    results = search_matches(demo_query, top_k=5)
    assert results, "Expected candidates for the demo query."
    assert results[0]["id"] == sony_headphones_id


def test_results_never_expose_hidden_features(demo_query):
    for result in search_matches(demo_query, top_k=5):
        assert "hidden_features" not in result
        assert "scratch on left earcup" not in str(result).lower()


def test_similarities_are_bounded_and_descending(demo_query):
    results = search_matches(demo_query, top_k=5)
    scores = [row["similarity"] for row in results]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_respected(demo_query):
    assert len(search_matches(demo_query, top_k=3)) <= 3
    assert len(search_matches(demo_query, top_k=1)) == 1


def test_empty_query_returns_nothing():
    assert search_matches("") == []
    assert search_matches("   ") == []


def test_unrelated_query_does_not_rank_headphones_first():
    results = search_matches("I lost a blue chemistry textbook in the library", top_k=3)
    assert results
    assert results[0]["category"] == "book"


def test_claimed_items_drop_out_of_search(demo_query, sony_headphones_id):
    assert search_matches(demo_query, top_k=5)[0]["id"] == sony_headphones_id

    create_pickup_request(item_id=sony_headphones_id)

    results = search_matches(demo_query, top_k=5)
    assert sony_headphones_id not in [row["id"] for row in results]


def test_category_filter_narrows_results():
    results = search_matches("black wireless audio device", top_k=3, category="earbuds")
    assert results
    assert all(row["category"] == "earbuds" for row in results)


def test_listing_found_items_hides_ownership_evidence():
    items = list_found_items(status="available", limit=100)
    assert items
    assert all("hidden_features" not in item for item in items)
