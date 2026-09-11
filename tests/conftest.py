"""Test fixtures.

Tests run against a real PostgreSQL database - the point is to exercise the
actual retrieval and persistence path, not mocks. A dedicated `*_test` database
is created and seeded once per session so the development data is never touched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _test_database_url() -> str:
    """Derive a sibling `*_test` database from DATABASE_URL."""
    if override := os.environ.get("TEST_DATABASE_URL"):
        return override

    from dotenv import dotenv_values

    configured = (
        os.environ.get("DATABASE_URL")
        or dotenv_values(ROOT / ".env").get("DATABASE_URL")
        or "postgresql+psycopg2://postgres:postgres@localhost:5432/lostfound"
    )
    parts = urlsplit(configured)
    name = parts.path.lstrip("/") or "lostfound"
    if not name.endswith("_test"):
        name = f"{name}_test"
    return urlunsplit((parts.scheme, parts.netloc, f"/{name}", parts.query, parts.fragment))


def _create_database_if_missing(url: str) -> None:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    parts = urlsplit(url)
    name = parts.path.lstrip("/")
    maintenance = urlunsplit(("postgresql", parts.netloc, "/postgres", "", ""))

    connection = psycopg2.connect(maintenance)
    try:
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
            if cursor.fetchone() is None:
                cursor.execute(f'CREATE DATABASE "{name}"')
    finally:
        connection.close()


# Must happen before anything imports app.config, which caches its settings.
TEST_URL = _test_database_url()
os.environ["DATABASE_URL"] = TEST_URL
os.environ["LLM_PROVIDER"] = "none"  # deterministic extraction in tests
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""


@pytest.fixture(scope="session", autouse=True)
def seeded_database():
    """Create and seed the test database once."""
    try:
        _create_database_if_missing(TEST_URL)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL is not reachable for tests: {exc}")

    from scripts.seed_database import seed

    seed(keep=False)
    yield


@pytest.fixture(autouse=True)
def clean_transactional_tables(seeded_database):
    """Roll the catalogue back to its seeded state between tests."""
    yield

    from sqlalchemy import text

    from app.database.connection import session_scope

    with session_scope() as session:
        for table in ("pickup_requests", "claims", "escalations", "notifications", "lost_items"):
            session.execute(text(f"DELETE FROM {table}"))
        session.execute(text("UPDATE found_items SET status = 'available'"))


@pytest.fixture
def demo_query() -> str:
    return "I lost my black Sony headphones near the library yesterday around 4 PM."


@pytest.fixture
def sony_headphones_id(seeded_database) -> int:
    """The id of the demo target item."""
    from sqlalchemy import select

    from app.database.connection import session_scope
    from app.database.models import FoundItem

    with session_scope() as session:
        item = session.scalar(
            select(FoundItem).where(
                FoundItem.category == "headphones",
                FoundItem.brand == "Sony",
                FoundItem.color == "black",
            )
        )
        assert item is not None, "Seed data is missing the black Sony headphones."
        return item.id
