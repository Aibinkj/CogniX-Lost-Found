"""Engine / session management."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.database.vector_support import resolve_backend

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Create the engine once and detect the vector backend on first use."""
    global _engine, _SessionFactory
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
        resolve_backend(_engine)
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _SessionFactory is not None  # set alongside the engine
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine() -> None:
    """Test hook: drop the cached engine and session factory."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None


def database_is_reachable() -> bool:
    try:
        with get_engine().connect():
            return True
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a status flag
        logger.warning("Database unreachable: %s", exc)
        return False
