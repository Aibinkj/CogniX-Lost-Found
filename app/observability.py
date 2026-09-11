"""Optional Langfuse tracing.

Enabled only when ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set
and the ``langfuse`` package is installed. Everything degrades to a no-op
otherwise, so Langfuse is never required to run the prototype.

The SDK's surface changed between v2 and v3; both are probed and any failure
falls back to the no-op rather than breaking a user's request.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class _NoopSpan:
    def update(self, **_: Any) -> None: ...
    def end(self, **_: Any) -> None: ...


NOOP_SPAN = _NoopSpan()


class Tracer:
    """Thin wrapper over the Langfuse client."""

    def __init__(self, client: Any | None) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def span(
        self,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Iterator[Any]:
        if self._client is None:
            yield NOOP_SPAN
            return

        span: Any = NOOP_SPAN
        try:
            if hasattr(self._client, "start_observation"):  # v4
                span = self._client.start_observation(
                    name=name,
                    as_type="span",
                    input=input,
                    metadata={**(metadata or {}), "session_id": session_id},
                )
            elif hasattr(self._client, "start_span"):  # v3
                span = self._client.start_span(name=name, input=input, metadata=metadata)
            elif hasattr(self._client, "trace"):  # v2
                span = self._client.trace(
                    name=name, input=input, metadata=metadata, session_id=session_id
                )
        except Exception as exc:  # noqa: BLE001 - tracing must never break a request
            logger.debug("Langfuse span %r could not start: %s", name, exc)
            span = NOOP_SPAN

        try:
            yield span
        finally:
            try:
                span.end()
            except Exception:  # noqa: BLE001
                pass

    def flush(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Langfuse flush failed: %s", exc)


@lru_cache(maxsize=1)
def get_tracer() -> Tracer:
    if not settings.langfuse_enabled:
        logger.info("Langfuse disabled (no credentials).")
        return Tracer(None)

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:  # noqa: BLE001 - missing package or bad credentials
        logger.warning("Langfuse unavailable (%s). Continuing without tracing.", exc)
        return Tracer(None)

    # A client object is not proof of a working connection. Verify, so the UI
    # never claims tracing is on while spans are silently going nowhere.
    if not _supports_spans(client):
        logger.warning(
            "Installed langfuse version exposes no known span API. Continuing without tracing."
        )
        return Tracer(None)

    try:
        if hasattr(client, "auth_check") and not client.auth_check():
            logger.warning("Langfuse credentials rejected. Continuing without tracing.")
            return Tracer(None)
    except Exception as exc:  # noqa: BLE001 - unreachable host, timeout, etc.
        # The SDK puts every response header in the message; the tail carries
        # the status code and body, which is the part worth logging.
        logger.warning(
            "Langfuse auth check failed (%s: ...%s). Continuing without tracing.",
            exc.__class__.__name__,
            str(exc)[-160:],
        )
        return Tracer(None)

    logger.info("Langfuse tracing enabled (%s).", settings.langfuse_host)
    return Tracer(client)


def _supports_spans(client: Any) -> bool:
    return any(
        hasattr(client, method) for method in ("start_observation", "start_span", "trace")
    )
