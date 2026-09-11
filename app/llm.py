"""Provider-agnostic LLM client.

Configured entirely through the environment (`LLM_PROVIDER`, `LLM_MODEL`,
`LLM_API_KEY`, `LLM_BASE_URL`). Two providers are implemented:

* ``openai``    - any OpenAI-compatible ``/chat/completions`` endpoint, which
  covers OpenAI, OpenRouter, Groq, Together, vLLM, Ollama and LM Studio.
* ``anthropic`` - the Anthropic Messages API.

With ``LLM_PROVIDER=none`` (or no API key) :func:`get_llm` returns ``None`` and
callers use their deterministic path instead. Nothing in this module invents
model output when no model is reachable.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The configured provider could not answer."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    raw_usage: dict[str, Any] | None = None


class LLMClient:
    """Minimal chat interface: a system prompt plus a user prompt in, text out."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        timeout: float,
        reasoning_effort: str = "",
    ) -> None:
        self.provider = provider
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout = timeout
        self._reasoning_effort = reasoning_effort.strip()

    def complete(self, system: str, user: str, max_tokens: int = 800) -> LLMResponse:
        try:
            if self.provider == "anthropic":
                return self._complete_anthropic(system, user, max_tokens)
            return self._complete_openai(system, user, max_tokens)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.provider} request failed: {exc}") from exc

    def complete_json(self, system: str, user: str, max_tokens: int = 800) -> dict[str, Any]:
        """Complete and parse the first JSON object in the reply."""
        response = self.complete(system, user, max_tokens)
        parsed = extract_json_object(response.text)
        if parsed is None:
            raise LLMError(f"Model did not return JSON. Got: {response.text[:200]!r}")
        return parsed

    def _complete_openai(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        payload = {
            "model": self.model,
            "temperature": self._temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.json()
        return LLMResponse(
            text=body["choices"][0]["message"]["content"] or "",
            model=body.get("model", self.model),
            provider=self.provider,
            raw_usage=body.get("usage"),
        )

    def _complete_anthropic(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.json()
        chunks = [block.get("text", "") for block in body.get("content", [])]
        return LLMResponse(
            text="".join(chunks),
            model=body.get("model", self.model),
            provider=self.provider,
            raw_usage=body.get("usage"),
        )


@lru_cache(maxsize=1)
def get_llm() -> LLMClient | None:
    """The configured client, or ``None`` when no LLM is available."""
    if not settings.llm_enabled:
        logger.info("LLM disabled (LLM_PROVIDER=%s). Using rule-based extraction.", settings.llm_provider)
        return None
    if not settings.llm_model:
        logger.warning("LLM_MODEL is empty; disabling the LLM.")
        return None
    return LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        reasoning_effort=settings.llm_reasoning_effort,
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model reply, tolerating code fences and prose."""
    if not raw:
        return None

    fenced = _FENCE_RE.search(raw)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(raw)

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start == -1 or end <= start:
                continue
            try:
                parsed = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            return parsed
    return None
