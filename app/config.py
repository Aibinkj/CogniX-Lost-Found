"""Central configuration, loaded from the environment / `.env`."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class ConfidenceWeights(BaseSettings):
    """Explicit, configurable weights for the confidence signals."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    semantic: float = Field(0.50, alias="WEIGHT_SEMANTIC")
    location: float = Field(0.20, alias="WEIGHT_LOCATION")
    time: float = Field(0.15, alias="WEIGHT_TIME")
    category: float = Field(0.10, alias="WEIGHT_CATEGORY")
    brand: float = Field(0.05, alias="WEIGHT_BRAND")

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "location": self.location,
            "time": self.time,
            "category": self.category,
            "brand": self.brand,
        }

    def normalized(self) -> dict[str, float]:
        weights = self.as_dict()
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Confidence weights must sum to a positive number.")
        return {key: value / total for key, value in weights.items()}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database ---------------------------------------------------------
    database_url: str = Field(
        "postgresql+psycopg2://postgres:postgres@localhost:5432/lostfound",
        alias="DATABASE_URL",
    )
    vector_backend: Literal["auto", "pgvector", "numpy"] = Field("auto", alias="VECTOR_BACKEND")

    # --- embeddings -------------------------------------------------------
    embedding_backend: Literal["auto", "sentence-transformers", "hashing"] = Field(
        "auto", alias="EMBEDDING_BACKEND"
    )
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(384, alias="EMBEDDING_DIM")

    # --- llm --------------------------------------------------------------
    llm_provider: Literal["openai", "anthropic", "none"] = Field("none", alias="LLM_PROVIDER")
    llm_model: str = Field("", alias="LLM_MODEL")
    llm_api_key: str = Field("", alias="LLM_API_KEY")
    llm_base_url: str = Field("https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    llm_timeout_seconds: float = Field(30.0, alias="LLM_TIMEOUT_SECONDS")
    # openai provider only; sent when set. Thinking models (e.g. Gemini) can
    # otherwise spend the whole max_tokens budget reasoning and truncate the JSON.
    llm_reasoning_effort: str = Field("", alias="LLM_REASONING_EFFORT")

    # --- matching / safety ------------------------------------------------
    confidence_high_threshold: float = Field(0.75, alias="CONFIDENCE_HIGH_THRESHOLD")
    confidence_low_threshold: float = Field(0.60, alias="CONFIDENCE_LOW_THRESHOLD")
    ambiguity_margin: float = Field(0.07, alias="AMBIGUITY_MARGIN")
    search_top_k: int = Field(5, alias="SEARCH_TOP_K")
    verification_threshold: float = Field(0.55, alias="VERIFICATION_THRESHOLD")
    verification_use_llm: bool = Field(False, alias="VERIFICATION_USE_LLM")
    # Help questions are answered only from a retrieved article at or above this
    # similarity; below it the agent says it does not know and points to staff.
    # Calibrated on all-MiniLM-L6-v2: on-topic questions scored 0.34-0.64,
    # unrelated ones 0.05-0.18.
    help_min_similarity: float = Field(0.30, alias="HELP_MIN_SIMILARITY")

    # --- observability ----------------------------------------------------
    langfuse_public_key: str = Field("", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field("", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    # --- app --------------------------------------------------------------
    api_url: str = Field("http://localhost:8000", alias="API_URL")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @field_validator("confidence_low_threshold")
    @classmethod
    def _low_below_high(cls, value: float, info) -> float:
        high = info.data.get("confidence_high_threshold")
        if high is not None and value > high:
            raise ValueError("CONFIDENCE_LOW_THRESHOLD must not exceed CONFIDENCE_HIGH_THRESHOLD.")
        return value

    @property
    def weights(self) -> dict[str, float]:
        return ConfidenceWeights().normalized()

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "none" and bool(self.llm_api_key.strip())

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key.strip() and self.langfuse_secret_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Model-download and HTTP chatter drowns out the agent's own logs.
_NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "huggingface_hub", "transformers",
    "sentence_transformers", "filelock",
)


def configure_logging(level: str | None = None) -> None:
    """Set up logging for an entry point (seed script, API, Streamlit)."""
    import logging

    logging.basicConfig(
        level=getattr(logging, (level or settings.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
