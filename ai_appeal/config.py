"""Typed configuration for the AI Appeal Assistant feature.

All values are read from environment variables so that no secrets or
deployment-specific settings are hardcoded. `main.py` (and the ingestion
script) call `load_dotenv()` before this module is used.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

# City label used for documents that apply to every municipality.
GENERAL_CITY = "general"


@dataclass(frozen=True)
class AIAppealSettings:
    openai_api_key: str | None
    embedding_model: str
    chat_model: str
    embedding_dimensions: int
    top_k: int


@lru_cache(maxsize=1)
def get_settings() -> AIAppealSettings:
    """Build settings once per process from environment variables."""
    return AIAppealSettings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embedding_model=os.getenv("AI_APPEAL_EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("AI_APPEAL_CHAT_MODEL", "gpt-4o-mini"),
        # text-embedding-3-small produces 1536-dimensional vectors.
        embedding_dimensions=int(os.getenv("AI_APPEAL_EMBEDDING_DIMENSIONS", "1536")),
        top_k=int(os.getenv("AI_APPEAL_TOP_K", "6")),
    )
