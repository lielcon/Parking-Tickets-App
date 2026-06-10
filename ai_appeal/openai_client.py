"""Thin, typed gateway around the OpenAI API.

Keeps all OpenAI specifics (client construction, models, error translation)
in one place so the service layer stays focused on RAG orchestration.
"""

import logging

from openai import OpenAI, OpenAIError

from ai_appeal.config import AIAppealSettings, get_settings

logger = logging.getLogger("ai_appeal.openai")

EMBEDDING_BATCH_SIZE = 64


class AIAppealConfigurationError(Exception):
    """Raised when the feature is not configured (e.g. missing API key)."""


class AIAppealUpstreamError(Exception):
    """Raised when the OpenAI API fails or returns an unusable response."""


class OpenAIGateway:
    def __init__(self, settings: AIAppealSettings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            raise AIAppealConfigurationError(
                "OPENAI_API_KEY is not configured. Add it to your .env file."
            )
        self._client = OpenAI(api_key=self._settings.openai_api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts, preserving order."""
        embeddings: list[list[float]] = []
        try:
            for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[start : start + EMBEDDING_BATCH_SIZE]
                response = self._client.embeddings.create(
                    model=self._settings.embedding_model,
                    input=batch,
                )
                embeddings.extend(item.embedding for item in response.data)
        except OpenAIError as exc:
            logger.error("OpenAI embeddings request failed: %s", exc)
            raise AIAppealUpstreamError("Embedding generation failed.") from exc
        return embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        """Run a chat completion that must return a single JSON object."""
        try:
            response = self._client.chat.completions.create(
                model=self._settings.chat_model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except OpenAIError as exc:
            logger.error("OpenAI chat completion failed: %s", exc)
            raise AIAppealUpstreamError("AI analysis request failed.") from exc

        content = response.choices[0].message.content
        if not content:
            raise AIAppealUpstreamError("AI analysis returned an empty response.")
        return content
