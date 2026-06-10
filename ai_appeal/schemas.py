"""Pydantic schemas for the AI Appeal Assistant feature."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

AppealStrength = Literal["Strong", "Medium", "Weak"]


class AIAppealAnalysisRequest(BaseModel):
    ticket_id: int
    user_explanation: str = Field(min_length=10, max_length=4000)


class LegalSourceUsed(BaseModel):
    title: str
    source: str


class AIAppealAnalysisResponse(BaseModel):
    appeal_strength: AppealStrength
    confidence: int = Field(ge=0, le=100)
    reasoning: str
    missing_evidence: list[str]
    recommended_action: str
    legal_sources_used: list[LegalSourceUsed]


class ModelVerdict(BaseModel):
    """Strict schema for the JSON the LLM must return.

    The model never outputs source titles directly; it only selects indices
    of the legal excerpts it was given (`used_excerpt_indices`). The server
    maps those indices back to real retrieval metadata, so the API can never
    return an invented legal citation.
    """

    appeal_strength: AppealStrength
    confidence: int = Field(ge=0, le=100)
    reasoning: str
    missing_evidence: list[str] = Field(default_factory=list)
    recommended_action: str
    used_excerpt_indices: list[int] = Field(default_factory=list)
