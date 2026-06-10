"""Service layer: orchestrates the full RAG appeal-analysis flow.

Flow: ticket details + user explanation
      -> query embedding
      -> vector search over legal_chunks
      -> OpenAI analysis grounded ONLY in the retrieved excerpts
      -> persist analysis history
      -> structured response

Anti-hallucination guarantee: the model only selects indices of the excerpts
it was shown. `legal_sources_used` is built server-side from retrieval
metadata, so the API can never return an invented citation.
"""

import json
import logging

from pydantic import ValidationError
from sqlalchemy.orm import Session

from ai_appeal.config import get_settings
from ai_appeal.models import AIAppealAnalysisDB, TicketRef
from ai_appeal.openai_client import AIAppealUpstreamError, OpenAIGateway
from ai_appeal.retrieval import (
    RetrievedChunk,
    count_chunks_for_city,
    search_legal_chunks,
)
from ai_appeal.schemas import (
    AIAppealAnalysisResponse,
    LegalSourceUsed,
    ModelVerdict,
)

logger = logging.getLogger("ai_appeal.service")


class KnowledgeBaseEmptyError(Exception):
    """Raised when no legal documents exist for the ticket's city."""


SYSTEM_PROMPT = """\
You are a legal analysis assistant for municipal parking ticket appeals in Israel.

Strict rules:
1. Base your analysis ONLY on the numbered legal excerpts provided by the user.
   Do not rely on outside legal knowledge and never invent regulations,
   sections, or citations.
2. If the excerpts do not clearly support an appeal, say so honestly and
   lower the appeal strength and confidence accordingly.
3. Write "reasoning" and "recommended_action" in the same language as the
   user's explanation.

Respond with a single JSON object with exactly these keys:
- "appeal_strength": one of "Strong", "Medium", "Weak"
- "confidence": integer 0-100
- "reasoning": concise explanation grounded in the excerpts
- "missing_evidence": array of strings (evidence the user should collect; may be empty)
- "recommended_action": one practical next step for the user
- "used_excerpt_indices": array of integers - the excerpt numbers you actually
  relied on (use only numbers that were provided)
"""


def _build_query_text(ticket: TicketRef, user_explanation: str) -> str:
    """Combine ticket context and the user's story into the retrieval query."""
    parts = [
        f"City: {ticket.city}",
        f"Parking ticket number: {ticket.ticket_number}",
        f"Issued at: {ticket.issued_at.isoformat()}",
    ]
    if ticket.fine_amount:
        parts.append(f"Fine amount: {ticket.fine_amount}")
    parts.append(f"User explanation: {user_explanation}")
    return "\n".join(parts)


def _build_user_prompt(
    ticket: TicketRef,
    user_explanation: str,
    chunks: list[RetrievedChunk],
) -> str:
    excerpt_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        excerpt_blocks.append(
            f"EXCERPT {index}\n"
            f"City: {chunk.city} | Document: {chunk.title} | Source: {chunk.source}\n"
            f"{chunk.chunk_text}"
        )
    excerpts_text = "\n\n---\n\n".join(excerpt_blocks)

    return (
        "TICKET DETAILS\n"
        f"City: {ticket.city}\n"
        f"Ticket number: {ticket.ticket_number}\n"
        f"Plate number: {ticket.plate_number}\n"
        f"Issued at: {ticket.issued_at.isoformat()}\n"
        f"Fine amount: {ticket.fine_amount or 'unknown'}\n\n"
        "USER EXPLANATION\n"
        f"{user_explanation}\n\n"
        "LEGAL EXCERPTS\n"
        f"{excerpts_text}\n\n"
        "Analyze whether the user has grounds for appeal and respond with the "
        "required JSON object."
    )


def _parse_verdict(raw_content: str) -> ModelVerdict:
    try:
        payload = json.loads(raw_content)
        return ModelVerdict.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.error("Model returned invalid analysis JSON: %s", exc)
        raise AIAppealUpstreamError("AI analysis returned an invalid response.") from exc


def _resolve_sources_used(
    verdict: ModelVerdict,
    chunks: list[RetrievedChunk],
) -> list[LegalSourceUsed]:
    """Map the model's excerpt indices back to real retrieval metadata."""
    valid_indices = [i for i in verdict.used_excerpt_indices if 1 <= i <= len(chunks)]
    # If the model did not select valid excerpts, attribute all retrieved ones.
    selected = valid_indices or list(range(1, len(chunks) + 1))

    sources: list[LegalSourceUsed] = []
    seen: set[str] = set()
    for index in selected:
        chunk = chunks[index - 1]
        if chunk.source in seen:
            continue
        seen.add(chunk.source)
        sources.append(LegalSourceUsed(title=chunk.title, source=chunk.source))
    return sources


def analyze_appeal(
    db: Session,
    ticket: TicketRef,
    user_explanation: str,
) -> AIAppealAnalysisResponse:
    """Run the full RAG flow for one ticket and persist the analysis."""
    settings = get_settings()

    # 1) Fail fast (before any OpenAI call) if the knowledge base is empty.
    if count_chunks_for_city(db, ticket.city) == 0:
        raise KnowledgeBaseEmptyError(
            f"No legal documents have been ingested for city '{ticket.city}'."
        )

    gateway = OpenAIGateway(settings)

    # 2) Embed the retrieval query (ticket context + user explanation).
    query_text = _build_query_text(ticket, user_explanation)
    query_embedding = gateway.embed_text(query_text)

    # 3) Vector search for the most relevant legal chunks.
    chunks = search_legal_chunks(db, ticket.city, query_embedding, settings.top_k)
    logger.info(
        "Retrieved %d legal chunks for ticket %d (city=%s).",
        len(chunks),
        ticket.id,
        ticket.city,
    )

    # 4) Grounded OpenAI analysis.
    raw_content = gateway.complete_json(
        SYSTEM_PROMPT,
        _build_user_prompt(ticket, user_explanation, chunks),
    )
    verdict = _parse_verdict(raw_content)
    sources_used = _resolve_sources_used(verdict, chunks)

    response = AIAppealAnalysisResponse(
        appeal_strength=verdict.appeal_strength,
        confidence=verdict.confidence,
        reasoning=verdict.reasoning,
        missing_evidence=verdict.missing_evidence,
        recommended_action=verdict.recommended_action,
        legal_sources_used=sources_used,
    )

    # 5) Persist analysis history.
    analysis = AIAppealAnalysisDB(
        user_id=ticket.user_id,
        ticket_id=ticket.id,
        user_explanation=user_explanation,
        appeal_strength=response.appeal_strength,
        confidence=response.confidence,
        reasoning=response.reasoning,
        missing_evidence=response.missing_evidence,
        recommended_action=response.recommended_action,
        legal_sources_used=[source.model_dump() for source in response.legal_sources_used],
    )
    db.add(analysis)
    db.commit()
    logger.info("Saved AI appeal analysis %d for ticket %d.", analysis.id, ticket.id)

    return response
