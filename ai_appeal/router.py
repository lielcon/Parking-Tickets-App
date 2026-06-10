"""API routes for the AI Appeal Assistant feature."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ai_appeal.db import get_db
from ai_appeal.models import TicketRef
from ai_appeal.openai_client import (
    AIAppealConfigurationError,
    AIAppealUpstreamError,
)
from ai_appeal.schemas import AIAppealAnalysisRequest, AIAppealAnalysisResponse
from ai_appeal.service import KnowledgeBaseEmptyError, analyze_appeal

logger = logging.getLogger("ai_appeal.router")

router = APIRouter(tags=["AI Appeal Assistant"])


@router.post("/ai-appeal-analysis", response_model=AIAppealAnalysisResponse)
def create_ai_appeal_analysis(
    payload: AIAppealAnalysisRequest,
    db: Session = Depends(get_db),
) -> AIAppealAnalysisResponse:
    ticket = db.query(TicketRef).filter(TicketRef.id == payload.ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")

    try:
        return analyze_appeal(db, ticket, payload.user_explanation.strip())
    except KnowledgeBaseEmptyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AIAppealConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AIAppealUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during AI appeal analysis.")
        raise HTTPException(status_code=500, detail="AI appeal analysis failed.") from exc
