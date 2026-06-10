"""SQLAlchemy models for the AI Appeal Assistant feature.

Three new tables are introduced:
- legal_documents:    full legal/municipal source documents
- legal_chunks:       chunked text + pgvector embeddings for retrieval
- ai_appeal_analyses: history of AI appeal analyses

`TicketRef` is a read-only mapping of the existing "tickets" table so this
package can validate and read tickets without importing `main.py`.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from ai_appeal.config import get_settings
from ai_appeal.db import Base


class LegalDocumentDB(Base):
    __tablename__ = "legal_documents"

    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    # Stable identifier of the original document (e.g. file path or URL).
    # Used by the ingestion pipeline to detect already-ingested documents.
    source = Column(String, nullable=False, unique=True, index=True)
    # Original web page URL for documents ingested from the web (null for files).
    source_url = Column(String, nullable=True)
    document_type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class LegalChunkDB(Base):
    __tablename__ = "legal_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer,
        ForeignKey("legal_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalized from the parent document so vector search can filter by
    # city without a join on the hot path.
    city = Column(String, nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(get_settings().embedding_dimensions), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AIAppealAnalysisDB(Base):
    __tablename__ = "ai_appeal_analyses"

    id = Column(Integer, primary_key=True, index=True)
    # Nullable because tickets themselves may not be linked to a user.
    # Plain integer (no FK constraint): the users table is owned by main.py
    # and is not part of this package's metadata.
    user_id = Column(Integer, nullable=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False, index=True)
    user_explanation = Column(Text, nullable=False)
    appeal_strength = Column(String, nullable=False)
    confidence = Column(Integer, nullable=False)
    reasoning = Column(Text, nullable=False)
    missing_evidence = Column(JSONB, nullable=False, default=list)
    recommended_action = Column(Text, nullable=False)
    legal_sources_used = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class TicketRef(Base):
    """Read-only view of the existing tickets table (owned by main.py)."""

    __tablename__ = "tickets"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    city = Column(String, nullable=False)
    plate_number = Column(String, nullable=False)
    ticket_number = Column(String, nullable=False)
    issued_at = Column(DateTime, nullable=False)
    payable_at = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)
    fine_amount = Column(String, nullable=True)
