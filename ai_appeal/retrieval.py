"""Vector retrieval over the legal knowledge base (pgvector)."""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from ai_appeal.config import GENERAL_CITY
from ai_appeal.models import LegalChunkDB, LegalDocumentDB


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    title: str
    source: str
    city: str
    chunk_text: str
    distance: float


def count_chunks_for_city(db: Session, city: str) -> int:
    """Cheap pre-check so we can fail clearly (and skip OpenAI calls)
    when nothing was ingested for this city."""
    return (
        db.query(LegalChunkDB)
        .filter(LegalChunkDB.city.in_([city, GENERAL_CITY]))
        .count()
    )


def search_legal_chunks(
    db: Session,
    city: str,
    query_embedding: list[float],
    top_k: int,
) -> list[RetrievedChunk]:
    """Return the top_k most similar chunks for this city (cosine distance).

    Documents ingested under the "general" city apply to all municipalities
    and are always included in the candidate set.
    """
    rows = (
        db.query(
            LegalChunkDB.id,
            LegalChunkDB.document_id,
            LegalChunkDB.city,
            LegalChunkDB.chunk_text,
            LegalDocumentDB.title,
            LegalDocumentDB.source,
            LegalChunkDB.embedding.cosine_distance(query_embedding).label("distance"),
        )
        .join(LegalDocumentDB, LegalDocumentDB.id == LegalChunkDB.document_id)
        .filter(LegalChunkDB.city.in_([city, GENERAL_CITY]))
        .order_by(LegalChunkDB.embedding.cosine_distance(query_embedding))
        .limit(top_k)
        .all()
    )
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            title=row.title,
            source=row.source,
            city=row.city,
            chunk_text=row.chunk_text,
            distance=float(row.distance),
        )
        for row in rows
    ]
