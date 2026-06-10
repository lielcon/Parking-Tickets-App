"""Database access for the AI Appeal Assistant feature.

This module is intentionally self-contained: it builds its own engine and
session factory from DATABASE_URL instead of importing anything from
`main.py`. This keeps the feature additive and avoids circular imports.
"""

import logging
import os
from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger("ai_appeal.db")

Base = declarative_base()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing. Add it to your .env file.")
    return create_engine(database_url)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: one DB session per request."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_ai_appeal_schema() -> None:
    """Enable pgvector and create the feature's tables if they do not exist."""
    # Import here so models are registered on Base before create_all.
    from ai_appeal import models

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(
        bind=engine,
        tables=[
            models.LegalDocumentDB.__table__,
            models.LegalChunkDB.__table__,
            models.AIAppealAnalysisDB.__table__,
        ],
    )

    # Lightweight idempotent migration: create_all does not alter existing
    # tables, so add columns introduced after the initial release here.
    with engine.connect() as conn:
        conn.execute(
            text("ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS source_url VARCHAR")
        )
        conn.commit()

    logger.info("AI appeal schema is ready (pgvector extension + tables).")
