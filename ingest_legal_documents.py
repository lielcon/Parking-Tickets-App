"""Legal document ingestion pipeline for the AI Appeal Assistant.

Reads legal documents from local files or official municipal web pages,
extracts text, chunks it, generates OpenAI embeddings, and stores everything
in the legal_documents and legal_chunks tables (PostgreSQL + pgvector).

Ingestion is a ONE-TIME process: the RAG analysis flow never fetches the web
at request time and relies exclusively on content stored in the database.

Folder convention (file mode)
-----------------------------
legal_docs/
    תל אביב/            <- subfolder name = city the documents apply to
        parking_regulations.md
        appeal_procedure.pdf
    ירושלים/
        parking_regulations.txt
    some_national_rule.md  <- files at the root apply to ALL cities ("general")

Supported file formats: .txt, .md, .pdf

Usage
-----
    # Ingest all local documents from /legal_docs
    python ingest_legal_documents.py

    # Ingest official municipal web pages (one-time fetch)
    python ingest_legal_documents.py --url https://www.tel-aviv.gov.il/... --city "תל אביב"
    python ingest_legal_documents.py --url https://... --url https://... --city "ירושלים"

    # Re-ingest documents that already exist (replaces their chunks)
    python ingest_legal_documents.py --force
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("ai_appeal.ingest")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class DiscoveredDocument:
    path: Path
    city: str
    source: str  # stable identifier: path relative to the docs dir
    title: str


def discover_documents(docs_dir: Path, general_city: str) -> list[DiscoveredDocument]:
    """Find supported documents. Subfolder name = city; root files = general."""
    documents: list[DiscoveredDocument] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(docs_dir)
        city = relative.parts[0] if len(relative.parts) > 1 else general_city
        documents.append(
            DiscoveredDocument(
                path=path,
                city=city,
                source=relative.as_posix(),
                title=path.stem.replace("_", " ").replace("-", " ").strip(),
            )
        )
    return documents


def extract_text(path: Path) -> str:
    """Extract plain text from a supported document file."""
    from ai_appeal.sanitization import sanitize_text

    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        raw_text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        raw_text = "\n\n".join(pages)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # PDF extraction in particular can emit NUL bytes that PostgreSQL rejects.
    return sanitize_text(raw_text, label=f"file text ({path.name})")


def ingest_text_content(
    db,
    gateway,
    *,
    city: str,
    title: str,
    source: str,
    source_url: str | None,
    document_type: str,
    text: str,
    force: bool,
) -> bool:
    """Shared save path for files and web pages: dedup, chunk, embed, store.

    Returns True if the document was (re)ingested.
    """
    from ai_appeal.chunking import chunk_text
    from ai_appeal.models import LegalChunkDB, LegalDocumentDB
    from ai_appeal.sanitization import sanitize_text

    existing = (
        db.query(LegalDocumentDB).filter(LegalDocumentDB.source == source).first()
    )
    if existing and not force:
        logger.info("Skipping already-ingested document: %s", source)
        return False

    # Final safety net before DB storage: PostgreSQL rejects NUL characters.
    title = sanitize_text(title, label=f"title ({source})").strip()
    text = sanitize_text(text, label=f"content ({source})").strip()
    if not text:
        logger.warning("Skipping empty document: %s", source)
        return False

    chunks = [
        sanitize_text(chunk, label=f"chunk ({source})") for chunk in chunk_text(text)
    ]
    chunks = [chunk for chunk in chunks if chunk.strip()]
    if not chunks:
        logger.warning("Skipping document with no usable chunks: %s", source)
        return False

    logger.info(
        "Ingesting %s (city=%s, %d chars, %d chunks)...",
        source,
        city,
        len(text),
        len(chunks),
    )
    embeddings = gateway.embed_texts(chunks)

    if existing:
        # --force: replace the document and its chunks atomically.
        db.query(LegalChunkDB).filter(
            LegalChunkDB.document_id == existing.id
        ).delete(synchronize_session=False)
        db.delete(existing)
        db.flush()

    document = LegalDocumentDB(
        city=city,
        title=title,
        source=source,
        source_url=source_url,
        document_type=document_type,
        content=text,
    )
    db.add(document)
    db.flush()  # assign document.id before inserting chunks

    for chunk, embedding in zip(chunks, embeddings):
        db.add(
            LegalChunkDB(
                document_id=document.id,
                city=city,
                chunk_text=chunk,
                embedding=embedding,
            )
        )
    db.commit()
    logger.info("Saved document %d with %d chunks.", document.id, len(chunks))
    return True


def ingest_file(db, gateway, doc: DiscoveredDocument, document_type: str, force: bool) -> bool:
    return ingest_text_content(
        db,
        gateway,
        city=doc.city,
        title=doc.title,
        source=doc.source,
        source_url=None,
        document_type=document_type,
        text=extract_text(doc.path),
        force=force,
    )


def ingest_url(db, gateway, url: str, city: str, document_type: str, force: bool) -> bool:
    from ai_appeal.web_extraction import extract_web_page

    page = extract_web_page(url)
    return ingest_text_content(
        db,
        gateway,
        city=city,
        title=page.title,
        source=url,
        source_url=url,
        document_type=document_type,
        text=page.text,
        force=force,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest legal documents for RAG.")
    parser.add_argument(
        "--docs-dir",
        default="legal_docs",
        help="Folder containing legal documents (default: legal_docs)",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        metavar="URL",
        help="Official municipal web page to ingest (repeatable). "
        "When given, only the URLs are ingested (the docs folder is skipped).",
    )
    parser.add_argument(
        "--city",
        default=None,
        help='City the ingested URLs apply to (default: "general" = all cities)',
    )
    parser.add_argument(
        "--document-type",
        default=None,
        help="document_type stored for documents in this run "
        "(default: municipal_regulation for files, municipal_webpage for URLs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest documents that already exist (replaces their chunks)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    load_dotenv()

    # Imported after load_dotenv so DATABASE_URL / OPENAI_API_KEY are available.
    from ai_appeal.config import GENERAL_CITY
    from ai_appeal.db import get_session_factory, init_ai_appeal_schema
    from ai_appeal.openai_client import AIAppealConfigurationError, OpenAIGateway

    url_mode = bool(args.url)

    if not url_mode:
        docs_dir = Path(args.docs_dir)
        if not docs_dir.is_dir():
            logger.error("Docs folder not found: %s", docs_dir.resolve())
            return 1
        documents = discover_documents(docs_dir, GENERAL_CITY)
        if not documents:
            logger.warning(
                "No supported documents (.txt/.md/.pdf) found in %s.", docs_dir.resolve()
            )
            return 0

    try:
        gateway = OpenAIGateway()
    except AIAppealConfigurationError as exc:
        logger.error("%s", exc)
        return 1

    init_ai_appeal_schema()

    document_type = args.document_type or (
        "municipal_webpage" if url_mode else "municipal_regulation"
    )
    city_for_urls = args.city or GENERAL_CITY

    ingested = 0
    skipped = 0
    failed = 0
    db = get_session_factory()()
    try:
        if url_mode:
            total = len(args.url)
            for url in args.url:
                try:
                    if ingest_url(db, gateway, url, city_for_urls, document_type, args.force):
                        ingested += 1
                    else:
                        skipped += 1
                except Exception:
                    db.rollback()
                    failed += 1
                    logger.exception("Failed to ingest URL: %s", url)
        else:
            total = len(documents)
            for doc in documents:
                try:
                    if ingest_file(db, gateway, doc, document_type, args.force):
                        ingested += 1
                    else:
                        skipped += 1
                except Exception:
                    db.rollback()
                    failed += 1
                    logger.exception("Failed to ingest document: %s", doc.source)
    finally:
        db.close()

    logger.info(
        "Done. Ingested: %d | Skipped: %d | Failed: %d (total discovered: %d)",
        ingested,
        skipped,
        failed,
        total,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
