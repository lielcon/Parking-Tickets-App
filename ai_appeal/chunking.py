"""Text chunking for the legal document ingestion pipeline.

Strategy: split on paragraph boundaries and pack paragraphs into chunks of
up to `max_chars`, carrying a small tail overlap between consecutive chunks
so legal context is not cut mid-thought at chunk boundaries.
"""

import re

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 150


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Hard-split a paragraph that is longer than one chunk, preferring
    sentence boundaries."""
    sentences = re.split(r"(?<=[.!?؟。])\s+", paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        # A single sentence longer than max_chars is sliced directly.
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if len(current) + len(sentence) + 1 > max_chars and current:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split raw document text into overlapping chunks for embedding."""
    units: list[str] = []
    for paragraph in _split_paragraphs(text):
        if len(paragraph) > max_chars:
            units.extend(_split_long_paragraph(paragraph, max_chars))
        else:
            units.append(paragraph)

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 2 > max_chars:
            chunks.append(current)
            # Seed the next chunk with the tail of the previous one.
            overlap = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{overlap}\n{unit}".strip()
        else:
            current = f"{current}\n\n{unit}".strip() if current else unit
    if current:
        chunks.append(current)
    return chunks
