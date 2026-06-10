"""Text sanitization for the ingestion pipeline.

PDF extraction (and occasionally web pages) can produce NUL bytes (\\x00)
and other control characters. PostgreSQL rejects strings containing NUL with
"A string literal cannot contain NUL (0x00) characters", so all text is
sanitized before being stored.

Only invalid control characters are removed: normal whitespace (tab, newline,
carriage return) and all printable text - including Hebrew - is preserved.
"""

import logging
import re

logger = logging.getLogger("ai_appeal.sanitize")

# C0 control characters except \t (\x09), \n (\x0a), \r (\x0d),
# plus DEL (\x7f) and C1 control characters (\x80-\x9f).
_INVALID_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def sanitize_text(text: str, *, label: str = "text") -> str:
    """Remove NUL and other unsafe control characters before DB storage."""
    cleaned = _INVALID_CONTROL_CHARS_RE.sub("", text)
    if len(cleaned) != len(text):
        logger.info(
            "Sanitized %s: removed %d invalid control character(s) (%d -> %d chars).",
            label,
            len(text) - len(cleaned),
            len(text),
            len(cleaned),
        )
    return cleaned
