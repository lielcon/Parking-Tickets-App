"""Clean text extraction from official municipal web pages.

Used only by the one-time ingestion pipeline. The RAG analysis flow never
fetches the web at request time - it relies exclusively on content already
stored in the database.
"""

import logging
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from ai_appeal.sanitization import sanitize_text

logger = logging.getLogger("ai_appeal.web")

REQUEST_TIMEOUT_SECONDS = 30
# Government/municipal sites often reject non-browser user agents, so the
# one-time fetch identifies as a regular browser.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.8",
}

# Tags that never contain legal content (navigation, chrome, scripts...).
# Note: <form> is NOT removed because ASP.NET sites (common for Israeli
# municipalities) wrap the entire page body in a single form element.
NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "iframe",
    "button",
    "svg",
    "input",
    "select",
    "textarea",
)

# Block-level tags whose text is collected as paragraphs.
CONTENT_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "blockquote"]


class WebExtractionError(Exception):
    """Raised when a page cannot be fetched or yields no usable text."""


@dataclass(frozen=True)
class ExtractedWebPage:
    url: str
    title: str
    text: str


def _fetch_html(url: str) -> str:
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers=REQUEST_HEADERS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WebExtractionError(f"Failed to fetch {url}: {exc}") from exc

    # Municipal pages sometimes omit the charset header; fall back to
    # detection so Hebrew content is decoded correctly.
    if "charset" not in (response.headers.get("Content-Type") or "").lower():
        response.encoding = response.apparent_encoding or response.encoding
    return response.text


def _pick_content_root(soup: BeautifulSoup):
    """Prefer <main>/<article>, but only when they actually hold content.

    Some municipal sites ship an empty <main> shell and render content
    elsewhere, so semantic containers are used only if they have real text.
    """
    for candidate in (soup.find("main"), soup.find("article")):
        if candidate is not None and len(candidate.get_text(strip=True)) >= 200:
            return candidate
    return soup.body or soup


def _is_menu_item(element) -> bool:
    """A list item whose entire text is a single link is navigation, not content."""
    if element.name != "li":
        return False
    links = element.find_all("a")
    if len(links) != 1:
        return False
    return element.get_text(" ", strip=True) == links[0].get_text(" ", strip=True)


def _extract_blocks(soup: BeautifulSoup) -> list[str]:
    """Collect paragraph-like text blocks from the main content area."""
    root = _pick_content_root(soup)

    blocks: list[str] = []
    for element in root.find_all(CONTENT_TAGS):
        # Skip containers whose text is already covered by a nested block
        # (e.g. an <li> wrapping a <p>) to avoid duplicated paragraphs.
        if element.find(CONTENT_TAGS):
            continue
        if _is_menu_item(element):
            continue
        text = element.get_text(" ", strip=True)
        # Skip client-side template fragments (e.g. Angular "{{ ... }}").
        if "{{" in text:
            continue
        if text and (not blocks or blocks[-1] != text):
            blocks.append(text)
    return blocks


def extract_web_page(url: str) -> ExtractedWebPage:
    """Fetch a URL and return its title and clean readable text."""
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else url

    for tag in soup(NOISE_TAGS):
        tag.decompose()

    blocks = _extract_blocks(soup)
    text = "\n\n".join(blocks).strip()

    # Fallback for pages with unusual markup: take all remaining text.
    if len(text) < 200:
        fallback_root = _pick_content_root(soup)
        lines = [
            line.strip()
            for line in fallback_root.get_text("\n").splitlines()
            if line.strip() and "{{" not in line
        ]
        fallback_text = "\n\n".join(lines).strip()
        if len(fallback_text) > len(text):
            text = fallback_text

    if not text:
        raise WebExtractionError(f"No readable text content extracted from {url}.")

    title = sanitize_text(title, label=f"page title ({url})").strip() or url
    text = sanitize_text(text, label=f"page text ({url})").strip()
    if not text:
        raise WebExtractionError(f"No readable text content extracted from {url}.")

    logger.info("Extracted %d characters from %s", len(text), url)
    return ExtractedWebPage(url=url, title=title, text=text)
