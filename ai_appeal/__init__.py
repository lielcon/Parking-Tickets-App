"""AI Appeal Assistant (RAG) - self-contained, additive feature package.

main.py only needs to call `setup_ai_appeal(app)`; everything else
(schema init, retrieval, OpenAI access, routes) lives inside this package.
"""

import logging

from fastapi import FastAPI

logger = logging.getLogger("ai_appeal")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def setup_ai_appeal(app: FastAPI) -> None:
    """Initialize the feature schema and register its routes.

    Initialization failures are logged but never crash the host app, so the
    rest of the API keeps working even if this feature is misconfigured.
    """
    from ai_appeal.db import init_ai_appeal_schema
    from ai_appeal.router import router

    try:
        init_ai_appeal_schema()
    except Exception:
        logger.exception(
            "AI appeal schema initialization failed. "
            "The /ai-appeal-analysis endpoint may not work until this is fixed."
        )

    app.include_router(router)
    logger.info("AI Appeal Assistant routes registered.")
