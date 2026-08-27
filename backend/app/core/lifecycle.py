"""Application lifecycle hooks (startup / shutdown)."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.config import Settings
from app.core.logging import setup_logging

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown events.

    The Settings instance is stored on ``app.state`` so it can be
    accessed from request handlers if needed.
    """
    settings: Settings = app.state.settings

    # --- Startup ---
    setup_logging(settings.LOG_LEVEL)
    logger.info(
        "Starting %s v%s on %s:%s",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.API_HOST,
        settings.API_PORT,
    )

    yield

    # --- Shutdown ---
    logger.info("Shutting down %s", settings.APP_NAME)
    if hasattr(app.state, "camera_manager"):
        app.state.camera_manager.stop()
