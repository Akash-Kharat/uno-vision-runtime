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
    
    try:
        # Restore active model
        registry = getattr(app.state, "model_registry", None)
        runtime_manager = getattr(app.state, "runtime_manager", None)
        if registry and runtime_manager:
            active_id = registry.get_active()
            if active_id:
                try:
                    from app.domain.runtime import ModelRuntimeDescriptor
                    meta = registry.get_metadata(active_id)
                    insp = registry.get_inspection(active_id)
                    prof = registry.get_profile(active_id)
                    if prof:
                        desc = ModelRuntimeDescriptor(
                            model_id=active_id,
                            model_path=registry.get_model_path(active_id),
                            inspection_result=insp,
                            profile=prof
                        )
                        runtime_manager.load_model(desc)
                        runtime_manager.activate_model()
                        logger.info(f"Restored active model {active_id}")
                except Exception as e:
                    logger.error(f"Failed to restore active model {active_id}: {str(e)}")

        yield
    finally:
        logger.info("Shutting down uno-vision-runtime")
        if hasattr(app.state, "inference_manager"):
            try:
                app.state.inference_manager.stop()
            except Exception:
                pass
        
        if hasattr(app.state, "camera_manager"):
            app.state.camera_manager.stop()
