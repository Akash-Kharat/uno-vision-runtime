"""UNO Vision Runtime — FastAPI application entry point."""

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.camera import router as camera_router
from app.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.lifecycle import lifespan
from app.services.camera_manager import CameraManager


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory.

    Args:
        settings: Optional pre-built settings (useful for testing).
                  When *None*, settings are loaded from the environment.

    Returns:
        A fully configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    # Attach settings and managers so lifecycle and handlers can access them
    app.state.settings = settings
    app.state.camera_manager = CameraManager(settings)

    # Exception handlers
    register_exception_handlers(app)

    # Root endpoint
    @app.get("/")
    async def root() -> dict[str, str]:
        """Return basic service information."""
        return {
            "service": "UNO Vision Runtime",
            "api_version": "v1",
        }

    # Versioned API routes
    app.include_router(health_router, prefix="/api/v1", tags=["health"])
    app.include_router(camera_router, prefix="/api/v1/camera", tags=["camera"])

    return app


# Default application instance used by `uvicorn app.main:app`
app: FastAPI = create_app()
