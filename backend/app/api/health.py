"""Health check API endpoint."""

from fastapi import APIRouter

from app.config import get_settings
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return application health status."""
    settings = get_settings()
    return HealthResponse(
        success=True,
        status="healthy",
        service=settings.APP_NAME,
    )
