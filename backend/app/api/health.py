"""Health check API endpoint."""

from fastapi import APIRouter, Request
from typing import Any

from app.config import get_settings
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health")
async def get_health(request: Request) -> dict[str, Any]:
    """Basic health check endpoint."""
    camera_manager = request.app.state.camera_manager
    runtime_manager = request.app.state.runtime_manager
    
    return {
        "success": True,
        "status": "healthy",
        "service": "uno-vision-runtime",
        "components": {
            "camera": camera_manager.state,
            "model_runtime": runtime_manager.get_status()
        }
    }
