"""Camera API endpoints."""

import cv2
from fastapi import APIRouter, Request, Response

from app.core.exceptions import AppError
from app.schemas.camera import CameraStatusResponse

router = APIRouter()


@router.get("/status", response_model=CameraStatusResponse)
async def get_status(request: Request) -> CameraStatusResponse:
    """Get current camera status without opening it."""
    manager = request.app.state.camera_manager
    return manager.get_status()


@router.post("/start", response_model=CameraStatusResponse)
async def start_camera(request: Request) -> CameraStatusResponse:
    """Start the configured camera."""
    manager = request.app.state.camera_manager
    return manager.start()


@router.post("/stop", response_model=CameraStatusResponse)
async def stop_camera(request: Request) -> CameraStatusResponse:
    """Stop the active camera."""
    manager = request.app.state.camera_manager
    return manager.stop()


@router.get("/frame", responses={200: {"content": {"image/jpeg": {}}}})
async def capture_frame(request: Request) -> Response:
    """Capture one frame from the active camera as JPEG."""
    manager = request.app.state.camera_manager
    
    # This might raise AppError if not running or read fails
    frame = manager.capture_frame()
    
    # Encode to JPEG
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise AppError(
            code="CAMERA_JPEG_ENCODE_FAILED",
            message="Failed to encode frame to JPEG.",
            status_code=500
        )
        
    return Response(content=encoded.tobytes(), media_type="image/jpeg")
