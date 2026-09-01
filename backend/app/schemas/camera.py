"""Camera API response schemas."""

from typing import Literal
from pydantic import BaseModel


class CameraStateDetail(BaseModel):
    """Details about the current camera state."""
    
    state: Literal["STOPPED", "STARTING", "RUNNING", "ERROR"]
    index: int
    device: str | None = None
    requested_width: int
    requested_height: int
    # Actual dimensions are set to null when the camera is not RUNNING
    actual_width: int | None = None
    actual_height: int | None = None


class CameraStatusResponse(BaseModel):
    """Response model for camera status."""

    success: bool
    camera: CameraStateDetail
