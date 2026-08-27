"""Detection endpoint."""

from fastapi import APIRouter, Request

from app.schemas.detection import DetectionResponse

router = APIRouter()

@router.post("", response_model=DetectionResponse)
async def detect_frame(request: Request) -> DetectionResponse:
    """Run detection inference on the current camera frame."""
    detection_service = request.app.state.detection_service
    return detection_service.detect_current_frame()

import cv2
import numpy as np
from fastapi.responses import Response

@router.get("/debug/frame", response_class=Response)
async def debug_frame(request: Request) -> Response:
    """Return an annotated JPEG visualizing bounding boxes mapped over the original physical camera frame."""
    detection_service = request.app.state.detection_service
    
    # Run the detection to get canonical boundaries and standard response
    resp = detection_service.detect_current_frame()
    
    # We must grab the same frame the camera just saw. 
    # Since we don't return the raw frame inside DetectionResponse directly to save bandwidth,
    # we can fetch the *latest* cached frame directly from CameraManager right now,
    # knowing it will closely match what we just ran detection on (or exactly match if it's the latest).
    frame = request.app.state.camera_manager.capture_frame()
    if frame is None:
        from app.core.exceptions import AppError
        raise AppError(code="CAMERA_FRAME_UNAVAILABLE", message="Failed to capture frame for debug", status_code=500)

    # Clone for drawing safely
    img = frame.copy()
    
    # Draw canonical detections directly onto original image
    for obj in resp.objects:
        x1 = int(obj.bbox.x)
        y1 = int(obj.bbox.y)
        x2 = int(obj.bbox.x + obj.bbox.width)
        y2 = int(obj.bbox.y + obj.bbox.height)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{obj.class_name} ({obj.confidence:.2f})"
        cv2.putText(img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    ret, buffer = cv2.imencode('.jpg', img)
    if not ret:
        from app.core.exceptions import AppError
        raise AppError(code="INTERNAL_ERROR", message="Failed to encode jpeg", status_code=500)
        
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
