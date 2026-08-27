"""API Schemas for detection results."""
from datetime import datetime
from typing import Any
from pydantic import BaseModel

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class DetectedObject(BaseModel):
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

class DetectionTimings(BaseModel):
    capture_time_ms: float
    preprocessing_time_ms: float
    inference_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float
    inner: dict[str, float] | None = None
    diagnostics: dict[str, Any] | None = None

class DetectionResponse(BaseModel):
    request_id: str
    success: bool
    timestamp: datetime

    model_id: str
    model_name: str | None = None

    inference_time_ms: float
    timings: DetectionTimings | None = None

    image_width: int
    image_height: int

    object_count: int
    class_counts: dict[str, int]
    objects: list[DetectedObject]
