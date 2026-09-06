"""Runtime state definitions for continuous inference."""

from enum import Enum
from dataclasses import dataclass
from typing import Any

from app.schemas.detection import DetectionResponse

class InferenceState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    INFERENCE = "INFERENCE"
    WAITING = "WAITING"
    DEGRADED = "DEGRADED"
    STOPPING = "STOPPING"
    ERROR = "ERROR"

@dataclass(frozen=True)
class InferenceResultSnapshot:
    """A thread-safe snapshot of the latest inference result."""
    sequence_id: int
    frame_sequence_id: int
    timestamp: float
    frame_timestamp: float
    detection_timestamp: float
    model_id: str
    response: DetectionResponse

@dataclass
class InferenceStats:
    """Mutable inference statistics."""
    total_inference_count: int = 0
    successful_inference_count: int = 0
    failed_inference_count: int = 0
    last_inference_time_ms: float = 0.0
    average_inference_time_ms: float = 0.0
    effective_inference_fps: float = 0.0
    last_error: str | None = None
    # Scheduler observability metrics
    captured_frames: int = 0
    inferred_frames: int = 0
    skipped_frames: int = 0
    inference_failures: int = 0
    latest_frame_age_ms: float | None = None
    inference_busy: bool = False
