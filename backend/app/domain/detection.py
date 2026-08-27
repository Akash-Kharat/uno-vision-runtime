"""Domain models for detection."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RawDetection:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass(frozen=True)
class PreprocessedInput:
    tensor: list | object  # The actual ndarray goes here, using object/list for typing simplicity
    original_width: int
    original_height: int
    model_width: int
    model_height: int
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int

@dataclass(frozen=True)
class DetectionResult:
    objects: list[RawDetection]
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
