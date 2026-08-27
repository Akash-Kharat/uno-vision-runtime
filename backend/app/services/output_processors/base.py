"""Base output processor protocol."""

from typing import Protocol
import numpy as np

from app.domain.detection import RawDetection, PreprocessedInput
from app.domain.runtime import ModelRuntimeDescriptor

class OutputProcessor(Protocol):
    """Protocol for transforming raw model tensors into canonical detections."""
    
    name: str

    def process(
        self,
        outputs: list[np.ndarray],
        descriptor: ModelRuntimeDescriptor,
        preprocessing: PreprocessedInput,
    ) -> list[RawDetection]:
        """Convert raw tensors to bounding boxes."""
        ...
