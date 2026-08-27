"""Abstract interface for a compute accelerator backend."""

from typing import Protocol, Any, Dict
import numpy as np
from app.schemas.profile import ModelProfile
from app.domain.detection import PreprocessedInput

class ComputeBackend(Protocol):
    """Generic interface for hardware acceleration backends."""
    
    def initialize(self) -> None:
        """Initialize the backend context and load kernels if necessary."""
        ...
        
    def is_available(self) -> bool:
        """Returns True if the backend is successfully initialized and available for use."""
        ...
        
    def get_device_info(self) -> Dict[str, Any]:
        """Returns diagnostic information about the underlying hardware device."""
        ...
        
    def preprocess_yolo(self, frame: np.ndarray, profile: ModelProfile, profiler: Any = None) -> PreprocessedInput:
        """
        Executes YOLO specific preprocessing on the hardware backend.
        Must match the CPU equivalent: Resize (Letterbox), BGR->RGB, Normalization, HWC->CHW.
        """
        ...
        
    def shutdown(self) -> None:
        """Release any held hardware resources."""
        ...
