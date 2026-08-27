"""Model runtime descriptor combining inspection and profile."""

from pathlib import Path
from pydantic import BaseModel
from app.schemas.model import ModelInspectionResult
from app.schemas.profile import ModelProfile

class ModelRuntimeDescriptor(BaseModel):
    """A descriptor for a model ready for inference."""
    model_id: str
    model_path: Path
    inspection_result: ModelInspectionResult
    profile: ModelProfile
