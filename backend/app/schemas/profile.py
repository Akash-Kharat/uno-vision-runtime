"""Model runtime configuration profiles."""

from pydantic import BaseModel, Field
from app.domain.enums import (
    ModelTask, InputLayout, ColorFormat, ResizeMethod, 
    NormalizationType, BoundingBoxFormat, ConfidenceInterpretation
)

class NormalizationProfile(BaseModel):
    """Profile for input normalization."""
    type: NormalizationType = NormalizationType.UNKNOWN
    scale: float | None = None
    mean: list[float] | None = None
    std: list[float] | None = None

class InputProfile(BaseModel):
    """Profile for model input constraints and formats."""
    name: str | None = None
    layout: InputLayout = InputLayout.UNKNOWN
    color_format: ColorFormat = ColorFormat.UNKNOWN
    width: int | None = None
    height: int | None = None
    dtype: str | None = None

class PreprocessingProfile(BaseModel):
    """Profile for pre-inference operations."""
    resize: ResizeMethod = ResizeMethod.CUSTOM
    normalization: NormalizationProfile = Field(default_factory=NormalizationProfile)

class OutputProfile(BaseModel):
    """Profile for post-inference processing."""
    processor: str = "UNKNOWN"
    bbox_format: BoundingBoxFormat = BoundingBoxFormat.UNKNOWN
    confidence_interpretation: ConfidenceInterpretation = ConfidenceInterpretation.UNKNOWN
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45

class ModelProfile(BaseModel):
    """Complete runtime profile for a model."""
    task: ModelTask = ModelTask.UNKNOWN
    input: InputProfile = Field(default_factory=InputProfile)
    preprocessing: PreprocessingProfile = Field(default_factory=PreprocessingProfile)
    output: OutputProfile = Field(default_factory=OutputProfile)
    classes: list[str] = Field(default_factory=list)
