"""Model metadata schemas."""

from pydantic import BaseModel, Field
from app.domain.enums import ModelFormat, ModelTask

class ModelTensorInfo(BaseModel):
    """Information about a single input or output tensor."""
    name: str
    shape: list[int | str | None]
    dtype: str

class ModelInspectionResult(BaseModel):
    """Result of inspecting a model file."""
    success: bool
    format: ModelFormat
    task: ModelTask
    inputs: list[ModelTensorInfo] = Field(default_factory=list)
    outputs: list[ModelTensorInfo] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    detected_fields: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
