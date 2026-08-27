"""Registry metadata schemas."""

from pydantic import BaseModel
from typing import Any
from app.domain.enums import ModelStatus

class ModelMetadata(BaseModel):
    """Metadata representing a registered model on disk."""
    id: str
    original_filename: str
    created_at: str
    updated_at: str
    status: ModelStatus
    file_size_bytes: int
    sha256: str
    inspection_file: str = "inspection.json"
    profile_file: str | None = None
    runtime: str = "onnx"
    active: bool = False

class ModelListResponse(BaseModel):
    success: bool
    models: list[dict[str, Any]]  # id, name, original_filename, status, task, active

class ModelDetailResponse(BaseModel):
    success: bool
    metadata: ModelMetadata
    inspection: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
