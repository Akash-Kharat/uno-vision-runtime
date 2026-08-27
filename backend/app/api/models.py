"""Models API endpoints."""

import hashlib
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile, Depends
from pydantic import BaseModel

from app.domain.enums import (
    ModelTask, InputLayout, ColorFormat, ResizeMethod, NormalizationType, BoundingBoxFormat, ModelStatus
)
from app.domain.runtime import ModelRuntimeDescriptor
from app.domain.lifecycle import validate_transition
from app.schemas.model import ModelInspectionResult
from app.schemas.profile import ModelProfile
from app.schemas.registry import ModelMetadata, ModelListResponse, ModelDetailResponse
from app.services.model_inspector import ONNXModelInspector
from app.services.model_profile_validator import ModelProfileValidator
from app.services.output_registry import output_registry
from app.core.exceptions import AppError

router = APIRouter()

class InspectRequest(BaseModel):
    path: str

class InspectResponse(BaseModel):
    success: bool
    inspection: ModelInspectionResult

@router.post("/inspect", response_model=InspectResponse)
async def inspect_model(request: InspectRequest) -> InspectResponse:
    """Inspect a model available on disk (Development/Testing only)."""
    inspector = ONNXModelInspector()
    result = inspector.inspect(Path(request.path))
    return InspectResponse(success=True, inspection=result)

@router.get("/config-options")
async def get_config_options() -> dict[str, list[str]]:
    """Return available configuration options for model profiles."""
    return {
        "tasks": [t.value for t in ModelTask],
        "input_layouts": [l.value for l in InputLayout],
        "color_formats": [c.value for c in ColorFormat],
        "resize_methods": [r.value for r in ResizeMethod],
        "normalization_types": [n.value for n in NormalizationType],
        "bbox_formats": [b.value for b in BoundingBoxFormat],
        "output_processors": output_registry.list_available()
    }

@router.post("/upload", response_model=ModelDetailResponse)
async def upload_model(request: Request, file: UploadFile = File(...)) -> ModelDetailResponse:
    settings = request.app.state.settings
    registry = request.app.state.model_registry
    
    if not file.filename or not file.filename.lower().endswith(".onnx"):
        raise AppError(code="MODEL_UPLOAD_INVALID", message="Only .onnx files are allowed", status_code=400)
        
    model_id = f"mdl_{uuid.uuid4().hex[:12]}"
    temp_path = Path(settings.MODEL_STORAGE_PATH) / "runtime" / f"temp_{model_id}.onnx"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_size = 0
    sha256_hash = hashlib.sha256()
    
    try:
        with open(temp_path, "wb") as f:
            while chunk := await file.read(8192):
                file_size += len(chunk)
                if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
                    raise AppError(code="MODEL_TOO_LARGE", message="File exceeds maximum upload size", status_code=413)
                sha256_hash.update(chunk)
                f.write(chunk)
                
        # Inspect model
        inspector = ONNXModelInspector()
        try:
            inspection_result = inspector.inspect(temp_path)
        except Exception as e:
            raise AppError(code="MODEL_INSPECTION_FAILED", message=str(e), status_code=400)
            
        now = datetime.now(timezone.utc).isoformat()
        metadata = ModelMetadata(
            id=model_id,
            original_filename=file.filename,
            created_at=now,
            updated_at=now,
            status=ModelStatus.CONFIGURATION_REQUIRED,
            file_size_bytes=file_size,
            sha256=sha256_hash.hexdigest(),
        )
        
        # Register pushes file to permanent location
        registry.register(metadata, inspection_result, temp_path)
        
        return ModelDetailResponse(
            success=True,
            metadata=metadata,
            inspection=inspection_result.model_dump(),
            profile=None
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()

@router.get("", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    registry = request.app.state.model_registry
    models = registry.list_models()
    
    out = []
    for m in models:
        out.append({
            "id": m.id,
            "name": m.original_filename,
            "original_filename": m.original_filename,
            "status": m.status.value,
            "active": m.active
        })
    return ModelListResponse(success=True, models=out)

@router.get("/{model_id}", response_model=ModelDetailResponse)
async def get_model(request: Request, model_id: str) -> ModelDetailResponse:
    registry = request.app.state.model_registry
    meta = registry.get_metadata(model_id)
    insp = registry.get_inspection(model_id)
    prof = registry.get_profile(model_id)
    
    return ModelDetailResponse(
        success=True,
        metadata=meta,
        inspection=insp.model_dump(),
        profile=prof.model_dump() if prof else None
    )

@router.put("/{model_id}/profile", response_model=ModelDetailResponse)
async def update_profile(request: Request, model_id: str, profile: ModelProfile) -> ModelDetailResponse:
    registry = request.app.state.model_registry
    
    meta = registry.get_metadata(model_id)
    insp = registry.get_inspection(model_id)
    
    validator = ModelProfileValidator()
    val_result = validator.validate(insp, profile)
    
    if not val_result.valid:
        raise AppError(
            code="MODEL_PROFILE_INVALID",
            message=f"Profile is incomplete or invalid. Missing: {val_result.missing_fields}, Errors: {val_result.errors}",
            status_code=400
        )
        
    registry.save_profile(model_id, profile)
    
    # Update status to READY
    validate_transition(meta.status, ModelStatus.READY)
    meta.status = ModelStatus.READY
    meta.profile_file = "profile.json"
    meta.updated_at = datetime.now(timezone.utc).isoformat()
    registry.update_metadata(meta)
    
    return ModelDetailResponse(
        success=True,
        metadata=meta,
        inspection=insp.model_dump(),
        profile=profile.model_dump()
    )

@router.post("/{model_id}/activate")
async def activate_model(request: Request, model_id: str) -> dict[str, Any]:
    registry = request.app.state.model_registry
    runtime_manager = request.app.state.runtime_manager
    
    meta = registry.get_metadata(model_id)
    insp = registry.get_inspection(model_id)
    prof = registry.get_profile(model_id)
    
    if meta.status not in [ModelStatus.READY, ModelStatus.ACTIVE]:
        raise AppError(code="MODEL_NOT_READY", message="Model is not READY", status_code=400)
    if not prof:
        raise AppError(code="MODEL_PROFILE_INVALID", message="Model profile missing", status_code=400)
        
    descriptor = ModelRuntimeDescriptor(
        model_id=model_id,
        model_path=registry.get_model_path(model_id),
        inspection_result=insp,
        profile=prof
    )
    
    # Load candidate separately to prevent breaking old model on failure
    runtime_manager.load_model(descriptor)
    
    # Swap active
    runtime_manager.activate_model()
    
    # Update active tracking
    registry.set_active(model_id)
    
    # Advance status
    validate_transition(meta.status, ModelStatus.ACTIVE)
    if meta.status != ModelStatus.ACTIVE:
        meta.status = ModelStatus.ACTIVE
        registry.update_metadata(meta)
        
    return {"success": True, "message": f"Model {model_id} activated"}

@router.delete("/{model_id}")
async def delete_model(request: Request, model_id: str) -> dict[str, Any]:
    registry = request.app.state.model_registry
    registry.delete(model_id)
    return {"success": True, "message": f"Model {model_id} deleted"}
