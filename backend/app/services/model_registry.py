"""Model registry for managing persistent storage."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any
import os

from app.config import Settings
from app.domain.enums import ModelStatus
from app.schemas.registry import ModelMetadata
from app.schemas.model import ModelInspectionResult
from app.schemas.profile import ModelProfile
from app.core.exceptions import AppError

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, settings: Settings):
        self.storage_path = Path(settings.MODEL_STORAGE_PATH).resolve()
        self.models_dir = self.storage_path / "models"
        self.runtime_dir = self.storage_path / "runtime"
        self.active_file = self.runtime_dir / "active_model.json"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_dir(self, model_id: str) -> Path:
        # Prevent path traversal
        clean_id = os.path.basename(model_id)
        if clean_id != model_id:
            raise AppError(code="INVALID_MODEL_ID", message="Invalid model ID format", status_code=400)
        return self.models_dir / model_id

    def register(self, metadata: ModelMetadata, inspection: ModelInspectionResult, temp_model_path: Path) -> None:
        model_dir = self._get_model_dir(metadata.id)
        if model_dir.exists():
            raise AppError(code="MODEL_ALREADY_EXISTS", message="Model already exists", status_code=409)
            
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Move model file
            target_model_path = model_dir / "model.onnx"
            shutil.move(str(temp_model_path), str(target_model_path))
            
            # Write inspection
            with open(model_dir / "inspection.json", "w", encoding="utf-8") as f:
                f.write(inspection.model_dump_json())
                
            # Write metadata
            self.update_metadata(metadata)
        except Exception as e:
            # Cleanup on failure
            if model_dir.exists():
                shutil.rmtree(model_dir)
            raise AppError(code="MODEL_REGISTRATION_FAILED", message=str(e), status_code=500)

    def get_metadata(self, model_id: str) -> ModelMetadata:
        model_dir = self._get_model_dir(model_id)
        meta_file = model_dir / "metadata.json"
        if not meta_file.exists():
            raise AppError(code="MODEL_NOT_FOUND", message=f"Model {model_id} not found", status_code=404)
            
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Add active status dynamically based on registry active file
            data["active"] = (self.get_active() == model_id)
            return ModelMetadata(**data)

    def get_inspection(self, model_id: str) -> ModelInspectionResult:
        model_dir = self._get_model_dir(model_id)
        file = model_dir / "inspection.json"
        if not file.exists():
            raise AppError(code="MODEL_NOT_FOUND", message="Inspection not found", status_code=404)
        with open(file, "r", encoding="utf-8") as f:
            return ModelInspectionResult(**json.load(f))

    def get_profile(self, model_id: str) -> ModelProfile | None:
        model_dir = self._get_model_dir(model_id)
        file = model_dir / "profile.json"
        if not file.exists():
            return None
        with open(file, "r", encoding="utf-8") as f:
            return ModelProfile(**json.load(f))

    def save_profile(self, model_id: str, profile: ModelProfile) -> None:
        model_dir = self._get_model_dir(model_id)
        if not model_dir.exists():
            raise AppError(code="MODEL_NOT_FOUND", message=f"Model {model_id} not found", status_code=404)
        with open(model_dir / "profile.json", "w", encoding="utf-8") as f:
            f.write(profile.model_dump_json())

    def update_metadata(self, metadata: ModelMetadata) -> None:
        model_dir = self._get_model_dir(metadata.id)
        if not model_dir.exists():
            raise AppError(code="MODEL_NOT_FOUND", message=f"Model {metadata.id} not found", status_code=404)
        # We don't save the 'active' field to disk metadata since active is global
        data = metadata.model_dump()
        data.pop("active", None)
        with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def list_models(self) -> list[ModelMetadata]:
        models = []
        if not self.models_dir.exists():
            return []
            
        active_id = self.get_active()
        
        for p in self.models_dir.iterdir():
            if p.is_dir() and (p / "metadata.json").exists():
                try:
                    with open(p / "metadata.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["active"] = (data.get("id") == active_id)
                        models.append(ModelMetadata(**data))
                except Exception:
                    logger.warning(f"Failed to load metadata for {p.name}")
        return models

    def delete(self, model_id: str) -> None:
        if self.get_active() == model_id:
            raise AppError(code="ACTIVE_MODEL_DELETE_FORBIDDEN", message="Cannot delete active model", status_code=400)
            
        model_dir = self._get_model_dir(model_id)
        if not model_dir.exists():
            raise AppError(code="MODEL_NOT_FOUND", message=f"Model {model_id} not found", status_code=404)
        shutil.rmtree(model_dir)

    def set_active(self, model_id: str | None) -> None:
        if model_id is None:
            if self.active_file.exists():
                self.active_file.unlink()
            return
            
        # Ensure model exists
        self.get_metadata(model_id)
        
        with open(self.active_file, "w", encoding="utf-8") as f:
            json.dump({"active_model_id": model_id}, f)

    def get_active(self) -> str | None:
        if not self.active_file.exists():
            return None
        try:
            with open(self.active_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("active_model_id")
        except Exception:
            return None

    def get_model_path(self, model_id: str) -> Path:
        """Get absolute path to model ONNX file without exposing to public API."""
        return self._get_model_dir(model_id) / "model.onnx"
