"""Model inspector service."""

import logging
from pathlib import Path
from typing import Protocol, Any

import onnxruntime as ort

from app.domain.enums import ModelFormat, ModelTask
from app.schemas.model import ModelInspectionResult, ModelTensorInfo
from app.core.exceptions import AppError

logger = logging.getLogger(__name__)

class ModelInspector(Protocol):
    """Protocol for a model inspector."""
    def inspect(self, model_path: Path) -> ModelInspectionResult:
        ...

class ONNXModelInspector:
    """Inspector for ONNX models."""
    def inspect(self, model_path: Path) -> ModelInspectionResult:
        if not model_path.exists():
            raise AppError(
                code="MODEL_NOT_FOUND", 
                message=f"Model not found at {model_path}", 
                status_code=404
            )
            
        if model_path.suffix.lower() != ".onnx":
            raise AppError(
                code="UNSUPPORTED_MODEL_FORMAT", 
                message="Only ONNX format is currently supported", 
                status_code=400
            )
            
        try:
            # We strictly use CPU for inspection.
            sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            
            inputs = []
            for item in sess.get_inputs():
                inputs.append(ModelTensorInfo(
                    name=item.name,
                    shape=item.shape,
                    dtype=item.type
                ))
                
            outputs = []
            for item in sess.get_outputs():
                outputs.append(ModelTensorInfo(
                    name=item.name,
                    shape=item.shape,
                    dtype=item.type
                ))
                
            meta = sess.get_modelmeta()
            metadata_dict = meta.custom_metadata_map if meta else {}
            
            detected = ["format"]
            if inputs:
                detected.extend(["input.name", "input.shape", "input.dtype"])
            if outputs:
                detected.extend(["output.names", "output.shapes", "output.dtype"])
                
            # Programmatically compute unknown fields for the profile
            all_profile_fields = [
                "task", "input.name", "input.shape", "input.dtype", 
                "input.color_format", "input.layout",
                "preprocessing.resize", "preprocessing.normalization",
                "output.names", "output.shapes", "output.dtype",
                "output.processor", "classes"
            ]
            
            unknown = [f for f in all_profile_fields if f not in detected]
            
            warnings = []
            # Heuristic check just to issue a warning, explicitly NOT changing the task
            if len(outputs) > 0 and isinstance(outputs[0].shape, list) and len(outputs[0].shape) == 3:
                warnings.append(
                    "Output structure resembles a common detection model but task cannot be reliably confirmed. "
                    "Manual configuration required."
                )
                
            return ModelInspectionResult(
                success=True,
                format=ModelFormat.ONNX,
                task=ModelTask.UNKNOWN,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata_dict,
                detected_fields=detected,
                unknown_fields=unknown,
                warnings=warnings
            )
            
        except AppError:
            raise
        except Exception as e:
            logger.exception("Failed to inspect ONNX model")
            raise AppError(
                code="MODEL_INSPECTION_FAILED",
                message=f"Failed to inspect model: {str(e)}",
                status_code=500
            )
