"""Model profile validator service."""

from pydantic import BaseModel
from app.schemas.model import ModelInspectionResult
from app.schemas.profile import ModelProfile
from app.domain.enums import ModelTask, InputLayout, ColorFormat, ResizeMethod, NormalizationType
from app.services.output_registry import output_registry

class ProfileValidationResult(BaseModel):
    """Result of profile validation."""
    valid: bool
    missing_fields: list[str]
    warnings: list[str]
    errors: list[str]

class ModelProfileValidator:
    """Validates whether a model profile contains enough information to run."""
    def validate(self, inspection: ModelInspectionResult, profile: ModelProfile) -> ProfileValidationResult:
        missing = []
        errors = []
        warnings = []
        
        if profile.task == ModelTask.UNKNOWN:
            missing.append("task")
            
        if profile.input.color_format == ColorFormat.UNKNOWN:
            missing.append("input.color_format")
            
        if profile.input.layout == InputLayout.UNKNOWN:
            missing.append("input.layout")
            
        if profile.preprocessing.resize == ResizeMethod.NONE:
            warnings.append("No resize method provided. Ensure inputs inherently match the model's expected resolution.")
            
        if profile.preprocessing.normalization.type == NormalizationType.UNKNOWN:
            missing.append("preprocessing.normalization")
            
        if profile.output.processor == "UNKNOWN":
            missing.append("output.processor")
        
        # Require classes for tasks that need it
        if not profile.classes and profile.task in [ModelTask.OBJECT_DETECTION, ModelTask.CLASSIFICATION]:
            missing.append("classes")
            
        # Check output processor against task
        if profile.output.processor != "UNKNOWN" and profile.task != ModelTask.UNKNOWN:
            proc_errors = output_registry.validate_profile(profile.output.processor, profile.task)
            errors.extend(proc_errors)
            
        # Normalization structural checks
        norm = profile.preprocessing.normalization
        if norm.type == NormalizationType.SCALE_0_1 and norm.scale is None:
            errors.append("SCALE_0_1 normalization requires a 'scale' value")
        elif norm.type == NormalizationType.MEAN_STD and (not norm.mean or not norm.std):
            errors.append("MEAN_STD normalization requires 'mean' and 'std' values")

        valid = len(missing) == 0 and len(errors) == 0
        
        return ProfileValidationResult(
            valid=valid,
            missing_fields=missing,
            warnings=warnings,
            errors=errors
        )
