"""Model lifecycle and state transitions."""

from app.domain.enums import ModelStatus
from app.core.exceptions import AppError

class ModelStateTransitionError(AppError):
    """Raised when an invalid model state transition is attempted."""
    def __init__(self, current: ModelStatus, target: ModelStatus) -> None:
        super().__init__(
            code="INVALID_MODEL_STATE_TRANSITION",
            message=f"Invalid transition from {current.value} to {target.value}",
            status_code=400
        )

VALID_TRANSITIONS = {
    ModelStatus.UPLOADED: {ModelStatus.INSPECTING, ModelStatus.ERROR},
    ModelStatus.INSPECTING: {ModelStatus.CONFIGURATION_REQUIRED, ModelStatus.ERROR},
    ModelStatus.CONFIGURATION_REQUIRED: {ModelStatus.VALIDATING, ModelStatus.READY, ModelStatus.ERROR},
    ModelStatus.VALIDATING: {ModelStatus.READY, ModelStatus.CONFIGURATION_REQUIRED, ModelStatus.ERROR},
    ModelStatus.READY: {ModelStatus.ACTIVE, ModelStatus.ERROR},
    ModelStatus.ACTIVE: {ModelStatus.READY, ModelStatus.ERROR},
    ModelStatus.ERROR: {ModelStatus.UPLOADED, ModelStatus.INSPECTING}
}

def validate_transition(current: ModelStatus, target: ModelStatus) -> None:
    """Validate that transitioning from current to target state is allowed."""
    allowed = VALID_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise ModelStateTransitionError(current, target)
