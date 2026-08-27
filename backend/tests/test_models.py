"""Tests for model discovery and configuration."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import create_app
from app.domain.enums import ModelTask, ModelFormat, ModelStatus, NormalizationType, ColorFormat, InputLayout, ResizeMethod, ConfidenceInterpretation, BoundingBoxFormat
from app.schemas.profile import ModelProfile, InputProfile, PreprocessingProfile, NormalizationProfile, OutputProfile
from app.schemas.model import ModelInspectionResult
from app.services.model_inspector import ONNXModelInspector
from app.services.model_profile_validator import ModelProfileValidator
from app.services.output_registry import output_registry
from app.domain.lifecycle import validate_transition, ModelStateTransitionError
from app.core.exceptions import AppError

@pytest.fixture
def app():
    return create_app()

@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client

# --- State Machine Tests ---
def test_valid_state_transitions():
    # Should not raise
    validate_transition(ModelStatus.UPLOADED, ModelStatus.INSPECTING)
    validate_transition(ModelStatus.INSPECTING, ModelStatus.CONFIGURATION_REQUIRED)
    validate_transition(ModelStatus.CONFIGURATION_REQUIRED, ModelStatus.VALIDATING)
    validate_transition(ModelStatus.VALIDATING, ModelStatus.READY)
    validate_transition(ModelStatus.READY, ModelStatus.ACTIVE)
    validate_transition(ModelStatus.ACTIVE, ModelStatus.READY)

def test_invalid_state_transitions():
    with pytest.raises(ModelStateTransitionError):
        validate_transition(ModelStatus.UPLOADED, ModelStatus.READY)
    with pytest.raises(ModelStateTransitionError):
        validate_transition(ModelStatus.ACTIVE, ModelStatus.INSPECTING)

# --- Output Registry Tests ---
def test_output_registry():
    assert "YOLO" in output_registry.list_available()
    
    proc = output_registry.get("YOLO")
    assert proc is not None
    assert ModelTask.OBJECT_DETECTION in proc.supported_tasks
    
    assert output_registry.get("NON_EXISTENT") is None

    # Processor / Task compatibility
    errors = output_registry.validate_profile("YOLO", ModelTask.CLASSIFICATION)
    assert len(errors) > 0
    assert "does not support task" in errors[0]
    
    errors = output_registry.validate_profile("INVALID_PROC", ModelTask.OBJECT_DETECTION)
    assert len(errors) > 0
    assert "Unknown output processor" in errors[0]
    
    errors = output_registry.validate_profile("YOLO", ModelTask.OBJECT_DETECTION)
    assert len(errors) == 0

# --- Inspector Tests ---
@patch("app.services.model_inspector.ort.InferenceSession")
def test_inspector_valid_model(mock_ort):
    mock_sess = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    mock_input.type = "tensor(float)"
    mock_sess.get_inputs.return_value = [mock_input]
    
    mock_output = MagicMock()
    mock_output.name = "output0"
    mock_output.shape = [1, 84, 8400]
    mock_output.type = "tensor(float)"
    mock_sess.get_outputs.return_value = [mock_output]
    
    mock_meta = MagicMock()
    mock_meta.custom_metadata_map = {"version": "8"}
    mock_sess.get_modelmeta.return_value = mock_meta
    
    mock_ort.return_value = mock_sess
    
    inspector = ONNXModelInspector()
    
    # We must patch Path.exists
    with patch.object(Path, "exists", return_value=True):
        res = inspector.inspect(Path("dummy.onnx"))
        
    assert res.success is True
    assert res.format == ModelFormat.ONNX
    assert res.task == ModelTask.UNKNOWN
    assert len(res.inputs) == 1
    assert res.inputs[0].name == "images"
    assert res.inputs[0].shape == [1, 3, 640, 640]
    assert len(res.outputs) == 1
    assert res.metadata["version"] == "8"
    assert "input.shape" in res.detected_fields
    assert "task" in res.unknown_fields
    # Check warning for [1, 84, 8400] detection signature
    assert any("detection model" in w for w in res.warnings)

def test_inspector_unsupported_extension():
    inspector = ONNXModelInspector()
    with patch.object(Path, "exists", return_value=True):
        with pytest.raises(AppError) as exc:
            inspector.inspect(Path("dummy.tflite"))
        assert exc.value.code == "UNSUPPORTED_MODEL_FORMAT"

def test_inspector_missing_file():
    inspector = ONNXModelInspector()
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(AppError) as exc:
            inspector.inspect(Path("dummy.onnx"))
        assert exc.value.code == "MODEL_NOT_FOUND"

# --- Validator Tests ---
def test_validator_complete_profile():
    insp = ModelInspectionResult(success=True, format=ModelFormat.ONNX, task=ModelTask.UNKNOWN)
    prof = ModelProfile(
        task=ModelTask.OBJECT_DETECTION,
        input=InputProfile(layout=InputLayout.NCHW, color_format=ColorFormat.RGB),
        preprocessing=PreprocessingProfile(
            resize=ResizeMethod.LETTERBOX,
            normalization=NormalizationProfile(type=NormalizationType.SCALE_0_1, scale=0.00392)
        ),
        output=OutputProfile(processor="YOLO"),
        classes=["person", "car"]
    )
    val = ModelProfileValidator().validate(insp, prof)
    assert val.valid is True
    assert len(val.missing_fields) == 0
    assert len(val.errors) == 0

def test_validator_incomplete_profile():
    insp = ModelInspectionResult(success=True, format=ModelFormat.ONNX, task=ModelTask.UNKNOWN)
    prof = ModelProfile() # default UNKNOWNs
    val = ModelProfileValidator().validate(insp, prof)
    assert val.valid is False
    assert "task" in val.missing_fields
    assert "input.color_format" in val.missing_fields
    assert "output.processor" in val.missing_fields

def test_validator_normalization_errors():
    insp = ModelInspectionResult(success=True, format=ModelFormat.ONNX, task=ModelTask.UNKNOWN)
    prof = ModelProfile(
        task=ModelTask.OBJECT_DETECTION,
        input=InputProfile(layout=InputLayout.NCHW, color_format=ColorFormat.RGB),
        preprocessing=PreprocessingProfile(
            resize=ResizeMethod.LETTERBOX,
            normalization=NormalizationProfile(type=NormalizationType.SCALE_0_1, scale=None) # missing scale
        ),
        output=OutputProfile(processor="YOLO"),
        classes=["person"]
    )
    val = ModelProfileValidator().validate(insp, prof)
    assert val.valid is False
    assert any("requires a 'scale' value" in e for e in val.errors)

# --- API Tests ---
def test_api_config_options(client):
    res = client.get("/api/v1/models/config-options")
    assert res.status_code == 200
    data = res.json()
    assert "tasks" in data
    assert "OBJECT_DETECTION" in data["tasks"]
    assert "output_processors" in data
    assert "YOLO" in data["output_processors"]

@patch("app.api.models.ONNXModelInspector.inspect")
def test_api_inspect(mock_inspect, client):
    mock_res = ModelInspectionResult(success=True, format=ModelFormat.ONNX, task=ModelTask.UNKNOWN)
    mock_inspect.return_value = mock_res
    
    res = client.post("/api/v1/models/inspect", json={"path": "/tmp/fake.onnx"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["inspection"]["format"] == "ONNX"
    mock_inspect.assert_called_once()
