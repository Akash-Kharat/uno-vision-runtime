import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.profile import ModelProfile
from app.domain.enums import ModelTask, ModelStatus

@pytest.fixture
def app():
    return create_app()

@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client

from app.schemas.model import ModelInspectionResult
from app.domain.enums import ModelFormat, ModelTask

def get_mock_inspection():
    return ModelInspectionResult(
        success=True,
        format=ModelFormat.ONNX,
        task=ModelTask.UNKNOWN,
        inputs=[],
        outputs=[]
    )

@patch("app.api.models.ONNXModelInspector")
def test_upload_valid_model(mock_inspector_cls, client):
    mock_inspector = MagicMock()
    mock_inspector.inspect.return_value = get_mock_inspection()
    mock_inspector_cls.return_value = mock_inspector
    
    res = client.post("/api/v1/models/upload", files={"file": ("test.onnx", b"dummy onnx content")})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["metadata"]["status"] == "CONFIGURATION_REQUIRED"
    assert data["metadata"]["original_filename"] == "test.onnx"

def test_upload_invalid_extension(client):
    res = client.post("/api/v1/models/upload", files={"file": ("test.txt", b"dummy content")})
    assert res.status_code == 400
    assert "Only .onnx files are allowed" in res.json()["error"]["message"]

@patch("app.api.models.ONNXModelInspector")
def test_upload_oversized_file(mock_inspector_cls, client, app):
    app.state.settings.MAX_UPLOAD_SIZE_BYTES = 10
    res = client.post("/api/v1/models/upload", files={"file": ("test.onnx", b"this is more than 10 bytes")})
    assert res.status_code == 413
    assert "exceeds maximum upload size" in res.json()["error"]["message"]

@patch("app.api.models.ONNXModelInspector")
def test_model_lifecycle(mock_inspector_cls, client, app):
    mock_inspector = MagicMock()
    mock_inspector.inspect.return_value = get_mock_inspection()
    mock_inspector_cls.return_value = mock_inspector
    
    res = client.post("/api/v1/models/upload", files={"file": ("test.onnx", b"dummy")})
    assert res.status_code == 200
    model_id = res.json()["metadata"]["id"]
    
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    assert any(m["id"] == model_id for m in res.json()["models"])
    
    res = client.put(f"/api/v1/models/{model_id}/profile", json={
        "task": "UNKNOWN",
        "input": {"layout": "UNKNOWN", "color_format": "UNKNOWN"},
        "preprocessing": {"resize": "NONE", "normalization": {"type": "UNKNOWN"}},
        "output": {"processor": "UNKNOWN"}
    })
    assert res.status_code == 400
    
    res = client.put(f"/api/v1/models/{model_id}/profile", json={
        "task": "OBJECT_DETECTION",
        "input": {"layout": "NCHW", "color_format": "RGB"},
        "preprocessing": {"resize": "LETTERBOX", "normalization": {"type": "SCALE_0_1", "scale": 0.00392}},
        "output": {"processor": "YOLO"},
        "classes": ["person"]
    })
    assert res.status_code == 200, res.json()
    
    res = client.get(f"/api/v1/models/{model_id}")
    assert res.status_code == 200
    assert res.json()["metadata"]["status"] == "READY"
    
    with patch("app.services.runtime_manager.ort.InferenceSession") as mock_ort:
        mock_ort.return_value = MagicMock()
        res = client.post(f"/api/v1/models/{model_id}/activate")
        assert res.status_code == 200
        
    res = client.get(f"/api/v1/models/{model_id}")
    assert res.json()["metadata"]["status"] == "ACTIVE"
    assert res.json()["metadata"]["active"] is True
    
    res = client.get("/api/v1/health")
    assert res.json()["components"]["model_runtime"] == "READY"
    
    res = client.delete(f"/api/v1/models/{model_id}")
    assert res.status_code == 400
    
    app.state.model_registry.set_active(None)
    
    res = client.delete(f"/api/v1/models/{model_id}")
    assert res.status_code == 200
