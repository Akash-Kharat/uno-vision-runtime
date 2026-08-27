import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.domain.runtime_state import InferenceState

@pytest.fixture
def app_and_client():
    app = create_app()
    return app, TestClient(app)

def test_system_hardware(app_and_client):
    app, client = app_and_client
    res = client.get("/api/v1/system/hardware")
    assert res.status_code == 200
    data = res.json()
    assert "cpu" in data
    assert "onnxruntime" in data
    assert "CPUExecutionProvider" in data["onnxruntime"]["available_providers"]

def test_system_providers(app_and_client):
    app, client = app_and_client
    res = client.get("/api/v1/system/providers")
    assert res.status_code == 200
    data = res.json()
    assert "active_provider" in data
    assert len(data["providers"]) > 0

def test_system_provider_activate(app_and_client):
    app, client = app_and_client
    app.state.inference_manager.state = InferenceState.STOPPED
    
    res = client.post("/api/v1/system/providers/CPUExecutionProvider/activate")
    assert res.status_code == 200
    assert res.json()["active_provider"] == "CPUExecutionProvider"
    
    # Try while running
    app.state.inference_manager.state = InferenceState.RUNNING
    res = client.post("/api/v1/system/providers/CPUExecutionProvider/activate")
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "PROVIDER_CHANGE_RUNTIME_CONFLICT"
