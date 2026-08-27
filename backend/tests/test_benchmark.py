import pytest
from unittest.mock import MagicMock
from app.main import create_app
from fastapi.testclient import TestClient
from app.domain.runtime_state import InferenceState

@pytest.fixture
def app_and_client():
    app = create_app()
    return app, TestClient(app)

def test_benchmark_api_rejects_conflict(app_and_client):
    app, client = app_and_client
    
    app.state.inference_manager.state = InferenceState.RUNNING
    
    res = client.post("/api/v1/benchmark/run", json={"iterations": 1, "warmup_iterations": 0})
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "BENCHMARK_RUNTIME_CONFLICT"

def test_benchmark_api_rejects_missing_camera(app_and_client):
    app, client = app_and_client
    
    app.state.inference_manager.state = InferenceState.STOPPED
    app.state.camera_manager.state = "STOPPED"
    
    res = client.post("/api/v1/benchmark/run", json={"iterations": 1, "warmup_iterations": 0})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "CAMERA_NOT_RUNNING"

def test_benchmark_api_rejects_missing_model(app_and_client):
    app, client = app_and_client
    
    app.state.inference_manager.state = InferenceState.STOPPED
    app.state.camera_manager.state = "RUNNING"
    app.state.detection_service.runtime_manager.get_active_runtime = MagicMock(return_value=(None, None))
    
    res = client.post("/api/v1/benchmark/run", json={"iterations": 1, "warmup_iterations": 0})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "MODEL_NOT_ACTIVE"

def test_benchmark_api_success(app_and_client):
    app, client = app_and_client
    
    app.state.inference_manager.state = InferenceState.STOPPED
    app.state.camera_manager.state = "RUNNING"
    
    # Mock successful detection loop
    frame_mock = MagicMock()
    app.state.camera_manager.get_latest_frame = MagicMock(return_value=frame_mock)
    
    session_mock = MagicMock()
    session_mock.get_inputs.return_value = [MagicMock(shape=[1, 3, 640, 640])]
    desc_mock = MagicMock()
    desc_mock.model_id = "test_id"
    
    app.state.detection_service.runtime_manager.get_active_runtime = MagicMock(return_value=(session_mock, desc_mock))
    
    metadata_mock = MagicMock()
    metadata_mock.original_filename = "yolo.onnx"
    app.state.model_registry.get_metadata = MagicMock(return_value=metadata_mock)
    
    resp_mock = MagicMock()
    timings_mock = MagicMock()
    timings_mock.capture_time_ms = 1.0
    timings_mock.preprocessing_time_ms = 2.0
    timings_mock.inference_time_ms = 3.0
    timings_mock.postprocessing_time_ms = 4.0
    timings_mock.total_time_ms = 10.0
    timings_mock.gpu_upload_ms = None
    timings_mock.gpu_kernel_ms = None
    timings_mock.gpu_download_ms = None
    timings_mock.total_gpu_time_ms = None
    resp_mock.timings = timings_mock
    
    app.state.detection_service.detect_current_frame = MagicMock(return_value=resp_mock)
    
    res = client.post("/api/v1/benchmark/run", json={"iterations": 3, "warmup_iterations": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["successful_iterations"] == 3
    assert data["failed_iterations"] == 0
    assert data["total_ms"]["mean"] == 10.0
    assert data["input_shape"] == [1, 3, 640, 640]
    
    # Called 5 times in total (2 warmup + 3 iterations)
    assert app.state.detection_service.detect_current_frame.call_count == 5
