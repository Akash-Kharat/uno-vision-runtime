import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.domain.runtime_state import InferenceState

@pytest.fixture
def app_and_client():
    app = create_app()
    return app, TestClient(app)

def test_diagnostics_returns_session_options(app_and_client):
    app, client = app_and_client
    
    # Mock an active model with options
    app.state.detection_service.runtime_manager.active_session_options = {
        "intra_op_num_threads": 4,
        "inter_op_num_threads": 2
    }
    app.state.detection_service.runtime_manager.active_session = True # Truthy mock
    
    import unittest.mock as mock
    mock_desc = mock.MagicMock()
    mock_desc.model_id = "test_model"
    
    app.state.detection_service.runtime_manager.get_active_runtime = mock.MagicMock(
        return_value=(mock.MagicMock(), mock_desc)
    )
    
    res = client.get("/api/v1/system/diagnostics")
    assert res.status_code == 200
    data = res.json()
    
    assert data["success"] is True
    assert "session_options" in data
    assert data["session_options"]["intra_op_num_threads"] == 4
    assert data["session_options"]["inter_op_num_threads"] == 2
