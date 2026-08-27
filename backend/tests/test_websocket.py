import pytest
from fastapi.testclient import TestClient
from app.main import create_app
import asyncio

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

def test_websocket_stream(client):
    # Just test that it connects and receives the initial state
    with client.websocket_connect("/api/v1/runtime/ws") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "runtime_state"
        assert "payload" in data
        assert data["payload"]["success"] is True
