"""Tests for the health and root endpoints."""

from fastapi.testclient import TestClient

from app.main import create_app

client = TestClient(create_app())


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_200(self) -> None:
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_body(self) -> None:
        data = client.get("/").json()
        assert data["service"] == "UNO Vision Runtime"
        assert data["api_version"] == "v1"


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(self) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_schema(self) -> None:
        data = client.get("/api/v1/health").json()
        assert data["success"] is True
        assert data["status"] == "healthy"
        assert data["service"] == "uno-vision-runtime"
