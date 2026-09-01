"""Tests for camera functionality."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import AppError
from app.main import create_app
from app.services.camera_manager import CameraManager

# Mock settings for predictable tests
MOCK_SETTINGS = {
    "CAMERA_INDEX": 0,
    "CAMERA_WIDTH": 1280,
    "CAMERA_HEIGHT": 720,
    "CAMERA_STARTUP_TIMEOUT_SECONDS": 1
}


def valid_mock_frame():
    """Return a mock frame with a valid pixel range."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[0, 0] = 255
    return frame


@pytest.fixture
def app():
    """Create test application."""
    app = create_app()
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def camera_manager(app):
    """Get the camera manager from the app."""
    return app.state.camera_manager


def test_camera_status_before_startup(client):
    """Test camera status before starting."""
    response = client.get("/api/v1/camera/status")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["camera"]["state"] == "STOPPED"
    assert data["camera"]["index"] == 0
    assert data["camera"]["requested_width"] == 1280
    assert data["camera"]["requested_height"] == 720
    assert data["camera"]["actual_width"] is None
    assert data["camera"]["actual_height"] is None


@patch("cv2.VideoCapture")
def test_successful_camera_startup(mock_video_capture, client, camera_manager):
    """Test starting the camera successfully."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, valid_mock_frame())
    mock_cap.get.side_effect = lambda prop: 1280 if prop == 3 else 720 # 3 is width, 4 is height
    mock_video_capture.return_value = mock_cap

    response = client.post("/api/v1/camera/start")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["camera"]["state"] == "RUNNING"
    assert data["camera"]["actual_width"] == 1280
    assert data["camera"]["actual_height"] == 720
    
    # Ensure capture is stored
    assert camera_manager.capture is mock_cap


@patch("cv2.VideoCapture")
def test_camera_startup_failure(mock_video_capture, client):
    """Test camera failure when it cannot be opened."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap

    response = client.post("/api/v1/camera/start")
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "CAMERA_START_FAILED"


@patch("cv2.VideoCapture")
def test_start_called_twice(mock_video_capture, client):
    """Test calling start when already running."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, valid_mock_frame())
    mock_cap.get.return_value = 1280
    mock_video_capture.return_value = mock_cap

    # First start
    client.post("/api/v1/camera/start")
    
    # Second start
    response = client.post("/api/v1/camera/start")
    assert response.status_code == 200
    data = response.json()
    assert data["camera"]["state"] == "RUNNING"
    
    # VideoCapture should only be called once
    mock_video_capture.assert_called_once()


@patch("cv2.VideoCapture")
def test_camera_startup_exception(mock_video_capture, client, camera_manager):
    """Test camera failure when cv2.VideoCapture raises an exception."""
    mock_video_capture.side_effect = Exception("OpenCV Error")

    response = client.post("/api/v1/camera/start")
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "CAMERA_START_FAILED"
    assert camera_manager.state == "ERROR"


@patch("cv2.VideoCapture")
def test_stop_called_while_running(mock_video_capture, client):
    """Test stopping an active camera."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, valid_mock_frame())
    mock_video_capture.return_value = mock_cap

    client.post("/api/v1/camera/start")
    
    response = client.post("/api/v1/camera/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["camera"]["state"] == "STOPPED"
    
    mock_cap.release.assert_called_once()


def test_stop_called_twice(client):
    """Test stopping a stopped camera."""
    # Already stopped
    response = client.post("/api/v1/camera/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["camera"]["state"] == "STOPPED"


def test_frame_capture_while_stopped(client):
    """Test capturing a frame when camera is not running."""
    response = client.get("/api/v1/camera/frame")
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == "CAMERA_NOT_RUNNING"


@patch("cv2.VideoCapture")
@patch("cv2.imencode")
def test_successful_frame_capture(mock_imencode, mock_video_capture, client):
    """Test successful frame capture and encoding."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, valid_mock_frame())
    mock_video_capture.return_value = mock_cap
    
    mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

    client.post("/api/v1/camera/start")
    
    response = client.get("/api/v1/camera/frame")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"\x01\x02\x03"


@patch("cv2.VideoCapture")
def test_failed_frame_capture(mock_video_capture, client):
    """Test handling of a failed frame read."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    # First read for startup succeeds, second for capture fails
    mock_cap.read.side_effect = [
        (True, valid_mock_frame()),
        (False, None)
    ]
    mock_video_capture.return_value = mock_cap

    client.post("/api/v1/camera/start")
    
    response = client.get("/api/v1/camera/frame")
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "CAMERA_READ_FAILED"


@patch("cv2.VideoCapture")
@patch("cv2.imencode")
def test_failed_jpeg_encoding(mock_imencode, mock_video_capture, client):
    """Test handling of failed JPEG encoding."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, valid_mock_frame())
    mock_video_capture.return_value = mock_cap
    
    # Encoding fails
    mock_imencode.return_value = (False, None)

    client.post("/api/v1/camera/start")
    
    response = client.get("/api/v1/camera/frame")
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "CAMERA_JPEG_ENCODE_FAILED"


@patch("platform.system")
@patch("cv2.VideoCapture")
def test_camera_startup_linux_backend(mock_video_capture, mock_platform, client):
    """Test backend selection on Linux."""
    mock_platform.return_value = "Linux"
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, valid_mock_frame())
    mock_video_capture.return_value = mock_cap
    
    import cv2
    v4l2_id = getattr(cv2, "CAP_V4L2", getattr(cv2, "CAP_ANY", 0))
    
    client.post("/api/v1/camera/start")
    mock_video_capture.assert_called_with(0, v4l2_id)


@patch("cv2.VideoCapture")
def test_camera_configuration(mock_video_capture, client):
    """Test FOURCC and FPS configuration."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, valid_mock_frame())
    mock_video_capture.return_value = mock_cap
    
    import cv2
    client.post("/api/v1/camera/start")
    
    calls = mock_cap.set.call_args_list
    props_set = [call[0][0] for call in calls]
    assert cv2.CAP_PROP_FOURCC in props_set
    assert cv2.CAP_PROP_FPS in props_set
    assert cv2.CAP_PROP_FRAME_WIDTH in props_set
    assert cv2.CAP_PROP_FRAME_HEIGHT in props_set


@patch("cv2.VideoCapture")
def test_camera_startup_black_frame_timeout(mock_video_capture, client, camera_manager):
    """Test that startup times out if only black frames are received."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    # Always return a black frame
    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, black_frame)
    mock_video_capture.return_value = mock_cap

    response = client.post("/api/v1/camera/start")
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "CAMERA_START_FAILED"
    assert "valid warm-up frame" in data["error"]["message"]
    assert mock_cap.read.call_count > 1


def test_camera_device_selection(app, client):
    """Test explicit CAMERA_DEVICE selection."""
    app.state.camera_manager.settings.CAMERA_DEVICE = "/dev/video2"
    
    with patch("cv2.VideoCapture") as mock_video_capture:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, valid_mock_frame())
        mock_video_capture.return_value = mock_cap
        
        response = client.post("/api/v1/camera/start")
        assert response.status_code == 200
        
        # Verify it passed the path
        args, kwargs = mock_video_capture.call_args
        assert args[0] == "/dev/video2"
        
        data = response.json()
        assert data["camera"]["device"] == "/dev/video2"

    app.state.camera_manager.settings.CAMERA_DEVICE = None
