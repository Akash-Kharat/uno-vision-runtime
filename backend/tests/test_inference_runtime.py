"""Tests for continuous inference manager and WebSocket streaming."""

import time
import pytest
from unittest.mock import MagicMock
from app.services.inference_runtime_manager import InferenceRuntimeManager
from app.domain.runtime_state import InferenceState
from app.core.exceptions import AppError

def test_inference_state_machine_invalid_transitions():
    cam = MagicMock()
    det = MagicMock()
    mgr = InferenceRuntimeManager(cam, det)
    
    assert mgr.state == InferenceState.STOPPED
    
    with pytest.raises(AppError) as e:
        mgr.pause()
    assert e.value.code == "INVALID_RUNTIME_TRANSITION"
    
    with pytest.raises(AppError) as e:
        mgr.resume()
    assert e.value.code == "INVALID_RUNTIME_TRANSITION"

def test_start_without_camera_fails():
    cam = MagicMock()
    cam.state = "STOPPED"
    det = MagicMock()
    
    mgr = InferenceRuntimeManager(cam, det)
    with pytest.raises(AppError) as e:
        mgr.start()
    assert e.value.code == "CAMERA_NOT_RUNNING"
    
def test_start_without_model_fails():
    cam = MagicMock()
    cam.state = "RUNNING"
    det = MagicMock()
    det.runtime_manager.get_active_runtime.return_value = (None, None)
    
    mgr = InferenceRuntimeManager(cam, det)
    with pytest.raises(AppError) as e:
        mgr.start()
    assert e.value.code == "MODEL_NOT_ACTIVE"

def test_inference_lifecycle():
    cam = MagicMock()
    cam.state = "RUNNING"
    
    det = MagicMock()
    session_mock = MagicMock()
    desc_mock = MagicMock()
    det.runtime_manager.get_active_runtime.return_value = (session_mock, desc_mock)
    
    mgr = InferenceRuntimeManager(cam, det)
    
    # Start it
    mgr.start()
    assert mgr.state in (InferenceState.STARTING, InferenceState.RUNNING)
    
    # Sleep tiny bit to let thread hit RUNNING
    time.sleep(0.05)
    assert mgr.state == InferenceState.RUNNING
    
    # Pause
    mgr.pause()
    assert mgr.state == InferenceState.PAUSED
    
    # Resume
    mgr.resume()
    assert mgr.state == InferenceState.RUNNING
    
    # Stop
    mgr.stop()
    assert mgr.state == InferenceState.STOPPED

def test_result_persistence_and_callback():
    from app.schemas.detection import DetectionResponse, DetectionTimings
    
    cam = MagicMock()
    cam.state = "RUNNING"
    frame_mock = MagicMock()
    frame_mock.sequence_id = 42
    cam.get_latest_frame.side_effect = [frame_mock, None] # One frame
    
    det = MagicMock()
    det.runtime_manager.get_active_runtime.return_value = (MagicMock(), MagicMock())
    
    resp_mock = DetectionResponse(
        request_id="123",
        success=True,
        timestamp="2024-01-01T00:00:00Z",
        model_id="mdl_abc",
        inference_time_ms=5.0,
        timings=DetectionTimings(capture_time_ms=1, preprocessing_time_ms=1, inference_time_ms=1, postprocessing_time_ms=1, total_time_ms=4),
        image_width=640,
        image_height=640,
        object_count=0,
        class_counts={},
        objects=[]
    )
    det.detect_current_frame.return_value = resp_mock
    
    mgr = InferenceRuntimeManager(cam, det)
    
    callbacks_fired = []
    def cb(snap):
        callbacks_fired.append(snap)
        
    mgr.register_callback(cb)
    
    mgr.start()
    time.sleep(0.1) # Let one loop run
    mgr.stop()
    
    assert len(callbacks_fired) == 1
    snap = callbacks_fired[0]
    assert snap.frame_sequence_id == 42
    assert snap.response == resp_mock
    
    res = mgr.get_latest_result()
    assert res is not None
    assert res["model_id"] == "mdl_abc"
