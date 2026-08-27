"""Detection integration tests."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from app.main import create_app
from app.services.preprocessing import Preprocessor
from app.domain.enums import ColorFormat, ResizeMethod, NormalizationType, InputLayout, BoundingBoxFormat, ConfidenceInterpretation
from app.schemas.profile import ModelProfile, InputProfile, PreprocessingProfile, NormalizationProfile, OutputProfile
from app.domain.detection import PreprocessedInput, RawDetection
from app.services.output_processors.yolo import YOLOOutputProcessor
from app.services.nms import class_aware_nms, compute_iou
from app.core.exceptions import AppError

@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as client:
        yield client

# --- Preprocessing Tests ---
def test_preprocessing_bgr_stretch_none_nchw():
    prep = Preprocessor()
    prof = ModelProfile(
        input=InputProfile(width=640, height=640, color_format=ColorFormat.BGR, layout=InputLayout.NCHW, dtype="tensor(float)"),
        preprocessing=PreprocessingProfile(resize=ResizeMethod.STRETCH, normalization=NormalizationProfile(type=NormalizationType.NONE))
    )
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    res = prep.preprocess_frame(frame, prof)
    
    assert res.original_width == 1280
    assert res.original_height == 720
    assert res.scale_x == 1280 / 640
    assert res.scale_y == 720 / 640
    assert res.pad_x == 0
    assert res.pad_y == 0
    assert res.tensor.shape == (1, 3, 640, 640)
    assert res.tensor.dtype == np.float32

def test_preprocessing_rgb_letterbox_scale_nhwc():
    prep = Preprocessor()
    prof = ModelProfile(
        input=InputProfile(width=640, height=640, color_format=ColorFormat.RGB, layout=InputLayout.NHWC, dtype="tensor(float16)"),
        preprocessing=PreprocessingProfile(resize=ResizeMethod.LETTERBOX, normalization=NormalizationProfile(type=NormalizationType.SCALE_0_1))
    )
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    res = prep.preprocess_frame(frame, prof)
    
    assert res.scale_x == 2.0  # 1280/640
    assert res.scale_y == 2.0
    assert res.pad_y == 140
    assert res.pad_x == 0
    assert res.tensor.shape == (1, 640, 640, 3)
    assert res.tensor.dtype == np.float16
    assert np.max(res.tensor) <= 1.0

# --- YOLO Output Processor Tests ---
def test_yolo_processor_1_84_8400():
    processor = YOLOOutputProcessor()
    
    prof = ModelProfile(
        output=OutputProfile(
            bbox_format=BoundingBoxFormat.CXCYWH,
            confidence_interpretation=ConfidenceInterpretation.DIRECT,
            confidence_threshold=0.5
        ),
        classes=["person", "car", "bike"]  # 3 classes, so 4 + 3 = 7 features minimum. We will use 7.
    )
    
    desc = MagicMock()
    desc.profile = prof
    
    prep = PreprocessedInput(tensor=[], original_width=1280, original_height=720, model_width=640, model_height=640, scale_x=2.0, scale_y=2.0, pad_x=0, pad_y=140)
    
    # [batch, features, boxes]
    # Features: cx, cy, w, h, cls0, cls1, cls2
    out_tensor = np.zeros((1, 7, 10), dtype=np.float32)
    # Box 1: cx=320, cy=320, w=100, h=100, cls=[0.9, 0.1, 0.1]
    out_tensor[0, :, 0] = [320, 320, 100, 100, 0.9, 0.1, 0.1]
    # Box 2: low confidence
    out_tensor[0, :, 1] = [320, 320, 100, 100, 0.2, 0.1, 0.1]
    
    dets = processor.process([out_tensor], desc, prep)
    
    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == 0
    assert d.confidence == pytest.approx(0.9)
    
    # Coordinate restoration check
    # Model box: cx=320, cy=320, w=100, h=100 -> x1=270, y1=270, x2=370, y2=370
    # Reverse pad: x1=270, y1=270-140=130, x2=370, y2=370-140=230
    # Reverse scale: x1=540, y1=260, x2=740, y2=460
    assert d.x1 == 540.0
    assert d.y1 == 260.0
    assert d.x2 == 740.0
    assert d.y2 == 460.0

def test_yolo_invalid_shape():
    processor = YOLOOutputProcessor()
    prof = ModelProfile(classes=["A", "B", "C"])
    desc = MagicMock()
    desc.profile = prof
    prep = MagicMock()
    
    # 3 classes + 4 boxes = 7 required, passing 6
    out_tensor = np.zeros((1, 6, 10), dtype=np.float32)
    with pytest.raises(AppError) as exc:
        processor.process([out_tensor], desc, prep)
    assert exc.value.code == "INVALID_MODEL_OUTPUT"

# --- NMS Tests ---
def test_nms_same_class_overlap():
    dets = [
        RawDetection(class_id=0, confidence=0.9, x1=10, y1=10, x2=100, y2=100),
        RawDetection(class_id=0, confidence=0.8, x1=12, y1=12, x2=98, y2=98), # High overlap
        RawDetection(class_id=0, confidence=0.7, x1=200, y1=200, x2=300, y2=300), # No overlap
    ]
    res = class_aware_nms(dets, 0.45)
    assert len(res) == 2
    assert res[0].confidence == 0.9
    assert res[1].confidence == 0.7

def test_nms_different_class():
    dets = [
        RawDetection(class_id=0, confidence=0.9, x1=10, y1=10, x2=100, y2=100),
        RawDetection(class_id=1, confidence=0.8, x1=12, y1=12, x2=98, y2=98), # High overlap but different class
    ]
    res = class_aware_nms(dets, 0.45)
    assert len(res) == 2

# --- API Tests ---
def test_detect_no_active_model(client):
    res = client.post("/api/v1/detect")
    # By default, camera isn't running in tests usually, wait we should mock it or check error order
    # Error order: 1. Camera check
    assert res.status_code == 503
    assert "Camera is not active" in res.json()["error"]["message"]

def test_detect_success(client):
    app = client.app
    app.state.camera_manager.state = "RUNNING"
    app.state.camera_manager.capture_frame = MagicMock(return_value=np.zeros((720, 1280, 3), dtype=np.uint8))
    
    desc = MagicMock()
    desc.model_id = "mdl_123"
    desc.profile = ModelProfile(
        classes=["person"],
        input=InputProfile(width=640, height=640, color_format=ColorFormat.RGB, layout=InputLayout.NCHW, dtype="tensor(float)"),
        preprocessing=PreprocessingProfile(resize=ResizeMethod.STRETCH, normalization=NormalizationProfile(type=NormalizationType.NONE)),
        output=OutputProfile(processor="YOLO", bbox_format=BoundingBoxFormat.XYWH)
    )
    
    mock_session = MagicMock()
    mock_node = MagicMock()
    mock_node.name = "images"
    mock_session.get_inputs.return_value = [mock_node]
    # Return fake output [1, 5, 10]
    fake_out = np.zeros((1, 5, 10), dtype=np.float32)
    fake_out[0, :, 0] = [320, 320, 100, 100, 0.9]
    mock_session.run.return_value = [fake_out]
    
    app.state.runtime_manager.get_active_runtime = MagicMock(return_value=(mock_session, desc))
    
    app.state.model_registry.get_metadata = MagicMock(return_value=MagicMock(original_filename="test.onnx"))
    
    res = client.post("/api/v1/detect")
    assert res.status_code == 200, res.json()
    data = res.json()
    assert data["success"] is True
    assert data["model_id"] == "mdl_123"
    assert data["object_count"] == 1
    assert data["class_counts"]["person"] == 1
    assert data["objects"][0]["class_name"] == "person"
    assert data["objects"][0]["bbox"]["width"] == 1280 * (100 / 640)  # stretched back
