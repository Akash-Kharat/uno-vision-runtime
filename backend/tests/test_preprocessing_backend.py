import pytest
import numpy as np
from unittest.mock import MagicMock
from app.config import get_settings
from app.services.preprocessing import Preprocessor
from app.schemas.profile import ModelProfile, InputProfile, PreprocessingProfile, NormalizationProfile, OutputProfile

def test_preprocessing_backend_auto_fallback():
    settings = get_settings()
    settings.PREPROCESSING_BACKEND = "AUTO"
    
    # Mock backend
    mock_backend = MagicMock()
    mock_backend.is_available.return_value = True
    
    # Force the backend to raise an exception to test fallback
    mock_backend.preprocess_yolo.side_effect = Exception("OpenCL crashed")
    
    preprocessor = Preprocessor(backend=mock_backend, config=settings)
    
    from app.domain.enums import InputLayout
    profile = ModelProfile(
        input=InputProfile(width=640, height=640, layout=InputLayout.NCHW),
        preprocessing=PreprocessingProfile(
            normalization=NormalizationProfile(type="SCALE_0_1")
        ),
        output=OutputProfile()
    )
    
    # Dummy frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Should not crash, should fall back to CPU implementation
    res = preprocessor.preprocess_frame(frame, profile)
    
    # Verify fallback executed
    mock_backend.preprocess_yolo.assert_called_once()
    assert res.tensor.shape == (1, 3, 640, 640)
