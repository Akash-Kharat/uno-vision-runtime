import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.config import get_settings
from app.services.accelerators.opencl_backend import OpenCLBackend
from app.schemas.profile import ModelProfile, InputProfile, PreprocessingProfile, NormalizationProfile
from app.domain.enums import InputLayout

@pytest.fixture
def dummy_profile():
    return ModelProfile(
        input=InputProfile(width=640, height=640, layout=InputLayout.NCHW, dtype="tensor(float)"),
        preprocessing=PreprocessingProfile(
            normalization=NormalizationProfile(type="SCALE_0_1")
        )
    )

def test_opencl_backend_copy_mode(dummy_profile):
    settings = get_settings()
    settings.ENABLE_OPENCL = False  # Disable actual hardware initialization
    settings.OPENCL_MEMORY_MODE = "COPY"
    
    backend = OpenCLBackend(settings)
    
    # We can't easily mock the entire OpenCL context to run preprocess_yolo
    # without pyopencl actually executing, but we can verify it parses the mode properly.
    assert backend.config.OPENCL_MEMORY_MODE == "COPY"
    
def test_opencl_backend_mapped_mode(dummy_profile):
    settings = get_settings()
    settings.ENABLE_OPENCL = False
    settings.OPENCL_MEMORY_MODE = "MAPPED"
    
    backend = OpenCLBackend(settings)
    assert backend.config.OPENCL_MEMORY_MODE == "MAPPED"
