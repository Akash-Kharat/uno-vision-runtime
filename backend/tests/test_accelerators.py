import pytest
from unittest.mock import MagicMock, patch
from app.config import get_settings
from app.services.accelerators.opencl_backend import OpenCLBackend

def test_opencl_backend_discovery():
    settings = get_settings()
    settings.ENABLE_OPENCL = False
    
    backend = OpenCLBackend(settings)
    backend.initialize()
    
    # Since ENABLE_OPENCL is False, it should not be available
    assert backend.is_available() is False

@patch("app.services.accelerators.opencl_context.HAS_OPENCL", False)
def test_opencl_backend_no_pyopencl():
    settings = get_settings()
    settings.ENABLE_OPENCL = True
    
    backend = OpenCLBackend(settings)
    backend.initialize()
    
    assert backend.is_available() is False
    assert backend.get_device_info() == {"available": False}
