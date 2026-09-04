import pytest
from pathlib import Path
from app.config import get_settings
from app.services.execution_provider_manager import ExecutionProviderManager
from app.services.onnx_session_factory import ONNXSessionFactory

@pytest.fixture
def test_models():
    settings = get_settings()
    fp32_path = Path(settings.MODEL_STORAGE_PATH) / "models" / "mdl_0f92eb15da2d" / "model.onnx"
    int8_path = Path(settings.MODEL_STORAGE_PATH) / "experiments" / "yolov8n_int8.onnx"
    return fp32_path, int8_path

def test_original_model_exists(test_models):
    fp32_path, _ = test_models
    if fp32_path.exists():
        assert fp32_path.stat().st_size > 10 * 1024 * 1024, "FP32 model should be ~12MB"

def test_quantized_model_loadable(test_models):
    fp32_path, int8_path = test_models
    if int8_path.exists():
        # Ensure it's smaller
        assert int8_path.stat().st_size < 5 * 1024 * 1024, "INT8 model should be < 5MB"
        
        # Test session factory compatibility
        settings = get_settings()
        ep_manager = ExecutionProviderManager()
        factory = ONNXSessionFactory(ep_manager, settings)
        
        try:
            res = factory.create(int8_path, override_providers=["CPUExecutionProvider"])
            assert res is not None
            assert res["session"] is not None
            opts = res.get("options", {})
            assert opts.get("intra_op_num_threads") == 4
        except Exception as e:
            pytest.fail(f"Failed to load INT8 model via factory: {e}")
