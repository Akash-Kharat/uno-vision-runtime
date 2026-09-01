import pytest
from app.config import get_settings, Settings
from app.services.execution_provider_manager import ExecutionProviderManager
from app.services.onnx_session_factory import ONNXSessionFactory
import onnxruntime as ort
import os

def test_onnx_session_factory_configures_options(tmp_path):
    # Test env override
    settings = get_settings()
    settings.ORT_INTRA_OP_THREADS = 3
    settings.ORT_INTER_OP_THREADS = 1
    settings.ORT_EXECUTION_MODE = "PARALLEL"
    
    ep_manager = ExecutionProviderManager()
    factory = ONNXSessionFactory(ep_manager, settings)
    
    dummy_path = str(tmp_path / "dummy.onnx")
    
    import unittest.mock as mock
    with mock.patch("onnxruntime.InferenceSession") as mock_session:
        mock_session.return_value.get_providers.return_value = ["CPUExecutionProvider"]
        
        res = factory.create(dummy_path, override_providers=["CPUExecutionProvider"])
        
        assert "options" in res
        opts = res["options"]
        assert opts["intra_op_num_threads"] == 3
        assert opts["inter_op_num_threads"] == 1
        assert opts["execution_mode"] == "PARALLEL"
        
        # Verify it passed sess_options to InferenceSession
        mock_session.assert_called_once()
        args, kwargs = mock_session.call_args
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        assert "sess_options" in kwargs
        sess_options = kwargs["sess_options"]
        assert sess_options.intra_op_num_threads == 3
        assert sess_options.inter_op_num_threads == 1
        assert sess_options.execution_mode == ort.ExecutionMode.ORT_PARALLEL

def test_default_threading_config():
    # Test that un-overridden config defaults to 4 and 2
    settings = Settings()
    assert settings.ORT_INTRA_OP_THREADS == 4
    assert settings.ORT_INTER_OP_THREADS == 2
    assert settings.ORT_EXECUTION_MODE == "SEQUENTIAL"

def test_cpu_fallback_inherits_config(tmp_path):
    settings = Settings()
    assert settings.ORT_INTRA_OP_THREADS == 4
    
    ep_manager = ExecutionProviderManager()
    factory = ONNXSessionFactory(ep_manager, settings)
    dummy_path = str(tmp_path / "dummy.onnx")
    
    import unittest.mock as mock
    with mock.patch("onnxruntime.InferenceSession") as mock_session:
        # First call (e.g. TensorRT) fails, second call (CPU fallback) succeeds
        mock_session.side_effect = [Exception("TensorRT failed"), mock.MagicMock()]
        
        # Ensure it resolves actual_providers properly in fallback
        mock_session.return_value.get_providers.return_value = ["CPUExecutionProvider"]
        
        with mock.patch.object(ep_manager, "get_providers", return_value=["TensorrtExecutionProvider"]):
            res = factory.create(dummy_path)
        
        assert mock_session.call_count == 2
        
        # Check fallback call
        args, kwargs = mock_session.call_args_list[1]
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        
        sess_options = kwargs["sess_options"]
        assert sess_options.intra_op_num_threads == 4
        assert sess_options.inter_op_num_threads == 2
        assert sess_options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
