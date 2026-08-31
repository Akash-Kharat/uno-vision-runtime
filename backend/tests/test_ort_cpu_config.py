import pytest
from app.config import get_settings
from app.services.execution_provider_manager import ExecutionProviderManager
from app.services.onnx_session_factory import ONNXSessionFactory
import onnxruntime as ort
import os

def test_onnx_session_factory_configures_options(tmp_path):
    settings = get_settings()
    settings.ORT_INTRA_OP_THREADS = 3
    settings.ORT_INTER_OP_THREADS = 2
    settings.ORT_EXECUTION_MODE = "PARALLEL"
    
    # We need a dummy onnx model to create a session, otherwise ort.InferenceSession fails
    # Let's mock ort.InferenceSession instead
    
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
        assert opts["inter_op_num_threads"] == 2
        assert opts["execution_mode"] == "PARALLEL"
        
        # Verify it passed sess_options to InferenceSession
        mock_session.assert_called_once()
        args, kwargs = mock_session.call_args
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        assert "sess_options" in kwargs
        sess_options = kwargs["sess_options"]
        assert sess_options.intra_op_num_threads == 3
        assert sess_options.inter_op_num_threads == 2
        assert sess_options.execution_mode == ort.ExecutionMode.ORT_PARALLEL
