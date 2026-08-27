import pytest
from pathlib import Path
from unittest.mock import MagicMock
from app.services.onnx_session_factory import ONNXSessionFactory
from app.services.execution_provider_manager import ExecutionProviderManager

def test_session_factory_fallback(tmp_path):
    # Mock model
    model_path = tmp_path / "dummy.onnx"
    
    manager = ExecutionProviderManager()
    manager.available_providers = ["CPUExecutionProvider", "DummyProvider"]
    manager.active_provider = "CPUExecutionProvider"
    
    factory = ONNXSessionFactory(manager)
    
    # Normally we need a valid ONNX to test full creation, 
    # but we can test the fallback mechanics by letting it throw ONNX error 
    # and catching it.
    import onnxruntime as ort
    original_init = ort.InferenceSession.__init__
    
    call_providers = []
    
    def mock_init(self, path, providers=None, **kwargs):
        call_providers.append(providers)
        if "DummyProvider" in providers:
            raise Exception("Dummy hardware failed")
        return None
        
    ort.InferenceSession.__init__ = mock_init
    
    try:
        # Override to dummy
        from app.core.exceptions import AppError
        with pytest.raises(AppError) as exc:
            factory.create(model_path, override_providers=["DummyProvider"])
        assert "Dummy hardware failed" in str(exc.value)
        
        # Now try normal create with dummy as active provider
        call_providers.clear()
        manager.active_provider = "DummyProvider"
        manager.get_providers = MagicMock(return_value=["DummyProvider"])
        
        # It should try dummy, fail, and fallback to CPU!
        # Oh wait, my fallback logic falls back to CPU if CPU is not in the list.
        # But if mock_init raises for CPU, it will fail again.
        def mock_init_fallback(self, path, providers=None, **kwargs):
            call_providers.append(providers)
            if "DummyProvider" in providers:
                raise Exception("Dummy hardware failed")
            # For CPU, let's just return to simulate success
            self.get_providers = lambda: ["CPUExecutionProvider"]
            return None
            
        ort.InferenceSession.__init__ = mock_init_fallback
        
        res = factory.create(model_path)
        assert res["requested_providers"] == ["DummyProvider"]
        assert call_providers == [["DummyProvider"], ["CPUExecutionProvider"]]
        
    finally:
        ort.InferenceSession.__init__ = original_init
