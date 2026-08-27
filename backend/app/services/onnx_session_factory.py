"""ONNX Session Factory."""

import logging
import time
from pathlib import Path
import onnxruntime as ort

from app.core.exceptions import AppError
from app.services.execution_provider_manager import ExecutionProviderManager

logger = logging.getLogger(__name__)

class ONNXSessionFactory:
    """Creates ONNX sessions decoupling hardware logic from business logic."""
    
    def __init__(self, provider_manager: ExecutionProviderManager):
        self.provider_manager = provider_manager

    def create(self, model_path: Path | str, override_providers: list[str] | None = None) -> dict:
        """
        Create a session.
        Returns a dict with:
          - session: ort.InferenceSession
          - requested_providers: list[str]
          - actual_providers: list[str]
          - creation_time_ms: float
        """
        path_str = str(model_path)
        
        providers = override_providers if override_providers else self.provider_manager.get_providers()
        
        t0 = time.perf_counter()
        
        try:
            session = ort.InferenceSession(path_str, providers=providers)
        except Exception as e:
            if override_providers:
                # If explicit benchmark failed
                raise AppError(code="SESSION_INIT_FAILED", message=str(e), status_code=500)
                
            # If default failed, we should fallback to CPU if not already trying it
            if "CPUExecutionProvider" not in providers:
                self.provider_manager.record_initialization_error(providers[0], str(e))
                logger.warning("Falling back to CPUExecutionProvider")
                try:
                    session = ort.InferenceSession(path_str, providers=["CPUExecutionProvider"])
                except Exception as e_cpu:
                    raise AppError(code="SESSION_INIT_FAILED", message=f"CPU fallback failed: {str(e_cpu)}", status_code=500)
            else:
                raise AppError(code="SESSION_INIT_FAILED", message=str(e), status_code=500)
                
        t1 = time.perf_counter()
        
        actual = []
        try:
            actual = session.get_providers()
        except Exception:
            pass
            
        return {
            "session": session,
            "requested_providers": providers,
            "actual_providers": actual,
            "creation_time_ms": (t1 - t0) * 1000.0
        }
