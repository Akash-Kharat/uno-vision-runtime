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
    
    def __init__(self, provider_manager: ExecutionProviderManager, config=None):
        self.provider_manager = provider_manager
        self.config = config

    def create(self, model_path: Path | str, override_providers: list[str] | None = None) -> dict:
        """
        Create a session.
        Returns a dict with:
          - session: ort.InferenceSession
          - requested_providers: list[str]
          - actual_providers: list[str]
          - creation_time_ms: float
          - options: dict
        """
        path_str = str(model_path)
        
        providers = override_providers if override_providers else self.provider_manager.get_providers()
        
        def build_opts():
            opts = ort.SessionOptions()
            applied = {}
            if self.config:
                exec_mode_map = {
                    "SEQUENTIAL": ort.ExecutionMode.ORT_SEQUENTIAL,
                    "PARALLEL": ort.ExecutionMode.ORT_PARALLEL
                }
                opt_level_map = {
                    "DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                    "BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                    "EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                    "ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                }
                
                intra = getattr(self.config, "ORT_INTRA_OP_THREADS", 4)
                inter = getattr(self.config, "ORT_INTER_OP_THREADS", 2)
                mode = getattr(self.config, "ORT_EXECUTION_MODE", "SEQUENTIAL").upper()
                opt_level = getattr(self.config, "ORT_GRAPH_OPTIMIZATION", "ALL").upper()
                arena = getattr(self.config, "ORT_ENABLE_CPU_MEM_ARENA", True)
                pattern = getattr(self.config, "ORT_ENABLE_MEM_PATTERN", True)
                
                opts.intra_op_num_threads = intra
                opts.inter_op_num_threads = inter
                opts.execution_mode = exec_mode_map.get(mode, ort.ExecutionMode.ORT_SEQUENTIAL)
                opts.graph_optimization_level = opt_level_map.get(opt_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
                opts.enable_cpu_mem_arena = arena
                opts.enable_mem_pattern = pattern
                
                applied = {
                    "intra_op_num_threads": opts.intra_op_num_threads,
                    "inter_op_num_threads": opts.inter_op_num_threads,
                    "execution_mode": mode,
                    "graph_optimization_level": opt_level,
                    "enable_cpu_mem_arena": opts.enable_cpu_mem_arena,
                    "enable_mem_pattern": opts.enable_mem_pattern
                }
            return opts, applied

        opts = ort.SessionOptions()
        applied_options = {}
        
        if "CPUExecutionProvider" in providers:
            opts, applied_options = build_opts()
            
        t0 = time.perf_counter()
        
        try:
            session = ort.InferenceSession(path_str, providers=providers, sess_options=opts)
        except Exception as e:
            if override_providers:
                # If explicit benchmark failed
                raise AppError(code="SESSION_INIT_FAILED", message=str(e), status_code=500)
                
            # If default failed, we should fallback to CPU if not already trying it
            if "CPUExecutionProvider" not in providers:
                self.provider_manager.record_initialization_error(providers[0], str(e))
                logger.warning("Falling back to CPUExecutionProvider")
                try:
                    fallback_opts, applied_options = build_opts()
                    session = ort.InferenceSession(path_str, providers=["CPUExecutionProvider"], sess_options=fallback_opts)
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
            "creation_time_ms": (t1 - t0) * 1000.0,
            "options": applied_options
        }
