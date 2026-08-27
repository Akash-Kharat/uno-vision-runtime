"""Provider Benchmark Manager."""

import time
import logging
import numpy as np
from app.services.execution_provider_manager import ExecutionProviderManager
from app.services.onnx_session_factory import ONNXSessionFactory
from app.domain.execution_provider import ProviderBenchmarkResult, ProviderBenchmarkComparison
from app.core.exceptions import AppError

logger = logging.getLogger(__name__)

class ProviderBenchmarkManager:
    def __init__(
        self, 
        provider_manager: ExecutionProviderManager,
        session_factory: ONNXSessionFactory,
        detection_service,
        camera_manager
    ):
        self.provider_manager = provider_manager
        self.session_factory = session_factory
        self.detection_service = detection_service
        self.camera_manager = camera_manager

    def run_benchmark(self, warmup: int, iterations: int) -> ProviderBenchmarkComparison:
        results = []
        
        # We must benchmark all available providers
        providers_to_test = self.provider_manager.available_providers
        
        # Save state
        original_active = self.provider_manager.active_provider
        
        for provider in providers_to_test:
            logger.info(f"Benchmarking {provider}...")
            
            try:
                res = self._benchmark_provider(provider, warmup, iterations)
                results.append(res)
            except Exception as e:
                logger.exception(f"Failed to benchmark {provider}")
                results.append(ProviderBenchmarkResult(
                    provider=provider,
                    available=True,
                    mean_inference_ms=0.0,
                    p50_inference_ms=0.0,
                    p95_inference_ms=0.0,
                    effective_fps=0.0,
                    successful_iterations=0,
                    failed_iterations=iterations,
                    error=str(e)
                ))
                
        # Restore state
        self.provider_manager.active_provider = original_active
        
        # Calculate recommended provider
        recommended = self._calculate_recommended(results)
        
        return ProviderBenchmarkComparison(
            success=True,
            results=results,
            recommended_provider=recommended
        )

    def _benchmark_provider(self, provider: str, warmup: int, iterations: int) -> ProviderBenchmarkResult:
        # Load a temporary session for this provider
        _, desc = self.detection_service.runtime_manager.get_active_runtime()
        if not desc:
            raise AppError(code="MODEL_NOT_ACTIVE", message="Cannot benchmark without an active model", status_code=400)
            
        try:
            session_data = self.session_factory.create(desc.model_path, override_providers=[provider])
            temp_session = session_data["session"]
            
            # Check if it silently fell back to CPU
            actual = session_data["actual_providers"]
            if provider != "CPUExecutionProvider" and ("CPUExecutionProvider" in actual and provider not in actual):
                raise Exception(f"Provider {provider} silently fell back to CPU")
                
        except Exception as e:
            raise Exception(f"Session initialization failed: {e}")
            
        # Swap session temporarily in runtime_manager
        original_session, _ = self.detection_service.runtime_manager.get_active_runtime()
        self.detection_service.runtime_manager.active_session = temp_session
        
        # Warmup
        for _ in range(warmup):
            while not self.camera_manager.get_latest_frame():
                time.sleep(0.01)
            try:
                self.detection_service.detect_current_frame()
            except Exception:
                pass
                
        # Measure
        successful = 0
        failed = 0
        inf_times = []
        
        t_start = time.perf_counter()
        
        for _ in range(iterations):
            while not self.camera_manager.get_latest_frame():
                time.sleep(0.01)
            try:
                resp = self.detection_service.detect_current_frame()
                inf_times.append(resp.inference_time_ms)
                successful += 1
            except Exception:
                failed += 1
                
        t_end = time.perf_counter()
        
        # Restore session
        self.detection_service.runtime_manager.active_session = original_session
        
        dt = t_end - t_start
        fps = successful / dt if dt > 0 else 0.0
        
        arr = np.array(inf_times) if inf_times else np.array([0.0])
        
        return ProviderBenchmarkResult(
            provider=provider,
            available=True,
            mean_inference_ms=float(np.mean(arr)),
            p50_inference_ms=float(np.percentile(arr, 50)),
            p95_inference_ms=float(np.percentile(arr, 95)),
            effective_fps=float(fps),
            successful_iterations=successful,
            failed_iterations=failed,
            error=None
        )

    def _calculate_recommended(self, results: list[ProviderBenchmarkResult]) -> str | None:
        best_score = float('-inf')
        best_provider = None
        
        for res in results:
            if res.failed_iterations > 0 or res.error:
                continue
                
            # Score formula: 
            # 1000 / mean_ms  (Speed)
            # - (p95 - mean) (Stability penalty)
            # - (failed * 100) (Failure penalty)
            
            speed = 1000.0 / max(res.mean_inference_ms, 1.0)
            stability_penalty = max(0, res.p95_inference_ms - res.mean_inference_ms) * 0.1
            
            score = speed - stability_penalty
            
            if score > best_score:
                best_score = score
                best_provider = res.provider
                
        # Fallback
        if not best_provider:
            for res in results:
                if res.provider == "CPUExecutionProvider":
                    return "CPUExecutionProvider"
                    
        return best_provider
