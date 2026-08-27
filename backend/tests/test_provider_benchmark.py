import pytest
from unittest.mock import MagicMock
from app.services.provider_benchmark_manager import ProviderBenchmarkManager
from app.domain.execution_provider import ProviderBenchmarkResult

def test_benchmark_manager_recommendation():
    manager = ProviderBenchmarkManager(None, None, None, None)
    
    # Test scoring logic
    results = [
        ProviderBenchmarkResult(
            provider="CPUExecutionProvider",
            available=True,
            mean_inference_ms=1180.0,
            p50_inference_ms=1180.0,
            p95_inference_ms=1200.0,
            effective_fps=0.8,
            successful_iterations=10,
            failed_iterations=0,
            error=None
        ),
        ProviderBenchmarkResult(
            provider="HardwareExecutionProvider",
            available=True,
            mean_inference_ms=300.0,
            p50_inference_ms=300.0,
            p95_inference_ms=320.0, # stable
            effective_fps=3.3,
            successful_iterations=10,
            failed_iterations=0,
            error=None
        ),
        ProviderBenchmarkResult(
            provider="UnstableProvider",
            available=True,
            mean_inference_ms=250.0,
            p50_inference_ms=250.0,
            p95_inference_ms=900.0, # very unstable
            effective_fps=4.0,
            successful_iterations=10,
            failed_iterations=0,
            error=None
        )
    ]
    
    recommended = manager._calculate_recommended(results)
    assert recommended == "HardwareExecutionProvider"

def test_benchmark_manager_recommendation_fallback():
    manager = ProviderBenchmarkManager(None, None, None, None)
    
    # Test fallback if all fail
    results = [
        ProviderBenchmarkResult(
            provider="CPUExecutionProvider",
            available=True,
            mean_inference_ms=1180.0,
            p50_inference_ms=1180.0,
            p95_inference_ms=1200.0,
            effective_fps=0.8,
            successful_iterations=10,
            failed_iterations=5, # failures
            error="Failed"
        )
    ]
    
    recommended = manager._calculate_recommended(results)
    assert recommended == "CPUExecutionProvider" # Fallback works
