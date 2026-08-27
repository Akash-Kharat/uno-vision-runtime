import pytest
import time
from app.domain.performance import PerformanceProfiler

def test_performance_profiler_enabled():
    profiler = PerformanceProfiler(enabled=True)
    
    with profiler.measure("test_stage"):
        time.sleep(0.01)
        
    timings = profiler.get_timings()
    assert "test_stage" in timings
    assert timings["test_stage"] >= 10.0 # ~10ms
    assert profiler.get_timing("test_stage") == timings["test_stage"]
    assert profiler.get_timing("missing") == 0.0

def test_performance_profiler_disabled():
    profiler = PerformanceProfiler(enabled=False)
    
    with profiler.measure("test_stage"):
        time.sleep(0.01)
        
    timings = profiler.get_timings()
    assert "test_stage" not in timings
    assert profiler.get_timing("test_stage") == 0.0

def test_empty_stage_handling():
    profiler = PerformanceProfiler(enabled=True)
    with profiler.measure("empty"):
        pass
    
    timings = profiler.get_timings()
    assert "empty" in timings
    assert timings["empty"] >= 0.0
