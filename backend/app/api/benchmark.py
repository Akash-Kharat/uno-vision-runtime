"""Benchmark API router."""

import time
import numpy as np
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.core.exceptions import AppError
from app.domain.performance import PerformanceProfiler

router = APIRouter()

class BenchmarkRequest(BaseModel):
    iterations: int = 100
    warmup_iterations: int = 10
    include_detailed_profiling: bool = True

@router.post("/run")
async def run_benchmark(req: BenchmarkRequest, request: Request):
    camera_manager = request.app.state.camera_manager
    inference_manager = request.app.state.inference_manager
    detection_service = request.app.state.detection_service

    # 1. Reject if background inference is running
    if inference_manager.state in ("RUNNING", "STARTING", "PAUSED"):
        raise AppError(code="BENCHMARK_RUNTIME_CONFLICT", message="Background continuous inference is currently active.", status_code=409)

    # 2. Check camera and model
    if camera_manager.state != "RUNNING":
        raise AppError(code="CAMERA_NOT_RUNNING", message="Camera must be running to benchmark.", status_code=400)
        
    session, desc = detection_service.runtime_manager.get_active_runtime()
    if not session or not desc:
        raise AppError(code="MODEL_NOT_ACTIVE", message="No active model loaded.", status_code=400)

    # 3. Warm-up
    for _ in range(req.warmup_iterations):
        # wait for frame
        while not camera_manager.get_latest_frame():
            time.sleep(0.01)
        try:
            detection_service.detect_current_frame()
        except AppError:
            pass
            
    # 4. Measure
    successful = 0
    failed = 0
    
    cap_times = []
    pre_times = []
    inf_times = []
    post_times = []
    tot_times = []
    
    profiler = PerformanceProfiler(enabled=req.include_detailed_profiling)

    t_start = time.perf_counter()
    
    # Memory tracking
    mem_start = 0.0
    mem_peak = 0.0
    mem_end = 0.0
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss / (1024 * 1024)
        mem_peak = mem_start
    except ImportError:
        process = None
        
    for _ in range(req.iterations):
        while not camera_manager.get_latest_frame():
            time.sleep(0.01)
            
        try:
            resp = detection_service.detect_current_frame(profiler=profiler)
            timings = resp.timings
            if timings:
                cap_times.append(timings.capture_time_ms)
                pre_times.append(timings.preprocessing_time_ms)
                inf_times.append(timings.inference_time_ms)
                post_times.append(timings.postprocessing_time_ms)
                tot_times.append(timings.total_time_ms)
            successful += 1
            
            if process:
                current_mem = process.memory_info().rss / (1024 * 1024)
                if current_mem > mem_peak:
                    mem_peak = current_mem
                    
        except AppError:
            failed += 1
            
    if process:
        mem_end = process.memory_info().rss / (1024 * 1024)
        
    t_end = time.perf_counter()
    dt = t_end - t_start
    fps = successful / dt if dt > 0 else 0
    
    def calc(arr):
        if not arr:
            return None
        npa = np.array(arr)
        return {
            "mean": float(np.mean(npa)),
            "p50": float(np.percentile(npa, 50)),
            "p95": float(np.percentile(npa, 95)),
            "p99": float(np.percentile(npa, 99)),
            "min": float(np.min(npa)),
            "max": float(np.max(npa))
        }

    # Shape
    input_shape = "unknown"
    if session:
        try:
            shape = session.get_inputs()[0].shape
            input_shape = [s if isinstance(s, int) else -1 for s in shape]
        except Exception:
            pass

    response_payload = {
        "success": True,
        "model_id": desc.model_id,
        "model_name": request.app.state.model_registry.get_metadata(desc.model_id).original_filename,
        "input_shape": input_shape,
        "iterations": req.iterations,
        "successful_iterations": successful,
        "failed_iterations": failed,
        "total_ms": calc(tot_times),
        "capture_ms": calc(cap_times),
        "preprocessing_ms": calc(pre_times),
        "inference_ms": calc(inf_times),
        "postprocessing_ms": calc(post_times),
        "effective_fps": float(fps)
    }
    
    if process:
        response_payload["memory"] = {
            "rss_memory_mb_start": mem_start,
            "rss_memory_mb_end": mem_end,
            "rss_memory_mb_peak": mem_peak
        }
        
    return response_payload

class ProviderBenchmarkRequest(BaseModel):
    warmup: int = 5
    iterations: int = 50

@router.post("/providers")
async def benchmark_providers(req: ProviderBenchmarkRequest, request: Request):
    inference_manager = request.app.state.inference_manager
    if inference_manager.state in ("RUNNING", "STARTING", "PAUSED"):
        raise AppError(code="BENCHMARK_RUNTIME_CONFLICT", message="Background continuous inference is currently active.", status_code=409)
        
    camera_manager = request.app.state.camera_manager
    if camera_manager.state != "RUNNING":
        raise AppError(code="CAMERA_NOT_RUNNING", message="Camera must be running to benchmark.", status_code=400)
        
    benchmark_manager = request.app.state.provider_benchmark_manager
    result = benchmark_manager.run_benchmark(req.warmup, req.iterations)
    return result

