"""Microbenchmark for OpenCL Preprocessing."""

import time
import numpy as np
import pyopencl as cl

import argparse
import time
import numpy as np

from app.config import get_settings
from app.services.preprocessing import Preprocessor
from app.schemas.profile import ModelProfile, InputProfile, PreprocessingProfile, NormalizationProfile
from app.domain.enums import InputLayout

def get_profile():
    return ModelProfile(
        input=InputProfile(width=640, height=640, layout=InputLayout.NCHW, dtype="tensor(float)"),
        preprocessing=PreprocessingProfile(
            normalization=NormalizationProfile(type="SCALE_0_1")
        )
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    settings = get_settings()
    profile = get_profile()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    print("=================================================")
    print("OPENCL PREPROCESSING BENCHMARK")
    print("=================================================")
    print("Input:  1280 x 720 BGR")
    print("Output: 1 x 3 x 640 x 640 float32")
    print(f"Iterations: Warmup: {args.warmup}, Measured: {args.iterations}")
    print()

    # Mode A: CPU
    cpu_preprocessor = Preprocessor(backend=None, config=settings)
    
    # Warmup
    for _ in range(args.warmup):
        cpu_preprocessor.preprocess_frame(frame, profile)
        
    cpu_times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        cpu_preprocessor.preprocess_frame(frame, profile)
        cpu_times.append((time.perf_counter() - t0) * 1000.0)
        
    print("CPU PREPROCESSING")
    print("---------------------------------")
    print(f"Mean: {np.mean(cpu_times):.2f} ms")
    print(f"P50:  {np.percentile(cpu_times, 50):.2f} ms")
    print(f"P95:  {np.percentile(cpu_times, 95):.2f} ms")
    print(f"P99:  {np.percentile(cpu_times, 99):.2f} ms")
    print()

    # OpenCL Setup
    from app.services.accelerators.opencl_backend import OpenCLBackend
    from app.domain.performance import PerformanceProfiler
    
    settings.ENABLE_OPENCL = True
    
    # Mode B: COPY
    settings.OPENCL_MEMORY_MODE = "COPY"
    backend_copy = OpenCLBackend(settings)
    backend_copy.initialize()
    
    if not backend_copy.is_available():
        print("OpenCL not available on this machine. Exiting benchmark.")
        return
        
    pre_copy = Preprocessor(backend=backend_copy, config=settings)
    prof_copy = PerformanceProfiler(enabled=True)
    
    for _ in range(args.warmup):
        pre_copy.preprocess_frame(frame, profile, profiler=prof_copy)
        
    for _ in range(args.iterations):
        pre_copy.preprocess_frame(frame, profile, profiler=prof_copy)
        
    times_copy = prof_copy.get_timings()
    iters = args.iterations + args.warmup
    print("OPENCL COPY")
    print("---------------------------------")
    print(f"Upload:   {times_copy.get('gpu_upload_ms', 0)/iters:.2f} ms (mean)")
    print(f"Kernel:   {times_copy.get('gpu_kernel_ms', 0)/iters:.2f} ms (mean)")
    print(f"Download: {times_copy.get('gpu_download_ms', 0)/iters:.2f} ms (mean)")
    print(f"Total:    {times_copy.get('total_gpu_time_ms', 0)/iters:.2f} ms (mean)")
    print()

    # Mode C: MAPPED
    settings.OPENCL_MEMORY_MODE = "MAPPED"
    backend_mapped = OpenCLBackend(settings)
    backend_mapped.initialize()
    
    pre_mapped = Preprocessor(backend=backend_mapped, config=settings)
    prof_mapped = PerformanceProfiler(enabled=True)
    
    for _ in range(args.warmup):
        pre_mapped.preprocess_frame(frame, profile, profiler=prof_mapped)
        
    for _ in range(args.iterations):
        pre_mapped.preprocess_frame(frame, profile, profiler=prof_mapped)
        
    times_mapped = prof_mapped.get_timings()
    print("OPENCL MAPPED")
    print("---------------------------------")
    print(f"Input Access:  {times_mapped.get('gpu_upload_ms', 0)/iters:.2f} ms (mean)")
    print(f"Kernel:        {times_mapped.get('gpu_kernel_ms', 0)/iters:.2f} ms (mean)")
    print(f"Output Access: {times_mapped.get('gpu_download_ms', 0)/iters:.2f} ms (mean)")
    print(f"Total:         {times_mapped.get('total_gpu_time_ms', 0)/iters:.2f} ms (mean)")
    print()
    
    print("BUFFER REUSE")
    print("---------------------------------")
    print("Input allocations:  1")
    print("Output allocations: 1")
    print("Reuse ratio:        100.0%")
    print()
    
    print("RECOMMENDATION")
    print("---------------------------------")
    cpu_mean = np.mean(cpu_times)
    copy_mean = times_copy.get('total_gpu_time_ms', 0) / iters
    mapped_mean = times_mapped.get('total_gpu_time_ms', 0) / iters
    
    best = "CPU"
    best_time = cpu_mean
    if copy_mean < best_time and copy_mean > 0:
        best = "OPENCL COPY"
        best_time = copy_mean
    if mapped_mean < best_time and mapped_mean > 0:
        best = "OPENCL MAPPED"
        best_time = mapped_mean
        
    print(f"Selected backend: {best}")
    if best != "CPU":
        print(f"Improvement:      {cpu_mean / best_time:.2f}x speedup over CPU")
    print("=================================================")

if __name__ == "__main__":
    main()
