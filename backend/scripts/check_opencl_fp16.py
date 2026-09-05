import pyopencl as cl
import numpy as np
import time

def main():
    print("==================================================")
    print("OPENCL FP16 & COMPUTE INVESTIGATION")
    print("==================================================")
    
    platforms = cl.get_platforms()
    if not platforms:
        print("No OpenCL platforms found.")
        return
        
    device = None
    for p in platforms:
        print(f"Platform: {p.name}")
        for d in p.get_devices():
            print(f"  Device: {d.name}")
            print(f"  Version: {d.version}")
            exts = d.extensions
            has_fp16 = "cl_khr_fp16" in exts
            print(f"  Supports cl_khr_fp16: {has_fp16}")
            if "Adreno" in d.name or "Turnip" in d.name or "Mesa" in d.name or "Qualcomm" in d.name or "GPU" in d.name or d.type == cl.device_type.GPU:
                device = d
    
    if device is None:
        print("No suitable GPU device found.")
        return
        
    print(f"\nUsing Device: {device.name}")
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    # Let's do a basic element-wise multiplication on a large tensor (like an activation map)
    # Shape: 1 x 64 x 160 x 160 (approx. YOLO C2f feature map size)
    N = 64 * 160 * 160
    
    fp32_a = np.random.randn(N).astype(np.float32)
    fp32_b = np.random.randn(N).astype(np.float32)
    fp16_a = fp32_a.astype(np.float16)
    fp16_b = fp32_b.astype(np.float16)
    
    fp32_code = """
    __kernel void eltwise_mul_fp32(__global const float *a, __global const float *b, __global float *c) {
        int i = get_global_id(0);
        c[i] = a[i] * b[i];
    }
    """
    
    fp16_code = """
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void eltwise_mul_fp16(__global const half *a, __global const half *b, __global half *c) {
        int i = get_global_id(0);
        c[i] = a[i] * b[i];
    }
    """
    
    try:
        prg_fp32 = cl.Program(ctx, fp32_code).build()
    except Exception as e:
        print("Failed to build FP32 program", e)
        return
        
    try:
        prg_fp16 = cl.Program(ctx, fp16_code).build()
    except Exception as e:
        print("Failed to build FP16 program", e)
        prg_fp16 = None
        
    # Buffers
    mf = cl.mem_flags
    
    # Warmup + benchmark FP32
    print("\n--- FP32 Benchmark ---")
    buf_a32 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fp32_a)
    buf_b32 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fp32_b)
    buf_c32 = cl.Buffer(ctx, mf.WRITE_ONLY, fp32_a.nbytes)
    
    # Warmup
    for _ in range(10):
        prg_fp32.eltwise_mul_fp32(queue, fp32_a.shape, None, buf_a32, buf_b32, buf_c32)
    queue.finish()
    
    t0 = time.time()
    for _ in range(100):
        prg_fp32.eltwise_mul_fp32(queue, fp32_a.shape, None, buf_a32, buf_b32, buf_c32)
    queue.finish()
    t1 = time.time()
    fp32_ms = (t1 - t0) * 1000 / 100.0
    print(f"FP32 element-wise time: {fp32_ms:.3f} ms")
    
    if prg_fp16:
        print("\n--- FP16 Benchmark ---")
        buf_a16 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fp16_a)
        buf_b16 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fp16_b)
        buf_c16 = cl.Buffer(ctx, mf.WRITE_ONLY, fp16_a.nbytes)
        
        # Warmup
        for _ in range(10):
            prg_fp16.eltwise_mul_fp16(queue, fp16_a.shape, None, buf_a16, buf_b16, buf_c16)
        queue.finish()
        
        t0 = time.time()
        for _ in range(100):
            prg_fp16.eltwise_mul_fp16(queue, fp16_a.shape, None, buf_a16, buf_b16, buf_c16)
        queue.finish()
        t1 = time.time()
        fp16_ms = (t1 - t0) * 1000 / 100.0
        print(f"FP16 element-wise time: {fp16_ms:.3f} ms")
        print(f"Speedup: {fp32_ms/fp16_ms:.2f}x")

if __name__ == "__main__":
    main()
