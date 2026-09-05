import pyopencl as cl
import numpy as np
import time

def main():
    print("==================================================")
    print("OPENCL MATMUL (1x1 CONV PROXY) BENCHMARK")
    print("==================================================")
    
    platforms = cl.get_platforms()
    device = None
    for p in platforms:
        for d in p.get_devices():
            if "Adreno" in d.name or "FD" in d.name or d.type == cl.device_type.GPU:
                device = d
    
    if not device:
        print("No GPU found.")
        return
        
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    # 1x1 Conv shape proxy
    # M = H*W, K = in_channels, N = out_channels
    M = 256
    K = 128
    N = 128
    
    print(f"Matrix Multiply: [{M}x{K}] @ [{K}x{N}] -> [{M}x{N}]")
    
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    
    # 1. CPU FP32
    print("\n--- CPU FP32 (NumPy/BLAS) ---")
    # Warmup
    for _ in range(10):
        c_np = a_np @ b_np
    
    t0 = time.time()
    for _ in range(100):
        c_np = a_np @ b_np
    t1 = time.time()
    cpu_ms = (t1 - t0) * 1000 / 100.0
    print(f"CPU FP32: {cpu_ms:.3f} ms")
    
    # 2. OpenCL Naive FP32
    fp32_code = """
    __kernel void matmul_fp32(__global const float *A, __global const float *B, __global float *C, int M, int K, int N) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    """
    
    fp16_code = """
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void matmul_fp16(__global const half *A, __global const half *B, __global half *C, int M, int K, int N) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < M && col < N) {
            half sum = 0.0h;
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    """
    
    prg_fp32 = cl.Program(ctx, fp32_code).build()
    prg_fp16 = cl.Program(ctx, fp16_code).build()
    
    mf = cl.mem_flags
    
    print("\n--- GPU FP32 (Naive OpenCL) ---")
    buf_a32 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    buf_b32 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    buf_c32 = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.shape[0] * b_np.shape[1] * 4)
    
    kernel_fp32 = cl.Kernel(prg_fp32, "matmul_fp32")
    kernel_fp32.set_arg(0, buf_a32)
    kernel_fp32.set_arg(1, buf_b32)
    kernel_fp32.set_arg(2, buf_c32)
    kernel_fp32.set_arg(3, np.int32(M))
    kernel_fp32.set_arg(4, np.int32(K))
    kernel_fp32.set_arg(5, np.int32(N))
    
    global_size = (M, N)
    
    c_32 = np.empty((M, N), dtype=np.float32)
    
    for _ in range(10):
        cl.enqueue_nd_range_kernel(queue, kernel_fp32, global_size, None)
        cl.enqueue_copy(queue, c_32, buf_c32)
    queue.finish()
    
    t0 = time.time()
    for _ in range(100):
        cl.enqueue_nd_range_kernel(queue, kernel_fp32, global_size, None)
    cl.enqueue_copy(queue, c_32, buf_c32)
    queue.finish()
    t1 = time.time()
    gpu32_ms = (t1 - t0) * 1000 / 100.0
    print(f"GPU FP32 (Kernel + queue overhead): {gpu32_ms:.3f} ms")
    
    print("\n--- GPU FP16 (Naive OpenCL) ---")
    a_16 = a_np.astype(np.float16)
    b_16 = b_np.astype(np.float16)
    buf_a16 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_16)
    buf_b16 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_16)
    buf_c16 = cl.Buffer(ctx, mf.WRITE_ONLY, a_16.shape[0] * b_16.shape[1] * 2)
    
    kernel_fp16 = cl.Kernel(prg_fp16, "matmul_fp16")
    kernel_fp16.set_arg(0, buf_a16)
    kernel_fp16.set_arg(1, buf_b16)
    kernel_fp16.set_arg(2, buf_c16)
    kernel_fp16.set_arg(3, np.int32(M))
    kernel_fp16.set_arg(4, np.int32(K))
    kernel_fp16.set_arg(5, np.int32(N))
    
    c_16 = np.empty((M, N), dtype=np.float16)
    
    for _ in range(10):
        cl.enqueue_nd_range_kernel(queue, kernel_fp16, global_size, None)
        cl.enqueue_copy(queue, c_16, buf_c16)
    queue.finish()
    
    t0 = time.time()
    for _ in range(100):
        cl.enqueue_nd_range_kernel(queue, kernel_fp16, global_size, None)
    cl.enqueue_copy(queue, c_16, buf_c16)
    queue.finish()
    t1 = time.time()
    gpu16_ms = (t1 - t0) * 1000 / 100.0
    print(f"GPU FP16 (Kernel + queue overhead): {gpu16_ms:.3f} ms")
    print(f"FP16 vs FP32 Speedup: {gpu32_ms/gpu16_ms:.2f}x")
    
    # Also test transfer times (one by one to avoid OUT_OF_RESOURCES)
    print("\n--- Transfer Overhead ---")
    t0 = time.time()
    for _ in range(100):
        cl.enqueue_copy(queue, buf_a16, a_16)
        queue.finish()
    t1 = time.time()
    print(f"H2D Transfer (FP16): {(t1 - t0) * 1000 / 100.0:.3f} ms")
    
    t0 = time.time()
    for _ in range(100):
        cl.enqueue_copy(queue, c_16, buf_c16)
        queue.finish()
    t1 = time.time()
    print(f"D2H Transfer (FP16): {(t1 - t0) * 1000 / 100.0:.3f} ms")

if __name__ == "__main__":
    main()
