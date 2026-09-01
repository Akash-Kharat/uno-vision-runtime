import sys
import time
import threading
from pathlib import Path
import numpy as np
import onnxruntime as ort

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.services.accelerators.opencl_backend import OpenCLBackend

def run_benchmark(session, dummy_input: np.ndarray, warmup: int = 10, iterations: int = 50) -> float:
    # Warmup
    for _ in range(warmup):
        session.run(None, {"images": dummy_input})
        
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {"images": dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
    return np.mean(latencies)

def main():
    print("==================================================")
    print("DIAGNOSTIC 04: OPENCL CONTENTION TEST")
    print("==================================================")
    
    settings = get_settings()
    model_path = Path(settings.MODEL_STORAGE_PATH) / "models" / "mdl_0f92eb15da2d" / "model.onnx"
    if not model_path.exists():
        model_path = Path(__file__).resolve().parent.parent / "data" / "models" / "yolov8n.onnx"
        
    # Setup ORT
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 2
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"], sess_options=opts)
    
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    print("Test A: CPU Inference with NO OpenCL Context")
    time_a = run_benchmark(sess, dummy_input)
    print(f"  Mean latency: {time_a:.2f} ms")
    
    print("\nInitializing OpenCL Backend...")
    try:
        ocl_backend = OpenCLBackend(settings)
        ocl_backend.initialize()
        print("  OpenCL Initialized Successfully.")
    except Exception as e:
        print(f"  Failed to initialize OpenCL: {e}")
        return
        
    print("\nTest B: CPU Inference with IDLE OpenCL Context")
    time_b = run_benchmark(sess, dummy_input)
    print(f"  Mean latency: {time_b:.2f} ms")
    
    print("\nStarting OpenCL Background Workload Thread...")
    stop_ocl = threading.Event()
    
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    def ocl_worker():
        while not stop_ocl.is_set():
            try:
                ocl_backend.preprocess_yolo(dummy_frame)
                # Small sleep to simulate camera fps (~30fps)
                time.sleep(0.033)
            except Exception:
                pass

    t = threading.Thread(target=ocl_worker)
    t.start()
    
    print("\nTest C: CPU Inference WITH ACTIVE OpenCL Preprocessing (~30FPS bg)")
    time_c = run_benchmark(sess, dummy_input)
    print(f"  Mean latency: {time_c:.2f} ms")
    
    stop_ocl.set()
    t.join()
    
    print("\n--- RESULTS ---")
    print(f"A (No OCL):   {time_a:.2f} ms")
    print(f"B (Idle OCL): {time_b:.2f} ms")
    print(f"C (Act OCL):  {time_c:.2f} ms")

if __name__ == "__main__":
    main()
