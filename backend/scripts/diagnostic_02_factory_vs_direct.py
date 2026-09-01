import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.services.execution_provider_manager import ExecutionProviderManager
from app.services.onnx_session_factory import ONNXSessionFactory

def run_benchmark(session, input_name: str, input_tensor: np.ndarray, warmup: int = 20, iterations: int = 100) -> dict:
    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: input_tensor})
        
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: input_tensor})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
    latencies = np.array(latencies)
    mean = np.mean(latencies)
    
    return {
        "mean": mean,
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "fps": 1000.0 / mean if mean > 0 else 0
    }

def main():
    print("==================================================")
    print("DIAGNOSTIC 02: FACTORY VS DIRECT ORT")
    print("==================================================")
    
    settings = get_settings()
    model_path = Path(settings.MODEL_STORAGE_PATH) / "models" / "mdl_0f92eb15da2d" / "model.onnx"
    if not model_path.exists():
        model_path = Path(__file__).resolve().parent.parent / "data" / "models" / "yolov8n.onnx"
        
    print(f"Target Model: {model_path}")
    
    # 1. Direct ORT Session
    print("\nInitializing Direct ORT Session...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 2
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena = True
    opts.enable_mem_pattern = True
    
    sess_direct = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"], sess_options=opts)
    
    # 2. Factory Session
    print("Initializing Factory ORT Session...")
    ep_manager = ExecutionProviderManager()
    factory = ONNXSessionFactory(ep_manager, settings)
    res_factory = factory.create(model_path, override_providers=["CPUExecutionProvider"])
    sess_factory = res_factory["session"]
    
    # Generate Dummy Input
    input_info = sess_direct.get_inputs()[0]
    input_name = input_info.name
    shape = input_info.shape
    shape = [1 if isinstance(d, str) or d is None else d for d in shape]
    print(f"Input Shape: {shape} | Name: {input_name}")
    
    dummy_input = np.random.randn(*shape).astype(np.float32)
    
    print("\nRunning Benchmarks (Warmup: 20, Iterations: 100)...")
    direct_stats = run_benchmark(sess_direct, input_name, dummy_input)
    factory_stats = run_benchmark(sess_factory, input_name, dummy_input)
    
    print("\nTest                         Mean     P50     P95     P99")
    print("-" * 65)
    print(f"Direct ORT                   {direct_stats['mean']:<8.2f} {direct_stats['p50']:<7.2f} {direct_stats['p95']:<7.2f} {direct_stats['p99']:<7.2f}")
    print(f"Production Factory           {factory_stats['mean']:<8.2f} {factory_stats['p50']:<7.2f} {factory_stats['p95']:<7.2f} {factory_stats['p99']:<7.2f}")

if __name__ == "__main__":
    main()
