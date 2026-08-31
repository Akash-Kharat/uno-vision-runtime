import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.services.execution_provider_manager import ExecutionProviderManager
from app.services.onnx_session_factory import ONNXSessionFactory

def main():
    parser = argparse.ArgumentParser(description="Isolate ONNX CPU Inference")
    parser.add_argument("--model", type=str, default="data/models/yolov8n.onnx")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--intra-op", type=int, default=0)
    parser.add_argument("--inter-op", type=int, default=0)
    parser.add_argument("--execution-mode", type=str, default="SEQUENTIAL")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model {model_path} not found. Please provide a valid model path.")
        return

    # Setup config
    settings = get_settings()
    settings.ORT_INTRA_OP_THREADS = args.intra_op
    settings.ORT_INTER_OP_THREADS = args.inter_op
    settings.ORT_EXECUTION_MODE = args.execution_mode
    
    ep_manager = ExecutionProviderManager()
    ep_manager.set_active_provider("CPUExecutionProvider")
    factory = ONNXSessionFactory(ep_manager, settings)
    
    print("=================================================")
    print("ONNX RUNTIME CPU INFERENCE BENCHMARK")
    print("=================================================")
    print(f"Model:  {model_path.name}")
    print("Input:  [1, 3, 640, 640]")
    print("Provider: CPUExecutionProvider")
    print("Configuration:")
    print(f"  Intra-op threads: {args.intra_op if args.intra_op > 0 else 'ORT Default'}")
    print(f"  Inter-op threads: {args.inter_op if args.inter_op > 0 else 'ORT Default'}")
    print(f"  Execution mode: {args.execution_mode}")
    print(f"  Graph optimization: ALL")
    print("---------------------------------")
    
    try:
        session_data = factory.create(model_path)
        session = session_data["session"]
    except Exception as e:
        print(f"Failed to create session: {e}")
        return
        
    # Prepare dummy input
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(args.warmup):
        session.run(None, {input_name: dummy_input})
        
    times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        
    mean = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000.0 / mean if mean > 0 else 0
    
    print("INFERENCE LATENCY")
    print("---------------------------------")
    print(f"Mean: {mean:.2f} ms")
    print(f"P50:  {p50:.2f} ms")
    print(f"P95:  {p95:.2f} ms")
    print(f"P99:  {p99:.2f} ms")
    print(f"Min:  {min_time:.2f} ms")
    print(f"Max:  {max_time:.2f} ms")
    print()
    print("PERFORMANCE")
    print("---------------------------------")
    print(f"Effective FPS: {fps:.2f}")
    print("=================================================")

if __name__ == "__main__":
    main()
