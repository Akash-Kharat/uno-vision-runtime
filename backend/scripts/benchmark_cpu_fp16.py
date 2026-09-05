import onnxruntime as ort
import time
import numpy as np
import os

def benchmark(model_path):
    print(f"\nBenchmarking {model_path} on CPU...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # Handle dynamic batch size
    if isinstance(input_shape[0], str):
        input_shape[0] = 1
        
    dtype = np.float16 if "fp16" in model_path else np.float32
    dummy_input = np.random.randn(*input_shape).astype(dtype)
    
    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy_input})
        
    t0 = time.time()
    for _ in range(20):
        session.run(None, {input_name: dummy_input})
    t1 = time.time()
    
    avg_ms = (t1 - t0) * 1000 / 20.0
    print(f"Mean Inference Time: {avg_ms:.2f} ms")
    return avg_ms

def main():
    fp32_path = "data/models/models/mdl_0f92eb15da2d/model.onnx"
    fp16_path = "data/models/experiments/fp16/yolov8n_fp16.onnx"
    
    fp32_time = benchmark(fp32_path)
    try:
        fp16_time = benchmark(fp16_path)
        print(f"\nFP16 is {fp32_time/fp16_time:.2f}x faster than FP32 on CPU")
    except Exception as e:
        print(f"\nFP16 benchmark FAILED: {e}")
        
if __name__ == "__main__":
    main()
