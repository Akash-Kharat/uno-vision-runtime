import time
import statistics
import numpy as np
import onnxruntime as ort
import os

model_path = os.path.expanduser("~/uno-vision-runtime/backend/data/models/experiments/yolo11/yolo11n_416.onnx")
print(f"Loading {model_path}...")

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
sess_opts.inter_op_num_threads = 2
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(model_path, sess_opts, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 416, 416).astype(np.float32)

print("Warmup 20 runs...")
for _ in range(20):
    session.run(None, {input_name: dummy_input})

print("Benchmarking 100 runs...")
latencies = []
for _ in range(100):
    t0 = time.time()
    session.run(None, {input_name: dummy_input})
    t1 = time.time()
    latencies.append((t1 - t0) * 1000)

print("Results:")
print(f"Mean: {statistics.mean(latencies):.2f} ms")
print(f"P50:  {np.percentile(latencies, 50):.2f} ms")
print(f"P95:  {np.percentile(latencies, 95):.2f} ms")
print(f"P99:  {np.percentile(latencies, 99):.2f} ms")
