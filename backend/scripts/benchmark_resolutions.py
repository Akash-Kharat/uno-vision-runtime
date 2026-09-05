import onnxruntime as ort
import time
import numpy as np
import os
import psutil

def benchmark(model_path, res):
    print(f"\n--- Benchmarking {model_path} ({res}x{res}) ---")
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    # dummy input
    dummy_input = np.random.randn(1, 3, res, res).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy_input})
        
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        
    times = np.array(times)
    mean = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)
    fps = 1000.0 / mean
    
    print(f"Mean: {mean:.2f} ms")
    print(f"P50:  {p50:.2f} ms")
    print(f"P95:  {p95:.2f} ms")
    print(f"P99:  {p99:.2f} ms")
    print(f"FPS:  {fps:.2f}")
    
    # Check CPU
    cpu_percent = psutil.cpu_percent()
    try:
        freq = psutil.cpu_freq().current
    except:
        freq = 0
    try:
        temps = psutil.sensors_temperatures()
        temp = list(temps.values())[0][0].current if temps else 0
    except:
        temp = 0
        
    print(f"CPU Util: {cpu_percent}% | Freq: {freq} MHz | Temp: {temp} C")

def main():
    print("==================================================")
    print("RESOLUTION BENCHMARK")
    print("==================================================")
    
    resolutions = [640, 512, 416, 384, 320]
    for res in resolutions:
        path = f"data/models/experiments/resolutions/yolov8n_{res}.onnx"
        if os.path.exists(path):
            benchmark(path, res)
        else:
            print(f"File not found: {path}")
            
    print("\n==================================================")
    print("YOLO11 BENCHMARK")
    print("==================================================")
    
    y11_resolutions = [640, 416]
    for res in y11_resolutions:
        path = f"data/models/experiments/yolo11/yolo11n_{res}.onnx"
        if os.path.exists(path):
            benchmark(path, res)
        else:
            print(f"File not found: {path}")

if __name__ == "__main__":
    main()
