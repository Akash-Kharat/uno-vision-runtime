import sys
import time
try:
    import psutil
except ImportError:
    psutil = None
from pathlib import Path
import numpy as np
import onnxruntime as ort

def get_cpu_frequencies():
    freqs = []
    try:
        if psutil:
            for freq in psutil.cpu_freq(percpu=True):
                freqs.append(freq.current)
    except Exception:
        pass
    
    # Fallback to sysfs if psutil fails or missing
    if not freqs:
        for i in range(16): # Check up to 16 cores
            path = Path(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq")
            if path.exists():
                try:
                    val = int(path.read_text().strip()) / 1000.0 # MHz
                    freqs.append(val)
                except Exception:
                    pass
    return freqs

def get_thermal_info():
    zones = {}
    try:
        thermal_dir = Path("/sys/class/thermal")
        if thermal_dir.exists():
            for zone in thermal_dir.glob("thermal_zone*"):
                try:
                    temp = int((zone / "temp").read_text().strip()) / 1000.0
                    type_name = (zone / "type").read_text().strip()
                    zones[type_name] = temp
                except Exception:
                    pass
    except Exception:
        pass
    return zones

def get_system_load():
    try:
        with open("/proc/loadavg", "r") as f:
            return f.read().strip()
    except Exception:
        return "N/A"

def run_benchmark(session, input_name: str, input_tensor: np.ndarray, warmup: int = 20, iterations: int = 100):
    for _ in range(warmup):
        session.run(None, {input_name: input_tensor})
        
    latencies = []
    
    start_load = get_system_load()
    start_freq = get_cpu_frequencies()
    start_temp = get_thermal_info()
    
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: input_tensor})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
    end_load = get_system_load()
    end_freq = get_cpu_frequencies()
    end_temp = get_thermal_info()
    
    latencies = np.array(latencies)
    mean = np.mean(latencies)
    
    return {
        "mean": mean,
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "fps": 1000.0 / mean if mean > 0 else 0,
        "start_load": start_load,
        "end_load": end_load,
        "start_freq": start_freq,
        "end_freq": end_freq,
        "start_temp": start_temp,
        "end_temp": end_temp
    }

def print_stats(name: str, stats: dict):
    print(f"\n==================================================")
    print(f"MODEL: {name}")
    print(f"==================================================")
    print(f"Latency Mean: {stats['mean']:.2f} ms")
    print(f"Latency P50:  {stats['p50']:.2f} ms")
    print(f"Latency P95:  {stats['p95']:.2f} ms")
    print(f"Latency P99:  {stats['p99']:.2f} ms")
    print(f"Latency Min:  {stats['min']:.2f} ms")
    print(f"Latency Max:  {stats['max']:.2f} ms")
    print(f"Effective FPS: {stats['fps']:.2f}")
    
    print("\n-- Resources --")
    print(f"Load Avg: {stats['start_load']} -> {stats['end_load']}")
    if stats['start_freq'] and stats['end_freq']:
        print(f"Freq (Core 0): {stats['start_freq'][0]:.1f} MHz -> {stats['end_freq'][0]:.1f} MHz")
    if stats['start_temp'] and stats['end_temp']:
        start_c = stats['start_temp'].get('cpuss0-thermal', 0)
        end_c = stats['end_temp'].get('cpuss0-thermal', 0)
        print(f"Temp (cpuss0): {start_c:.1f} °C -> {end_c:.1f} °C")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", default="data/models/models/mdl_0f92eb15da2d/model.onnx")
    parser.add_argument("--int8", default="data/models/experiments/yolov8n_int8.onnx")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 2
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena = True
    opts.enable_mem_pattern = True
    
    # 1. Load FP32
    print(f"Loading FP32: {args.fp32}")
    sess_fp32 = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"], sess_options=opts)
    input_info = sess_fp32.get_inputs()[0]
    shape = input_info.shape
    shape = [1 if isinstance(d, str) or d is None else d for d in shape]
    dummy_input = np.random.randn(*shape).astype(np.float32)
    
    # Run FP32
    fp32_stats = run_benchmark(sess_fp32, input_info.name, dummy_input, args.warmup, args.iterations)
    print_stats("FP32 Model", fp32_stats)
    
    # Cooldown
    print("\nCooling down for 10 seconds...")
    time.sleep(10)
    
    # 2. Load INT8
    print(f"\nLoading INT8: {args.int8}")
    sess_int8 = ort.InferenceSession(args.int8, providers=["CPUExecutionProvider"], sess_options=opts)
    # Run INT8
    int8_stats = run_benchmark(sess_int8, input_info.name, dummy_input, args.warmup, args.iterations)
    print_stats("INT8 Model", int8_stats)
    
    print("\nModel       Mean    P50    P95    P99    FPS")
    print("-" * 50)
    print(f"FP32        {fp32_stats['mean']:<7.2f} {fp32_stats['p50']:<6.2f} {fp32_stats['p95']:<6.2f} {fp32_stats['p99']:<6.2f} {fp32_stats['fps']:<6.2f}")
    print(f"INT8        {int8_stats['mean']:<7.2f} {int8_stats['p50']:<6.2f} {int8_stats['p95']:<6.2f} {int8_stats['p99']:<6.2f} {int8_stats['fps']:<6.2f}")

if __name__ == "__main__":
    main()
