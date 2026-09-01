import sys
import time
import requests
import subprocess
from pathlib import Path

def run_isolated_benchmark():
    script = Path(__file__).resolve().parent / "benchmark_ort_cpu.py"
    cmd = [sys.executable, str(script), "--warmup", "10", "--iterations", "30"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if line.startswith("Mean:"):
            return float(line.split()[1])
    return -1.0

def wait_for_server():
    for _ in range(20):
        try:
            res = requests.get("http://127.0.0.1:8000/api/v1/health")
            if res.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False

def main():
    print("==================================================")
    print("DIAGNOSTIC 05: CAMERA / PIPELINE CONTENTION TEST")
    print("==================================================")
    
    print("\nTest A: Inference with Server STOPPED (Standalone)")
    time_a = run_isolated_benchmark()
    print(f"  Mean latency: {time_a:.2f} ms")
    
    print("\nStarting Uvicorn Server in background...")
    server_cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
    server_proc = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not wait_for_server():
        print("  Failed to start server!")
        server_proc.terminate()
        return
        
    # Ensure camera is stopped
    requests.post("http://127.0.0.1:8000/api/v1/camera/stop")
    
    print("\nTest B: Inference with Server RUNNING (Camera STOPPED)")
    time_b = run_isolated_benchmark()
    print(f"  Mean latency: {time_b:.2f} ms")
    
    # Start Camera
    print("\nStarting Camera via API...")
    res = requests.post("http://127.0.0.1:8000/api/v1/camera/start")
    if res.status_code != 200:
        print(f"  Failed to start camera: {res.text}")
    else:
        time.sleep(2.0) # Let it warm up
        print("Test C: Inference with Server RUNNING (Camera RUNNING background thread)")
        time_c = run_isolated_benchmark()
        print(f"  Mean latency: {time_c:.2f} ms")
        
        print("\nTest D: Production Full Pipeline Benchmark (Via API)")
        # Run production benchmark
        prod_script = Path(__file__).resolve().parent / "benchmark_detection.py"
        cmd = [sys.executable, str(prod_script), "--warmup", "10", "--iterations", "30"]
        prod_res = subprocess.run(cmd, capture_output=True, text=True)
        time_d = -1.0
        for line in prod_res.stdout.splitlines():
            if line.strip().startswith("Mean") and "ms" in line:
                try:
                    time_d = float(line.split()[1])
                except:
                    pass
        print(f"  Production Mean latency: {time_d:.2f} ms")
        
    print("\nShutting down server...")
    server_proc.terminate()
    server_proc.wait()
    
    print("\n--- RESULTS ---")
    print(f"A (Server Stopped):     {time_a:.2f} ms")
    print(f"B (Server Run, Cam Off):{time_b:.2f} ms")
    if res.status_code == 200:
        print(f"C (Server Run, Cam On): {time_c:.2f} ms")
        print(f"D (Full Prod Pipeline): {time_d:.2f} ms")
        
if __name__ == "__main__":
    main()
