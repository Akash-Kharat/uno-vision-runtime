#!/usr/bin/env python3
"""
Registers the promoted YOLO11n@416 model into the production registry
and launches a 30-minute sustained validation test via the HTTP API.
Run this on the UNO Q (via ssh or nohup).
"""
import subprocess
import sys
import os
import time
import requests
import json
import psutil

API = "http://127.0.0.1:8000"
MODEL_ONNX = "data/models/experiments/yolo11/yolo11n_416.onnx"
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def wait_for_server(timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{API}/", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def register_model():
    """Upload yolo11n_416.onnx, set profile, activate."""
    print("==> Uploading model...")
    with open(MODEL_ONNX, "rb") as f:
        r = requests.post(f"{API}/api/v1/models/upload", files={"file": ("yolo11n_416.onnx", f, "application/octet-stream")})
    if r.status_code != 200:
        print("Upload failed:", r.text)
        return None
    model_id = r.json()["metadata"]["id"]
    print(f"  Model ID: {model_id}")

    print("==> Setting profile...")
    profile = {
        "task": "OBJECT_DETECTION",
        "input": {"layout": "NCHW", "color_format": "RGB"},
        "preprocessing": {
            "resize": "LETTERBOX",
            "normalization": {"type": "SCALE_0_1", "scale": 0.00392156862745098}
        },
        "output": {
            "processor": "YOLO",
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45
        },
        "classes": COCO80
    }
    r = requests.put(f"{API}/api/v1/models/{model_id}/profile", json=profile)
    if r.status_code != 200:
        print("Profile failed:", r.text)
        return None

    print("==> Activating model...")
    r = requests.post(f"{API}/api/v1/models/{model_id}/activate")
    if r.status_code != 200:
        print("Activate failed:", r.text)
        return None
    print(f"  Model {model_id} activated.")
    return model_id

def run_sustained_test(duration_secs=1800):
    """Start camera + inference, then poll status for the test duration."""
    print("\n==> Starting camera...")
    r = requests.post(f"{API}/api/v1/camera/start")
    print("  Camera:", r.json().get("success", r.text))

    print("==> Starting inference scheduler...")
    r = requests.post(f"{API}/api/v1/runtime/start")
    print("  Runtime:", r.json().get("success", r.text))

    print(f"\n==> Sustained test for {duration_secs}s...\n")
    t_start = time.time()
    results = []

    try:
        while time.time() - t_start < duration_secs:
            time.sleep(10)
            elapsed = int(time.time() - t_start)

            r = requests.get(f"{API}/api/v1/runtime/status", timeout=5)
            if r.status_code != 200:
                print(f"  [{elapsed}s] Status error: {r.status_code}")
                continue

            data = r.json()
            stats = data.get("runtime", {}).get("stats", {})

            eff_fps = stats.get("effective_fps", 0.0)
            inferred = stats.get("frames_inferred", 0)
            skipped = stats.get("frames_skipped", 0)
            captured = stats.get("frames_captured", 0)
            dropped = stats.get("dropped_frames", 0)
            age_ms = stats.get("latest_detection_age_ms")
            busy = stats.get("inference_busy", False)
            avg_inf_ms = stats.get("avg_inference_ms", 0.0)
            p95_total_ms = stats.get("p95_total_ms", 0.0)

            cpu = psutil.cpu_percent(interval=0.5)
            try:
                temps_d = psutil.sensors_temperatures()
                temp = list(temps_d.values())[0][0].current if temps_d else 0.0
            except:
                temp = 0.0

            age_str = f"{age_ms:.0f}ms" if age_ms is not None else "N/A"
            row = {
                "elapsed_s": elapsed, "eff_fps": round(eff_fps, 2),
                "captured": captured, "inferred": inferred, "skipped": skipped, "dropped": dropped,
                "avg_inf_ms": round(avg_inf_ms, 1), "p95_total_ms": round(p95_total_ms, 1),
                "age_ms": round(age_ms, 0) if age_ms else None,
                "cpu": cpu, "temp_c": temp, "busy": busy
            }
            results.append(row)

            print(f"  [{elapsed:5d}s] fps={eff_fps:.2f} inf={inferred} skip={skipped} "
                  f"avg_inf={avg_inf_ms:.0f}ms age={age_str} CPU={cpu:.1f}% T={temp:.1f}°C")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print("\n==> Stopping runtime and camera...")
    requests.post(f"{API}/api/v1/runtime/stop")
    requests.post(f"{API}/api/v1/camera/stop")

    # Summary
    if results:
        import statistics
        fps_vals = [r["eff_fps"] for r in results if r["eff_fps"] > 0]
        cpu_vals = [r["cpu"] for r in results]
        temp_vals = [r["temp_c"] for r in results]
        inf_vals = [r["avg_inf_ms"] for r in results if r["avg_inf_ms"] > 0]

        print("\n" + "="*60)
        print("SUSTAINED TEST SUMMARY")
        print("="*60)
        print(f"Duration:          {results[-1]['elapsed_s']}s")
        print(f"Total inferred:    {results[-1]['inferred']}")
        print(f"Total skipped:     {results[-1]['skipped']}")
        print(f"Mean actual FPS:   {statistics.mean(fps_vals):.2f}")
        print(f"Mean CPU:          {statistics.mean(cpu_vals):.1f}%")
        print(f"Max  CPU:          {max(cpu_vals):.1f}%")
        print(f"Mean Temp:         {statistics.mean(temp_vals):.1f}°C")
        print(f"Max  Temp:         {max(temp_vals):.1f}°C")
        print(f"Mean Inf ms:       {statistics.mean(inf_vals):.1f}ms")
        print("="*60)
    
    # Save JSON results
    with open("/tmp/sustained_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to /tmp/sustained_results.json")


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 1800

    print(f"UNO Q Production Validation — {duration}s test")
    print("Waiting for API server...")

    if not wait_for_server(timeout=60):
        print("ERROR: API not responding. Start uvicorn first.")
        sys.exit(1)

    print("API is up.")
    model_id = register_model()
    if not model_id:
        print("ERROR: Model registration failed.")
        sys.exit(1)

    run_sustained_test(duration_secs=duration)
    print("\nDONE.")
