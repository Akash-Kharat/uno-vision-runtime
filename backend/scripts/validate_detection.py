#!/usr/bin/env python3
"""Hardware detection validation script."""
import argparse
import requests
import time
import numpy as np
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Validate hardware inference detection timings.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/detect", help="API URL to ping")
    parser.add_argument("--iterations", type=int, default=10, help="Number of inference calls")
    args = parser.parse_args()

    capture_times = []
    pre_times = []
    inf_times = []
    post_times = []
    total_times = []

    print(f"Starting hardware validation against {args.url} for {args.iterations} iterations...")
    
    for i in range(1, args.iterations + 1):
        try:
            resp = requests.post(args.url)
        except Exception as e:
            print(f"Iteration {i} failed: {e}")
            continue

        if resp.status_code != 200:
            print(f"Iteration {i} failed with status {resp.status_code}: {resp.text}")
            continue

        data = resp.json()
        
        objects = data.get("object_count", 0)
        classes = data.get("class_counts", {})
        
        timings = data.get("timings", {})
        c_time = timings.get("capture_time_ms", 0.0)
        p_time = timings.get("preprocessing_time_ms", 0.0)
        i_time = timings.get("inference_time_ms", 0.0)
        po_time = timings.get("postprocessing_time_ms", 0.0)
        t_time = timings.get("total_time_ms", 0.0)

        capture_times.append(c_time)
        pre_times.append(p_time)
        inf_times.append(i_time)
        post_times.append(po_time)
        total_times.append(t_time)

        print(f"\nIteration: {i}")
        print(f"Objects: {objects}")
        print("Classes:")
        for cls, count in classes.items():
            print(f"  {cls}: {count}")

        print(f"\nCapture:        {c_time:.2f} ms")
        print(f"Preprocessing:  {p_time:.2f} ms")
        print(f"Inference:      {i_time:.2f} ms")
        print(f"Postprocessing: {po_time:.2f} ms")
        print(f"Total:          {t_time:.2f} ms")

        # Give a short breather to the target hardware
        time.sleep(0.5)

    if not total_times:
        print("No successful iterations recorded.")
        return

    print("\n" + "="*40)
    print("Aggregate Statistics (ms)")
    print("="*40)

    def print_stats(name, arr):
        arr = np.array(arr)
        print(f"{name}:")
        print(f"  Min: {np.min(arr):.2f}")
        print(f"  Max: {np.max(arr):.2f}")
        print(f"  Avg: {np.mean(arr):.2f}")
        print(f"  P50: {np.percentile(arr, 50):.2f}")
        print(f"  P95: {np.percentile(arr, 95):.2f}")

    print_stats("Capture Time", capture_times)
    print_stats("Preprocessing Time", pre_times)
    print_stats("Inference Time", inf_times)
    print_stats("Postprocessing Time", post_times)
    print_stats("Total Time", total_times)

if __name__ == "__main__":
    main()
