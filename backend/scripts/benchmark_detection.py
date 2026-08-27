"""CLI Benchmark Tool for UNO Vision Runtime."""

import argparse
import json
import urllib.request
import urllib.error
import sys

def main():
    parser = argparse.ArgumentParser(description="UNO Vision Runtime Benchmark Tool")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Base URL of the runtime")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warm-up iterations")
    parser.add_argument("--detailed", action="store_true", help="Include detailed profiling")
    parser.add_argument("--json-output", type=str, help="Path to write JSON results")
    
    args = parser.parse_args()
    
    endpoint = f"{args.url}/api/v1/benchmark/run"
    payload = {
        "iterations": args.iterations,
        "warmup_iterations": args.warmup,
        "include_detailed_profiling": args.detailed
    }
    
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Running benchmark on {args.url} (Warmup: {args.warmup}, Iterations: {args.iterations})...")
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"Error: Server returned {e.code}")
        body = e.read().decode('utf-8')
        print(body)
        sys.exit(1)
    except Exception as e:
        print(f"Error connecting to {args.url}: {e}")
        sys.exit(1)
        
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Wrote raw JSON to {args.json_output}\n")
        
    print("=================================================")
    print("UNO VISION RUNTIME BENCHMARK")
    print("=================================================")
    print()
    print("Model:")
    print(f"  {data.get('model_name', data.get('model_id', 'Unknown'))}")
    
    shape = data.get('input_shape')
    if shape:
        print(f"  Shape: {shape}")
    print()
    print("Iterations:")
    print(f"  Warmup:     {args.warmup}")
    print(f"  Measured:   {data['iterations']}")
    print(f"  Success:    {data['successful_iterations']}")
    print(f"  Failed:     {data['failed_iterations']}")
    print()
    print("TOTAL LATENCY")
    print("---------------------------------")
    t = data['total_ms']
    print(f"Mean:         {t['mean']:.2f} ms")
    print(f"P50:          {t['p50']:.2f} ms")
    print(f"P95:          {t['p95']:.2f} ms")
    print(f"P99:          {t['p99']:.2f} ms")
    print()
    print("PIPELINE")
    print("---------------------------------")
    print(f"Capture:      {data['capture_ms']['mean']:.2f} ms")
    print(f"Preprocess:   {data['preprocessing_ms']['mean']:.2f} ms")
    print(f"Inference:    {data['inference_ms']['mean']:.2f} ms")
    print(f"Postprocess:  {data['postprocessing_ms']['mean']:.2f} ms")
    print()
    print("PERFORMANCE")
    print("---------------------------------")
    print(f"Effective FPS:{data['effective_fps']:.2f}")
    print("=================================================")

if __name__ == "__main__":
    main()
