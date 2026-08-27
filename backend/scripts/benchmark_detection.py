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
    def get_val(stats, key):
        if stats is None or key not in stats:
            return "N/A"
        return f"{stats[key]:.2f} ms"

    def print_stage(name, stats):
        print(f"{name}:")
        print(f"  Mean:       {get_val(stats, 'mean')}")
        print(f"  P50:        {get_val(stats, 'p50')}")
        print(f"  P95:        {get_val(stats, 'p95')}")
        print(f"  P99:        {get_val(stats, 'p99')}")

    print("TOTAL LATENCY")
    print("---------------------------------")
    print_stage("Total", data.get("total_ms"))
    print()
    print("PIPELINE")
    print("---------------------------------")
    print_stage("Capture", data.get("capture_ms"))
    print_stage("Preprocess", data.get("preprocessing_ms"))
    print_stage("Inference", data.get("inference_ms"))
    print_stage("Postprocess", data.get("postprocessing_ms"))
    print()
    
    backend = data.get("preprocessing_backend", "CPU")
    if backend == "AUTO" or backend == "OPENCL":
        # Usually it should resolve to OPENCL or CPU in practice, but API might just echo config
        pass
        
    print("PREPROCESSING BACKEND")
    print("---------------------------------")
    print(f"Backend: {backend}")
    if data.get("total_gpu_ms"):
        print_stage("Upload", data.get("gpu_upload_ms"))
        print_stage("Kernel", data.get("gpu_kernel_ms"))
        print_stage("Download", data.get("gpu_download_ms"))
        print_stage("Total GPU", data.get("total_gpu_ms"))
        print()
        print("BUFFER REUSE")
        print("---------------------------------")
        in_ru = data.get("input_buffer_reused", {}).get("mean", 0) * 100
        out_ru = data.get("output_buffer_reused", {}).get("mean", 0) * 100
        print(f"Input Reuse:  {in_ru:.1f}%")
        print(f"Output Reuse: {out_ru:.1f}%")
    else:
        print("Upload:       N/A")
        print("Kernel:       N/A")
        print("Download:     N/A")
        print("Total GPU:    N/A")
    print()
    print("PERFORMANCE")
    print("---------------------------------")
    fps = data.get('effective_fps')
    print(f"Effective FPS:{fps:.2f}" if fps is not None else "Effective FPS: N/A")
    print("=================================================")

if __name__ == "__main__":
    main()
