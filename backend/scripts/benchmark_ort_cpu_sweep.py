import sys
import subprocess
import re
from pathlib import Path

def run_sweep():
    print("=================================================")
    print("ONNX RUNTIME CPU THREAD SWEEP")
    print("=================================================")
    
    script_path = Path(__file__).parent / "benchmark_ort_cpu.py"
    
    intra_configs = [1, 2, 3, 4]
    inter_configs = [1, 2]
    
    results = []
    
    for inter in inter_configs:
        for intra in intra_configs:
            print(f"Testing config: intra={intra}, inter={inter}")
            model_path = Path(__file__).resolve().parent.parent / "data" / "models" / "yolov8n.onnx"
            cmd = [
                sys.executable, str(script_path),
                "--model", str(model_path),
                "--intra-op", str(intra),
                "--inter-op", str(inter),
                "--warmup", "2",
                "--iterations", "5"
            ]
            
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                mean_match = re.search(r"Mean:\s+([\d.]+)", res.stdout)
                p50_match = re.search(r"P50:\s+([\d.]+)", res.stdout)
                p95_match = re.search(r"P95:\s+([\d.]+)", res.stdout)
                p99_match = re.search(r"P99:\s+([\d.]+)", res.stdout)
                fps_match = re.search(r"Effective FPS:\s+([\d.]+)", res.stdout)
                
                if mean_match and p95_match and fps_match:
                    results.append({
                        "intra": intra,
                        "inter": inter,
                        "mean": float(mean_match.group(1)),
                        "p50": float(p50_match.group(1)),
                        "p95": float(p95_match.group(1)),
                        "p99": float(p99_match.group(1)),
                        "fps": float(fps_match.group(1)),
                        "status": "OK"
                    })
                else:
                    print(f"Failed to parse output for config intra={intra}, inter={inter}")
                    print("Output was:")
                    print(res.stdout)
                    results.append({
                        "intra": intra,
                        "inter": inter,
                        "status": "FAILED_PARSE"
                    })
            except subprocess.CalledProcessError as e:
                print(f"Failed to run config intra={intra}, inter={inter}")
                results.append({
                        "intra": intra,
                        "inter": inter,
                        "status": "FAILED_EXEC"
                })

    # Sort results
    valid_results = [r for r in results if r["status"] == "OK"]
    
    # Sort by P95, then by Mean
    valid_results.sort(key=lambda x: (x["p95"], x["mean"]))
    
    print("\n=================================================")
    print("SWEEP LEADERBOARD (Ranked by P95)")
    print("=================================================")
    print(f"{'Intra':<8} {'Inter':<8} {'Mean':<10} {'P50':<10} {'P95':<10} {'P99':<10} {'FPS':<10}")
    print("-" * 75)
    for r in valid_results:
        print(f"{r['intra']:<8} {r['inter']:<8} {r['mean']:<10.2f} {r['p50']:<10.2f} {r['p95']:<10.2f} {r['p99']:<10.2f} {r['fps']:<10.2f}")
    
    if valid_results:
        best = valid_results[0]
        print("\nRECOMMENDED CONFIGURATION:")
        print(f"ORT_INTRA_OP_THREADS = {best['intra']}")
        print(f"ORT_INTER_OP_THREADS = {best['inter']}")
        
if __name__ == "__main__":
    run_sweep()
