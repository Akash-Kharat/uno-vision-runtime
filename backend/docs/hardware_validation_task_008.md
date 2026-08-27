# UNO Vision Runtime — TASK 008 Hardware Validation

## Validation Procedure
After compiling optimizations, use `scripts/benchmark_detection.py` on the Arduino UNO Q.

Stop background inference before proceeding:
```bash
curl -X POST http://localhost:8000/api/v1/runtime/stop
```

## Levels

### Level 1 — Functional Performance
Quick validation of pipeline optimizations.
```bash
python scripts/benchmark_detection.py --warmup 10 --iterations 50
```

### Level 2 — Standard Benchmark
Detailed performance and standard configuration baseline comparing new YOLO post-processing against earlier latency stats (targeting <100ms Post-Processing).
```bash
python scripts/benchmark_detection.py --warmup 20 --iterations 100 --detailed
```

### Level 3 — Sustained Hardware Benchmark
Thermal and sustained performance validation verifying no significant P50 / P95 drift over an extended time interval.
```bash
python scripts/benchmark_detection.py --warmup 20 --iterations 500
```
> **Note:** Record the Initial and Final P50 metrics to monitor thermal throttling on the Linux ARM cores.
