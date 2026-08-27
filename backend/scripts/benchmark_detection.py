#!/usr/bin/env python3
"""
UNO Vision Runtime Benchmark CLI.

Thin client for:
    POST /api/v1/benchmark/run

The benchmark API is the single source of truth. This script only:
- parses CLI arguments
- sends the benchmark request
- validates the response
- formats the benchmark report
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
BENCHMARK_ENDPOINT = "/api/v1/benchmark/run"


def format_ms(value: Any) -> str:
    """Format a millisecond value without converting missing values to zero."""
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2f} ms"
    except (TypeError, ValueError):
        return "N/A"


def format_number(value: Any, decimals: int = 2) -> str:
    """Format a numeric value safely."""
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def get_metric(data: dict[str, Any], section: str, key: str) -> Any:
    """
    Safely retrieve nested benchmark metrics.

    Example:
        get_metric(data, "total_ms", "mean")
    """
    metrics = data.get(section)

    if not isinstance(metrics, dict):
        return None

    return metrics.get(key)


def print_metric_block(
    data: dict[str, Any],
    title: str,
    section: str,
) -> None:
    """Print mean and percentile values for a benchmark stage."""

    print(f"{title}:")

    for label, key in (
        ("Mean", "mean"),
        ("P50", "p50"),
        ("P95", "p95"),
        ("P99", "p99"),
    ):
        value = get_metric(data, section, key)
        print(f"  {label:<11} {format_ms(value)}")


def print_gpu_metrics(data: dict[str, Any]) -> None:
    """Print optional OpenCL GPU timing metrics."""

    backend = data.get("preprocessing_backend", "UNKNOWN")

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


def print_memory_metrics(data: dict[str, Any]) -> None:
    """Print memory metrics when available."""

    memory = data.get("memory")

    if not isinstance(memory, dict):
        return

    print("MEMORY")
    print("---------------------------------")

    print(
        "RSS Start:    "
        f"{format_number(memory.get('rss_memory_mb_start'))} MB"
    )
    print(
        "RSS End:      "
        f"{format_number(memory.get('rss_memory_mb_end'))} MB"
    )
    print(
        "RSS Peak:     "
        f"{format_number(memory.get('rss_memory_mb_peak'))} MB"
    )
    print()


def print_report(data: dict[str, Any], warmup: int) -> None:
    """Render the complete benchmark report."""

    print("=" * 49)
    print("UNO VISION RUNTIME BENCHMARK")
    print("=" * 49)
    print()

    print("Model:")
    print(f"  {data.get('model_name', 'N/A')}")
    print(f"  Shape: {data.get('input_shape', 'N/A')}")
    print()

    print("Iterations:")
    print(f"  Warmup:     {warmup}")
    print(f"  Measured:   {data.get('iterations', 'N/A')}")
    print(f"  Success:    {data.get('successful_iterations', 'N/A')}")
    print(f"  Failed:     {data.get('failed_iterations', 'N/A')}")
    print()

    print("TOTAL LATENCY")
    print("---------------------------------")
    print_metric_block(data, "Total", "total_ms")
    print()

    print("PIPELINE")
    print("---------------------------------")
    print_metric_block(data, "Capture", "capture_ms")
    print()

    print_metric_block(data, "Preprocess", "preprocessing_ms")
    print()

    print_metric_block(data, "Inference", "inference_ms")
    print()

    print_metric_block(data, "Postprocess", "postprocessing_ms")
    print()

    print_gpu_metrics(data)
    print_memory_metrics(data)

    print("PERFORMANCE")
    print("---------------------------------")
    print(
        f"Effective FPS: "
        f"{format_number(data.get('effective_fps'))}"
    )

    print("=" * 49)


def run_benchmark(
    base_url: str,
    warmup: int,
    iterations: int,
    timeout: float,
) -> int:
    """Call the benchmark API and print the result."""

    url = f"{base_url.rstrip('/')}{BENCHMARK_ENDPOINT}"

    payload = {
        "warmup": warmup,
        "iterations": iterations,
    }

    print(
        f"Running benchmark on {base_url} "
        f"(Warmup: {warmup}, Iterations: {iterations})..."
    )

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
        )

    except requests.ConnectionError:
        print(
            f"Error: Cannot connect to UNO Vision Runtime at {base_url}",
            file=sys.stderr,
        )
        print(
            "Make sure the server is running:",
            file=sys.stderr,
        )
        print(
            "  uvicorn app.main:app --host 0.0.0.0 --port 8000",
            file=sys.stderr,
        )
        return 1

    except requests.Timeout:
        print(
            f"Error: Benchmark request timed out after {timeout:.0f} seconds.",
            file=sys.stderr,
        )
        return 1

    except requests.RequestException as exc:
        print(
            f"Error: Benchmark request failed: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        data = response.json()
    except ValueError:
        print(
            f"Error: Server returned HTTP {response.status_code}",
            file=sys.stderr,
        )
        print(response.text, file=sys.stderr)
        return 1

    if response.status_code != 200:
        print(
            f"Error: Server returned {response.status_code}",
            file=sys.stderr,
        )

        error = data.get("error")
        detail = data.get("detail")

        if error:
            if isinstance(error, dict):
                print(
                    f"{error.get('code', 'ERROR')}: "
                    f"{error.get('message', 'Unknown error')}",
                    file=sys.stderr,
                )
            else:
                print(error, file=sys.stderr)

        elif detail:
            print(detail, file=sys.stderr)

        else:
            print(data, file=sys.stderr)

        return 1

    if data.get("success") is not True:
        print(
            "Error: Benchmark API returned an unsuccessful result.",
            file=sys.stderr,
        )
        print(data, file=sys.stderr)
        return 1

    print_report(data, warmup)

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="UNO Vision Runtime benchmark client."
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_BASE_URL,
        help=(
            "UNO Vision Runtime base URL "
            f"(default: {DEFAULT_BASE_URL})"
        ),
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of measured iterations (default: 100)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds (default: 600)",
    )

    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    if args.iterations <= 0:
        parser.error("--iterations must be > 0")

    if args.timeout <= 0:
        parser.error("--timeout must be > 0")

    return args


def main() -> int:
    args = parse_args()

    return run_benchmark(
        base_url=args.url,
        warmup=args.warmup,
        iterations=args.iterations,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
