"""Performance profiling and benchmarking domain models."""

import time
from contextlib import contextmanager
from typing import Iterator

class PerformanceProfiler:
    """Measures named execution stages."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self.timings[name] = (t1 - t0) * 1000.0
            
    def get_timings(self) -> dict[str, float]:
        return self.timings

    def get_timing(self, name: str) -> float:
        return self.timings.get(name, 0.0)

    def record_custom(self, name: str, value_ms: float) -> None:
        if self.enabled:
            self.timings[name] = value_ms
