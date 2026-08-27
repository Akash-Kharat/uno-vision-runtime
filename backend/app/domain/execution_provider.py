"""Execution Provider domain models."""

from enum import Enum
from pydantic import BaseModel

class ProviderStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    ACTIVE = "ACTIVE"
    ERROR = "ERROR"

class ExecutionProviderInfo(BaseModel):
    name: str
    available: bool
    active: bool = False
    priority: int
    status: ProviderStatus
    error: str | None = None

class ExecutionProviderStatus(BaseModel):
    providers: list[ExecutionProviderInfo]
    active_provider: str | None = None

class HardwareCpuInfo(BaseModel):
    architecture: str
    cores: int
    logical_cores: int | None = None
    frequency_mhz: float | None = None

class HardwareMemoryInfo(BaseModel):
    total_mb: float
    available_mb: float

class HardwareOnnxRuntimeInfo(BaseModel):
    version: str
    available_providers: list[str]

class HardwareCapabilities(BaseModel):
    cpu: HardwareCpuInfo
    memory: HardwareMemoryInfo
    onnxruntime: HardwareOnnxRuntimeInfo
    accelerators: dict = {}

class ProviderBenchmarkResult(BaseModel):
    provider: str
    available: bool
    mean_inference_ms: float
    p50_inference_ms: float
    p95_inference_ms: float
    effective_fps: float
    successful_iterations: int
    failed_iterations: int
    error: str | None = None

class ProviderBenchmarkComparison(BaseModel):
    success: bool
    results: list[ProviderBenchmarkResult]
    recommended_provider: str | None = None
