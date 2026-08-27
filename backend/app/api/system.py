"""System and Provider API router."""

import psutil
import platform
import onnxruntime as ort
from fastapi import APIRouter, Request, Path as FastPathParam
from pydantic import BaseModel

from app.core.exceptions import AppError
from app.domain.execution_provider import (
    ExecutionProviderStatus,
    HardwareCapabilities,
    HardwareCpuInfo,
    HardwareMemoryInfo,
    HardwareOnnxRuntimeInfo
)

router = APIRouter()

@router.get("/providers", response_model=ExecutionProviderStatus)
async def get_providers(request: Request):
    """Get the status of all known execution providers."""
    return request.app.state.execution_provider_manager.get_status()

@router.post("/providers/{provider}/activate")
async def activate_provider(provider: str, request: Request):
    """Switch the active execution provider."""
    manager = request.app.state.execution_provider_manager
    manager.set_active_provider(provider)
    
    # Check if a model is currently loaded as active.
    # If so, we should recreate the session with the new provider.
    runtime_manager = request.app.state.inference_manager
    detection_service = request.app.state.detection_service
    
    # We can only safely reload if runtime is STOPPED. 
    # The set_active_provider call already verified inference manager is STOPPED.
    session, desc = detection_service.runtime_manager.get_active_runtime()
    if desc:
        # Reload it to create a new session with the new active provider.
        detection_service.runtime_manager.load_model(desc)
        detection_service.runtime_manager.activate_model()
        
    return {"success": True, "active_provider": provider}

@router.get("/hardware", response_model=HardwareCapabilities)
async def get_hardware(request: Request):
    """Get hardware capabilities and environment snapshot."""
    cpu_freq = None
    try:
        cpu_freq = psutil.cpu_freq().current
    except Exception:
        pass
        
    mem = psutil.virtual_memory()
    
    accelerators = {}
    if hasattr(request.app.state, 'opencl_backend'):
        ocl_backend = request.app.state.opencl_backend
        if ocl_backend.is_available():
            accelerators["opencl"] = ocl_backend.get_device_info()
    
    return {
        "cpu": HardwareCpuInfo(
            architecture=platform.machine(),
            cores=psutil.cpu_count(logical=False) or 1,
            logical_cores=psutil.cpu_count(logical=True),
            frequency_mhz=cpu_freq
        ).model_dump(),
        "memory": HardwareMemoryInfo(
            total_mb=mem.total / (1024 * 1024),
            available_mb=mem.available / (1024 * 1024)
        ).model_dump(),
        "onnxruntime": HardwareOnnxRuntimeInfo(
            version=ort.__version__,
            available_providers=ort.get_available_providers()
        ).model_dump(),
        "accelerators": accelerators
    }

@router.get("/diagnostics")
async def get_diagnostics(request: Request):
    """Get detailed diagnostics of the active model and session."""
    manager = request.app.state.execution_provider_manager
    detection_service = request.app.state.detection_service
    
    session, desc = detection_service.runtime_manager.get_active_runtime()
    
    actual_providers = []
    if session:
        try:
            actual_providers = session.get_providers()
        except Exception:
            pass
            
    return {
        "success": True,
        "active_provider": manager.active_provider,
        "session_provider_chain": actual_providers,
        "model_id": desc.model_id if desc else None,
        "session_initialized": session is not None
    }
