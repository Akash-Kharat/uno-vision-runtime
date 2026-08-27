"""Execution Provider Manager."""

import logging
import threading
import onnxruntime as ort
from app.domain.execution_provider import (
    ProviderStatus, 
    ExecutionProviderInfo, 
    ExecutionProviderStatus
)
from app.core.exceptions import AppError

logger = logging.getLogger(__name__)

PREFERRED_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "V4L2ExecutionProvider",
    "CPUExecutionProvider"
]

class ExecutionProviderManager:
    """Manages discovery and selection of execution providers."""
    
    def __init__(self, inference_runtime_manager=None):
        self.lock = threading.Lock()
        self.inference_runtime_manager = inference_runtime_manager
        self.available_providers = ort.get_available_providers()
        self.active_provider = "CPUExecutionProvider" # Fallback initially
        
        # Determine initial best provider
        for p in PREFERRED_PROVIDERS:
            if p in self.available_providers:
                self.active_provider = p
                break

    def get_providers(self) -> list[str]:
        """Return the prioritized list of providers to try."""
        with self.lock:
            # Always put the active provider first, followed by CPU as fallback
            providers = [self.active_provider]
            if "CPUExecutionProvider" not in providers and "CPUExecutionProvider" in self.available_providers:
                providers.append("CPUExecutionProvider")
            return providers
            
    def get_status(self) -> ExecutionProviderStatus:
        with self.lock:
            provider_infos = []
            
            # First map preferred providers
            seen = set()
            for priority, p_name in enumerate(PREFERRED_PROVIDERS):
                seen.add(p_name)
                is_avail = p_name in self.available_providers
                is_active = p_name == self.active_provider
                
                status = ProviderStatus.UNAVAILABLE
                if is_active:
                    status = ProviderStatus.ACTIVE
                elif is_avail:
                    status = ProviderStatus.AVAILABLE
                    
                provider_infos.append(ExecutionProviderInfo(
                    name=p_name,
                    available=is_avail,
                    active=is_active,
                    priority=priority,
                    status=status
                ))
                
            # Map any remaining available providers
            for p_name in self.available_providers:
                if p_name not in seen:
                    is_active = p_name == self.active_provider
                    provider_infos.append(ExecutionProviderInfo(
                        name=p_name,
                        available=True,
                        active=is_active,
                        priority=999,
                        status=ProviderStatus.ACTIVE if is_active else ProviderStatus.AVAILABLE
                    ))
                    
            return ExecutionProviderStatus(
                providers=provider_infos,
                active_provider=self.active_provider
            )
            
    def set_active_provider(self, provider: str) -> None:
        with self.lock:
            if self.inference_runtime_manager and self.inference_runtime_manager.state.value in ("RUNNING", "STARTING", "PAUSED"):
                raise AppError(
                    code="PROVIDER_CHANGE_RUNTIME_CONFLICT",
                    message="Cannot change provider while runtime is active.",
                    status_code=409
                )
                
            if provider not in self.available_providers:
                raise AppError(
                    code="PROVIDER_UNAVAILABLE",
                    message=f"Provider {provider} is not available.",
                    status_code=400
                )
                
            self.active_provider = provider
            logger.info(f"Switched active provider to {provider}")

    def record_initialization_error(self, provider: str, error_msg: str) -> None:
        """Called if a provider fails to initialize."""
        logger.warning(f"Provider {provider} failed to initialize: {error_msg}")
        # In a more advanced version, we might mark it as ERROR and auto-fallback.
