"""Runtime manager for inference sessions."""

import logging
import threading
from typing import Any

import onnxruntime as ort

from app.core.exceptions import AppError
from app.domain.runtime import ModelRuntimeDescriptor

logger = logging.getLogger(__name__)

class RuntimeManager:
    """Manages the active model runtime."""
    
    def __init__(self, session_factory=None) -> None:
        self.lock = threading.Lock()
        self.session_factory = session_factory
        self.active_descriptor: ModelRuntimeDescriptor | None = None
        self.active_session: ort.InferenceSession | None = None
        self.active_session_options: dict = {}
        
        self.candidate_descriptor: ModelRuntimeDescriptor | None = None
        self.candidate_session: ort.InferenceSession | None = None
        self.candidate_session_options: dict = {}

    def load_model(self, descriptor: ModelRuntimeDescriptor) -> None:
        """Load a model into memory as a candidate. Does not activate it."""
        try:
            if self.session_factory:
                session_data = self.session_factory.create(descriptor.model_path)
                session = session_data["session"]
                options = session_data.get("options", {})
            else:
                session = ort.InferenceSession(str(descriptor.model_path), providers=['CPUExecutionProvider'])
                options = {}
            
            with self.lock:
                self.candidate_descriptor = descriptor
                self.candidate_session = session
                self.candidate_session_options = options
                
            logger.info(f"Successfully loaded candidate model {descriptor.model_id}")
            
        except Exception as e:
            logger.exception("Failed to load model runtime")
            raise AppError(
                code="MODEL_ACTIVATION_FAILED", 
                message=f"Failed to load ONNX session: {str(e)}", 
                status_code=500
            )

    def activate_model(self) -> None:
        """Atomically swap the candidate model to become active."""
        with self.lock:
            if not self.candidate_session or not self.candidate_descriptor:
                raise AppError(
                    code="MODEL_NOT_READY",
                    message="No candidate model loaded to activate.",
                    status_code=400
                )
            
            # Atomic swap
            self.active_session = self.candidate_session
            self.active_descriptor = self.candidate_descriptor
            self.active_session_options = self.candidate_session_options
            
            # Clear candidate
            self.candidate_session = None
            self.candidate_descriptor = None
            self.candidate_session_options = {}
            
            logger.info(f"Activated model {self.active_descriptor.model_id}")

    def get_active_runtime(self) -> tuple[ort.InferenceSession | None, ModelRuntimeDescriptor | None]:
        """Get the active session and descriptor safely."""
        with self.lock:
            return self.active_session, self.active_descriptor

    def unload_model(self) -> None:
        """Unload the active model."""
        with self.lock:
            self.active_session = None
            self.active_descriptor = None
            self.active_session_options = {}
            logger.info("Unloaded active model")

    def get_status(self) -> str:
        """Get the status of the runtime manager."""
        with self.lock:
            if self.active_session:
                return "READY"
            return "STOPPED"
