"""UNO Vision Runtime — FastAPI application entry point."""

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.camera import router as camera_router
from app.api.models import router as models_router
from app.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.lifecycle import lifespan
from app.services.camera_manager import CameraManager


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory.

    Args:
        settings: Optional pre-built settings (useful for testing).
                  When *None*, settings are loaded from the environment.

    Returns:
        A fully configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    from app.services.model_registry import ModelRegistry
    from app.services.runtime_manager import RuntimeManager
    from app.services.detection_service import DetectionService
    from app.services.inference_runtime_manager import InferenceRuntimeManager
    from app.services.execution_provider_manager import ExecutionProviderManager
    from app.services.onnx_session_factory import ONNXSessionFactory
    from app.services.provider_benchmark_manager import ProviderBenchmarkManager
    from app.services.accelerators.opencl_backend import OpenCLBackend
    
    # Attach settings and managers so lifecycle and handlers can access them
    app.state.settings = settings
    
    camera_manager = CameraManager(settings)
    model_registry = ModelRegistry(settings)
    
    # Initialize OpenCL backend for preprocessing
    opencl_backend = OpenCLBackend(settings)
    opencl_backend.initialize()
    
    # Initialize execution provider manager
    ep_manager = ExecutionProviderManager()
    session_factory = ONNXSessionFactory(ep_manager, settings)
    
    runtime_manager = RuntimeManager(session_factory=session_factory)
    detection_service = DetectionService(
        camera_manager=camera_manager, 
        runtime_manager=runtime_manager,
        registry=model_registry,
        opencl_backend=opencl_backend,
        config=settings
    )
    inference_manager = InferenceRuntimeManager(
        camera_manager,
        detection_service,
        target_fps=getattr(settings, "INFERENCE_TARGET_FPS", 5)
    )
    # Complete circular injection for safe provider switching checks
    ep_manager.inference_runtime_manager = inference_manager
    
    provider_benchmark_manager = ProviderBenchmarkManager(
        provider_manager=ep_manager,
        session_factory=session_factory,
        detection_service=detection_service,
        camera_manager=camera_manager
    )

    app.state.camera_manager = camera_manager
    app.state.model_registry = model_registry
    app.state.runtime_manager = runtime_manager
    app.state.detection_service = detection_service
    app.state.inference_manager = inference_manager
    app.state.execution_provider_manager = ep_manager
    app.state.onnx_session_factory = session_factory
    app.state.provider_benchmark_manager = provider_benchmark_manager
    app.state.opencl_backend = opencl_backend

    # Exception handlers
    register_exception_handlers(app)

    # Root endpoint
    @app.get("/")
    async def root() -> dict[str, str]:
        """Return basic service information."""
        return {
            "service": "UNO Vision Runtime",
            "api_version": "v1",
        }

    # Versioned API routes
    app.include_router(health_router, prefix="/api/v1", tags=["health"])
    app.include_router(camera_router, prefix="/api/v1/camera", tags=["camera"])
    app.include_router(models_router, prefix="/api/v1/models", tags=["models"])
    
    from app.api.detect import router as detect_router
    app.include_router(detect_router, prefix="/api/v1/detect", tags=["inference"])
    
    from app.api.runtime import router as runtime_router
    app.include_router(runtime_router, prefix="/api/v1/runtime", tags=["runtime"])
    
    from app.api.benchmark import router as benchmark_router
    app.include_router(benchmark_router, prefix="/api/v1/benchmark", tags=["benchmark"])

    from app.api.system import router as system_router
    app.include_router(system_router, prefix="/api/v1/system", tags=["system"])

    return app


# Default application instance used by `uvicorn app.main:app`
app: FastAPI = create_app()
