"""UNO Vision Runtime application configuration."""

from typing import Literal
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings have sensible defaults for local development.
    A .env file is optional.
    """

    APP_NAME: str = "uno-vision-runtime"
    APP_VERSION: str = "0.1.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    CAMERA_BACKEND: str = "V4L2"
    CAMERA_PIXEL_FORMAT: str = "MJPG"
    CAMERA_FPS: int = 30
    CAMERA_MIN_PIXEL_RANGE: int = 10
    CAMERA_RECONNECT_INTERVAL_SECONDS: int = 5
    CAMERA_STARTUP_TIMEOUT_SECONDS: int = 5

    MODEL_STORAGE_PATH: str = "./data/models"
    MAX_UPLOAD_SIZE_BYTES: int = Field(100 * 1024 * 1024, description="Maximum ONNX upload size.")
    
    INFERENCE_TARGET_FPS: int = Field(5, description="Target continuous inference frame rate.")
    
    # Performance & Optimization Defaults
    PERFORMANCE_HISTORY_SIZE: int = Field(500, description="Size of rolling history deque for stats")
    MAX_PRE_NMS_DETECTIONS: int = Field(300, description="Max candidate boxes to feed into NMS")
    MAX_DETECTIONS: int = Field(100, description="Max final detections to output")
    ENABLE_DETAILED_PROFILING: bool = Field(False, description="Turn on inner profiling steps by default")
    
    # Preprocessing & Hardware Acceleration
    ENABLE_OPENCL: bool = Field(True, description="Enable OpenCL backend discovery")
    OPENCL_DEVICE_TYPE: str = Field("ALL", description="GPU, CPU, or ALL")
    OPENCL_PREFERRED_VENDOR: str = Field("Qualcomm", description="Preferred OpenCL vendor")
    PREPROCESSING_BACKEND: str = Field("AUTO", description="AUTO, OPENCL, or CPU")
    ENABLE_OPENCL_PROFILING: bool = Field(True, description="Expose precise OpenCL queue timing")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings() -> Settings:
    """Create and return application settings."""
    return Settings()
