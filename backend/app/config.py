"""UNO Vision Runtime application configuration."""

from typing import Literal

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings() -> Settings:
    """Create and return application settings."""
    return Settings()
