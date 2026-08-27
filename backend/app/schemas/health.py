"""Health endpoint response schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    success: bool
    status: str
    service: str
