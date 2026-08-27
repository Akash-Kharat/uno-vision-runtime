"""Common response schemas."""

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Structured error detail."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response returned by all API error handlers."""

    success: bool = False
    error: ErrorDetail
