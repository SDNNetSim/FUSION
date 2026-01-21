"""
Pydantic schemas for request/response validation.

Defines API contracts for runs, configs, and common types.
"""

from .common import ErrorResponse, PaginatedResponse
from .config import ConfigSchema, ConfigValidationResponse, TemplateInfo
from .run import RunCreate, RunListResponse, RunProgress, RunResponse

__all__ = [
    # Run schemas
    "RunCreate",
    "RunResponse",
    "RunListResponse",
    "RunProgress",
    # Config schemas
    "TemplateInfo",
    "ConfigSchema",
    "ConfigValidationResponse",
    # Common schemas
    "ErrorResponse",
    "PaginatedResponse",
]
