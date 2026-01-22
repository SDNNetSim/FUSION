"""
Common Pydantic schemas shared across endpoints.

Includes error responses and pagination.
"""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response format."""

    detail: str
    code: str | None = None


class PaginatedResponse(BaseModel):
    """Base class for paginated responses."""

    total: int
    limit: int
    offset: int
