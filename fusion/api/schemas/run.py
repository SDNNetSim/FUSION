"""
Pydantic schemas for simulation runs.

Defines request/response models for the runs API.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class RunProgress(BaseModel):
    """Progress information for a running simulation."""

    current_erlang: float | None = None
    total_erlangs: int | None = None
    current_iteration: int | None = None
    total_iterations: int | None = None
    percent_complete: float | None = None


class RunCreate(BaseModel):
    """Request body for creating a new run."""

    name: str | None = Field(None, max_length=255, description="Optional run name")
    template: str = Field("default", description="Configuration template to use")
    config: dict = Field(default_factory=dict, description="Config overrides")


class RunResponse(BaseModel):
    """Response body for a single run."""

    id: str
    name: str | None
    status: str
    template: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    progress: RunProgress | None = None

    class Config:
        """Pydantic config."""

        from_attributes = True


class RunListResponse(BaseModel):
    """Response body for listing runs."""

    runs: list[RunResponse]
    total: int
    limit: int
    offset: int
