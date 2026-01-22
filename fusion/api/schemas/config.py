"""
Pydantic schemas for configuration endpoints.

Defines models for templates and config validation.
"""

from pydantic import BaseModel


class TemplateInfo(BaseModel):
    """Information about a configuration template."""

    name: str
    description: str | None = None
    path: str


class TemplateListResponse(BaseModel):
    """Response body for listing templates."""

    templates: list[TemplateInfo]


class ConfigSchema(BaseModel):
    """JSON Schema representation of configuration options."""

    schema_version: str = "1.0"
    properties: dict


class ConfigValidationResponse(BaseModel):
    """Response body for config validation."""

    valid: bool
    errors: list[str] = []
    warnings: list[str] = []
