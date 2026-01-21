"""
Configuration template endpoints.

Provides access to configuration templates for creating runs.
"""

import configparser
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import settings
from ..schemas.config import ConfigValidationResponse, TemplateInfo, TemplateListResponse


class ConfigValidationRequest(BaseModel):
    """Request body for config validation."""

    content: str


router = APIRouter()


@router.get("/templates", response_model=TemplateListResponse)
def list_templates() -> TemplateListResponse:
    """
    List available configuration templates.

    :returns: All .ini files in the templates directory.
    """
    templates_dir = Path(settings.templates_dir)

    if not templates_dir.exists():
        return TemplateListResponse(templates=[])

    templates = []
    for path in sorted(templates_dir.glob("*.ini")):
        # Extract description from first comment line if present
        description = None
        try:
            with open(path) as f:
                first_line = f.readline().strip()
                if first_line.startswith(";") or first_line.startswith("#"):
                    description = first_line.lstrip(";# ").strip()
        except OSError:
            pass

        templates.append(
            TemplateInfo(
                name=path.stem,
                description=description,
                path=str(path),
            )
        )

    return TemplateListResponse(templates=templates)


@router.get("/templates/{name}")
def get_template(name: str) -> dict:
    """
    Get the contents of a specific template.

    :param name: Template name (without .ini extension).
    :returns: Template content as raw text.
    """
    template_path = Path(settings.templates_dir) / f"{name}.ini"

    if not template_path.exists():
        raise HTTPException(status_code=404, detail=f"Template not found: {name}")

    try:
        content = template_path.read_text()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to read template: {e}")

    return {"name": name, "content": content}


@router.post("/validate", response_model=ConfigValidationResponse)
def validate_config(request: ConfigValidationRequest) -> ConfigValidationResponse:
    """
    Validate a configuration without running a simulation.

    :param request: Configuration content to validate.
    :returns: Validation result with any errors or warnings.
    """
    errors = []
    warnings = []

    try:
        parser = configparser.ConfigParser()
        parser.read_string(request.content)

        # Check for required sections
        required_sections = ["general_settings"]
        for section in required_sections:
            if section not in parser.sections():
                errors.append(f"Missing required section: [{section}]")

        # Check general_settings if present
        if "general_settings" in parser.sections():
            general = parser["general_settings"]

            # Check for common required fields
            if "network" not in general:
                warnings.append("No 'network' specified in [general_settings]")

        # Additional validation can be added here

    except configparser.Error as e:
        errors.append(f"INI parsing error: {e}")

    return ConfigValidationResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


@router.get("/schema")
def get_config_schema() -> dict:
    """
    Get JSON Schema for configuration options.

    :returns: JSON Schema describing configuration structure.
    """
    # Basic schema for FUSION configuration
    # This can be expanded based on the actual config structure
    return {
        "schema_version": "1.0",
        "type": "object",
        "properties": {
            "general_settings": {
                "type": "object",
                "description": "General simulation settings",
                "properties": {
                    "network": {
                        "type": "string",
                        "description": "Network topology to simulate",
                    },
                    "max_iters": {
                        "type": "integer",
                        "description": "Maximum iterations per erlang",
                        "default": 100,
                    },
                    "num_requests": {
                        "type": "integer",
                        "description": "Number of requests to simulate",
                    },
                },
            },
            "traffic_settings": {
                "type": "object",
                "description": "Traffic generation settings",
                "properties": {
                    "arrival_rate": {
                        "type": "number",
                        "description": "Request arrival rate",
                    },
                    "holding_time": {
                        "type": "number",
                        "description": "Average holding time",
                    },
                },
            },
        },
    }
