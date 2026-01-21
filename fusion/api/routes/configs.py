"""
Configuration template endpoints.

Provides access to configuration templates for creating runs.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..schemas.config import TemplateInfo, TemplateListResponse

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
