"""
Artifact endpoints for browsing and downloading run outputs.

Provides secure file access with path traversal protection.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..services.artifact_service import get_safe_path, list_directory

router = APIRouter()


@router.get("/runs/{run_id}/artifacts")
def list_artifacts(run_id: str, path: str = "") -> dict:
    """
    List files and directories in a run's output.

    :param run_id: The run identifier.
    :param path: Relative path within the run directory.
    :returns: List of file/directory entries.
    """
    entries = list_directory(run_id, path)
    return {"path": path, "entries": entries}


@router.get("/runs/{run_id}/artifacts/{file_path:path}")
def download_artifact(run_id: str, file_path: str) -> FileResponse:
    """
    Download a specific artifact file.

    :param run_id: The run identifier.
    :param file_path: Relative path to the file within the run directory.
    :returns: The file content.
    """
    safe_path = get_safe_path(run_id, file_path)

    if safe_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Cannot download directory. Use the list endpoint.",
        )

    # Determine media type
    suffix = safe_path.suffix.lower()
    media_types = {
        ".json": "application/json",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".log": "text/plain",
        ".ini": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf": "application/pdf",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=safe_path,
        filename=safe_path.name,
        media_type=media_type,
    )
