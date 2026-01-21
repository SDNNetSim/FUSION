"""
Artifact service for safe file access.

Provides secure access to run artifacts with path traversal protection.
"""

from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from ..config import settings


def get_safe_path(run_id: str, relative_path: str) -> Path:
    """
    Validate and return safe artifact path.

    Security checks:

    1. Normalize path to prevent traversal (../)
    2. Resolve symlinks via realpath()
    3. Verify resolved path is within run directory
    4. Reject symlinks that escape the run directory

    :param run_id: The run identifier.
    :param relative_path: Relative path within the run directory.
    :returns: Validated absolute path to the artifact.
    :raises HTTPException: 403 for path traversal, 404 for not found.
    """
    base = (settings.runs_dir / run_id).resolve()

    # Ensure the run directory exists
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    # Join and resolve (follows symlinks)
    requested = (base / relative_path).resolve()

    # Security: ensure resolved path is within run directory
    if not requested.is_relative_to(base):
        raise HTTPException(
            status_code=403, detail="Access denied: path escapes run directory"
        )

    if not requested.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return requested


def list_directory(run_id: str, relative_path: str = "") -> list[dict]:
    """
    List directory contents safely.

    :param run_id: The run identifier.
    :param relative_path: Relative path within the run directory.
    :returns: List of file/directory entries with metadata.
    """
    if relative_path:
        dir_path = get_safe_path(run_id, relative_path)
    else:
        dir_path = (settings.runs_dir / run_id).resolve()
        if not dir_path.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")

    entries = []
    for item in sorted(dir_path.iterdir()):
        try:
            stat = item.stat()
            entries.append(
                {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size_bytes": stat.st_size if item.is_file() else None,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )
        except OSError:
            # Skip files we can't stat
            continue

    return entries
