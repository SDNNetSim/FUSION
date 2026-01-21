"""
fusion.api: FastAPI backend for the FUSION GUI.

Provides a REST API and Server-Sent Events (SSE) for:
- Creating and managing simulation runs
- Streaming logs and progress updates
- Browsing and downloading artifacts
- Configuration management

Entry Point:
    Run via CLI: `fusion gui` or `python -m fusion.cli.run_gui`
    Direct: `uvicorn fusion.api.main:app --host 127.0.0.1 --port 8765`
"""

from .config import settings
from .main import app

__all__ = ["app", "settings"]
