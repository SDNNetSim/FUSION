"""
FastAPI application entry point.

Configures the app, routes, middleware, and static file serving.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db.database import init_db
from .routes import artifacts, codebase, configs, runs, system, topology
from .services.run_manager import recover_orphaned_runs

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Initializing FUSION GUI API...")
    init_db()
    recover_orphaned_runs()
    logger.info("FUSION GUI API ready")

    yield

    # Shutdown
    logger.info("Shutting down FUSION GUI API...")


app = FastAPI(
    title="FUSION GUI API",
    description="REST API for the FUSION optical network simulator GUI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(configs.router, prefix="/api/configs", tags=["configs"])
app.include_router(artifacts.router, prefix="/api", tags=["artifacts"])
app.include_router(system.router, prefix="/api", tags=["system"])
app.include_router(topology.router, prefix="/api", tags=["topology"])
app.include_router(codebase.router, prefix="/api", tags=["codebase"])


# Static file serving with SPA fallback
static_dir = Path(__file__).parent / "static"


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions with a consistent error format."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": "INTERNAL_ERROR"},
    )


# Only set up static serving if the static directory exists with content
if static_dir.exists() and (static_dir / "index.html").exists():
    # Mount assets directory
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{path:path}")
    async def serve_spa(request: Request, path: str) -> FileResponse:
        """
        Serve static files with SPA fallback.

        All non-API routes serve index.html for client-side routing.
        """
        # API routes are handled by the routers above
        if path.startswith("api/"):
            return JSONResponse(status_code=404, content={"detail": "Not found"})

        # Try to serve static file
        file_path = static_dir / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # SPA fallback: serve index.html for all other routes
        return FileResponse(static_dir / "index.html")
