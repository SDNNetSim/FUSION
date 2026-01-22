# Backend Standards

This document defines conventions for the Python/FastAPI backend.

## Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | FastAPI | Async, auto-docs, Pydantic native |
| Server | Uvicorn | ASGI, production-ready |
| Database | SQLite + SQLAlchemy 2.0 | Zero config, already in deps |
| Validation | Pydantic v2 | FastAPI native, excellent DX |
| SSE | sse-starlette | Mature, async, FastAPI integration |
| Async Files | aiofiles | Non-blocking file I/O |

## Project Structure

```
fusion/api/
├── __init__.py
├── main.py                     # FastAPI app, lifespan, static serving
├── config.py                   # Settings (pydantic-settings)
├── dependencies.py             # Dependency injection
│
├── routes/                     # API endpoints
│   ├── __init__.py             # Router aggregation
│   ├── runs.py
│   ├── configs.py
│   ├── artifacts.py
│   ├── topology.py
│   └── system.py
│
├── schemas/                    # Pydantic models (request/response)
│   ├── __init__.py
│   ├── run.py
│   ├── config.py
│   └── common.py
│
├── services/                   # Business logic
│   ├── __init__.py
│   ├── run_manager.py          # Job lifecycle
│   ├── config_service.py       # Config validation
│   ├── artifact_service.py     # Safe file access
│   └── progress_watcher.py     # Progress monitoring
│
├── db/                         # Database layer
│   ├── __init__.py
│   ├── database.py             # Engine, session factory
│   └── models.py               # SQLAlchemy ORM models
│
└── static/                     # Built React (gitignored except .gitkeep)
    └── .gitkeep
```

## Application Setup

### Main Application

```python
# fusion/api/main.py
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db.database import init_db
from .routes import runs, configs, artifacts, topology, system
from .services.run_manager import recover_orphaned_runs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    init_db()
    recover_orphaned_runs()
    yield
    # Shutdown (cleanup if needed)


app = FastAPI(
    title="FUSION GUI API",
    version="1.0.0",
    lifespan=lifespan,
)

# API routes
app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(configs.router, prefix="/api/configs", tags=["configs"])
app.include_router(artifacts.router, prefix="/api", tags=["artifacts"])
app.include_router(topology.router, prefix="/api/topology", tags=["topology"])
app.include_router(system.router, prefix="/api", tags=["system"])


# Static file serving with SPA fallback
static_dir = Path(__file__).parent / "static"

if static_dir.exists() and (static_dir / "index.html").exists():
    # Serve static files for known extensions
    @app.get("/{path:path}")
    async def serve_spa(request: Request, path: str):
        # Check if it's an API route (shouldn't reach here, but safety)
        if path.startswith("api/"):
            return {"detail": "Not found"}, 404

        # Try to serve static file
        file_path = static_dir / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # SPA fallback: serve index.html for all other routes
        return FileResponse(static_dir / "index.html")

    # Mount static assets (js, css, images)
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")
```

### Configuration

```python
# fusion/api/config.py
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server
    host: str = "127.0.0.1"
    port: int = 8765

    # Database
    database_url: str = "sqlite:///data/gui_runs/runs.db"

    # Paths
    runs_dir: Path = Path("data/gui_runs")
    templates_dir: Path = Path("fusion/configs/templates")

    # Limits
    max_concurrent_runs: int = 1
    max_log_size_bytes: int = 10 * 1024 * 1024  # 10MB

    class Config:
        env_prefix = "FUSION_GUI_"


settings = Settings()
```

### Database Setup

```python
# fusion/api/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from ..config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # SQLite specific
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create tables if they don't exist."""
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### ORM Models

```python
# fusion/api/db/models.py
from datetime import datetime
from sqlalchemy import String, Integer, Text, DateTime, Float
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(12), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Process tracking
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pgid: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Error info
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Progress cache
    current_erlang: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_erlangs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_iterations: Mapped[int | None] = mapped_column(Integer, nullable=True)
```

## Pydantic Schemas

```python
# fusion/api/schemas/run.py
from datetime import datetime
from pydantic import BaseModel, Field


class RunProgress(BaseModel):
    current_erlang: float | None = None
    total_erlangs: int | None = None
    current_iteration: int | None = None
    total_iterations: int | None = None
    percent_complete: float | None = None
    latest_metrics: dict | None = None


class RunBase(BaseModel):
    name: str | None = None


class RunCreate(RunBase):
    template: str = "default"
    config: dict = Field(default_factory=dict)


class RunResponse(RunBase):
    id: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    progress: RunProgress | None = None

    class Config:
        from_attributes = True


class RunListResponse(BaseModel):
    runs: list[RunResponse]
    total: int
    limit: int
    offset: int
```

## Route Conventions

```python
# fusion/api/routes/runs.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ..db.database import get_db
from ..db.models import Run
from ..schemas.run import RunCreate, RunResponse, RunListResponse
from ..services.run_manager import RunManager

router = APIRouter()


@router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_run(
    data: RunCreate,
    db: Session = Depends(get_db),
):
    """Create and start a new simulation run."""
    manager = RunManager(db)
    run = manager.create_run(data)
    return run


@router.get("", response_model=RunListResponse)
def list_runs(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """List all runs with optional filtering."""
    query = db.query(Run).order_by(Run.created_at.desc())

    if status:
        statuses = status.split(",")
        query = query.filter(Run.status.in_(statuses))

    total = query.count()
    runs = query.offset(offset).limit(min(limit, 100)).all()

    return RunListResponse(runs=runs, total=total, limit=limit, offset=offset)


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: str, db: Session = Depends(get_db)):
    """Get details for a specific run."""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


@router.delete("/{run_id}", response_model=RunResponse)
def cancel_run(run_id: str, db: Session = Depends(get_db)):
    """Cancel a running job or delete a completed one."""
    manager = RunManager(db)
    run = manager.cancel_or_delete(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


@router.get("/{run_id}/logs")
async def stream_logs(run_id: str, from_start: bool = True, db: Session = Depends(get_db)):
    """Stream logs via Server-Sent Events."""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    from ..services.run_manager import stream_run_logs
    return EventSourceResponse(stream_run_logs(run_id, from_start))
```

## Service Layer

```python
# fusion/api/services/run_manager.py
import asyncio
import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import aiofiles
from sqlalchemy.orm import Session

from ..config import settings
from ..db.models import Run
from ..schemas.run import RunCreate


class RunManager:
    def __init__(self, db: Session):
        self.db = db

    def create_run(self, data: RunCreate) -> Run:
        """Create a new run and start the simulation."""
        # Check concurrency limit
        active = self.db.query(Run).filter(Run.status == "RUNNING").count()
        if active >= settings.max_concurrent_runs:
            raise ValueError("Maximum concurrent runs reached")

        # Generate ID and paths
        run_id = uuid.uuid4().hex[:12]
        run_dir = settings.runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "output").mkdir()

        # Write config
        config_path = run_dir / "config.ini"
        self._write_config(data, config_path)

        # Create database record
        run = Run(
            id=run_id,
            name=data.name or f"Run {run_id[:6]}",
            status="PENDING",
            config_json=json.dumps(data.config),
        )
        self.db.add(run)
        self.db.commit()

        # Start simulation
        self._start_process(run, config_path)
        return run

    def _start_process(self, run: Run, config_path: Path):
        """Start the simulation subprocess."""
        run_dir = settings.runs_dir / run.id
        log_path = run_dir / "logs" / "sim.log"
        progress_path = run_dir / "progress.jsonl"

        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                [
                    sys.executable, "-m", "fusion.cli.run_sim",
                    "--config", str(config_path),
                    "--output-dir", str(run_dir / "output"),
                    "--progress-file", str(progress_path),
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # New process group
            )

        run.status = "RUNNING"
        run.pid = process.pid
        run.pgid = os.getpgid(process.pid)
        run.started_at = datetime.utcnow()
        self.db.commit()

        # Start background watcher (in production, use proper task queue)
        # For MVP, this is handled by periodic polling

    def cancel_or_delete(self, run_id: str) -> Run | None:
        """Cancel running job or delete completed job."""
        run = self.db.query(Run).filter(Run.id == run_id).first()
        if not run:
            return None

        if run.status == "RUNNING":
            self._kill_process(run)

        if run.status in ("COMPLETED", "FAILED", "CANCELLED"):
            # Delete artifacts
            run_dir = settings.runs_dir / run_id
            if run_dir.exists():
                import shutil
                shutil.rmtree(run_dir)

        self.db.delete(run)
        self.db.commit()
        return run

    def _kill_process(self, run: Run):
        """Kill the entire process group."""
        if not run.pgid:
            return

        try:
            os.killpg(run.pgid, signal.SIGTERM)
            # Give graceful shutdown time
            import time
            time.sleep(2)
            try:
                os.killpg(run.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass

        run.status = "CANCELLED"
        run.completed_at = datetime.utcnow()

    def _write_config(self, data: RunCreate, path: Path):
        """Write configuration to INI file."""
        # Load template
        template_path = settings.templates_dir / f"{data.template}.ini"
        if not template_path.exists():
            template_path = settings.templates_dir / "default.ini"

        # TODO: Merge template with overrides
        import shutil
        shutil.copy(template_path, path)


def recover_orphaned_runs():
    """Mark stale RUNNING jobs as FAILED on startup."""
    from ..db.database import SessionLocal

    db = SessionLocal()
    try:
        running = db.query(Run).filter(Run.status == "RUNNING").all()
        for run in running:
            if not _is_process_alive(run.pgid):
                run.status = "FAILED"
                run.error_message = "Server restarted while run was active"
                run.completed_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


def _is_process_alive(pgid: int | None) -> bool:
    if not pgid:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


async def stream_run_logs(run_id: str, from_start: bool) -> AsyncGenerator[dict, None]:
    """Stream log file content via SSE."""
    log_path = settings.runs_dir / run_id / "logs" / "sim.log"

    if not log_path.exists():
        yield {"event": "error", "data": "Log file not found"}
        return

    async with aiofiles.open(log_path, mode="r") as f:
        if from_start:
            content = await f.read()
            if content:
                yield {"event": "log", "data": content}
        else:
            await f.seek(0, 2)  # End of file

        while True:
            line = await f.readline()
            if line:
                yield {"event": "log", "data": line.rstrip()}
            else:
                # Check if run is still active
                from ..db.database import SessionLocal
                db = SessionLocal()
                run = db.query(Run).filter(Run.id == run_id).first()
                db.close()

                if run and run.status not in ("PENDING", "RUNNING"):
                    yield {"event": "end", "data": run.status}
                    break

                await asyncio.sleep(0.3)
```

## Process Tree Termination

The simulator uses `multiprocessing.Pool` internally, creating child processes. We must kill the entire process tree, not just the parent.

### POSIX (Linux/macOS)

```python
import os
import signal
import subprocess

def start_simulation(cmd: list[str], log_file) -> subprocess.Popen:
    """Start simulation in a new process session."""
    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Creates new process group (PGID = PID)
    )

def kill_process_tree_posix(pgid: int) -> None:
    """Kill entire process group on POSIX systems."""
    try:
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(2)  # Grace period
        os.killpg(pgid, signal.SIGKILL)  # Force kill stragglers
    except ProcessLookupError:
        pass  # Already dead
```

### Windows

Windows does not have process groups like POSIX. Options:

**Option A: Job Objects (recommended)**

```python
import subprocess
import ctypes
from ctypes import wintypes

# Create a job object that auto-kills children when handle closes
def start_simulation_windows(cmd: list[str], log_file) -> subprocess.Popen:
    """Start simulation with Windows Job Object for tree termination."""
    # CREATE_NEW_PROCESS_GROUP allows Ctrl+Break signal
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    # Assign to job object (requires win32job or ctypes)
    # Job object configured with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    # See: https://docs.microsoft.com/en-us/windows/win32/procthread/job-objects

    return process
```

**Option B: taskkill fallback (simpler, less reliable)**

```python
import subprocess

def kill_process_tree_windows(pid: int) -> None:
    """Kill process tree using taskkill."""
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(pid)],
        capture_output=True,
    )
```

**Option C: psutil (cross-platform, adds dependency)**

```python
import psutil

def kill_process_tree_psutil(pid: int) -> None:
    """Kill process tree using psutil (cross-platform)."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        gone, alive = psutil.wait_procs(children + [parent], timeout=3)
        for p in alive:
            p.kill()
    except psutil.NoSuchProcess:
        pass
```

### Platform Detection

```python
import platform
import os

def kill_simulation(run: Run) -> None:
    """Kill simulation process tree (cross-platform)."""
    if platform.system() == "Windows":
        kill_process_tree_windows(run.pid)
    else:
        kill_process_tree_posix(run.pgid)
```

## Artifact Security

```python
# fusion/api/services/artifact_service.py
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

    Raises:
        403: Path traversal or symlink escape attempt
        404: File does not exist
    """
    base = (settings.runs_dir / run_id).resolve()

    # Join and resolve (follows symlinks)
    requested = (base / relative_path).resolve()

    # Security: ensure resolved path is within run directory
    if not requested.is_relative_to(base):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path escapes run directory"
        )

    if not requested.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return requested


def list_directory(run_id: str, relative_path: str = "") -> list[dict]:
    """List directory contents safely."""
    dir_path = get_safe_path(run_id, relative_path) if relative_path else (
        settings.runs_dir / run_id
    ).resolve()

    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")

    entries = []
    for item in sorted(dir_path.iterdir()):
        stat = item.stat()
        entries.append({
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
            "size_bytes": stat.st_size if item.is_file() else None,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    return entries
```

### Artifact Security Tests

```python
# fusion/api/tests/test_artifact_security.py
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from fusion.api.main import app

@pytest.fixture
def run_with_symlink(tmp_path):
    """Create a run directory with a malicious symlink."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    # Create a symlink pointing outside the run directory
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("sensitive data")

    symlink = run_dir / "escape.txt"
    symlink.symlink_to(secret_file)

    return run_dir

def test_symlink_escape_blocked(client, run_with_symlink):
    """Symlinks pointing outside run directory should be rejected."""
    response = client.get(f"/api/runs/test_run/artifacts/escape.txt")
    assert response.status_code == 403

def test_path_traversal_blocked(client):
    """Path traversal attempts should be rejected."""
    response = client.get("/api/runs/test_run/artifacts/../../../etc/passwd")
    assert response.status_code == 403

def test_double_encoded_traversal_blocked(client):
    """Double-encoded traversal should be rejected."""
    response = client.get("/api/runs/test_run/artifacts/..%252f..%252fetc/passwd")
    assert response.status_code in (403, 404)
```

## Linting and Testing

### Ruff Configuration

The project already uses ruff. Add API-specific paths to existing config:

```toml
# pyproject.toml (additions)
[tool.ruff]
extend-include = ["fusion/api/**/*.py"]
```

### Pytest Tests

```python
# fusion/api/tests/test_runs.py
import pytest
from fastapi.testclient import TestClient

from fusion.api.main import app
from fusion.api.db.database import get_db, Base, engine


@pytest.fixture
def client():
    """Test client with clean database."""
    Base.metadata.create_all(bind=engine)
    yield TestClient(app)
    Base.metadata.drop_all(bind=engine)


def test_create_run(client):
    response = client.post("/api/runs", json={
        "name": "Test Run",
        "config": {"general_settings": {"network": "nsfnet"}}
    })

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Run"
    assert data["status"] == "PENDING"


def test_list_runs(client):
    # Create a run first
    client.post("/api/runs", json={"name": "Test"})

    response = client.get("/api/runs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1


def test_artifact_path_traversal(client):
    # Create a run
    create_resp = client.post("/api/runs", json={"name": "Test"})
    run_id = create_resp.json()["id"]

    # Attempt path traversal
    response = client.get(f"/api/runs/{run_id}/artifacts/../../../etc/passwd")
    assert response.status_code == 403
```

## Type Hints

All functions must have type hints:

```python
# Good
def get_run(run_id: str, db: Session) -> Run | None:
    return db.query(Run).filter(Run.id == run_id).first()

# Bad
def get_run(run_id, db):
    return db.query(Run).filter(Run.id == run_id).first()
```

## Logging

Use the existing FUSION logging setup:

```python
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)

def create_run(data: RunCreate) -> Run:
    logger.info("Creating run: %s", data.name)
    # ...
    logger.debug("Run created with ID: %s", run.id)
```
