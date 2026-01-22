# FUSION API Module

FastAPI backend for the FUSION GUI web interface.

## Overview

This module provides a REST API for managing simulation runs, streaming logs, and accessing artifacts. It serves as the backend for the FUSION GUI.

## Quick Start

```bash
# Install GUI dependencies
pip install -r requirements-gui.txt

# Start the server
python -m fusion.cli.run_gui

# Or with reload for development
python -m fusion.cli.run_gui --reload
```

The API will be available at `http://127.0.0.1:8765`.

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://127.0.0.1:8765/docs`
- ReDoc: `http://127.0.0.1:8765/redoc`

## Module Structure

```
fusion/api/
├── __init__.py           # Package exports
├── main.py               # FastAPI app, lifespan, middleware
├── config.py             # Settings (pydantic-settings)
├── dependencies.py       # Dependency injection
├── routes/               # API endpoints
│   ├── runs.py           # /api/runs - Run management
│   ├── configs.py        # /api/configs - Templates
│   ├── artifacts.py      # /api/runs/{id}/artifacts
│   └── system.py         # /api/health, /api/version
├── schemas/              # Pydantic models
│   ├── run.py            # Run request/response models
│   ├── config.py         # Config models
│   └── common.py         # Shared models
├── services/             # Business logic
│   ├── run_manager.py    # Simulation lifecycle
│   └── artifact_service.py # Secure file access
├── db/                   # Database layer
│   ├── database.py       # SQLite + SQLAlchemy
│   └── models.py         # ORM models
├── static/               # Built React frontend
└── tests/                # Unit tests
```

## Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/runs | Create and start a run |
| GET | /api/runs | List all runs |
| GET | /api/runs/{id} | Get run details |
| DELETE | /api/runs/{id} | Cancel/delete run |
| GET | /api/runs/{id}/logs | Stream logs (SSE) |
| GET | /api/runs/{id}/artifacts | List artifacts |
| GET | /api/configs/templates | List config templates |
| GET | /api/health | Health check |

## Configuration

Environment variables (prefix: `FUSION_GUI_`):

| Variable | Default | Description |
|----------|---------|-------------|
| HOST | 127.0.0.1 | Server bind address |
| PORT | 8765 | Server port |
| DATABASE_URL | sqlite:///data/gui_runs/runs.db | SQLite database path |
| MAX_CONCURRENT_RUNS | 1 | Maximum simultaneous runs |

## Development

```bash
# Run with auto-reload
python -m fusion.cli.run_gui --reload --log-level debug

# Run tests
pytest fusion/api/tests/
```

## See Also

- [GUI Documentation](../../docs/gui/) - Full GUI design documentation
- [Backend Standards](../../docs/gui/06-backend-standards.md) - Coding conventions
