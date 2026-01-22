# Architecture

## System Overview

FUSION GUI uses a **React SPA + FastAPI backend** architecture. The backend serves both the API and the pre-built React static files, enabling single-command startup for end users.

```
See: diagrams/architecture.txt for visual representation
```

## Why This Architecture?

| Alternative | Why Not |
|------------|---------|
| Electron/Tauri | Adds native packaging complexity, 200MB+ bundle size, not needed since browser is acceptable |
| PyQt5 WebEngine | Cross-platform Qt issues, ties to Qt ecosystem, complicates pip distribution |
| Separate repos | API drift, coordinated releases, users must install two things |

**React + FastAPI wins because:**
- Single `pip install` includes everything
- FastAPI serves static files + API on one port
- Subprocess isolation keeps server stable during long runs
- REST API naturally extends to future HPC job submission
- Industry-standard tooling with excellent documentation

## Component Architecture

### Backend (Python)

```
fusion/api/
├── __init__.py
├── main.py                 # FastAPI app, lifespan, static serving
├── config.py               # Settings via pydantic-settings
├── dependencies.py         # Dependency injection (DB sessions, etc.)
├── routes/
│   ├── __init__.py
│   ├── runs.py             # /api/runs endpoints
│   ├── configs.py          # /api/configs endpoints
│   ├── artifacts.py        # /api/artifacts endpoints
│   ├── topology.py         # /api/topology endpoints
│   └── system.py           # /api/health, /api/version
├── schemas/
│   ├── __init__.py
│   ├── run.py              # Run request/response models
│   ├── config.py           # Config models
│   └── common.py           # Shared models (pagination, errors)
├── services/
│   ├── __init__.py
│   ├── run_manager.py      # Job lifecycle, process management
│   ├── config_service.py   # Config loading/validation
│   ├── progress_watcher.py # Progress file monitoring
│   └── artifact_service.py # Safe artifact access
├── db/
│   ├── __init__.py
│   ├── database.py         # SQLite connection, session factory
│   └── models.py           # SQLAlchemy ORM models
└── static/                 # Built React files (generated, gitignored)
    └── .gitkeep
```

### Frontend (React)

```
frontend/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── index.html
├── src/
│   ├── main.tsx            # Entry point
│   ├── App.tsx             # Root component, routing
│   ├── api/
│   │   ├── client.ts       # Axios/fetch wrapper, error handling
│   │   ├── runs.ts         # Run API functions
│   │   ├── configs.ts      # Config API functions
│   │   └── types.ts        # Generated/shared API types
│   ├── components/
│   │   ├── ui/             # Primitive UI components (Button, Card, etc.)
│   │   ├── layout/         # Layout components (Sidebar, Header)
│   │   ├── runs/           # Run-specific components
│   │   ├── config/         # Config editor components
│   │   ├── artifacts/      # File browser components
│   │   └── topology/       # Network visualization
│   ├── pages/
│   │   ├── RunListPage.tsx
│   │   ├── RunDetailPage.tsx
│   │   ├── NewRunPage.tsx
│   │   ├── ConfigEditorPage.tsx
│   │   └── TopologyPage.tsx
│   ├── hooks/
│   │   ├── useRuns.ts      # Run queries/mutations
│   │   ├── useSSE.ts       # SSE connection hook
│   │   └── useProgress.ts  # Progress subscription
│   ├── stores/
│   │   └── ui.ts           # UI state (sidebar, theme)
│   ├── lib/
│   │   └── utils.ts        # Utility functions
│   └── styles/
│       └── globals.css     # Tailwind imports, custom styles
└── public/
    └── favicon.ico
```

## Repository Layout (Monorepo)

```
FUSION/
├── fusion/                 # Python package (existing)
│   ├── api/                # NEW: FastAPI server
│   ├── cli/
│   │   └── run_gui.py      # MODIFY: Launch GUI server
│   └── ...
├── frontend/               # NEW: React application
├── docs/
│   └── gui/                # This documentation
├── data/
│   └── gui_runs/           # NEW: GUI-managed run data
├── pyproject.toml          # MODIFY: Add [gui] extras
├── Makefile                # MODIFY: Add GUI targets
└── .github/
    └── workflows/
        └── gui.yml         # NEW: GUI CI workflow
```

## Why Monorepo?

| Benefit | Explanation |
|---------|-------------|
| Atomic commits | Frontend + backend changes in one PR |
| No version drift | Single version number, coordinated releases |
| Simpler CI | One pipeline builds and tests everything |
| Shared tooling | Common pre-commit, Makefile targets |
| IDE flexibility | Devs can open `frontend/` in WebStorm, root in PyCharm |

## Communication Model

### REST for CRUD Operations

Standard request/response for:
- Creating/listing/deleting runs
- Fetching configs and templates
- Downloading artifacts

### SSE for Real-Time Updates

Server-Sent Events for:
- Log streaming (tail of `sim.log`)
- Progress updates (watching `progress.jsonl`)
- Run status changes

**Why SSE over WebSocket?**
- One-way data flow (server to client only)
- Auto-reconnect built into browsers
- Simpler implementation, no connection state
- Works through corporate proxies

## Process Model

### Simulation Subprocess

```
FastAPI Server (PID 1000)
    |
    +-- subprocess.Popen() with start_new_session=True
            |
            +-- fusion-sim process (PID 2000, PGID 2000)
                    |
                    +-- multiprocessing.Pool workers (PIDs 2001, 2002, ...)
```

**Critical**: Simulations run in a **new process session** so we can kill the entire process tree with `os.killpg()`.

### Concurrency Model

- **MVP**: Sequential runs only (one active run at a time)
- **Later**: Configurable concurrency limit, job queue

### Persistence

| Data | Storage | Survives Restart? |
|------|---------|-------------------|
| Run metadata | SQLite (`gui_runs/runs.db`) | Yes |
| Run status | SQLite + filesystem | Yes |
| Logs | Filesystem (`runs/<id>/logs/`) | Yes |
| Progress | Filesystem (`runs/<id>/progress.jsonl`) | Yes |
| Artifacts | Filesystem (`runs/<id>/output/`) | Yes |

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | Full support | Primary development platform |
| macOS | Full support | Tested on Apple Silicon and Intel |
| Windows | Supported (M2+) | Requires platform-specific process termination |

**Process termination by platform:**

- **POSIX (Linux/macOS)**: Use `start_new_session=True` + `os.killpg(pgid, signal)` to kill entire process tree
- **Windows**: Use Job Objects or `taskkill /T /F /PID` fallback (see [06-backend-standards.md](06-backend-standards.md))

Windows support is tested in CI starting M2. MVP (M1) may work on Windows but is not the primary target.

## Security Considerations

1. **Localhost only**: Bind to `127.0.0.1`, not `0.0.0.0`
2. **No auth**: Single-user local use, no credentials needed
3. **Path traversal protection**: Artifact downloads validate paths are within run directory, reject symlinks escaping the run directory
4. **No secrets in logs**: Config sanitization before display
