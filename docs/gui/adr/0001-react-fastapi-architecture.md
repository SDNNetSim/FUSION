# ADR 0001: React + FastAPI Architecture

## Status

Accepted

## Date

2024-01-15

## Context

FUSION is a Python-based optical network simulator that currently uses a CLI interface. We want to add a graphical user interface to:

- Allow users to configure and run simulations without CLI knowledge
- Provide real-time monitoring of long-running simulations
- Enable browsing and downloading of output artifacts
- Visualize network topology and results

**Constraints:**
- Primary use is local, single-user, no accounts/auth required
- Simulations run from minutes to weeks
- Users should not need to install Node.js
- We may add HPC job submission in the future
- Browser-based UI is acceptable (no desktop app requirement)

**Options Considered:**

1. **React SPA + FastAPI backend** - Web UI served by Python
2. **Electron desktop app** - Native app with embedded Chromium
3. **Tauri desktop app** - Rust-based native wrapper
4. **PyQt5/PySide6** - Native Python GUI
5. **PyQt5 WebEngine** - Embedded web view in Qt

## Decision

We will use **React SPA + FastAPI backend** architecture.

The FastAPI server:
- Serves the pre-built React static files
- Provides REST API for CRUD operations
- Streams real-time updates via Server-Sent Events (SSE)
- Manages simulation subprocesses

The React frontend:
- Built with Vite, TypeScript, and Tailwind CSS
- Uses TanStack Query for server state management
- Ships pre-built inside the Python wheel

## Rationale

### Why React + FastAPI?

| Factor | React + FastAPI | Electron | Tauri | PyQt5 | Qt WebEngine |
|--------|----------------|----------|-------|-------|--------------|
| Bundle size | ~5MB static | 200MB+ | 20MB+ | Varies | Varies |
| User install | `pip install` | Download installer | Download installer | `pip install` | `pip install` |
| Node.js needed | No (pre-built) | Yes (dev) | Yes (dev) | No | No |
| Cross-platform | Yes | Yes | Yes | Issues | Issues |
| Modern UI | Excellent | Excellent | Excellent | Limited | Requires web skills |
| Developer pool | Large | Large | Small | Small | Small |
| HPC extensibility | Excellent | Poor | Poor | Poor | Poor |

**Key advantages of React + FastAPI:**

1. **Zero user friction**: `pip install fusion[gui]` then `fusion gui`. No Node.js, no Electron download, no native installers.

2. **Clean subprocess isolation**: Long-running simulations in separate processes. If a simulation crashes or leaks memory, the server remains stable.

3. **Future HPC support**: REST API naturally extends to remote job submission. The same UI can talk to local or remote backends.

4. **Modern developer experience**: React and FastAPI are industry-standard with excellent documentation, tooling, and developer availability.

5. **Easy updates**: Update Python package, get new UI automatically. No separate app update mechanism.

### Why NOT Electron/Tauri?

- **Installation complexity**: Users must download and run installers, deal with OS security warnings, manage separate update mechanisms.
- **Bundle size**: Electron bundles entire Chromium (~200MB). Tauri is smaller but requires Rust toolchain for development.
- **Network model**: Desktop apps connecting to Python backend adds complexity. Browser already knows how to talk HTTP.
- **Not needed**: We don't need desktop integration (system tray, file associations). Browser on localhost is fine.

### Why NOT PyQt5?

- **UI quality**: Building modern, responsive UIs in Qt is harder than in React. Qt widgets look dated without significant effort.
- **Cross-platform issues**: PyQt5 has platform-specific bugs, especially on macOS. WebEngine is particularly problematic.
- **Developer expertise**: Web developers are more common than Qt developers.

## Consequences

### Positive

- Simple installation and distribution
- Modern, maintainable frontend codebase
- Clear separation between UI and backend
- Easy to add features (API endpoints + UI components)
- Familiar technology for most developers

### Negative

- Two build systems (Python + Node.js for development)
- Need to bundle built React in Python wheel
- Browser-based, so no deep OS integration
- SSE reconnection needs handling for long sessions

### Neutral

- Monorepo structure with `frontend/` directory
- Developers may open subfolders in different IDEs
- Need pre-commit hooks for both Python and JS linting

## Implementation Notes

1. **Static file serving**: FastAPI serves React build from `fusion/api/static/`. SPA fallback ensures deep links work.

2. **Process management**: Simulations run as subprocesses in new process sessions (`start_new_session=True`). This enables clean cancellation of entire process trees.

3. **Real-time updates**: SSE for logs and progress. File-based IPC (stdout to file, tail for streaming) avoids pipe buffer deadlocks.

4. **Build pipeline**: CI builds frontend, copies to `fusion/api/static/`, then builds Python wheel. Users get pre-built UI.

## Related Decisions

- ADR 0002: TanStack Query for server state (planned)
- ADR 0003: SQLite for run persistence (planned)
