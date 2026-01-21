# Contributing to FUSION GUI

This guide covers how to contribute to the GUI components of FUSION.

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 20+ (for development only, not required for users)
- Git

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/SDNNetSim/FUSION.git
cd FUSION

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install Python dependencies (including GUI extras)
pip install -e ".[gui,dev]"

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Running in Development Mode

**Terminal 1: Backend (with hot reload)**
```bash
make api-dev
# or directly:
uvicorn fusion.api.main:app --reload --port 8765
```

**Terminal 2: Frontend (with hot reload)**
```bash
make frontend-dev
# or directly:
cd frontend && npm run dev
```

The frontend dev server runs on `http://localhost:5173` and proxies API requests to `http://localhost:8765`.

### IDE Setup

**For Python (PyCharm, VS Code):**
- Open the repository root
- Configure interpreter to use the `venv`
- Enable ruff for linting

**For Frontend (WebStorm, VS Code):**
- Open the `frontend/` directory as a separate project/workspace
- VS Code will auto-detect TypeScript and ESLint configs

## Branching Strategy

```
main
  └── develop
        ├── feature/gui-run-list
        ├── feature/gui-log-viewer
        └── fix/gui-sse-reconnect
```

- `main`: Stable releases only
- `develop`: Integration branch for GUI features
- `feature/*`: Feature branches (branch from `develop`)
- `fix/*`: Bug fix branches

### Branch Naming

```
feature/gui-<description>    # New features
fix/gui-<description>        # Bug fixes
refactor/gui-<description>   # Code improvements
docs/gui-<description>       # Documentation
```

## Pull Request Process

### Before Opening a PR

1. **Run all checks locally:**
   ```bash
   make validate-gui
   ```

2. **Update documentation** if you:
   - Added/changed API endpoints
   - Added new components
   - Changed configuration options

3. **Add tests** for:
   - New API endpoints (backend)
   - New components with complex logic (frontend)

### PR Checklist

Copy this checklist into your PR description:

```markdown
## PR Checklist

### General
- [ ] Branch is up-to-date with `develop`
- [ ] All CI checks pass
- [ ] No unrelated changes included

### Backend (if applicable)
- [ ] Added/updated Pydantic schemas
- [ ] Added/updated API tests
- [ ] Updated API documentation in `docs/gui/04-api.md`
- [ ] No new ruff/mypy warnings

### Frontend (if applicable)
- [ ] Added/updated TypeScript types
- [ ] Added component tests (if complex)
- [ ] No new ESLint warnings
- [ ] Tested in both light and dark mode
- [ ] Keyboard accessible
- [ ] No critical axe violations on affected pages

### Documentation
- [ ] Updated relevant docs in `docs/gui/`
- [ ] Added ADR if architectural decision (optional)
```

### PR Size Guidelines

- **Small PRs** (< 200 lines): Can be merged quickly
- **Medium PRs** (200-500 lines): Need thorough review
- **Large PRs** (> 500 lines): Consider splitting

### Review Process

1. At least one approval required
2. All CI checks must pass
3. No unresolved conversations
4. Squash merge to `develop`

## Coding Standards

### Python

Follow existing FUSION standards (see `CODING_STANDARDS.md`), plus:

- Use type hints on all functions
- Use Pydantic for request/response validation
- Use dependency injection for database sessions
- Log at appropriate levels (debug, info, warning, error)

### TypeScript/React

See [05-frontend-standards.md](05-frontend-standards.md) for detailed conventions.

Key points:
- Strict TypeScript (`"strict": true`)
- Functional components with hooks
- TanStack Query for server state
- Zustand for UI state only
- Named exports (not default)

### Commit Messages

Follow conventional commits:

```
feat(gui): add run cancellation button
fix(gui): handle SSE reconnection on network error
docs(gui): update API documentation for artifacts
refactor(gui): extract LogViewer into separate component
test(gui): add tests for run creation flow
```

Prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code change that doesn't fix bug or add feature
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Testing

### Backend Tests

```bash
# Run all API tests
pytest fusion/api/tests/ -v

# Run with coverage
pytest fusion/api/tests/ --cov=fusion/api --cov-report=html
```

**What to test:**
- API endpoint responses (status codes, response shapes)
- Service layer logic (run creation, cancellation)
- Security (path traversal prevention)

### Frontend Tests

```bash
cd frontend

# Run tests
npm test

# Run with coverage
npm run test:coverage

# Watch mode
npm run test:watch
```

**What to test:**
- Component rendering (does it render without crashing?)
- User interactions (clicks, form submissions)
- Conditional rendering (loading, error, empty states)

### End-to-End Tests

Playwright tests in `frontend/e2e/` cover critical user flows:

```bash
# Run E2E tests (optional locally; required in CI for M2+)
cd frontend && npm run test:e2e

# Run with browser UI for debugging
cd frontend && npm run test:e2e:ui
```

E2E tests use a fake simulator (`FUSION_GUI_FAKE_SIMULATOR=true`) for fast, deterministic execution. See [10-testing.md](10-testing.md) for:

- **Testing Ladder**: Which test type to write for each PR
- **Flake Policy**: How to handle intermittent failures
- **Accessibility**: Running axe checks in Playwright

## Adding New Features

### Adding a New API Endpoint

1. **Define schema** in `fusion/api/schemas/`:
   ```python
   class MyRequest(BaseModel):
       field: str

   class MyResponse(BaseModel):
       result: str
   ```

2. **Add route** in `fusion/api/routes/`:
   ```python
   @router.post("/my-endpoint", response_model=MyResponse)
   def my_endpoint(data: MyRequest, db: Session = Depends(get_db)):
       # Implementation
       return MyResponse(result="done")
   ```

3. **Add service logic** if complex (in `fusion/api/services/`)

4. **Add tests** in `fusion/api/tests/`

5. **Update docs** in `docs/gui/04-api.md`

### Adding a New UI Component

1. **Create component** in `frontend/src/components/<category>/`:
   ```typescript
   // MyComponent.tsx
   interface MyComponentProps {
     value: string;
   }

   export function MyComponent({ value }: MyComponentProps) {
     return <div>{value}</div>;
   }
   ```

2. **Add API function** if needed in `frontend/src/api/`

3. **Add hook** if complex state in `frontend/src/hooks/`

4. **Add to page** in `frontend/src/pages/`

5. **Add tests** if complex logic

### Adding an ADR

For significant architectural decisions, create an ADR:

```bash
# Create new ADR
touch docs/gui/adr/NNNN-<title>.md
```

Template:
```markdown
# NNNN: Title

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
What is the issue we're addressing?

## Decision
What have we decided to do?

## Consequences
What are the positive and negative outcomes?
```

## Makefile Targets

```makefile
# Development
make api-dev          # Run backend with hot reload
make frontend-dev     # Run frontend with hot reload

# Building
make frontend-build   # Build frontend for production
make build-gui        # Build complete GUI package

# Testing
make test-api         # Run backend tests
make test-frontend    # Run frontend tests
make validate-gui     # Run all GUI checks

# Linting
make lint-api         # Lint backend
make lint-frontend    # Lint frontend
```

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Features**: Open a GitHub Issue describing the use case

## Code of Conduct

Be respectful and constructive. We're all here to build something useful.
