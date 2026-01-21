# CI/CD Pipeline

This document describes the continuous integration and deployment pipeline for FUSION GUI.

## Pipeline Overview

```
Push/PR
   │
   ├─► Backend Checks
   │     ├── Lint (ruff)
   │     ├── Type check (mypy) [optional]
   │     └── Tests (pytest)
   │
   ├─► Frontend Checks
   │     ├── Lint (ESLint)
   │     ├── Type check (tsc)
   │     ├── Tests (vitest)
   │     └── Build (vite)
   │
   └─► Integration
         └── Build wheel with static assets
```

## GitHub Actions Workflow

```yaml
# .github/workflows/gui.yml
name: GUI CI

on:
  push:
    branches: [main, develop]
    paths:
      - 'fusion/api/**'
      - 'frontend/**'
      - 'docs/gui/**'
      - '.github/workflows/gui.yml'
  pull_request:
    branches: [main, develop]
    paths:
      - 'fusion/api/**'
      - 'frontend/**'
      - 'docs/gui/**'

jobs:
  backend:
    name: Backend Checks
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -e ".[gui,dev]"

      - name: Lint with ruff
        run: |
          ruff check fusion/api/
          ruff format --check fusion/api/

      - name: Type check with mypy (optional)
        run: |
          mypy fusion/api/ --ignore-missing-imports || true
        continue-on-error: true

      - name: Run tests
        run: |
          pytest fusion/api/tests/ -v --tb=short

  frontend:
    name: Frontend Checks
    runs-on: ubuntu-latest

    defaults:
      run:
        working-directory: frontend

    steps:
      - uses: actions/checkout@v4

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        run: npm ci

      - name: Lint
        run: npm run lint

      - name: Type check
        run: npm run typecheck

      - name: Run tests
        run: npm run test:ci

      - name: Build
        run: npm run build

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: frontend-build
          path: frontend/dist/
          retention-days: 1

  e2e:
    name: E2E Tests (Playwright)
    needs: [backend, frontend]
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install Python dependencies
        run: pip install -e ".[gui,dev]"

      - name: Install frontend dependencies
        run: cd frontend && npm ci

      - name: Install Playwright browsers
        run: cd frontend && npx playwright install --with-deps chromium

      - name: Download frontend build
        uses: actions/download-artifact@v4
        with:
          name: frontend-build
          path: fusion/api/static/

      - name: Run E2E tests
        run: |
          cd frontend && npm run test:e2e
        env:
          # Use fake simulator for fast, deterministic tests
          FUSION_GUI_FAKE_SIMULATOR: "true"

      - name: Upload Playwright report
        uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: playwright-report
          path: frontend/playwright-report/
          retention-days: 7

  integration:
    name: Integration Build
    needs: [backend, frontend, e2e]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Download frontend build
        uses: actions/download-artifact@v4
        with:
          name: frontend-build
          path: fusion/api/static/

      - name: Build wheel
        run: |
          pip install build
          python -m build

      - name: Verify wheel contents
        run: |
          pip install dist/*.whl
          # Check that static files are included
          python -c "from pathlib import Path; import fusion.api; p = Path(fusion.api.__file__).parent / 'static' / 'index.html'; assert p.exists(), 'Static files missing'"

      - name: Upload wheel
        uses: actions/upload-artifact@v4
        with:
          name: fusion-wheel
          path: dist/*.whl
          retention-days: 7

  release:
    name: Release
    needs: [integration]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    steps:
      - uses: actions/checkout@v4

      - name: Download wheel
        uses: actions/download-artifact@v4
        with:
          name: fusion-wheel
          path: dist/

      # Add PyPI publish step when ready
      # - name: Publish to PyPI
      #   uses: pypa/gh-action-pypi-publish@release/v1
      #   with:
      #     password: ${{ secrets.PYPI_API_TOKEN }}
```

## Package Configuration

### pyproject.toml Additions

```toml
[project.optional-dependencies]
gui = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "sse-starlette>=1.8.0",
    "aiofiles>=23.2.0",
    # sqlalchemy already in base deps
]

[project.scripts]
fusion-gui = "fusion.cli.run_gui:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["fusion*"]

[tool.setuptools.package-data]
"fusion.api" = ["static/**/*"]
```

### MANIFEST.in

```
include fusion/api/static/**/*
recursive-include fusion/api/static *
```

## Build Process

### Development Build

No build needed - run dev servers directly:

```bash
# Backend
uvicorn fusion.api.main:app --reload --port 8765

# Frontend (separate terminal)
cd frontend && npm run dev
```

### Production Build

```bash
# Build frontend
cd frontend
npm ci
npm run build

# Copy to static directory
rm -rf ../fusion/api/static/*
cp -r dist/* ../fusion/api/static/

# Build Python wheel
cd ..
python -m build
```

### Makefile Targets

```makefile
# Makefile additions

.PHONY: frontend-dev
frontend-dev:
	cd frontend && npm run dev

.PHONY: api-dev
api-dev:
	uvicorn fusion.api.main:app --reload --port 8765

.PHONY: frontend-build
frontend-build:
	cd frontend && npm ci && npm run build
	rm -rf fusion/api/static/*
	cp -r frontend/dist/* fusion/api/static/

.PHONY: build-gui
build-gui: frontend-build
	python -m build

.PHONY: test-api
test-api:
	pytest fusion/api/tests/ -v

.PHONY: test-frontend
test-frontend:
	cd frontend && npm test

.PHONY: lint-api
lint-api:
	ruff check fusion/api/
	ruff format --check fusion/api/

.PHONY: lint-frontend
lint-frontend:
	cd frontend && npm run lint

.PHONY: validate-gui
validate-gui: lint-api lint-frontend test-api test-frontend
	@echo "All GUI checks passed"
```

## Wheel vs Source Distribution

### Wheel (Recommended for Users)

- Pre-built, fast to install
- Includes compiled frontend assets
- No Node.js required

```bash
pip install fusion-6.1.0-py3-none-any.whl
```

### Source Distribution (sdist)

- Requires Node.js to build frontend during install
- Needed for:
  - Users who want to modify frontend
  - Platforms not supported by wheel

For sdist to work, we need a build hook:

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

# Note: sdist won't include built frontend
# Users building from source need Node.js
```

**Recommendation**: Always distribute wheels with pre-built frontend.

## CI Gates

### PR Merge Requirements

| Check | Required | Blocking |
|-------|----------|----------|
| Backend lint (ruff) | Yes | Yes |
| Backend tests | Yes | Yes |
| Frontend lint (ESLint) | Yes | Yes |
| Frontend typecheck | Yes | Yes |
| Frontend tests | Yes | Yes |
| Frontend build | Yes | Yes |
| E2E tests (Playwright) | Yes (M2+) | Yes |
| Accessibility (axe) | Advisory (M2), Blocking (M4) | M4+ |
| Backend mypy | No | No |

### E2E Flake Policy

- Playwright retries: **1** (configured in `playwright.config.ts`)
- Tests that pass only on retry are considered flaky
- Repeated flakes (2+ in a week) require a tracking issue labeled `flaky-test`
- See [10-testing.md](10-testing.md#flake-policy) for handling guidelines

### Branch Protection Rules

For `main` and `develop`:

```
- Require pull request reviews (1)
- Require status checks to pass
  - Backend Checks
  - Frontend Checks
- Require branches to be up to date
- Do not allow bypassing settings
```

## Release Process

### Version Bumping

1. Update version in `pyproject.toml`
2. Update version in `frontend/package.json`
3. Update CHANGELOG.md
4. Create git tag

```bash
# Example
git checkout main
git pull
# Update versions
git add .
git commit -m "chore: bump version to 6.1.0"
git tag v6.1.0
git push origin main --tags
```

### Release Checklist

```markdown
## Release Checklist

- [ ] All CI checks pass on `main`
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Version bumped in frontend/package.json
- [ ] Git tag created
- [ ] Wheel built and tested locally
- [ ] GitHub Release created
- [ ] PyPI package published (when ready)
```

## Local Development Shortcuts

### VS Code Tasks

```json
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "API Dev Server",
      "type": "shell",
      "command": "uvicorn fusion.api.main:app --reload --port 8765",
      "group": "build",
      "problemMatcher": []
    },
    {
      "label": "Frontend Dev Server",
      "type": "shell",
      "command": "npm run dev",
      "options": { "cwd": "${workspaceFolder}/frontend" },
      "group": "build",
      "problemMatcher": []
    }
  ]
}
```

### Pre-commit Hooks

Add GUI checks to existing pre-commit config:

```yaml
# .pre-commit-config.yaml additions
repos:
  - repo: local
    hooks:
      - id: frontend-lint
        name: Frontend Lint
        entry: bash -c 'cd frontend && npm run lint'
        language: system
        files: ^frontend/
        pass_filenames: false
```
