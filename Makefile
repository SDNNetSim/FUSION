# FUSION Project Makefile (cross-platform: Windows/macOS/Linux)
# Uses Python for portability (no bashisms like uname/find/rm/if [ ])

.PHONY: help install install-dev install-manual lint test validate clean check-env \
        precommit-install precommit-run pr-ready style-check

# -------- Config --------
PY ?= python
PIP := $(PY) -m pip

# Detect Windows via Make's built-in OS variable (set by Windows environments)
ifeq ($(OS),Windows_NT)
  IS_WINDOWS := 1
else
  IS_WINDOWS := 0
endif

# -------- Help --------
help:
	@echo "FUSION Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  install            Install all dependencies (automated)"
	@echo "  install-dev        Install development tools only"
	@echo "  install-manual     Manual installation (advanced users)"
	@echo "  precommit-install  Install pre-commit hooks"
	@echo "  check-env          Check if virtual environment is activated"
	@echo ""
	@echo "Validation (run before submitting PR):"
	@echo "  validate           Run all pre-commit checks on all files + tests"
	@echo "  lint               Run pre-commit checks on all files"
	@echo "  precommit-run      Run pre-commit on staged files only"
	@echo "  test               Run unit tests with pytest"
	@echo ""
	@echo "Utilities:"
	@echo "  clean              Clean up generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make validate          # Full validation before PR"
	@echo "  make lint              # Run all linting checks"
	@echo "  make test              # Run tests only"

# -------- Environment --------
check-env:
	@$(PY) -c "import os,sys; v=os.environ.get('VIRTUAL_ENV'); \
print('✅ Virtual environment active: '+v if v else '⚠️  Warning: Virtual environment not detected\\n   Windows (PowerShell): .\\\\venv\\\\Scripts\\\\Activate.ps1\\n   Windows (cmd):        .\\\\venv\\\\Scripts\\\\activate.bat\\n   macOS/Linux:          source venv/bin/activate\\n'); \
sys.exit(0)"

# -------- Install --------
install: check-env
ifeq ($(IS_WINDOWS),1)
	@echo "Running Windows installation..."
	@echo "NOTE: install.sh is not runnable in native Windows shells."
	@echo "Installing base requirements via pip instead..."
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements.txt
	@echo "✅ Base dependencies installed (Windows)."
	@echo "If you need the exact install.sh behavior, run it via WSL/Git-Bash, or add an install.ps1 equivalent."
else
	@echo "Running automated installation..."
	./install.sh
endif

install-dev: check-env
	@echo "Installing development dependencies..."
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -e .[dev]
	@echo "✅ Development dependencies installed"

# Manual install (kept mostly as-is, but Windows-safe)
install-manual: check-env
ifeq ($(IS_WINDOWS),1)
	@echo "Installing dependencies manually on Windows..."
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install torch==2.2.2
	@echo "⚠️  PyG wheels can be tricky on Windows depending on Python/CUDA."
	@echo "    Recommended: follow PyTorch Geometric's official install instructions for your setup,"
	@echo "    then run:"
	@echo "      pip install torch-geometric==2.6.1"
	@echo "      pip install -e .[rl,dev]"
	@echo "✅ Manual (partial) install complete on Windows."
else
	@echo "Installing dependencies manually..."
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install torch==2.2.2
	# Platform-specific PyG installation (macOS vs Linux)
	@$(PY) -c "import platform; print(platform.system())" | grep -q Darwin && \
		( \
		  echo "Installing PyG packages for macOS..."; \
		  MACOSX_DEPLOYMENT_TARGET=10.15 $(PIP) install --no-build-isolation torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html; \
		) || \
		( \
		  echo "Installing PyG packages for Linux..."; \
		  $(PIP) install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html; \
		)
	@$(PIP) install torch-geometric==2.6.1
	@$(PIP) install -e .[rl,dev]
	@echo "✅ Dependencies installed successfully"
endif

# -------- Pre-commit --------
precommit-install: check-env
	@echo "📦 Installing pre-commit hooks..."
	@$(PIP) install pre-commit
	@pre-commit install
	@pre-commit install --hook-type commit-msg
	@echo "✅ Pre-commit hooks installed"

precommit-run: check-env
	@echo "🔍 Running pre-commit checks on staged files..."
	@pre-commit run

# -------- Validation --------
validate: check-env
	@echo "🚀 Running complete validation (pre-commit + tests)..."
	@echo "Running pre-commit checks on all files..."
	@pre-commit run --all-files
	@echo "Running unit tests..."
	@$(PY) -m pytest

lint: check-env
	@echo "🔍 Running all pre-commit checks on all files..."
	@pre-commit run --all-files

test: check-env
	@echo "🧪 Running unit tests..."
	@$(PY) -m pytest

# -------- Clean --------
# Cross-platform clean implemented in Python (no find/rm)
clean:
	@echo "🧹 Cleaning up generated files..."
	@$(PY) -c "from pathlib import Path; import shutil; \
root=Path('.'); \
files = ['.coverage']; \
dirs  = ['.pytest_cache','.mypy_cache']; \
# delete *.pyc
for p in root.rglob('*.pyc'): \
    p.unlink(missing_ok=True); \
# delete __pycache__ and *.egg-info
for d in root.rglob('__pycache__'): \
    shutil.rmtree(d, ignore_errors=True); \
for d in root.rglob('*.egg-info'): \
    shutil.rmtree(d, ignore_errors=True); \
# delete known files/dirs
for f in files: \
    (root/f).unlink(missing_ok=True); \
for d in dirs: \
    shutil.rmtree(root/d, ignore_errors=True); \
print('✅ Cleanup complete')"

# Legacy aliases
pr-ready: validate
style-check: lint
