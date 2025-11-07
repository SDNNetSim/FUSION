# FUSION Project Makefile
# Provides convenient commands for development and validation

.PHONY: help install lint test validate clean check-env precommit-install precommit-run

# Default target
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
	@echo "  validate           Run all pre-commit checks on all files"
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

# Check if virtual environment is activated
check-env:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "⚠️  Warning: Virtual environment not detected"; \
		echo "   Consider running: source venv/bin/activate"; \
		echo ""; \
	else \
		echo "✅ Virtual environment active: $$VIRTUAL_ENV"; \
	fi

# Install dependencies using automated script
install: check-env
	@echo "Running automated installation..."
	./install.sh

# Install for development only (no PyG)
install-dev: check-env
	@echo "Installing development dependencies..."
	python -m pip install --upgrade pip setuptools wheel
	pip install -e .[dev]
	@echo "✅ Development dependencies installed"

# Manual installation (legacy)
install-manual: check-env
	@echo "Installing dependencies manually..."
	python -m pip install --upgrade pip setuptools wheel
	pip install torch==2.2.2
	# Platform-specific PyG installation
	@if [ "$$(uname)" = "Darwin" ]; then \
		if [ "$$(uname -m)" = "arm64" ]; then \
			echo "Installing PyG packages for Apple Silicon..."; \
			MACOSX_DEPLOYMENT_TARGET=11.0 pip install --no-build-isolation torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html; \
		else \
			echo "Installing PyG packages for Intel Mac..."; \
			MACOSX_DEPLOYMENT_TARGET=10.15 pip install --no-build-isolation torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html; \
		fi; \
	else \
		echo "Installing PyG packages for Linux..."; \
		pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html; \
	fi
	pip install torch-geometric==2.6.1
	pip install -e .[rl,dev]
	@echo "✅ Dependencies installed successfully"

# Install pre-commit hooks
precommit-install: check-env
	@echo "📦 Installing pre-commit hooks..."
	pip install pre-commit
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "✅ Pre-commit hooks installed"

# Run pre-commit on staged files
precommit-run: check-env
	@echo "🔍 Running pre-commit checks on staged files..."
	pre-commit run

# Full PR validation - run all pre-commit checks on all files
validate: check-env
	@echo "🚀 Running complete validation (pre-commit + tests)..."
	@echo "Running pre-commit checks on all files..."
	pre-commit run --all-files
	@echo "Running unit tests..."
	python -m pytest

# Lint only - run all pre-commit checks
lint: check-env
	@echo "🔍 Running all pre-commit checks on all files..."
	pre-commit run --all-files

# Test only
test: check-env
	@echo "🧪 Running unit tests..."
	python -m pytest

# Clean up generated files
clean:
	@echo "🧹 Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Legacy aliases for common workflows
pr-ready: validate
style-check: lint
