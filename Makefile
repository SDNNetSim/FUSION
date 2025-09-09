# FUSION Project Makefile
# Provides convenient commands for development and validation

.PHONY: help install lint test validate quick-validate clean check-env

# Default target
help:
	@echo "FUSION Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  install         Install dependencies"
	@echo "  check-env       Check if virtual environment is activated"
	@echo ""
	@echo "Validation (run before submitting PR):"
	@echo "  validate        Run complete PR validation (lint + test + cross-platform)"
	@echo "  quick-validate  Run quick validation (faster, stops on first failure)"
	@echo "  lint            Run only linting checks"
	@echo "  test            Run only unit tests"
	@echo "  cross-platform  Run only cross-platform compatibility test"
	@echo ""
	@echo "Utilities:"
	@echo "  clean           Clean up generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make validate          # Full validation before PR"
	@echo "  make quick-validate    # Quick check during development"
	@echo "  make lint              # Check code style only"

# Check if virtual environment is activated
check-env:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "⚠️  Warning: Virtual environment not detected"; \
		echo "   Consider running: source venv/bin/activate"; \
		echo ""; \
	else \
		echo "✅ Virtual environment active: $$VIRTUAL_ENV"; \
	fi

# Install dependencies
install: check-env
	@echo "Installing dependencies..."
	python -m pip install --upgrade pip setuptools wheel
	pip install torch==2.2.2
	# Platform-specific PyG installation
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "Installing PyG packages for macOS..."; \
		MACOSX_DEPLOYMENT_TARGET=10.15 pip install --no-build-isolation torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html; \
	else \
		echo "Installing PyG packages for Linux..."; \
		pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html; \
	fi
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

# Full PR validation
validate: check-env
	@echo "🚀 Running complete PR validation..."
	python tools/validate_pr.py

# Quick validation for development
quick-validate: check-env
	@echo "⚡ Running quick validation..."
	python tools/validate_pr.py --quick

# Lint only
lint: check-env
	@echo "🔍 Running linting checks..."
	python tools/validate_pr.py --lint-only

# Test only  
test: check-env
	@echo "🧪 Running unit tests..."
	python tools/validate_pr.py --test-only

# Cross-platform test only
cross-platform: check-env
	@echo "🌐 Running cross-platform compatibility test..."
	python tools/validate_pr.py --cross-platform-only

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
dev-check: quick-validate
style-check: lint