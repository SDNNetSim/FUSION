# FUSION Project Makefile
# Provides convenient commands for development and validation

.PHONY: help install lint test validate quick-validate clean check-env format lint-new test-new analyze profile setup-hooks check-all

# Default target
help:
	@echo "FUSION Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  install         Install dependencies"
	@echo "  check-env       Check if virtual environment is activated"
	@echo ""
	@echo "Code Quality (new development workflow):"
	@echo "  format          Format code with black and isort"
	@echo "  lint-new        Run modern linting (ruff + mypy)"
	@echo "  test-new        Run tests with coverage"
	@echo "  analyze         Run dependency and dead code analysis"
	@echo "  profile         Run performance profiling"
	@echo "  setup-hooks     Install and update pre-commit hooks"
	@echo "  check-all       Run all code quality checks"
	@echo ""
	@echo "Validation (legacy - run before submitting PR):"
	@echo "  validate        Run complete PR validation (lint + test + cross-platform)"
	@echo "  quick-validate  Run quick validation (faster, stops on first failure)"
	@echo "  lint            Run only linting checks (legacy)"
	@echo "  test            Run only unit tests (legacy)"
	@echo "  cross-platform  Run only cross-platform compatibility test"
	@echo ""
	@echo "Utilities:"
	@echo "  clean           Clean up generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make check-all         # Run all new code quality checks"
	@echo "  make format            # Format code before committing"
	@echo "  make lint-new          # Modern linting and type checking"
	@echo "  make validate          # Full validation before PR (legacy)"
	@echo "  make quick-validate    # Quick check during development (legacy)"

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

# New development workflow targets
format: check-env
	@echo "🎨 Formatting code..."
	black fusion/ tests/
	isort fusion/ tests/
	@echo "✅ Code formatting complete"

lint-new: check-env
	@echo "🔍 Running modern linting..."
	ruff check fusion/
	mypy fusion/
	@echo "✅ Modern linting complete"

test-new: check-env
	@echo "🧪 Running tests with coverage..."
	pytest --cov=fusion --cov-report=html --cov-report=term-missing
	@echo "✅ Tests with coverage complete"

analyze: check-env
	@echo "📊 Running code analysis..."
	./scripts/analyze_dependencies.sh
	@echo "✅ Code analysis complete"

profile: check-env
	@echo "⚡ Running performance profiling..."
	./scripts/profile_performance.sh
	@echo "✅ Performance profiling complete"

setup-hooks: check-env
	@echo "🔗 Setting up pre-commit hooks..."
	pre-commit install
	pre-commit autoupdate
	@echo "✅ Pre-commit hooks setup complete"

check-all: format lint-new test-new analyze
	@echo "🎯 All code quality checks complete!"
	@echo ""
	@echo "📊 Summary:"
	@echo "  ✅ Code formatting"
	@echo "  ✅ Modern linting (ruff + mypy)"
	@echo "  ✅ Tests with coverage"
	@echo "  ✅ Dependency and dead code analysis"
	@echo ""
	@echo "🚀 Your code is ready for review!"

# Legacy aliases for common workflows
pr-ready: validate
dev-check: quick-validate
style-check: lint
