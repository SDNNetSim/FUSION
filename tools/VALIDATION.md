# PR Validation Guide

This guide explains how to validate your pull requests locally before submitting them, ensuring they will pass all CI/CD pipeline checks.

## Quick Start

The fastest way to validate your PR:

```bash
# Complete validation (recommended before submitting PR)
make validate

# Quick validation during development
make quick-validate

# Lint only (fastest)
make lint
```

## Available Tools

We provide multiple ways to run validation:

### 1. Makefile (Recommended)
```bash
make validate         # Complete validation
make quick-validate   # Quick check (stops on first failure)
make lint            # Linting only
make test            # Unit tests only
make cross-platform  # Cross-platform compatibility test
```

### 2. Python Script (Full-featured)
```bash
python validate_pr.py                 # Complete validation
python validate_pr.py --quick         # Quick validation
python validate_pr.py --lint-only     # Linting only
python validate_pr.py --test-only     # Unit tests only
python validate_pr.py --cross-platform-only  # Cross-platform test
```

### 3. Shell Script (Simple)
```bash
./validate_pr.sh            # Complete validation
./validate_pr.sh quick      # Quick validation
./validate_pr.sh lint       # Linting only
./validate_pr.sh test       # Unit tests only
./validate_pr.sh cross-platform  # Cross-platform test
```

## What Gets Validated

The validation process runs the same checks as our GitHub Actions CI/CD pipelines:

### 1. **Python Syntax Validation**
- Checks all Python files for syntax errors
- Ensures code can be compiled

### 2. **Import Tests**
- Verifies all key modules can be imported
- Catches missing dependencies or circular imports

### 3. **Configuration Validation**
- Tests all config files in `fusion/configs/templates/`
- Ensures config loading works properly

### 4. **Linting (Pylint)**
- Runs pylint on `fusion/` and `tests/` packages
- Checks code style and quality
- Matches `.github/workflows/pylint.yml`

### 5. **Unit Tests (Pytest)**
- Runs complete test suite
- Matches `.github/workflows/unittests.yml`

### 6. **Cross-Platform Compatibility**
- Tests the same command that runs in GitHub Actions
- Ensures configuration parsing works across platforms
- Matches `.github/workflows/cross_platform.yml`

## Prerequisites

### Dependencies
```bash
# Install required tools
pip install pylint pytest

# Or install all project dependencies
make install
```

### Virtual Environment (Recommended)
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
```

## Validation Workflow

### During Development
```bash
# Quick check while coding
make quick-validate

# Check just linting
make lint
```

### Before Committing
```bash
# Full validation
make validate
```

### Before Submitting PR
```bash
# Complete validation + cleanup
make clean
make validate
```

## Understanding Results

### ✅ Success
When all checks pass, you'll see:
```
🎉 ALL CHECKS PASSED!
Your PR is ready for submission.
```

### ❌ Failures
When checks fail, you'll see detailed output showing:
- Which check failed
- Specific error messages
- File locations of issues

Example failure output:
```
❌ FAILED CHECKS (2):
  • Pylint on fusion package
  • Unit tests

Please fix the issues above before submitting your PR.
```

### ⚠️ Warnings
Some checks may produce warnings (like pylint style warnings) but still pass. These are shown in yellow.

## Common Issues & Solutions

### Pylint Errors
```bash
# Run pylint directly to see detailed output
pylint fusion/
pylint tests/

# Focus on specific files
pylint fusion/core/simulation.py
```

### Test Failures
```bash
# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_engine.py

# Run tests and stop on first failure
pytest -x
```

### Import Errors
```bash
# Check if you're in the right directory
pwd  # Should be FUSION project root

# Verify virtual environment
echo $VIRTUAL_ENV

# Check if dependencies are installed
pip list | grep pylint
```

### Cross-Platform Issues
The cross-platform test may show numpy architecture warnings locally - this is expected. The test passes if it gets past configuration parsing.

## Integration with Git Workflow

### Pre-commit Hook (Optional)
Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
make quick-validate
```

### Git Alias (Optional)
```bash
git config alias.validate '!make validate'
# Usage: git validate
```

## Performance Tips

### Speed Up Validation
- Use `--quick` flag to stop on first failure
- Use specific checks (`--lint-only`, `--test-only`) during development
- Run `make clean` periodically to remove cache files

### Parallel Development
- Run `make lint` while coding
- Run `make test` after making changes
- Run full `make validate` before committing

## Troubleshooting

### Command Not Found
```bash
# Ensure you're in project root
ls Makefile validate_pr.py

# Check if tools are installed
which python pylint pytest
```

### Permission Denied
```bash
# Make scripts executable
chmod +x validate_pr.sh
```

### Virtual Environment Issues
```bash
# Check if venv is active
echo $VIRTUAL_ENV

# Activate venv
source venv/bin/activate

# Verify dependencies
pip list
```

### Path Issues
```bash
# Ensure you're in the right directory
cd /path/to/FUSION
pwd

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

## CI/CD Pipeline Mapping

Our local validation mirrors the GitHub Actions workflows:

| Local Check | GitHub Workflow | Purpose |
|-------------|----------------|---------|
| `python validate_pr.py --lint-only` | `.github/workflows/pylint.yml` | Code quality |
| `python validate_pr.py --test-only` | `.github/workflows/unittests.yml` | Unit testing |
| `python validate_pr.py --cross-platform-only` | `.github/workflows/cross_platform.yml` | Platform compatibility |

## Advanced Usage

### Custom Configuration
The validation script respects environment variables:
```bash
# Skip certain tests
export SKIP_SLOW_TESTS=1
make validate

# Use different config
export CONFIG_PATH="path/to/config.ini"
make cross-platform
```

### Integration with IDEs
Most IDEs can run these commands:
- **VS Code**: Configure tasks in `.vscode/tasks.json`
- **PyCharm**: Add external tools
- **Vim/Neovim**: Add key mappings

### Continuous Validation
```bash
# Watch for changes and auto-validate
while inotifywait -e modify -r fusion/ tests/; do
    make quick-validate
done
```

## Support

If you encounter issues with the validation tools:

1. Check this guide first
2. Ensure all prerequisites are installed
3. Try the troubleshooting steps
4. Ask for help in the project's communication channels

Remember: The goal is to catch issues early and ensure your PR will pass CI/CD checks. These tools save time by finding problems before submission!
