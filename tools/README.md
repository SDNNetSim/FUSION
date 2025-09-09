# Development Tools

This directory contains development utilities and tools for the FUSION project.

## Contents

### PR Validation Tools
- **`validate_pr.py`** - Complete Python validation script with full features
- **`validate_pr.sh`** - Simple shell script wrapper for basic validation
- **`VALIDATION.md`** - Comprehensive documentation for validation tools

## Usage

### Quick Start
```bash
# From project root directory
make validate          # Complete validation
make quick-validate    # Quick validation
make lint              # Linting only
```

### Direct Usage
```bash
# Python script (from project root)
python tools/validate_pr.py --quick

# Shell script (can be run from anywhere)
./tools/validate_pr.sh quick
```

## What These Tools Do

The validation tools run the same checks as our GitHub Actions CI/CD pipelines:

1. **Python Syntax Validation** - Ensures all Python files are syntactically correct
2. **Import Tests** - Verifies all modules can be imported
3. **Configuration Validation** - Tests config file loading
4. **Linting (Pylint)** - Code style and quality checks
5. **Unit Tests (Pytest)** - Complete test suite
6. **Cross-Platform Compatibility** - Platform-specific configuration tests

## Integration

These tools are designed to be used:
- **During development** - Quick validation while coding
- **Before committing** - Full validation to catch issues early
- **In IDEs** - Integration with development environments
- **In CI/CD** - Mirror the same checks that run in GitHub Actions

## Adding New Tools

When adding new development tools to this directory:

1. Follow the naming convention: `tool_name.py` or `tool_name.sh`
2. Add documentation to this README
3. Consider adding Makefile targets for convenience
4. Ensure tools work from the project root directory
5. Include help/usage information in the tool itself

## Support

For issues with these tools, see the main project documentation or create an issue in the project repository.