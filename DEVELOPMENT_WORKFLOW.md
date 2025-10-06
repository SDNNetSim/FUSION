# FUSION Development Workflow Guide

This guide describes the modern development workflow for the FUSION project, including code quality tools, testing, and automation.

## Quick Start

For immediate development workflow:

```bash
# Setup (one-time)
pip install -r requirements-dev.txt
make setup-hooks

# Daily development workflow
make format          # Format your code
make lint-new        # Check for issues
make test-new        # Run tests with coverage
make check-all       # Run all quality checks

# Before submitting PR
make validate        # Legacy validation (still supported)
# OR
make check-all       # New comprehensive checks
```

## Directory Structure

The project uses two distinct directories for automation:

### `tools/` - Project Validation and CI/CD
- **Purpose**: Validation tools for PR checking and CI/CD integration
- **Contents**:
  - `validate_pr.py` - Main PR validation script
  - `validate_pr.sh` - Shell wrapper for validation
  - `scripts/` - Utility scripts for validation and CI
- **Usage**: Used by `make validate`, `make lint` (legacy), and CI/CD pipelines

### `scripts/` - Development Workflow Automation
- **Purpose**: Modern development workflow automation and code quality tools
- **Contents**:
  - `analyze_dependencies.sh` - Dependency analysis and dead code detection
  - `generate_diagrams.sh` - Architecture visualization generation
  - `profile_performance.sh` - Performance profiling and optimization
- **Usage**: Used by `make analyze`, `make profile`, and other quality targets

## Code Quality Tools

### Modern Stack (Recommended)
- **black**: Code formatting
- **ruff**: Fast, modern linting (replaces flake8 + isort + many plugins)
- **mypy**: Type checking
- **pytest + pytest-cov**: Testing with coverage
- **vulture**: Dead code detection
- **bandit**: Security vulnerability scanning

### Tool Configuration
- `pyproject.toml` - Configuration for black, ruff, pytest, coverage
- `mypy.ini` - MyPy type checking configuration
- `.pre-commit-config.yaml` - Git hooks configuration
- `.vulture_whitelist.py` - Dead code detection whitelist

## Development Commands

### Core Workflow
```bash
make format      # Format code with black and isort
make lint-new    # Modern linting with ruff and mypy
make test-new    # Run tests with coverage reporting
make analyze     # Dependency analysis and dead code detection
make profile     # Performance profiling
make check-all   # Run all quality checks
```

### Setup and Maintenance
```bash
make setup-hooks     # Install and update pre-commit hooks
make install         # Install project dependencies
make clean          # Clean up generated files
```

### Legacy Commands (Still Supported)
```bash
make validate        # Full PR validation (legacy)
make quick-validate  # Quick validation (legacy)
make lint           # Legacy linting
make test           # Legacy testing
```

## Pre-commit Hooks

Pre-commit hooks automatically run when you commit code:

- **black**: Code formatting
- **isort**: Import sorting
- **ruff**: Modern linting and formatting
- **mypy**: Type checking
- **pylint**: Advanced code analysis
- **vulture**: Dead code detection
- **bandit**: Security scanning
- **Standard hooks**: Trailing whitespace, YAML validation, etc.

### Managing Hooks
```bash
pre-commit install           # Install hooks
pre-commit run --all-files   # Run hooks on all files
pre-commit autoupdate        # Update hook versions
```

## CI/CD Integration

### GitHub Actions
The `.github/workflows/quality.yml` workflow runs on every push and PR:

1. **Environment Setup**: Python 3.11, system dependencies
2. **Code Quality**: Format check, linting, type checking
3. **Testing**: Unit tests with coverage reporting
4. **Security**: Vulnerability scanning
5. **Analysis**: Dead code detection, dependency analysis
6. **Artifacts**: Coverage reports, analysis results

### Coverage Reporting
- **Local**: `reports/coverage/htmlcov/index.html` after running tests
- **CI**: Uploaded to Codecov automatically
- **Target**: Maintain high test coverage

## File Organization

### Configuration Files
```
├── pyproject.toml              # Tool configuration (black, ruff, pytest)
├── mypy.ini                    # Type checking configuration
├── .pre-commit-config.yaml     # Git hooks configuration
├── .vulture_whitelist.py       # Dead code detection whitelist
├── requirements-dev.txt        # Development dependencies
└── .github/workflows/quality.yml # CI/CD pipeline
```

### Generated Output
```
├── reports/                   # All development tool outputs
│   ├── analysis/              # Dependency and code analysis
│   │   ├── dependencies.png
│   │   ├── circular_dependencies.txt
│   │   ├── dead_code.txt
│   │   └── dependency_report.txt
│   ├── diagrams/              # Architecture diagrams
│   │   ├── dependencies.png
│   │   ├── classes_fusion.png
│   │   ├── architecture_overview.png
│   │   └── deps_*.png
│   ├── profiling/             # Performance analysis
│   │   ├── profile.prof
│   │   ├── memory_profile.txt
│   │   ├── runtime_profile.svg
│   │   └── profile_simulation.sh
│   └── coverage/              # Test coverage reports
│       ├── htmlcov/
│       ├── coverage.xml
│       └── .coverage
├── docs/                      # Sphinx documentation source
└── data/                      # Simulation input/output data
```

## Best Practices

### Daily Development
1. **Start**: `make format` to ensure consistent formatting
2. **Develop**: Write code with type hints
3. **Check**: `make lint-new` to catch issues early
4. **Test**: `make test-new` to ensure functionality
5. **Commit**: Pre-commit hooks run automatically

### Before PR Submission
1. **Comprehensive Check**: `make check-all`
2. **Legacy Validation**: `make validate` (if required)
3. **Review**: Check generated reports in `reports/`
4. **Clean**: `make clean` if needed

### Performance Optimization
1. **Profile**: `make profile` to identify bottlenecks
2. **Analyze**: Review `reports/profiling/` results
3. **Optimize**: Focus on high-impact areas
4. **Verify**: Re-run profiling to confirm improvements

## Troubleshooting

### Common Issues

**Pre-commit hooks failing:**
```bash
pre-commit clean
pre-commit install
pre-commit run --all-files
```

**Type checking errors:**
- Add type hints to function signatures
- Use `# type: ignore` for complex cases
- Update `mypy.ini` for project-specific rules

**Dead code false positives:**
- Add entries to `.vulture_whitelist.py`
- Use specific patterns for dynamic code

**Performance profiling not working:**
- Ensure target modules exist
- Check `fusion/cli/main.py` is available
- Use custom profiling scripts in `reports/profiling/`

### Getting Help

1. **Documentation**: Check tool-specific docs (ruff, mypy, etc.)
2. **Configuration**: Review `pyproject.toml` and other config files
3. **Examples**: Look at existing code patterns in the project
4. **Community**: Refer to tool communities for advanced usage

## Migration from Legacy Tools

### Gradual Migration
The new tools complement existing validation:
- Legacy `make validate` still works
- New `make check-all` provides modern workflow
- Pre-commit hooks prevent issues before commit
- CI runs both legacy and modern checks

### Key Differences
- **ruff** replaces flake8 + many plugins (faster, more comprehensive)
- **pytest-cov** provides better coverage reporting
- **Pre-commit hooks** catch issues earlier in development
- **Automated formatting** reduces style discussions

## Future Enhancements

Planned improvements:
- Integration with IDE tooling
- Additional performance profiling tools
- Enhanced dependency analysis
- Automated code quality metrics tracking
