# CLI Tests

## Test Coverage
- **config_setup.py**: Configuration loading, validation, and management
- **constants.py**: CLI constants and exit codes
- **main_parser.py**: Argument parser construction and configuration
- **registry.py**: Argument registry management
- **run_sim.py**: Simulation entry point and error handling
- **run_train.py**: Training entry point and error handling
- **utils.py**: Shared utilities for CLI entry points

## Running Tests
```bash
# Run all module tests
pytest fusion/cli/tests/

# Run with coverage
pytest --cov=fusion.cli fusion/cli/tests/
```

## Test Data
- `conftest.py`: Shared fixtures and mock configurations

## Test Categories
- **Unit tests**: Test individual functions in isolation
