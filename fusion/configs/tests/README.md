# Configs Tests

## Test Coverage
- **config.py**: ConfigManager and SimulationConfig functionality
- **cli_to_config.py**: CLI argument mapping
- **errors.py**: Configuration exception classes
- **constants.py**: Configuration constants
- **registry.py**: ConfigRegistry template management
- **schema.py**: Schema validation functionality
- **validate.py**: SchemaValidator class

## Running Tests
```bash
# Run all module tests
pytest fusion/configs/tests/

# Run with coverage
pytest --cov=fusion.configs fusion/configs/tests/
```

## Test Data
- `fixtures/`: Test data files for configuration testing

## Test Categories
- **Unit tests**: Test individual functions in isolation
