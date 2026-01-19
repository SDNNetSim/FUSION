# Failures Module Tests

Unit and integration tests for the FUSION failures module.

## Running Tests

Run all tests:
```bash
pytest fusion/modules/failures/tests/ -v
```

Run with coverage:
```bash
pytest fusion/modules/failures/tests/ -v --cov=fusion.modules.failures
```

Run specific test file:
```bash
pytest fusion/modules/failures/tests/test_failure_manager.py -v
```

## Test Files

- `test_failure_manager.py`: Tests for FailureManager class
- `test_failure_types.py`: Tests for individual failure type functions

## Coverage Target

Minimum coverage: 85%
