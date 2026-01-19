# Unity Module Tests

Unit tests for the `fusion.unity` module, which provides Unity cluster management functionality for FUSION simulations.

## Test Coverage

- **test_constants.py**: Constants and configuration values
- **test_errors.py**: Custom exception classes and error hierarchy
- **test_fetch_results.py**: Remote result fetching and synchronization functions
- **test_make_manifest.py**: Manifest generation from specification files
- **test_submit_manifest.py**: SLURM job submission functionality

## Running Tests

```bash
# Run all unity module tests
pytest fusion/unity/tests/

# Run specific test file
pytest fusion/unity/tests/test_constants.py

# Run with coverage
pytest --cov=fusion.unity fusion/unity/tests/

# Run with verbose output
pytest fusion/unity/tests/ -v

# Run specific test class
pytest fusion/unity/tests/test_fetch_results.py::TestConvertOutputToInputPath

# Run specific test function
pytest fusion/unity/tests/test_constants.py::TestConstants::test_resource_keys_contains_expected_values
```

## Test Organization

All tests follow the AAA (Arrange-Act-Assert) pattern and use descriptive naming:
- Test classes: `Test<FunctionOrFeatureName>`
- Test methods: `test_<what>_<when>_<expected>`

## Test Categories

- **Unit tests**: Test individual functions in isolation with mocked dependencies
- **Integration tests**: None (unity module tests are pure unit tests)

## Key Testing Principles

1. **Isolation**: Each test is independent with mocked external dependencies
2. **Fast execution**: All tests run in milliseconds
3. **Deterministic**: Tests produce consistent results
4. **Comprehensive**: Cover normal cases, edge cases, and error conditions

## Coverage Targets

- **Unity module**: 90%+ coverage target (critical module)
- Current coverage can be viewed by running tests with coverage reporting

## Dependencies

Tests use the following testing tools:
- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `unittest.mock`: Mocking external dependencies
