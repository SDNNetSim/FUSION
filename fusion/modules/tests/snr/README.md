# SNR Module Tests

## Test Coverage

This test suite provides comprehensive unit test coverage for the SNR (Signal-to-Noise Ratio) measurement module.

### Modules Tested

- **utils.py**: Utility functions for SNR calculations
  - File loading for modulation formats and GSNR data
  - Slot index computation across different bands
  - SNR response validation

- **snr.py**: StandardSNRMeasurer class implementation
  - Path and link SNR calculations
  - Linear noise (ASE) modeling
  - Nonlinear noise (SCI, XCI) calculations
  - Cross-talk calculations for multi-core fibers
  - SNR threshold determination and acceptability checks
  - Metrics tracking and algorithm reset

- **registry.py**: SNR algorithm registry system
  - Algorithm registration and retrieval
  - Dynamic algorithm instantiation
  - Multi-core algorithm filtering
  - Global registry convenience functions

## Running Tests

### Run All SNR Module Tests
```bash
pytest fusion/modules/tests/snr/
```

### Run Specific Test File
```bash
# Test utilities
pytest fusion/modules/tests/snr/test_utils.py

# Test SNR measurer
pytest fusion/modules/tests/snr/test_snr.py

# Test registry
pytest fusion/modules/tests/snr/test_registry.py
```

### Run with Coverage
```bash
# Generate coverage report
pytest --cov=fusion.modules.snr fusion/modules/tests/snr/

# Generate HTML coverage report
pytest --cov=fusion.modules.snr --cov-report=html fusion/modules/tests/snr/
```

### Run Specific Test Class or Method
```bash
# Run specific test class
pytest fusion/modules/tests/snr/test_snr.py::TestCalculateSNR

# Run specific test method
pytest fusion/modules/tests/snr/test_snr.py::TestCalculateSNR::test_calculate_snr_with_valid_path_returns_positive_value
```

### Run with Verbose Output
```bash
pytest fusion/modules/tests/snr/ -v
```

## Test Categories

### Unit Tests
All tests in this module are **unit tests** that:
- Test individual functions and methods in isolation
- Mock external dependencies (file I/O, network operations, numpy operations)
- Execute quickly (milliseconds per test)
- Are deterministic and independent
- Follow the AAA (Arrange-Act-Assert) pattern

### Test Organization

Tests are organized into classes by functionality:

#### test_utils.py
- `TestGetLoadedFiles`: Tests for file loading functionality
- `TestGetSlotIndex`: Tests for slot index computation
- `TestComputeResponse`: Tests for SNR response validation

#### test_snr.py
- `TestStandardSNRMeasurerInitialization`: Initialization and property tests
- `TestCalculateSNR`: Path-level SNR calculation tests
- `TestCalculateLinkSNR`: Link-level SNR calculation tests
- `TestASENoiseCalculation`: Amplified spontaneous emission noise tests
- `TestNonlinearNoiseCalculation`: Nonlinear noise component tests
- `TestCrosstalkCalculation`: Multi-core fiber crosstalk tests
- `TestSNRThresholdMethods`: SNR threshold and acceptability tests
- `TestMetricsAndReset`: Metrics and state management tests
- `TestSetupSNRCalculation`: Setup and initialization tests

#### test_registry.py
- `TestSNRRegistry`: Core registry functionality tests
- `TestGlobalRegistryFunctions`: Global convenience function tests
- `TestSNRRegistryEdgeCases`: Edge cases and error handling tests

## Test Data

### Fixtures
The test suite uses pytest fixtures for:
- Mock engine properties (`mock_engine_props`)
- Mock SDN properties (`mock_sdn_props`)
- Mock spectrum properties (`mock_spectrum_props`)
- Mock route properties (`mock_route_props`)
- SNR measurer instances (`snr_measurer`)

### Mock Objects
Tests extensively use mocking to:
- Isolate units under test
- Avoid file system operations
- Simulate network topology
- Control numpy behavior

## Key Testing Patterns

### Parametrized Tests
Many tests use `@pytest.mark.parametrize` to test multiple scenarios:
- Different band types (L, C, S)
- Various modulation formats (BPSK, QPSK, 16QAM, etc.)
- Different link lengths and configurations
- Edge cases and boundary conditions

### Error Testing
Comprehensive error handling tests verify:
- ValueError for invalid inputs
- KeyError for missing registry entries
- TypeError for incorrect class types
- Proper error messages and context

### Property Testing
Tests verify:
- Return types match specifications
- Values fall within expected ranges
- Relationships between inputs and outputs
- State changes and side effects

## Coverage Targets

Based on FUSION testing standards:
- **Target Coverage**: 90%+ (critical module)
- **Current Focus**: All public methods and critical paths
- **Edge Cases**: Invalid inputs, boundary conditions, error states

## Dependencies

### Required
- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `numpy`: Numerical operations
- `unittest.mock`: Mocking functionality

### Module Dependencies
- `fusion.core.properties`: SNR properties classes
- `fusion.interfaces.snr`: AbstractSNRMeasurer interface
- `fusion.modules.snr`: SNR module implementation

## Best Practices Applied

1. **Isolation**: Each test is independent with no shared state
2. **Clear Naming**: Test names follow `test_<what>_<when>_<expected>` pattern
3. **AAA Pattern**: Arrange-Act-Assert structure in all tests
4. **Mocking**: External dependencies are mocked appropriately
5. **Specific Assertions**: Clear, specific assertions for each test
6. **Documentation**: Comprehensive docstrings for all test methods

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- Fast execution (all tests complete in seconds)
- No external dependencies
- Deterministic results
- Clear failure messages

## Troubleshooting

### Import Errors
If you encounter import errors, ensure you're running from the project root:
```bash
cd /path/to/FUSION
pytest fusion/modules/tests/snr/
```

### Coverage Not Generated
Install coverage tools:
```bash
pip install pytest-cov
```

### Mock-Related Failures
Ensure `unittest.mock` is available (Python 3.3+) or install `mock` package for older versions.

## Contributing

When adding new tests:
1. Follow the existing naming conventions
2. Use appropriate fixtures
3. Mock external dependencies
4. Add docstrings to test methods
5. Ensure tests are independent
6. Run full test suite before committing

## Additional Resources

- [FUSION Testing Standards](../../../../TESTING_STANDARDS.md)
- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
