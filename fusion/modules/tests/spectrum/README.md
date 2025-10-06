# Spectrum Module Tests

## Test Coverage

This directory contains comprehensive unit tests for the spectrum assignment module:

- **test_utils.py**: SpectrumHelpers utility class tests
- **test_first_fit.py**: FirstFitSpectrum algorithm tests
- **test_last_fit.py**: LastFitSpectrum algorithm tests
- **test_best_fit.py**: BestFitSpectrum algorithm tests
- **test_light_path_slicing.py**: LightPathSlicingManager tests
- **test_registry.py**: SpectrumRegistry and global functions tests

## Running Tests

### Run all spectrum module tests
```bash
pytest fusion/modules/tests/spectrum/
```

### Run specific test file
```bash
pytest fusion/modules/tests/spectrum/test_first_fit.py
```

### Run specific test class
```bash
pytest fusion/modules/tests/spectrum/test_first_fit.py::TestAssignMethod
```

### Run specific test function
```bash
pytest fusion/modules/tests/spectrum/test_first_fit.py::TestAssignMethod::test_assign_with_valid_path_returns_assignment
```

### Run with coverage
```bash
pytest --cov=fusion.modules.spectrum fusion/modules/tests/spectrum/
```

### Run with verbose output
```bash
pytest fusion/modules/tests/spectrum/ -v
```

## Test Categories

### Unit Tests
All tests in this module are isolated unit tests that mock external dependencies:
- **FirstFitSpectrum**: Tests first-available spectrum slot allocation
- **LastFitSpectrum**: Tests last-available spectrum slot allocation
- **BestFitSpectrum**: Tests fragmentation-minimizing spectrum allocation
- **LightPathSlicingManager**: Tests request slicing functionality
- **SpectrumHelpers**: Tests spectrum utility functions
- **SpectrumRegistry**: Tests algorithm registration and retrieval

## Test Data

Tests use pytest fixtures to provide:
- Mock engine properties
- Mock SDN properties
- Mock routing properties
- Mock spectrum objects
- Pre-configured network spectrum matrices

## Coverage Goals

- **Critical modules**: 90%+ coverage
- **All public methods**: Tested
- **Error conditions**: Tested
- **Edge cases**: Tested

## Standards

All tests follow the FUSION testing standards:
- AAA pattern (Arrange, Act, Assert)
- Clear test names: `test_<what>_<when>_<expected>`
- Proper type annotations for Mypy compliance
- Ruff-compliant code formatting
- Isolated unit tests with mocked dependencies
