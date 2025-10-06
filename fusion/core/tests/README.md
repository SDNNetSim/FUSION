# Core Tests

## Test Coverage
- **test_metrics.py**: SimStats class for simulation statistics collection
- **test_routing.py**: Routing coordination and algorithm dispatch  
- **test_sdn_controller.py**: SDN controller for network resource management
- **test_snr_measurements.py**: Signal-to-noise ratio calculations
- **test_spectrum_assignment.py**: Spectrum allocation algorithms
- **test_properties.py**: Core data properties and structures

## Running Tests
```bash
# Run all core module tests
pytest fusion/core/tests/

# Run with coverage
pytest --cov=fusion.core fusion/core/tests/

# Run specific test file
pytest fusion/core/tests/test_metrics.py

# Run specific test class
pytest fusion/core/tests/test_metrics.py::TestSimStats

# Run specific test method
pytest fusion/core/tests/test_metrics.py::TestSimStats::test_init_with_valid_parameters_creates_instance
```

## Test Data
- Uses mocked external dependencies for isolation
- NetworkX graphs for topology testing
- NumPy arrays for spectrum matrix testing

## Test Categories
- **Unit tests**: Test individual functions and methods in isolation
- **Property tests**: Test core data structure initialization and validation
- **Integration points**: Test interaction between core components (mocked)

## Test Design Principles
- **Isolation**: Each test is independent with proper setup/teardown
- **Mocking**: External dependencies are mocked for true unit testing
- **Focused**: Tests single units of functionality
- **Descriptive**: Test names follow `test_<what>_<when>_<expected>` pattern

## Coverage Requirements
This is a critical module with 90%+ coverage target covering:
- All public methods and functions
- Error conditions and edge cases
- Property initialization and validation
- Core simulation workflows