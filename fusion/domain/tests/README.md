# Domain Tests

## Test Coverage

### Domain Objects
- **test_config.py**: SimulationConfig creation, validation, and legacy conversion
- **test_request.py**: Request lifecycle, status transitions, and enums
- **test_network_state.py**: NetworkState and LinkSpectrum spectrum management
- **test_results.py**: Result dataclasses for pipeline stages

### Not Yet Covered
- **test_lightpath.py**: Lightpath capacity management and protection (TODO)

## Running Tests

```bash
# Run all domain module tests
pytest fusion/domain/tests/

# Run with coverage
pytest --cov=fusion.domain fusion/domain/tests/

# Run specific test file
pytest fusion/domain/tests/test_config.py

# Run specific test class
pytest fusion/domain/tests/test_config.py::TestSimulationConfig

# Run specific test method
pytest fusion/domain/tests/test_config.py::TestSimulationConfig::test_from_engine_props_creates_valid_config
```

## Test Data
- Uses minimal fixtures for dataclass instantiation
- NumPy arrays for spectrum matrix testing
- NetworkX graphs for topology-dependent state tests

## Test Categories
- **Validation tests**: Test `__post_init__` validation catches invalid inputs
- **Property tests**: Test computed properties return correct values
- **Conversion tests**: Test legacy dict conversion round-trips correctly
- **State mutation tests**: Test mutable operations (allocate, release) maintain invariants

## Test Design Principles
- **Isolation**: Each test is independent with fresh object instances
- **Boundary testing**: Test edge cases for validation (e.g., empty paths, negative values)
- **Round-trip testing**: Verify `from_legacy_dict` and `to_legacy_dict` are inverses
- **Descriptive**: Test names follow `test_<what>_<when>_<expected>` pattern

## Coverage Requirements
Target 80%+ coverage for domain module covering:
- All `__post_init__` validation paths
- All computed properties
- All public methods
- Legacy conversion methods
- Error conditions and edge cases

## Key Test Scenarios

### SimulationConfig
- Creation from engine_props dictionary
- Validation of required fields
- Round-trip conversion to/from legacy format

### Request
- Status enum values and transitions
- Block reason tracking
- Request type handling (ARRIVAL, DEPARTURE, FAILURE, RECOVERY)

### NetworkState
- Spectrum allocation and release
- Lightpath registry management
- Link spectrum queries across cores and bands

### Results
- Immutability of result objects
- Factory method creation
- Success/failure state representation
