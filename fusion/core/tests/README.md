# Core Tests

## Test Coverage

### Core Components
- **test_simulation.py**: Simulation engine and event processing
- **test_sdn_controller.py**: SDN controller for network resource management
- **test_orchestrator.py**: SDNOrchestrator pipeline-based request handling
- **test_orchestrator_policy.py**: Orchestrator policy and heuristic handling
- **test_pipeline_factory.py**: PipelineFactory and PipelineSet creation

### Routing and Spectrum
- **test_routing.py**: Routing coordination and algorithm dispatch
- **test_spectrum_assignment.py**: Spectrum allocation algorithms
- **test_snr_measurements.py**: Signal-to-noise ratio calculations

### Metrics
- **test_metrics.py**: SimStats class for simulation statistics collection
- **test_ml_metrics.py**: Machine learning metrics (SL/USL)
- **test_recovery_metrics.py**: Recovery and restoration metrics
- **test_survivability_metrics.py**: Survivability and failure metrics

### Supporting Components
- **test_grooming.py**: Traffic grooming functionality
- **test_properties.py**: Core data properties and structures
- **test_request.py**: Request generation and handling
- **test_persistence.py**: State persistence and checkpointing

### Quality and Behavior
- **test_determinism.py**: Simulation reproducibility verification
- **test_feature_flag.py**: Feature flag functionality

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

## Not Tested
- **adapters/**: Adapter classes are temporary migration layers and are tested indirectly through orchestrator tests
