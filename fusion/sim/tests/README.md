# FUSION Sim Tests

## Test Coverage

### Utils Module Tests
- **test_io.py**: I/O operations (YAML/JSON parsing and modification)
- **test_network.py**: Network analysis and path calculations
- **test_spectrum.py**: Spectrum allocation and channel management
- **test_formatting.py**: String formatting utilities
- **test_data_utils.py**: Data processing and matrix operations
- **test_simulation.py**: Simulation setup and management

### Main Module Tests
- **test_batch_runner.py**: Batch simulation orchestration
- **test_run_simulation.py**: Simulation execution entry points
- **test_network_simulator.py**: Legacy network simulator
- **test_train_pipeline.py**: RL training pipeline

## Running Tests

```bash
# Run all sim module tests
pytest fusion/sim/tests/

# Run with coverage
pytest --cov=fusion.sim fusion/sim/tests/

# Run specific test file
pytest fusion/sim/tests/test_network.py

# Run specific test
pytest fusion/sim/tests/test_network.py::TestFindPathLength::test_find_path_length_with_single_hop_returns_correct_length
```

## Test Categories

- **Unit tests**: Test individual functions in isolation with mocked dependencies
- All tests follow AAA pattern (Arrange, Act, Assert)
- Comprehensive edge case and error condition coverage
