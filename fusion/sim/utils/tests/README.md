# Simulation Utils Tests

## Test Coverage
- **network.py**: Network path analysis, congestion metrics, and fragmentation
- **io.py**: File I/O operations for YAML and JSON
- **spectrum.py**: Spectrum allocation and channel management
- **formatting.py**: String formatting and conversion utilities
- **data.py**: Data processing and matrix operations
- **simulation.py**: Simulation setup and management

## Running Tests
```bash
# Run all module tests
pytest fusion/sim/utils/tests/

# Run with coverage
pytest --cov=fusion.sim.utils fusion/sim/utils/tests/

# Run specific test file
pytest fusion/sim/utils/tests/test_network.py
```

## Test Categories
- **Unit tests**: Test individual functions in isolation with mocked dependencies
