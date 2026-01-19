# Utils Tests

## Test Coverage

- **config.py**: Configuration utility functions including string-to-bool conversion, dictionary parsing, CLI overrides, and type conversion
- **logging_config.py**: Centralized logging configuration including logger setup, simulation logging, and log adapters
- **os.py**: Operating system utilities including directory creation and project root discovery
- **random.py**: Random number generation utilities for simulations (uniform and exponential distributions)
- **spectrum.py**: Spectrum utility functions for finding free slots, channels, and analyzing channel overlaps
- **network.py**: Network utility functions for path analysis, congestion calculation, and modulation selection
- **data.py**: Data structure manipulation utilities for sorting dictionaries

## Running Tests

```bash
# Run all utils tests
pytest fusion/utils/tests/

# Run specific test file
pytest fusion/utils/tests/test_config.py

# Run with coverage
pytest --cov=fusion.utils fusion/utils/tests/

# Run with verbose output
pytest -v fusion/utils/tests/

# Run specific test class
pytest fusion/utils/tests/test_config.py::TestStrToBool

# Run specific test function
pytest fusion/utils/tests/test_config.py::TestStrToBool::test_str_to_bool_with_various_inputs_returns_expected
```

## Test Categories

- **Unit tests**: All tests are isolated unit tests that mock external dependencies
- **Coverage target**: 80%+ (utility modules standard)

## Test Standards

All tests follow the FUSION testing standards:

- **AAA Pattern**: Arrange, Act, Assert structure
- **Descriptive naming**: `test_<what>_<when>_<expected>` pattern
- **Mocking**: External dependencies (file system, network, time) are mocked
- **Type hints**: Full type annotations for mypy compliance
- **Independence**: Each test is independent and can run in any order
- **Parametrization**: Related test cases use `@pytest.mark.parametrize`

## Key Testing Patterns

### Configuration Testing
Tests verify type conversion, CLI overrides, and error handling for configuration utilities.

### Logging Testing
Tests mock file system operations and verify handler creation, logger caching, and formatting.

### Random Number Testing
Tests verify deterministic behavior with seeds, proper distributions, and error handling.

### Spectrum Testing
Tests use numpy arrays to simulate network spectrum and verify slot/channel finding algorithms.

### Network Testing
Tests use NetworkX graphs and numpy arrays to verify path metrics and congestion calculations.

### Data Structure Testing
Tests verify sorting operations preserve data integrity and follow expected ordering.
