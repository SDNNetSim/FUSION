# ML Module Tests

## Test Coverage
- **constants.py**: Constants and shared values validation
- **preprocessing.py**: Data preprocessing and transformation functions
- **model_io.py**: Model saving, loading, and export functionality
- **evaluation.py**: Model evaluation metrics and analysis
- **visualization.py**: Plotting and visualization utilities
- **feature_engineering.py**: Feature extraction and engineering

## Running Tests
```bash
# Run all ML module tests
pytest fusion/modules/tests/ml/

# Run specific test file
pytest fusion/modules/tests/ml/test_preprocessing.py

# Run with coverage
pytest --cov=fusion.modules.ml fusion/modules/tests/ml/

# Run with verbose output
pytest fusion/modules/tests/ml/ -v
```

## Test Data
- Uses pytest fixtures for mock data
- Mocks external dependencies (file I/O, plotting, network utilities)
- Tests are isolated and independent

## Test Categories
- **Unit tests**: Test individual functions in isolation with mocked dependencies
- **Edge cases**: Empty inputs, boundary values, invalid data
- **Error conditions**: Custom exceptions and error handling
