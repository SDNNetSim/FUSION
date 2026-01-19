# fusion.io Tests

## Test Coverage

This directory contains comprehensive unit tests for the `fusion.io` module:

- **test_exporter.py**: Data export functionality (JSON, CSV exporters, registry, simulation data exporter)
- **test_structure.py**: Network structure creation (link length assignment, core node assignment, network loading)
- **test_generate.py**: Data generation utilities (physical topology creation, bandwidth/modulation info)

## Running Tests

```bash
# Run all io module tests
pytest fusion/io/tests/

# Run specific test file
pytest fusion/io/tests/test_exporter.py
pytest fusion/io/tests/test_structure.py
pytest fusion/io/tests/test_generate.py

# Run with coverage
pytest --cov=fusion.io fusion/io/tests/

# Run with verbose output
pytest -v fusion/io/tests/

# Run specific test
pytest fusion/io/tests/test_exporter.py::TestJSONExporter::test_export_with_dict_creates_json_file
```

## Test Categories

- **Unit tests**: All tests are unit tests that mock external dependencies
- **Fixtures**: Uses pytest fixtures and temporary directories for file I/O operations
- **Mocking**: Extensively uses unittest.mock to isolate units under test

## Test Structure

All tests follow the AAA (Arrange-Act-Assert) pattern and use descriptive naming:
- `test_<what>_<when>_<expected>`

Example:
```python
def test_export_with_dict_creates_json_file(self) -> None:
    """Test exporting dictionary data to JSON file."""
    # Arrange
    exporter = JSONExporter()
    data = {"key": "value"}

    # Act
    exporter.export(data, output_path)

    # Assert
    assert output_path.exists()
```

## Coverage Targets

- **Critical modules**: 90%+ coverage
- **Current coverage**: Run `pytest --cov=fusion.io --cov-report=term-missing fusion/io/tests/` to view

## Dependencies

- pytest
- pytest-cov
- unittest.mock (standard library)
- tempfile (standard library)
