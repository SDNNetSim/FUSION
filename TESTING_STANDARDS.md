# FUSION Testing Standards

> **AI/LLM Usage Note**: This document is optimized for Claude to understand and apply FUSION project testing standards consistently.

## 1. Unit Testing with Testing Modules

### 1.1 Test Organization
- **Type**: Unit testing for individual functions/methods in isolation
- **Location**: Testing module within each package
- **Structure**: `tests/` subdirectory within each module
- **Documentation**: Each `tests/` directory MUST have a README.md

```
fusion/
├── core/
│   ├── __init__.py
│   ├── config.py
│   └── tests/
│       ├── __init__.py
│       ├── README.md           # Test documentation
│       └── test_config.py      # Unit tests for config.py
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   └── tests/
│       ├── __init__.py
│       ├── README.md           # Test documentation
│       └── test_helpers.py     # Unit tests for helpers.py
```

### 1.4 Test Directory README.md
**Every `tests/` directory MUST include README.md**:

```markdown
# [Module Name] Tests

## Test Coverage
- **config.py**: Configuration loading and validation
- **helpers.py**: Utility functions

## Running Tests
```bash
# Run all module tests
pytest fusion/[module_name]/tests/

# Run with coverage
pytest --cov=fusion.[module_name] fusion/[module_name]/tests/
```

## Test Data
- `fixtures/`: Test data files
- `conftest.py`: Shared fixtures (if present)

## Test Categories
- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test component interactions (if any)
```

### 1.2 Unit Testing Principles
- **Isolation**: Test one function/method at a time
- **Independence**: No dependencies on external systems
- **Fast execution**: Each test should run in milliseconds
- **Deterministic**: Same input always produces same output
- **Focused**: Test single unit of functionality

### 1.3 Test Discovery
pytest automatically finds:
- Files matching `test_*.py` in `tests/` directories
- Functions starting with `test_`
- Classes starting with `Test`

## 2. Test Structure

### 2.1 Naming Pattern
`test_<what>_<when>_<expected>`

```python
def test_config_loading_with_valid_file_returns_dict():
    """Test that valid configuration file loads successfully."""
    pass

def test_config_loading_with_missing_file_raises_error():
    """Test that missing file raises ConfigFileNotFoundError."""
    pass
```

### 2.2 AAA Pattern
```python
def test_function_name():
    """Clear description of what is being tested."""
    # Arrange - Set up test data
    input_data = {"key": "value"}
    expected_result = True

    # Act - Execute function
    result = function_under_test(input_data)

    # Assert - Verify results
    assert result == expected_result
```

### 2.3 Class Organization
```python
import pytest
from unittest.mock import Mock, patch
from ..config import load_configuration, ConfigFileNotFoundError

class TestConfigurationLoading:
    """Tests for configuration loading functionality."""

    def test_valid_file_loads_successfully(self):
        """Test successful loading of valid config file."""
        # Arrange
        config_path = "tests/fixtures/valid_config.ini"

        # Act
        result = load_configuration(config_path)

        # Assert
        assert isinstance(result, dict)
        assert "database" in result

    def test_missing_file_raises_error(self):
        """Test that missing file raises appropriate error."""
        # Arrange
        config_path = "nonexistent.ini"

        # Act & Assert
        with pytest.raises(ConfigFileNotFoundError):
            load_configuration(config_path)
```

## 3. Test Coverage Requirements

### 3.1 What to Test
- **Public functions**: All public methods must have tests
- **Error conditions**: Test all custom exceptions
- **Edge cases**: Empty inputs, boundary values, invalid data
- **Critical paths**: Core functionality and business logic

### 3.2 Coverage Targets
- **Critical modules**: 90%+ coverage
- **Utility modules**: 80%+ coverage
- **Integration modules**: 70%+ coverage

## 4. Mocking

### 4.1 When to Mock (Unit Testing)
- **Always mock external dependencies**:
  - External APIs and services
  - File system operations
  - Database connections
  - Network requests
  - Time-dependent operations
  - Other modules/functions
- **Mock principle**: Unit tests should only test the unit itself

### 4.2 Mock Examples
```python
@patch('fusion.core.config.Path.exists')
def test_config_file_validation_when_missing(mock_exists):
    """Test validation when config file doesn't exist (unit test)."""
    # Arrange - Mock external dependency
    mock_exists.return_value = False

    # Act & Assert - Test only the unit's logic
    with pytest.raises(ConfigFileNotFoundError):
        validate_config_file("missing.ini")
```

## 5. Fixtures and Test Data

### 5.1 Fixtures
```python
@pytest.fixture
def sample_config():
    """Provide sample configuration for tests."""
    return {
        "database": {"host": "localhost", "port": 5432},
        "logging": {"level": "INFO"}
    }

def test_config_validation_with_valid_data(sample_config):
    """Test configuration validation with valid data."""
    result = validate_config(sample_config)
    assert result is True
```

### 5.2 Parametrized Tests
```python
@pytest.mark.parametrize("input_value,expected", [
    ("valid_config.ini", True),
    ("empty_config.ini", False),
    ("malformed.ini", False),
])
def test_config_validation(input_value, expected):
    """Test config validation with various inputs."""
    result = is_valid_config(input_value)
    assert result == expected
```

## 6. Test Execution

### 6.1 Running Tests
```bash
# Run all tests in module
pytest fusion/core/tests/

# Run specific test file
pytest fusion/core/tests/test_config.py

# Run specific test
pytest fusion/core/tests/test_config.py::test_config_loading_with_valid_file

# Run with coverage
pytest --cov=fusion/core fusion/core/tests/
```

### 6.2 Coverage Reporting
```bash
# Generate coverage report
pytest --cov=fusion --cov-report=html

# Check coverage percentage
pytest --cov=fusion --cov-fail-under=80
```

## 7. Best Practices

### 7.1 Test Independence
- Each test should be independent
- No shared state between tests
- Use fixtures for common setup

### 7.2 Clear Assertions
```python
# ✅ Good - Specific assertions
assert len(results) == 3
assert "error" not in response
assert response.status_code == 200

# ❌ Avoid - Generic assertions
assert results
assert response
```

### 7.3 Error Testing
```python
def test_invalid_input_raises_validation_error():
    """Test that invalid input raises ValidationError with clear message."""
    with pytest.raises(ValidationError) as exc_info:
        validate_input("")

    assert "Input cannot be empty" in str(exc_info.value)
```

## 8. Testing Tools

### 8.1 Required Tools
- **pytest**: Primary testing framework
- **pytest-cov**: Coverage reporting
- **unittest.mock**: Mocking functionality

### 8.2 Optional Tools
- **pytest-xdist**: Parallel test execution
- **pytest-benchmark**: Performance testing
- **hypothesis**: Property-based testing

---

## Quick Reference Checklist

### Test Organization ✓
- [ ] Testing module (`tests/`) within each package
- [ ] README.md in every `tests/` directory
- [ ] Naming: `test_<module_name>.py`
- [ ] Unit tests for individual functions/methods
- [ ] Fast, isolated, deterministic tests

### Test Quality ✓
- [ ] Clear test function names following pattern
- [ ] All public functions tested
- [ ] Error conditions tested
- [ ] Edge cases covered
- [ ] Appropriate use of mocks
- [ ] Independent tests
- [ ] Clear, specific assertions

### Coverage ✓
- [ ] Critical modules: 90%+
- [ ] Utility modules: 80%+
- [ ] Integration modules: 70%+

---

*Follow these standards for comprehensive, maintainable tests that ensure code quality and reliability.*
