# FUSION Coding Standards & Guidelines

## Table of Contents
1. [General Principles](#general-principles)
2. [Naming Conventions](#naming-conventions)
3. [Code Organization](#code-organization)
4. [Documentation Standards](#documentation-standards)
5. [Error Handling](#error-handling)
6. [Type Annotations](#type-annotations)
7. [Import Standards](#import-standards)
8. [Testing Standards](#testing-standards)

## General Principles

### Code Quality Pillars
- **Readability**: Code is read more often than it's written
- **Consistency**: Follow established patterns throughout the codebase
- **Simplicity**: Prefer simple, clear solutions over clever ones
- **Maintainability**: Write code that's easy to modify and extend

### File Organization
- Keep files under 500 lines when possible
- Use meaningful file and directory names
- Group related functionality in packages/modules
- Separate concerns (errors, constants, schemas, utilities)

## Naming Conventions

### Functions and Variables
- Use `snake_case` for all functions and variables
- Be descriptive and avoid abbreviations
- Use verb phrases for functions: `load_config()`, `validate_structure()`
- Use noun phrases for variables: `config_path`, `user_settings`

#### Data Type Suffixes (Recommended)
For complex data types, append the type suffix to improve readability:

```python
# Recommended - Clear data structure intent
user_settings_dict: Dict[str, Any] = {}
active_connections_set: Set[str] = set()
pending_requests_list: List[Request] = []
error_message_queue: Queue[str] = Queue()
thread_config_map: Dict[str, ThreadConfig] = {}

# Less clear
user_settings: Dict[str, Any] = {}
active_connections: Set[str] = set()  # Could be list, tuple, etc.
```

**When to use suffixes:**
- ✅ Complex collections (`_dict`, `_set`, `_list`, `_queue`, `_map`)  
- ✅ When the base name doesn't indicate structure
- ❌ Simple variables where context is clear
- ❌ When type hints make it obvious

### Constants
- Use `SCREAMING_SNAKE_CASE` for module-level constants
- Group related constants together
- Use descriptive names that indicate purpose and scope

```python
# Good
DEFAULT_CONFIG_PATH = "config.ini"
MAX_RETRY_ATTEMPTS = 3
SUPPORTED_FILE_EXTENSIONS = ['.json', '.yaml', '.ini']

# Avoid
PATH = "config.ini"  # Too generic
MAX = 3  # Unclear what this limits
```

### Classes
- Use `PascalCase` for class names
- Choose names that represent the entity or concept
- Avoid generic names like `Manager`, `Handler` unless they truly manage/handle

```python
# Good
class ConfigurationManager:  # Clear purpose
class NetworkTopology:       # Clear domain concept
class RequestProcessor:      # Clear action

# Less ideal  
class Manager:              # Too generic
class Helper:               # Unclear purpose
```

### Private Methods and Variables
- Use single underscore `_` for internal use
- Use double underscore `__` only for name mangling (rare)
- Private methods don't need docstrings unless complex

### Function Parameters
- Use descriptive parameter names
- Avoid single letters except for common math operations
- Use full words over abbreviations

```python
# Good
def process_configuration(config_path: str, output_directory: str) -> bool:

# Avoid
def process_config(path: str, out_dir: str) -> bool:  # Abbreviations
def process(p: str, o: str) -> bool:  # Single letters
```

## Code Organization

### Module Structure
```
package/
├── __init__.py          # Public API exports
├── constants.py         # Package constants
├── errors.py           # Custom exceptions
├── schema.py           # Data schemas/models
├── utils.py            # Utility functions
└── core.py             # Main functionality
```

### Function Organization
- Keep functions under 50 lines when possible
- Single responsibility principle
- Extract complex logic into helper functions
- Group related functions together

### Class Organization
```python
class ExampleClass:
    """Class docstring."""
    
    # Class variables
    CLASS_CONSTANT = "value"
    
    def __init__(self):
        """Constructor."""
        pass
    
    # Public methods first
    def public_method(self):
        """Public method."""
        pass
    
    @classmethod
    def class_method(cls):
        """Class method."""
        pass
    
    @staticmethod
    def static_method():
        """Static method."""
        pass
    
    # Private methods last
    def _private_method(self):
        pass
```

## Documentation Standards

### Docstring Format (Sphinx)
Use Sphinx format for all public functions, classes, and methods:

```python
def load_configuration(config_path: str, validate: bool = True) -> Dict[str, Any]:
    """Load and parse configuration file.
    
    Detailed description of what the function does, including any
    important behavior, side effects, or assumptions.
    
    :param config_path: Path to configuration file (absolute or relative)
    :type config_path: str
    :param validate: Whether to validate configuration structure, defaults to True
    :type validate: bool
    :return: Parsed configuration data with normalized values
    :rtype: Dict[str, Any]
    :raises ConfigFileNotFoundError: If config file doesn't exist
    :raises ConfigParseError: If file format is invalid
    
    Example:
        >>> config = load_configuration("settings.ini")
        >>> print(config['database']['host'])
        localhost
    """
```

### Comments
- Use comments sparingly - good code should be self-documenting
- Explain **why**, not **what**
- Use comments for complex algorithms or business logic

```python
# Good - explains the why
# Retry with exponential backoff to handle network instability
time.sleep(2 ** attempt)

# Less useful - explains the what (obvious from code)
# Increment the counter
counter += 1
```

### README and Documentation
- Every package should have a README explaining its purpose
- Include usage examples for complex modules
- Document configuration options and their effects

## Error Handling

### Custom Exceptions
Create specific exceptions for different error conditions:

```python
# Good - Specific exceptions
class ConfigurationError(Exception):
    """Base exception for configuration errors."""
    pass

class ConfigFileNotFoundError(ConfigurationError):
    """Raised when configuration file cannot be found."""
    pass

class InvalidConfigFormatError(ConfigurationError):
    """Raised when configuration file format is invalid."""
    pass
```

### Exception Handling Patterns
```python
# Good - Specific exception handling
try:
    config = load_configuration(path)
except ConfigFileNotFoundError:
    logger.error(f"Config file not found: {path}")
    return default_config()
except InvalidConfigFormatError as e:
    logger.error(f"Invalid config format: {e}")
    raise

# Avoid - Broad exception catching
try:
    config = load_configuration(path)
except Exception as e:  # Too broad
    print(f"Something went wrong: {e}")
```

### Error Messages
- Be specific and actionable
- Include relevant context (file paths, values, etc.)
- Suggest solutions when possible

```python
# Good
raise ConfigFileNotFoundError(
    f"Configuration file not found at '{config_path}'. "
    f"Please ensure the file exists or provide a valid path."
)

# Less helpful
raise Exception("Config error")
```

## Type Annotations

### Required Annotations
- All function parameters and return values
- Class attributes and instance variables
- Module-level variables and constants

```python
from typing import Dict, List, Optional, Union, Any

# Function annotations
def process_data(
    input_data: List[Dict[str, Any]], 
    output_format: str = "json"
) -> Optional[str]:
    pass

# Variable annotations
config_cache: Dict[str, Any] = {}
retry_count: int = 0
is_initialized: bool = False
```

### Type Hint Best Practices
- Use `Optional[T]` for values that can be None
- Use `Union[T1, T2]` sparingly - consider refactoring instead
- Use `Any` as a last resort
- Import types from `typing` module

## Import Standards

### Import Organization
```python
# 1. Standard library imports
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

# 2. Third-party library imports
import numpy as np
import pandas as pd

# 3. Local application imports
from fusion.core.config import ConfigManager
from fusion.utils.helpers import normalize_path
```

### Import Guidelines
- Use absolute imports for clarity
- Group imports logically with blank lines
- Use `from module import specific_items` for frequently used items
- Avoid `import *` except in `__init__.py` files

## Testing Standards

### Test File Organization
```
tests/
├── unit/
│   ├── test_config.py
│   └── test_utils.py
├── integration/
│   └── test_pipeline.py
└── fixtures/
    └── sample_config.ini
```

### Test Naming
- Use descriptive test names: `test_config_loading_with_missing_file`
- Follow pattern: `test_[what]_[when]_[expected]`
- Group related tests in classes

### Test Structure
```python
def test_configuration_loading_with_valid_file():
    """Test that valid configuration file loads successfully."""
    # Arrange
    config_path = "tests/fixtures/valid_config.ini"
    expected_values = {"database": {"host": "localhost"}}
    
    # Act
    result = load_configuration(config_path)
    
    # Assert
    assert result["database"]["host"] == "localhost"
    assert "database" in result
```

## Code Quality Tools

### Required Tools
- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

### Pre-commit Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Performance Guidelines

### General Performance
- Avoid premature optimization
- Profile before optimizing
- Use appropriate data structures
- Cache expensive operations when beneficial

### Memory Management
- Use generators for large datasets
- Close file handles and connections
- Avoid circular references
- Clean up resources in finally blocks

## Security Guidelines

### Input Validation
- Validate all external input
- Sanitize file paths
- Use parameterized queries for databases
- Avoid eval() and exec()

### Configuration Security
- Never commit secrets or API keys
- Use environment variables for sensitive data
- Validate configuration values
- Set appropriate file permissions

---

## Quick Reference

### Naming Checklist
- [ ] Functions use verb phrases (`load_config`, `validate_data`)
- [ ] Variables use noun phrases (`config_path`, `user_settings`)  
- [ ] Complex data types use suffixes (`settings_dict`, `users_list`)
- [ ] Constants are SCREAMING_SNAKE_CASE
- [ ] Classes are PascalCase
- [ ] Private members start with `_`

### Code Quality Checklist
- [ ] Functions are under 50 lines
- [ ] Files are under 500 lines
- [ ] All public functions have docstrings
- [ ] Type annotations on all parameters/returns
- [ ] Specific exception handling
- [ ] Imports organized correctly
- [ ] Tests cover main functionality

---

*This document should be updated as the codebase evolves and new patterns emerge.*