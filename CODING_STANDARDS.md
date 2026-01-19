# FUSION Coding Standards

> **AI/LLM Usage Note**: This document is optimized for Claude to understand and apply FUSION project coding standards consistently.

## 1. Naming Conventions

### 1.1 Basic Rules
- **Functions**: `snake_case` verbs (`load_config`, `validate_data`)
- **Variables**: `snake_case` nouns (`config_path`, `user_settings`)
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Classes**: `PascalCase`
- **Private**: Prefix with `_`

### 1.2 Data Type Suffixes (Deprecated)
**Note**: Type suffixes like `_dict`, `_list`, `_set` are now discouraged. Modern Python type hints make the type clear without redundant suffixes.

```python
# Good - let type hints show the type
user_settings: dict[str, Any] = {}
active_connections: set[str] = set()
pending_requests: list[Request] = []

# Bad - redundant type suffixes
user_settings_dict: dict[str, Any] = {}  # Redundant
active_connections_set: set[str] = set()  # Redundant
pending_requests_list: list[Request] = []  # Redundant
```

**Exception**: Use suffixes only when you have multiple representations of the same data:
```python
# OK when differentiating between formats
user_data_json: str = '{"name": "John"}'
user_data_dict: dict = {"name": "John"}
```

### 1.3 Variable Best Practices
- **Descriptive**: `blocked_requests` not `blocked_reqs`
- **Full words**: `current_congestion` not `curr_cong`
- **Units**: `timeout_seconds`, `distance_km`
- **Booleans**: `is_valid`, `has_data`, `was_routed`

### 1.4 CRITICAL: Refactoring Names
When changing variable/function names:
1. **Search entire codebase** for ALL occurrences
2. Check dictionary keys, config files, tests
3. Preserve external API compatibility
4. Update consistently across all files

## 2. Code Organization

### 2.1 Module Structure
```
package/
├── __init__.py          # Public API exports
├── constants.py         # Module constants
├── errors.py           # Custom exceptions
├── core.py             # Main functionality
├── utils.py            # Utility functions
├── registry.py         # Component registry (if needed)
├── README.md           # Module documentation
└── tests/              # Unit tests
    ├── __init__.py
    └── test_*.py
```

### 2.1.1 __init__.py Standards
```python
"""Package description.

Brief description of what this package does and its main components.
"""

# Public API exports
from .core import MainClass, primary_function
from .errors import CustomError, ValidationError
from .constants import DEFAULT_CONFIG, MAX_RETRIES

# Version info
__version__ = "1.0.0"

# Public API - explicitly define what's exported
__all__ = [
    "MainClass",
    "primary_function",
    "CustomError",
    "ValidationError",
    "DEFAULT_CONFIG",
    "MAX_RETRIES",
]
```

### 2.1.2 Registry.py Guidelines
**When to include `registry.py`**:
- Yes: Modules with multiple component types (e.g., algorithms, strategies)
- Yes: Plugin-style architectures
- Yes: Factory pattern implementations
- No: Simple utility modules
- No: Single-purpose modules

```python
# Example registry.py
class ComponentRegistry:
    """Registry for component discovery and instantiation."""

    _components = {}

    @classmethod
    def register(cls, name: str, component_class):
        """Register a component class."""
        cls._components[name] = component_class

    @classmethod
    def get_component(cls, name: str):
        """Get registered component by name."""
        return cls._components.get(name)
```

### 2.1.3 README.md Requirements
**Every module MUST have a README.md** with:
```markdown
# Module Name

## Purpose
Brief description of what this module does.

## Key Components
- `core.py`: Main functionality
- `utils.py`: Helper functions

## Usage
Basic usage examples.

## Dependencies
Internal and external dependencies.
```

### 2.2 Class Organization
```python
class ExampleClass:
    """Docstring."""

    CLASS_CONSTANT = "value"    # Class variables

    def __init__(self): pass    # Constructor
    def public_method(self): pass    # Public methods
    @classmethod
    def class_method(cls): pass      # Class methods
    @staticmethod
    def static_method(): pass       # Static methods
    def _private_method(self): pass  # Private methods last
```

### 2.3 Single Responsibility
```python
# Good - Single responsibility
class SimStats:
    def collect_metrics(self): pass
    def calculate_statistics(self): pass

class StatsPersistence:
    def save_stats(self): pass

# Bad - Multiple responsibilities
class SimStats:
    def collect_metrics(self): pass
    def save_to_file(self): pass      # File I/O responsibility
    def generate_ml_data(self): pass   # ML responsibility
```

## 3. Path Handling & State Management

### 3.1 Path Handling Guidelines
**NEVER hardcode paths** - Use configuration and path utilities

```python
# Bad - never do this
def load_data():
    with open("/home/user/data/config.txt") as f:
        return f.read()

# Good - always do this
from pathlib import Path
from fusion.core.config import get_data_dir

def load_data(config_filename: str = "config.txt"):
    data_dir = get_data_dir()  # From configuration
    config_path = Path(data_dir) / config_filename
    with open(config_path) as f:
        return f.read()
```

**Path handling best practices**:
- Use `pathlib.Path` for all path operations
- Get base paths from configuration/environment
- Use relative paths from known base directories
- Validate paths exist before using
- Use forward slashes (/) - `pathlib` handles OS differences

```python
# Path construction
base_dir = Path(config.get_base_directory())
output_file = base_dir / "results" / f"simulation_{timestamp}.json"

# Path validation
if not output_file.parent.exists():
    output_file.parent.mkdir(parents=True, exist_ok=True)
```

### 3.2 State Management with StateWrapper
**Use StateWrapper for mutable state objects like `engine_props`**

```python
# fusion/core/state_wrapper.py
class StateWrapper:
    """Minimal wrapper that preserves dict interface but adds safety."""

    def __init__(self, data: dict, name: str = "state"):
        self._data = data
        self._name = name
        self._original = data.copy()  # Keep original for debugging
        self._frozen = False
        self._log_mutations = False

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if self._frozen:
            raise RuntimeError(f"Cannot modify frozen state: {self._name}")

        if self._log_mutations:
            old_value = self._data.get(key, "<missing>")
            print(f"[{self._name}] {key}: {old_value} -> {value}")

        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def freeze(self):
        """Temporarily prevent mutations during critical sections."""
        self._frozen = True

    def unfreeze(self):
        """Allow mutations again."""
        self._frozen = False

    def get_changes(self):
        """See what changed since initialization."""
        return {k: v for k, v in self._data.items()
                if k not in self._original or self._original[k] != v}

    # Make it dict-like
    def __getattr__(self, name):
        return getattr(self._data, name)
```

**Usage**:
```python
# In simulation initialization
def __init__(self, engine_props: dict):
    # Wrap mutable state for safety
    self.engine_props = StateWrapper(engine_props, "engine_props")

    # Enable mutation logging during development
    if DEBUG_MODE:
        self.engine_props._log_mutations = True

# During critical sections
def critical_calculation(self):
    self.engine_props.freeze()  # Prevent accidental mutations
    try:
        result = complex_calculation(self.engine_props)
        return result
    finally:
        self.engine_props.unfreeze()
```

## 4. Type Annotations & Imports

### 4.1 Required Annotations
- All function parameters and returns
- Class attributes
- Module-level variables

```python
def process_data(input_data: List[Dict[str, Any]], format: str = "json") -> Optional[str]:
    pass
```

### 4.2 Import Organization
```python
# 1. Standard library
import os
from pathlib import Path

# 2. Third-party
import numpy as np

# 3. Local
from fusion.core.config import ConfigManager
```

## 5. Error Handling

### 5.1 Custom Exceptions
```python
class ConfigurationError(Exception):
    """Base exception for configuration errors."""
    pass

class ConfigFileNotFoundError(ConfigurationError):
    """Configuration file not found."""
    pass
```

### 5.2 Exception Handling
```python
# Good - specific exceptions
try:
    config = load_configuration(path)
except ConfigFileNotFoundError:
    logger.error(f"Config file not found: {path}")
    return default_config()

# Bad - broad catching
except Exception as e:  # Too broad
    pass
```

## 6. Documentation

### 6.1 Docstrings (Sphinx)
```python
def load_configuration(config_path: str) -> Dict[str, Any]:
    """Load and parse configuration file.

    :param config_path: Path to configuration file
    :type config_path: str
    :return: Parsed configuration data
    :rtype: Dict[str, Any]
    :raises ConfigFileNotFoundError: If config file doesn't exist
    """
```

### 6.2 Comments
- Explain **why**, not **what**
- Use sparingly - code should be self-documenting

## 7. Logging

### 7.1 Setup
```python
from fusion.utils.logging_config import get_logger
logger = get_logger(__name__)
```

### 7.2 Usage
```python
# Good - always use logger
logger.info(f"Processing request {request_id}")
logger.error(f"Failed to connect: {e}", exc_info=True)

# Bad - never use print
print("Processing request")
```

## 8. Testing

**Note**: See `testing_standards.md` for details

### 8.1 Module-Level Testing
- Tests in same directory as code
- Test files: `test_<module_name>.py`
- One test file per module

## 9. Bash Scripts

### 9.1 Structure
```bash
#!/bin/bash
set -e

# Usage: ./script.sh <param1> <param2>

function_name() {
    local param="$1"
    # Function body
}

main() {
    # Script logic
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

### 9.2 Standards
- Quote variables: `"$variable"`
- `UPPER_CASE` for constants
- `snake_case` for local variables
- Include error checking

## 10. Code Quality Tools

### 10.1 Pre-commit Tools
- **ruff**: Code formatting and linting (replaces black, isort, flake8)
- **mypy**: Type checking
- **pytest**: Testing
- **pydeps**: Dependencies
- **vulture**: Dead code
- **mccabe**: Complexity
- **sphinx**: Documentation

### 10.2 Additional Tools
- **graphviz**: Dependency graphs
- **pyreverse**: UML diagrams
- **snakeviz**: Performance profiling
- **pyspy**: Python profiler

---

## Quick Reference Checklist

### Naming
- [ ] Functions: verb phrases (`load_config`)
- [ ] Variables: noun phrases (`config_path`)
- [ ] Avoid redundant type suffixes (use `settings` not `settings_dict`)
- [ ] Constants: `SCREAMING_SNAKE_CASE`
- [ ] Classes: `PascalCase`

### Code Quality
- [ ] Type annotations on all params/returns
- [ ] Specific exception handling
- [ ] No print statements (use logging)
- [ ] Single responsibility classes
- [ ] Search codebase when refactoring names
- [ ] No hardcoded paths (use pathlib + config)
- [ ] StateWrapper for mutable state objects

### Module Organization
- [ ] __init__.py with proper exports and __all__
- [ ] README.md in every module
- [ ] registry.py only when needed (multi-component modules)
- [ ] tests/ subdirectory with proper structure

---

*Follow these standards for consistent, maintainable code that AI can easily understand and work with.*
