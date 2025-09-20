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

### Variable Naming Best Practices
- **Be descriptive**: `blocked_requests` instead of `blocked_reqs`
- **Use full words**: `current_congestion` instead of `curr_cong`
- **Indicate units when relevant**: `timeout_seconds`, `distance_km`
- **Boolean variables should sound like questions**: `is_valid`, `has_data`, `was_routed`

### ⚠️ CRITICAL: Refactoring Variable Names

When changing variable or function names, you MUST:

1. **Search entire codebase** for all occurrences:
   - Variable assignments and usage
   - Function/method calls
   - Dictionary keys that might reference the name
   - Configuration files that might use the name
   - Tests that might depend on the name

2. **Be especially careful with**:
   - Dictionary keys (e.g., `stats_dict['blocking_prob']`)
   - Public API methods that external code might call
   - Configuration parameters that users might have in their files
   - Serialized data formats that might break compatibility

3. **Example of careful refactoring**:
```python
# BEFORE: Check if 'req_num' is used as:
# - Variable: self.req_num
# - Parameter: def process(req_num: int)
# - Dict key: data['req_num']
# - Config: config.req_num
# - External API: might be called by other modules

# AFTER: Change ALL occurrences consistently
# BUT preserve dictionary keys if they're part of data format:
data['req_num'] = request_number  # Keep key for compatibility
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

### Single Responsibility Principle

Each class should have **one reason to change** and **one clear responsibility**:

```python
# BAD: Too many responsibilities
class SimStats:
    def collect_metrics(self): pass        # ✅ Core responsibility
    def calculate_statistics(self): pass   # ✅ Related to metrics
    def save_to_file(self): pass          # ❌ File I/O responsibility
    def generate_ml_data(self): pass       # ❌ ML-specific responsibility
    def analyze_network(self): pass        # ❌ Network analysis responsibility

# GOOD: Single responsibility
class SimStats:
    def collect_metrics(self): pass
    def calculate_statistics(self): pass
    def get_blocking_statistics(self): pass
    
class StatsPersistence:
    def save_stats(self): pass
    def load_stats(self): pass
    
class MLMetricsCollector:
    def update_train_data(self): pass
    def save_train_data(self): pass
```

### File and Class Responsibility Guidelines

**Before adding a method to a class, ask:**
- Does this method directly relate to the class's core purpose?
- Would this method make sense in a different, more specialized class?
- Does this method introduce dependencies that don't belong?

**When to split classes:**
- File I/O operations (→ dedicated persistence classes)
- Format-specific operations (→ formatter classes)
- Domain-specific logic (→ specialized domain classes)
- External integrations (→ adapter/connector classes)

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

## Logging Guidelines

### Logging Architecture
- Use the centralized logging configuration from `fusion.utils.logging_config`
- Separate presentation logic from data collection
- Use the reporting module for output formatting

### Logger Setup
```python
# At module level
from fusion.utils.logging_config import get_logger
logger = get_logger(__name__)

# For simulation-specific logging
from fusion.utils.logging_config import configure_simulation_logging
logger = configure_simulation_logging(sim_name, erlang, thread_num)
```

### Logging Levels
- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General informational messages
- **WARNING**: Warning messages for potentially harmful situations
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical messages for very serious errors

### Best Practices
- Never use `print()` statements - always use logging
- Log at appropriate levels (don't log everything as INFO)
- Include context in log messages
- Use structured logging for complex data

```python
# Good
logger.info(f"Processing request {request_id} for user {user_id}")
logger.error(f"Failed to connect to database: {e}", exc_info=True)

# Avoid
print(f"Processing request")  # Use logger instead
logger.info("Error occurred")  # Too vague, wrong level
```

### Output Organization
- Runtime logs go to `logs/` directory (gitignored)
- Simulation results go to `data/output/`
- Reporting code goes in `fusion/reporting/`

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
- [ ] No print statements (use logging)
- [ ] Proper separation of concerns
- [ ] Variable names are descriptive and complete
- [ ] All name changes verified across codebase

## Bash Script Coding Standards

### File Organization
- Store bash scripts in dedicated directories (`bash_scripts/`, `scripts/`)
- Use descriptive filenames that indicate the script's purpose
- Include `.sh` extension for all bash scripts
- Group related scripts together (e.g., cluster management, environment setup)

### Script Structure
```bash
#!/bin/bash

# Script description: What this script does and when to use it
# Usage: ./script_name.sh <param1> <param2>
# Example: ./script_name.sh /path/to/file "argument"

# Exit on any error
set -e

# Function definitions
function_name() {
    local param1="$1"
    local param2="$2"
    # Function body
}

# Main script logic
main() {
    # Script implementation
}

# Call main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

### Header Requirements
Every bash script MUST include:
- Shebang line: `#!/bin/bash`
- Purpose description comment
- Usage instructions with examples
- Parameter descriptions for complex scripts

### Variable Naming
- Use `UPPER_CASE` for environment variables and constants
- Use `snake_case` for local variables
- Use `readonly` for constants when appropriate
- Quote all variable references: `"$variable"`

```bash
# Good
readonly SCRIPT_DIR="$(dirname "$0")"
user_name="$1"
target_directory="$2"

# Avoid
SCRIPTDIR=$(dirname $0)  # No quotes, inconsistent case
userName=$1              # camelCase not preferred
```

### Error Handling
- Use `set -e` to exit on errors
- Check for required parameters
- Validate file/directory existence before operations
- Provide helpful error messages

```bash
# Good error handling
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <username> <partition>" >&2
    exit 1
fi

if [[ ! -f "$config_file" ]]; then
    echo "Error: Configuration file not found: $config_file" >&2
    exit 1
fi
```

### Function Guidelines
- Use functions for reusable code blocks
- Use `local` for function variables
- Return meaningful exit codes
- Document complex functions

```bash
# Good function structure
validate_input() {
    local input="$1"
    local context="$2"
    
    if [[ -z "$input" ]]; then
        echo "Error: Empty $context provided" >&2
        return 1
    fi
    
    return 0
}
```

### Command Execution
- Use full paths for commands when possible
- Check command availability with `command -v`
- Handle command failures gracefully
- Quote arguments that may contain spaces

```bash
# Good
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found" >&2
    exit 1
fi

python3 -m venv "$target_directory/venv"
```

### Output and Logging
- Use `echo` for informational messages
- Use `echo ... >&2` for error messages
- Include timestamps for long-running operations
- Use consistent formatting for status messages

```bash
# Good output formatting
echo "🔧 Creating virtual environment..."
echo "✅ Virtual environment created successfully"
echo "❌ Error: Operation failed" >&2
```

### Security Best Practices
- Validate all input parameters
- Use `readonly` for sensitive variables
- Avoid executing user-provided strings
- Use proper file permissions (755 for executables)

### SLURM Integration Standards
For cluster-related scripts:
- Use environment variables for SLURM parameters
- Include job array handling when applicable
- Implement proper resource management
- Log job progress and completion status

```bash
# SLURM script standards
echo "🌟 Starting SLURM Job ${SLURM_ARRAY_TASK_ID}"
echo "Manifest: ${MANIFEST}"
echo "Job Directory: ${JOB_DIR}"
```

### Testing and Validation
- Include dry-run options where applicable
- Test scripts with various input combinations
- Validate script behavior in different environments
- Document any environment-specific requirements

---

*This document should be updated as the codebase evolves and new patterns emerge.*