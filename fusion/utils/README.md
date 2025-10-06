# Utils Module

## Purpose
Common utility functions used throughout the FUSION codebase, providing standardized implementations for configuration management, logging, file operations, and random number generation.

## Key Components
- `config.py`: Configuration parsing and type conversion utilities
- `logging_config.py`: Centralized logging setup and configuration
- `os.py`: Operating system utilities (path handling, directory operations)
- `random.py`: Random number generation for simulations

## Usage
```python
from fusion.utils import setup_logger, create_directory, set_random_seed

# Set up logging for a module
logger = setup_logger(__name__, level="INFO")

# Create directories safely
create_directory("/path/to/output")

# Ensure reproducible random numbers
set_random_seed(42)
```

## Dependencies
Internal dependencies:
- `fusion.configs.constants` (for configuration utilities)
- `fusion.configs.errors` (for custom exceptions)

External dependencies:
- `numpy` (for random number generation)
- Standard library: `ast`, `logging`, `pathlib`, `datetime`