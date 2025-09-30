"""
Utility modules for FUSION.

This package provides common utilities used across the FUSION codebase.
Import directly from specific modules to avoid circular dependencies.

Example:
    from fusion.utils.logging_config import get_logger
    from fusion.utils.network import find_path_length
"""

# Only import the most commonly used utilities that don't create circular dependencies
from fusion.utils.logging_config import get_logger, setup_logger

__all__ = [
    "setup_logger",
    "get_logger",
]
