"""
FUSION Failures Module

Provides failure injection and tracking for network survivability testing.

Supports:
- F1: Link failures
- F2: Node failures
- F3: SRLG (Shared Risk Link Group) failures
- F4: Geographic failures (hop-radius based)

Example usage:
    >>> from fusion.modules.failures import FailureManager
    >>> manager = FailureManager(engine_props, topology)
    >>> event = manager.inject_failure(
    ...     'geo',
    ...     t_fail=100.0,
    ...     t_repair=200.0,
    ...     center_node=5,
    ...     hop_radius=2
    ... )
    >>> is_feasible = manager.is_path_feasible([0, 1, 2])
"""

from .errors import (
    FailureConfigError,
    FailureError,
    FailureNotFoundError,
    InvalidFailureTypeError,
)
from .failure_manager import FailureManager
from .failure_types import fail_geo, fail_link, fail_node, fail_srlg
from .registry import get_failure_handler, register_failure_type

__all__ = [
    # Main classes
    "FailureManager",
    # Failure type functions
    "fail_link",
    "fail_node",
    "fail_srlg",
    "fail_geo",
    # Registry
    "get_failure_handler",
    "register_failure_type",
    # Exceptions
    "FailureError",
    "FailureConfigError",
    "FailureNotFoundError",
    "InvalidFailureTypeError",
]

__version__ = "1.0.0"
