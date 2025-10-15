"""
Custom exceptions for the failures module.
"""


class FailureError(Exception):
    """Base exception for failure-related errors."""

    pass


class FailureConfigError(FailureError):
    """Raised when failure configuration is invalid."""

    pass


class FailureNotFoundError(FailureError):
    """Raised when a requested failure cannot be found."""

    pass


class InvalidFailureTypeError(FailureError):
    """Raised when an unknown failure type is requested."""

    pass
