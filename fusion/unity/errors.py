"""
Custom exceptions for the Unity cluster management module.

This module defines specific exceptions for various error conditions that can
occur during Unity cluster operations, including manifest processing, job
submission, and result synchronization.
"""


class UnityError(Exception):
    """Base exception for all Unity-related errors."""

    pass


class ManifestError(UnityError):
    """Base exception for manifest-related errors."""

    pass


class ManifestNotFoundError(ManifestError):
    """Raised when a required manifest file cannot be found."""

    pass


class ManifestValidationError(ManifestError):
    """Raised when manifest data fails validation."""

    pass


class SpecificationError(UnityError):
    """Base exception for specification file errors."""

    pass


class SpecNotFoundError(SpecificationError):
    """Raised when a specification file cannot be found."""

    pass


class SpecValidationError(SpecificationError):
    """Raised when specification data is invalid."""

    pass


class JobSubmissionError(UnityError):
    """Raised when job submission to the cluster fails."""

    pass


class SynchronizationError(UnityError):
    """Base exception for file synchronization errors."""

    pass


class RemotePathError(SynchronizationError):
    """Raised when there are issues with remote path operations."""

    pass


class ConfigurationError(UnityError):
    """Raised when configuration is missing or invalid."""

    pass
