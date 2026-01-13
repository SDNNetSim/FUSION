"""
Custom exceptions for the RL module utilities.

This module defines specific exception classes for different error conditions
that can occur in the RL module, providing better error handling and debugging.
"""


class RLUtilsError(Exception):
    """Base exception for RL utilities module errors."""


class ConfigurationError(RLUtilsError):
    """Raised when RL configuration is invalid or missing."""


class ModelSetupError(RLUtilsError):
    """Raised when model setup or initialization fails."""


class HyperparameterError(RLUtilsError):
    """Raised when hyperparameter configuration is invalid."""


class FeatureExtractorError(RLUtilsError):
    """Raised when feature extractor setup fails."""


class DataLoadingError(RLUtilsError):
    """Raised when data loading operations fail."""


class SimulationDataError(RLUtilsError):
    """Raised when simulation data processing encounters errors."""


class CacheError(RLUtilsError):
    """Raised when caching operations fail."""
