"""Custom exceptions for the RL module."""


class RLError(Exception):
    """Base exception for RL module errors."""


class RLConfigurationError(RLError):
    """Raised when RL configuration is invalid."""


class AlgorithmNotFoundError(RLError):
    """Raised when requested algorithm is not registered."""


class ModelLoadError(RLError):
    """Raised when model loading fails."""


class TrainingError(RLError):
    """Raised during training failures."""


class RLEnvironmentError(RLError):
    """Raised when RL environment setup fails."""


class AgentError(RLError):
    """Raised when agent operations fail."""


class InvalidActionError(AgentError):
    """Raised when an invalid action is attempted."""


class RouteSelectionError(AgentError):
    """Raised when route selection fails."""
