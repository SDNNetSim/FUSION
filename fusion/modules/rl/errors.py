"""
Custom exceptions for the reinforcement learning module.

This module defines a hierarchy of exceptions used throughout the RL components
for proper error handling and debugging.
"""


class RLError(Exception):
    """
    Base exception for all RL module errors.

    This serves as the root exception class for all reinforcement learning
    related errors, allowing for broad exception handling when needed.
    """


class RLConfigurationError(RLError):
    """
    Raised when RL configuration parameters are invalid or inconsistent.

    This exception is used when there are issues with simulation dictionaries,
    hyperparameters, or other configuration-related problems.
    """


class AlgorithmNotFoundError(RLError):
    """
    Raised when a requested algorithm is not found in the registry.

    This occurs when attempting to use an algorithm that hasn't been
    registered or doesn't exist in the available algorithm set.
    """


class ModelLoadError(RLError):
    """
    Raised when model loading operations fail.

    This exception covers failures in loading pre-trained models from disk,
    including file not found, corrupted models, or version incompatibilities.
    """


class TrainingError(RLError):
    """
    Raised during training process failures.

    This covers errors that occur during the training loop, model updates,
    or other training-specific operations.
    """


class RLEnvironmentError(RLError):
    """
    Raised when RL environment setup or operations fail.

    This includes errors in environment initialization, state transitions,
    or environment-specific operations.
    """


class AgentError(RLError):
    """
    Raised when agent operations fail.

    This serves as a base class for agent-specific errors and covers
    general agent operation failures.
    """


class InvalidActionError(AgentError):
    """
    Raised when an invalid action is attempted by an agent.

    This occurs when an agent tries to take an action that is not valid
    in the current state or is outside the action space.
    """


class RouteSelectionError(AgentError):
    """
    Raised when route selection operations fail.

    This covers errors in path selection algorithms, routing decisions,
    or route-related computations.
    """
