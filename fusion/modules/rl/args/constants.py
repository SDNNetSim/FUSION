"""
Comprehensive constants definition for the FUSION RL module.

This module provides a centralized location for all RL-related constants,
organized by category for better maintainability and discoverability.
This file is intended for new code; existing code should continue using
the individual files for backward compatibility.
"""

from enum import Enum


class AlgorithmType(Enum):
    """Enumeration of algorithm types for better type safety."""

    # Traditional RL algorithms
    Q_LEARNING = "q_learning"
    EPSILON_GREEDY_BANDIT = "epsilon_greedy_bandit"
    UCB_BANDIT = "ucb_bandit"

    # Deep RL algorithms
    PPO = "ppo"  # Proximal Policy Optimization
    A2C = "a2c"  # Advantage Actor-Critic
    DQN = "dqn"  # Deep Q-Network
    QR_DQN = "qr_dqn"  # Quantile Regression DQN


class EpisodicStrategy(Enum):
    """Enumeration of episodic exploration strategies."""

    EXPONENTIAL_DECAY = "exp_decay"
    LINEAR_DECAY = "linear_decay"


class ObservationFeature(Enum):
    """Enumeration of all possible observation features."""

    SOURCE = "source"
    DESTINATION = "destination"
    REQUEST_BANDWIDTH = "request_bandwidth"
    HOLDING_TIME = "holding_time"
    SLOTS_NEEDED = "slots_needed"
    PATH_LENGTHS = "path_lengths"
    PATH_CONGESTION = "paths_cong"  # Note: abbreviated for backward compatibility
    AVAILABLE_SLOTS = "available_slots"
    IS_FEASIBLE = "is_feasible"


# Algorithm categorizations
# All traditional (non-deep) RL algorithms
TRADITIONAL_RL_ALGORITHMS: tuple[str, ...] = (
    AlgorithmType.Q_LEARNING.value,
    AlgorithmType.EPSILON_GREEDY_BANDIT.value,
    AlgorithmType.UCB_BANDIT.value,
)

# All deep reinforcement learning algorithms
DEEP_RL_ALGORITHMS: tuple[str, ...] = (
    AlgorithmType.PPO.value,
    AlgorithmType.A2C.value,
    AlgorithmType.DQN.value,
    AlgorithmType.QR_DQN.value,
)

# Algorithms suitable for path selection
PATH_SELECTION_ALGORITHMS: tuple[str, ...] = TRADITIONAL_RL_ALGORITHMS + DEEP_RL_ALGORITHMS

# Algorithms suitable for core network decisions
CORE_DECISION_ALGORITHMS: tuple[str, ...] = TRADITIONAL_RL_ALGORITHMS


# Observation space templates
# Minimal observation spaces
BASIC_ROUTING_OBSERVATION: list[str] = [
    ObservationFeature.SOURCE.value,
    ObservationFeature.DESTINATION.value,
]

# Standard observation spaces
ROUTING_WITH_BANDWIDTH_OBSERVATION: list[str] = BASIC_ROUTING_OBSERVATION + [
    ObservationFeature.REQUEST_BANDWIDTH.value,
]

ROUTING_WITH_TIME_OBSERVATION: list[str] = BASIC_ROUTING_OBSERVATION + [
    ObservationFeature.HOLDING_TIME.value,
]

ROUTING_STANDARD_OBSERVATION: list[str] = BASIC_ROUTING_OBSERVATION + [
    ObservationFeature.REQUEST_BANDWIDTH.value,
    ObservationFeature.HOLDING_TIME.value,
]

# Extended observation spaces
ROUTING_WITH_PATHS_OBSERVATION: list[str] = ROUTING_STANDARD_OBSERVATION + [
    ObservationFeature.SLOTS_NEEDED.value,
    ObservationFeature.PATH_LENGTHS.value,
]

ROUTING_WITH_CONGESTION_OBSERVATION: list[str] = ROUTING_WITH_PATHS_OBSERVATION + [
    ObservationFeature.PATH_CONGESTION.value,
]

ROUTING_WITH_RESOURCES_OBSERVATION: list[str] = ROUTING_WITH_CONGESTION_OBSERVATION + [
    ObservationFeature.AVAILABLE_SLOTS.value,
]

# Complete observation space
ROUTING_COMPLETE_OBSERVATION: list[str] = ROUTING_WITH_RESOURCES_OBSERVATION + [
    ObservationFeature.IS_FEASIBLE.value,
]


# Validation sets
VALID_FEATURES: set[str] = {feature.value for feature in ObservationFeature}
VALID_STRATEGIES: set[str] = {strategy.value for strategy in EpisodicStrategy}
VALID_ALGORITHMS: set[str] = {algo.value for algo in AlgorithmType}


# Default configurations
# Default exploration parameters
DEFAULT_EPSILON: float = 1.0
DEFAULT_EPSILON_MIN: float = 0.01
DEFAULT_EPSILON_DECAY: float = 0.995

# Default learning parameters
DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_DISCOUNT_FACTOR: float = 0.99

# Default training parameters
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_BUFFER_SIZE: int = 10000
DEFAULT_UPDATE_FREQUENCY: int = 4

# Default observation space
DEFAULT_OBSERVATION_SPACE: str = "obs_4"  # Routing with bandwidth and time
