"""
Reinforcement Learning configuration constants and arguments.

This package contains configuration constants used throughout the FUSION RL module:
- Algorithm definitions and registries
- Observation space configurations
- Strategy definitions
"""

from fusion.modules.rl.args.general_args import (
    EPISODIC_STRATEGIES,
    VALID_CORE_ALGORITHMS,
    VALID_DEEP_REINFORCEMENT_LEARNING_ALGORITHMS,
    VALID_DRL_ALGORITHMS,
    VALID_PATH_ALGORITHMS,
)

from fusion.modules.rl.args.observation_args import (
    OBS_DICT,
    OBSERVATION_SPACE_DEFINITIONS,
    VALID_OBSERVATION_FEATURES,
)

__all__ = [
    # From general_args
    "EPISODIC_STRATEGIES",
    "VALID_CORE_ALGORITHMS",
    "VALID_DEEP_REINFORCEMENT_LEARNING_ALGORITHMS",
    "VALID_DRL_ALGORITHMS",
    "VALID_PATH_ALGORITHMS",
    # From observation_args
    "OBS_DICT",
    "OBSERVATION_SPACE_DEFINITIONS",
    "VALID_OBSERVATION_FEATURES",
]
