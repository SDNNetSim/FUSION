"""
Algorithm registry configuration for reinforcement learning implementations.

This module defines the registry mapping that associates algorithm names
with their setup functions and implementation classes.
"""

from collections.abc import Callable
from typing import Any

from fusion.modules.rl.algorithms.a2c import A2C
from fusion.modules.rl.algorithms.dqn import DQN

# Import algorithm classes
from fusion.modules.rl.algorithms.ppo import PPO
from fusion.modules.rl.algorithms.qr_dqn import QrDQN

# Import setup functions
from fusion.modules.rl.utils.setup import setup_a2c, setup_dqn, setup_ppo, setup_qr_dqn

# Type alias for clarity
AlgorithmConfig = dict[str, Callable[..., Any] | None]

# Algorithm registry mapping algorithm names to their configurations
ALGORITHM_REGISTRY: dict[str, AlgorithmConfig] = {
    "ppo": {
        "setup": setup_ppo,
        "load": None,  # TODO: Implement model loading functionality
        "class": PPO,
    },
    "a2c": {
        "setup": setup_a2c,
        "load": None,  # TODO: Implement model loading functionality
        "class": A2C,
    },
    "dqn": {
        "setup": setup_dqn,
        "load": None,  # TODO: Implement model loading functionality
        "class": DQN,
    },
    "qr_dqn": {
        "setup": setup_qr_dqn,
        "load": None,  # TODO: Implement model loading functionality
        "class": QrDQN,
    },
}
