"""
Algorithm registry configuration for reinforcement learning implementations.

This module defines the registry mapping that associates algorithm names
with their setup functions and implementation classes.
"""
from collections.abc import Callable
from typing import Any

# Import algorithm classes
from fusion.modules.rl.algorithms.a2c import A2C
from fusion.modules.rl.algorithms.dqn import DQN
from fusion.modules.rl.algorithms.ppo import PPO
from fusion.modules.rl.algorithms.qr_dqn import QrDQN

# Import setup functions
from fusion.modules.rl.utils.setup import setup_a2c, setup_dqn, setup_ppo, setup_qr_dqn

# Type alias for clarity
AlgorithmConfig = dict[str, Callable[..., Any] | None]

# Algorithm registry mapping algorithm names to their configurations
ALGORITHM_REGISTRY: dict[str, AlgorithmConfig] = {
    "a2c": {
        "class": A2C,
        "load": None,  # TODO: Implement model loading functionality
        "setup": setup_a2c,
    },
    "dqn": {
        "class": DQN,
        "load": None,  # TODO: Implement model loading functionality
        "setup": setup_dqn,
    },
    "ppo": {
        "class": PPO,
        "load": None,  # TODO: Implement model loading functionality
        "setup": setup_ppo,
    },
    "qr_dqn": {
        "class": QrDQN,
        "load": None,  # TODO: Implement model loading functionality
        "setup": setup_qr_dqn,
    },
}
