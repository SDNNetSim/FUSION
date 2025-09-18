"""
Algorithm registry configuration for reinforcement learning implementations.

This module defines the registry mapping that associates algorithm names
with their setup functions and implementation classes.
"""
from typing import Dict, Any, Optional, Callable

# Import setup functions
from fusion.modules.rl.utils.setup import setup_ppo
from fusion.modules.rl.utils.setup import setup_a2c
from fusion.modules.rl.utils.setup import setup_dqn
from fusion.modules.rl.utils.setup import setup_qr_dqn

# Import algorithm classes
from fusion.modules.rl.algorithms.ppo import PPO
from fusion.modules.rl.algorithms.a2c import A2C
from fusion.modules.rl.algorithms.dqn import DQN
from fusion.modules.rl.algorithms.qr_dqn import QrDQN

# Type alias for clarity
AlgorithmConfig = Dict[str, Optional[Callable[..., Any]]]

# Algorithm registry mapping algorithm names to their configurations
ALGORITHM_REGISTRY: Dict[str, AlgorithmConfig] = {
    'ppo': {
        'setup': setup_ppo,
        'load': None,  # TODO: Implement model loading functionality
        'class': PPO,
    },
    'a2c': {
        'setup': setup_a2c,
        'load': None,  # TODO: Implement model loading functionality
        'class': A2C,
    },
    'dqn': {
        'setup': setup_dqn,
        'load': None,  # TODO: Implement model loading functionality
        'class': DQN,
    },
    'qr_dqn': {
        'setup': setup_qr_dqn,
        'load': None,  # TODO: Implement model loading functionality
        'class': QrDQN,
    },
}
