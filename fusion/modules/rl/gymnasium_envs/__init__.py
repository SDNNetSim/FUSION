"""
Gymnasium environments for FUSION simulation.

This package provides Gymnasium-compatible environment implementations
for reinforcement learning with FUSION network simulations.
"""

from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
from fusion.modules.rl.gymnasium_envs.constants import (
    DEFAULT_SIMULATION_KEY,
    DEFAULT_SAVE_SIMULATION,
    SUPPORTED_SPECTRAL_BANDS,
    ARRIVAL_DICT_KEYS,
    DEFAULT_ITERATION,
    DEFAULT_ARRIVAL_COUNT
)

__all__ = [
    # Main environment class
    'SimEnv',
    # Constants
    'DEFAULT_SIMULATION_KEY',
    'DEFAULT_SAVE_SIMULATION',
    'SUPPORTED_SPECTRAL_BANDS',
    'ARRIVAL_DICT_KEYS',
    'DEFAULT_ITERATION',
    'DEFAULT_ARRIVAL_COUNT'
]
