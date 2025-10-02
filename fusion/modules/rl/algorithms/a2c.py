"""Advantage Actor-Critic (A2C) algorithm implementation."""

# Import spaces at module level for test compatibility
from gymnasium import spaces
from fusion.modules.rl.algorithms.base_drl import BaseDRLAlgorithm


class A2C(BaseDRLAlgorithm):
    """
    Facilitates Advantage Actor-Critic (A2C) for reinforcement learning.

    This class provides functionalities for handling observation space and action
    space specific to the A2C framework for reinforcement learning. It inherits
    common DRL functionality from BaseDRLAlgorithm to eliminate code duplication.
    """
