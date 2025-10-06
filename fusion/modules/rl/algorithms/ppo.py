"""Proximal Policy Optimization (PPO) algorithm implementation."""

# Import spaces at module level for test compatibility
from gymnasium import spaces
from fusion.modules.rl.algorithms.base_drl import (
    BaseDRLAlgorithm,  # pylint: disable=unused-import
)


class PPO(BaseDRLAlgorithm):
    """
    Facilitates Proximal Policy Optimization (PPO) for reinforcement learning.

    This class provides functionalities for handling observation space, action space,
    and rewards specific to the PPO framework for reinforcement learning. It inherits
    common DRL functionality from BaseDRLAlgorithm to eliminate code duplication.
    """
