"""Quantile Regression Deep Q-Network (QR-DQN) algorithm implementation."""

# Import spaces at module level for test compatibility
from fusion.modules.rl.algorithms.base_drl import (  # pylint: disable=unused-import
    BaseDRLAlgorithm,
)


class QrDQN(BaseDRLAlgorithm):
    """
    Facilitates Quantile Regression Deep Q-Network (QR-DQN) for reinforcement learning.

    This class provides QR-DQN-specific functionality while inheriting common
    DRL operations from BaseDRLAlgorithm to eliminate code duplication.
    """
