"""
RL Algorithms Module.

Provides reinforcement learning algorithm implementations for the FUSION
network simulation framework, including Q-learning, bandits, and deep RL methods.
"""

# Public API exports
from .a2c import A2C
from .algorithm_props import BanditProps, PPOProps, QProps, RLProps
from .bandits import EpsilonGreedyBandit, UCBBandit
from .base_drl import BaseDRLAlgorithm
from .dqn import DQN
from .persistence import BanditModelPersistence, QLearningModelPersistence
from .ppo import PPO
from .q_learning import QLearning
from .qr_dqn import QrDQN

__version__ = "1.0.0"

__all__ = [
    "BaseDRLAlgorithm",
    "QLearning",
    "EpsilonGreedyBandit",
    "UCBBandit",
    "A2C",
    "PPO",
    "DQN",
    "QrDQN",
    "RLProps",
    "QProps",
    "BanditProps",
    "PPOProps",
    "BanditModelPersistence",
    "QLearningModelPersistence",
]
