"""
Configuration constants for reinforcement learning algorithms.

This module defines the valid algorithm types and strategies
available in the FUSION RL module.
"""

# Strategies for epsilon decay in exploration-exploitation trade-off
EPISODIC_STRATEGIES: list[str] = ["exp_decay", "linear_decay"]

# Algorithms valid for path selection tasks
# Includes both traditional RL and deep RL algorithms
VALID_PATH_ALGORITHMS: list[str] = [
    # Traditional RL algorithms
    "q_learning",
    "epsilon_greedy_bandit",
    "ucb_bandit",
    # Deep RL algorithms
    "ppo",  # Proximal Policy Optimization
    "a2c",  # Advantage Actor-Critic
    "dqn",  # Deep Q-Network
    "qr_dqn",  # Quantile Regression DQN
]

# Traditional RL algorithms for core network decisions
VALID_CORE_ALGORITHMS: list[str] = [
    "q_learning",
    "epsilon_greedy_bandit",
    "ucb_bandit",
]

# Deep reinforcement learning algorithms
# Note: Abbreviated name kept for backward compatibility
VALID_DRL_ALGORITHMS: list[str] = [
    "ppo",  # Proximal Policy Optimization
    "a2c",  # Advantage Actor-Critic
    "dqn",  # Deep Q-Network
    "qr_dqn",  # Quantile Regression DQN
]

# Full name alias for better clarity in new code
VALID_DEEP_REINFORCEMENT_LEARNING_ALGORITHMS: list[str] = VALID_DRL_ALGORITHMS
