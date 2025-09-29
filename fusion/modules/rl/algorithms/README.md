# RL Algorithms Module

This module provides reinforcement learning algorithm implementations for the FUSION network simulation framework.

## Overview

The algorithms module contains implementations of various reinforcement learning algorithms used for network optimization, including traditional multi-armed bandits, Q-learning, and deep reinforcement learning methods.

## Key Components

### Base Classes
- **BaseDRLAlgorithm**: Base class for deep reinforcement learning algorithms, providing common functionality for observation and action space handling

### Multi-Armed Bandits
- **EpsilonGreedyBandit**: Implements epsilon-greedy strategy for exploration vs exploitation
- **UCBBandit**: Upper Confidence Bound bandit for optimistic action selection

### Q-Learning
- **QLearning**: Traditional tabular Q-learning implementation

### Deep RL Algorithms
- **DQN**: Deep Q-Network implementation
- **QrDQN**: Quantile Regression Deep Q-Network
- **PPO**: Proximal Policy Optimization
- **A2C**: Advantage Actor-Critic

### Supporting Components
- **Algorithm Properties**: Configuration classes (RLProps, QProps, BanditProps, PPOProps)
- **Model Persistence**: Classes for saving and loading trained models
- **Utility Functions**: Helper functions for model management

## Usage

### Basic Bandit Usage
```python
from fusion.modules.rl.algorithms import EpsilonGreedyBandit

# Initialize bandit for path selection
bandit = EpsilonGreedyBandit(rl_props, engine_props, is_path=True)

# Select action
action = bandit.select_path_arm(source=0, dest=5)

# Update with reward
bandit.update(arm=action, reward=1.0, iteration=0, trial=0)
```

### Deep RL Usage
```python
from fusion.modules.rl.algorithms import DQN

# Initialize DQN
dqn = DQN(rl_props, engine_obj)

# Get observation and action spaces
obs_space = dqn.get_obs_space()
action_space = dqn.get_action_space()
```

## Dependencies

- NumPy: Numerical computations
- Gymnasium: RL environment interfaces
- JSON: Model serialization
- Pathlib: Modern path handling

## Architecture

The module is designed with clear separation of concerns:

1. **Base functionality** in `BaseDRLAlgorithm` for common DRL operations
2. **Algorithm-specific implementations** for different RL methods
3. **Persistence layer** for model saving/loading
4. **Configuration classes** for algorithm parameters
5. **Utility functions** for shared operations

## File Organization

```
algorithms/
├── __init__.py              # Public API exports
├── README.md               # This file
├── base_drl.py            # Base DRL functionality
├── bandits.py             # Multi-armed bandit algorithms
├── q_learning.py          # Q-learning implementation
├── dqn.py                 # Deep Q-Network
├── qr_dqn.py              # Quantile Regression DQN
├── ppo.py                 # Proximal Policy Optimization
├── a2c.py                 # Advantage Actor-Critic
├── algorithm_props.py     # Configuration classes
└── persistence.py         # Model persistence utilities
```

## Future Extensions

The module is designed to easily accommodate additional RL algorithms while maintaining consistent interfaces and coding standards.