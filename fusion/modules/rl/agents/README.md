# RL Agents Module

## Purpose
This module provides reinforcement learning agents for intelligent decision-making in the FUSION network simulation framework. It includes agents for path selection, core assignment, and spectrum allocation.

## Key Components
- `base_agent.py`: Base class providing common functionality for all RL agents
- `path_agent.py`: Agent for intelligent path selection using various RL algorithms
- `core_agent.py`: Agent for core assignment (currently not implemented)
- `spectrum_agent.py`: Agent for spectrum allocation (currently not implemented)

## Usage
```python
from fusion.modules.rl.agents import PathAgent

# Create a path agent with Q-learning
path_agent = PathAgent(
    path_algorithm='q_learning',
    rl_props=rl_props_object,
    rl_help_obj=rl_helper_object
)

# Setup environment
path_agent.engine_props = engine_properties
path_agent.setup_env(is_path=True)

# Get route for a request
path_agent.get_route()
```

## Dependencies
- Internal: fusion.modules.rl.algorithms, fusion.modules.rl.utils, fusion.modules.rl.errors
- External: numpy

## Supported Algorithms
- Q-Learning
- Epsilon-Greedy Bandit
- UCB Bandit
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- DQN (Deep Q-Network)
- QR-DQN (Quantile Regression DQN)