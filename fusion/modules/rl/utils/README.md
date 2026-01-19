# RL Utils Module

Utility functions and classes supporting the FUSION reinforcement learning framework.

## Overview

This module provides essential infrastructure for RL training including:

- **Custom Callbacks**: Specialized SB3 callbacks for episodic reward tracking and dynamic hyperparameter adjustment
- **Hyperparameter Management**: Configuration classes and Optuna integration for hyperparameter optimization
- **Model Setup**: Factory functions for initializing SB3 models (PPO, A2C, DQN, QR-DQN)
- **Environment Utilities**: Observation space construction, topology conversion, simulation data handling
- **Error Hierarchy**: Custom exception classes for granular error handling

## Key Components

### Custom Callbacks

The `callbacks.py` module provides powerful SB3-compatible callbacks:

| Callback | Purpose |
|----------|---------|
| `EpisodicRewardCallback` | Tracks and saves episode rewards with configurable save intervals |
| `LearnRateEntCallback` | Dynamically decays learning rate and entropy coefficient during training |
| `GetModelParams` | Extracts model parameters and value function estimates for monitoring |

### Hyperparameter Optimization

The `hyperparams.py` module provides:

- `HyperparamConfig`: Manages hyperparameter schedules with multiple decay strategies (linear, exponential, softmax, reward-based, state-based)
- `get_optuna_hyperparams()`: Generates Optuna search spaces for all supported algorithms

### Model Setup

The `setup.py` module provides factory functions:

- `setup_ppo()`, `setup_a2c()`, `setup_dqn()`, `setup_qr_dqn()`: Initialize SB3 models from YAML configs
- `setup_feature_extractor()`: Configure GNN feature extractors with caching support
- `SetupHelper`: Class for managing simulation environment initialization

## Module Structure

```
utils/
|-- __init__.py              # Public API exports
|-- callbacks.py             # Custom SB3 callbacks
|-- hyperparams.py           # Hyperparameter configuration and Optuna
|-- setup.py                 # Model and environment setup utilities
|-- deep_rl.py               # Deep RL algorithm utilities
|-- observation_space.py     # Observation space construction
|-- topology.py              # Network topology conversion
|-- sim_env.py               # Simulation environment utilities
|-- sim_data.py              # Simulation data handling
|-- sim_filters.py           # Data filtering utilities
|-- general_utils.py         # General helper functions
|-- gym_envs.py              # Gymnasium environment utilities
|-- errors.py                # Custom exception hierarchy
|-- cache_gnn_once.py        # GNN embedding caching
|-- unity_hyperparams.py     # Unity cluster hyperparameter utilities
`-- rl_zoo.py                # RLZoo3 integration utilities
```

## Usage

```python
from fusion.modules.rl.utils import (
    # Callbacks
    EpisodicRewardCallback,
    LearnRateEntCallback,
    GetModelParams,
    # Hyperparameters
    HyperparamConfig,
    get_optuna_hyperparams,
    # Setup
    setup_ppo,
    setup_dqn,
    setup_feature_extractor,
)
```

## Related Documentation

- Parent module: `fusion/modules/rl/README.md`
- Sphinx docs: `docs/developer/fusion/modules/rl/utils.rst`
