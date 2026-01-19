# Reinforcement Learning Module

> **Status: Transitioning to UnifiedSimEnv**
>
> This module is transitioning from the legacy `GeneralSimEnv` to the new `UnifiedSimEnv`.
> New work should use `UnifiedSimEnv`.

## Overview

The RL module enables intelligent path selection, spectrum allocation, and network
optimization through reinforcement learning. Key capabilities include:

- Traditional RL algorithms (Q-learning, epsilon-greedy bandits, UCB bandits)
- Deep RL via Stable-Baselines3 (PPO, A2C, DQN, QR-DQN)
- GNN-based feature extractors for graph-structured observations
- Offline RL policies (Behavioral Cloning, Implicit Q-Learning)
- Action masking for safe RL deployment
- Hyperparameter optimization with Optuna

## Current Integration

| Environment | Status | Notes |
|-------------|--------|-------|
| `GeneralSimEnv` | **Pre v6.0** | Deprecated, removal planned for v6.1+ |
| `UnifiedSimEnv` | **Post v6.0** | Recommended for new work |

## Module Structure

```
fusion/modules/rl/
├── __init__.py              # Public API exports
├── README.md                # This file
├── TODO.md                  # Development roadmap
├── errors.py                # Custom exception hierarchy
├── model_manager.py         # Model creation and persistence
├── workflow_runner.py       # Training orchestration
├── adapter/                 # Simulation integration layer
├── agents/                  # RL agent implementations
├── algorithms/              # Core algorithm implementations
├── args/                    # Configuration and constants
├── environments/            # Unified environment
├── feat_extrs/              # GNN-based feature extractors
├── gymnasium_envs/          # Legacy environment (deprecated)
├── policies/                # Path selection policies
├── sb3/                     # Stable-Baselines3 integration
├── utils/                   # Utility modules
└── visualization/           # RL-specific visualization
```

## Quick Start

### Training with Deep RL

```python
from fusion.modules.rl import (
    WorkflowRunner,
    get_model,
    save_model,
)
from fusion.modules.rl.environments import UnifiedSimEnv

# Create environment
env = UnifiedSimEnv(engine_props=engine_props, sim_params=sim_params)

# Get model (loads hyperparameters from YAML)
model, hyperparams = get_model(
    sim_dict=sim_dict,
    device="cuda",
    env=env,
    yaml_dict=None,
)

# Train
model.learn(total_timesteps=100_000)

# Save
save_model(model, sim_dict, "ppo", erlang="100")
```

### Using Policies

```python
from fusion.modules.rl.policies import (
    PathPolicy,
    KSPFFPolicy,
    BCPolicy,
    compute_action_mask,
)

# Heuristic policy
policy = KSPFFPolicy()
action = policy.select_path(state, action_mask)

# Offline RL policy
bc_policy = BCPolicy.load("path/to/model.pt")
action = bc_policy.select_path(state, action_mask)
```

### Using the Adapter Layer

```python
from fusion.modules.rl.adapter import (
    RLSimulationAdapter,
    RLConfig,
    PathOption,
)

# Configure adapter
config = RLConfig(
    k_paths=3,
    use_action_masking=True,
)

# Create adapter
adapter = RLSimulationAdapter(
    orchestrator=orchestrator,
    config=config,
)

# Get path options for a request
path_options = adapter.get_path_options(request)
action_mask = path_options.compute_action_mask()
```

## Key Components

### Agents

| Agent | Purpose | Status |
|-------|---------|--------|
| `PathAgent` | Intelligent path selection | Implemented |
| `CoreAgent` | Core assignment | Placeholder |
| `SpectrumAgent` | Spectrum allocation | Placeholder |

### Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| `QLearning` | Tabular | Classic Q-learning |
| `EpsilonGreedyBandit` | Bandit | Epsilon-greedy exploration |
| `UCBBandit` | Bandit | Upper Confidence Bound |
| `PPO` | Deep RL | Proximal Policy Optimization |
| `A2C` | Deep RL | Advantage Actor-Critic |
| `DQN` | Deep RL | Deep Q-Network |
| `QrDQN` | Deep RL | Quantile Regression DQN |

### Policies

| Policy | Type | Description |
|--------|------|-------------|
| `KSPFFPolicy` | Heuristic | K-Shortest Path with First Fit |
| `OnePlusOnePolicy` | Protection | 1+1 disjoint path protection |
| `BCPolicy` | Offline RL | Behavioral Cloning |
| `IQLPolicy` | Offline RL | Implicit Q-Learning |
| `PointerPolicy` | Attention | Pointer network for path selection |

### Feature Extractors

| Extractor | Description |
|-----------|-------------|
| `PathGNN` | Standard GNN (GAT, GraphConv, SAGE) |
| `CachedPathGNN` | Cached GNN for static graphs |
| `Graphormer` | Transformer-based GNN |

## Dependencies

### Internal
- `fusion.core.simulation`: Simulation engine
- `fusion.core.orchestrator`: SDN orchestrator
- `fusion.interfaces.pipelines`: Routing/spectrum pipelines
- `fusion.visualization`: Plugin system

### External
- `gymnasium`: RL environment interface
- `stable-baselines3`: Deep RL algorithms
- `sb3-contrib`: Additional SB3 algorithms (MaskablePPO)
- `torch`: Neural networks and GNNs
- `optuna`: Hyperparameter optimization
- `numpy`, `psutil`: Utilities

## Testing

```bash
# Run all RL module tests
pytest fusion/modules/rl/ -v

# Run specific submodule tests
pytest fusion/modules/rl/environments/tests/ -v
pytest fusion/modules/rl/adapter/tests/ -v

# Run with coverage
pytest fusion/modules/rl/ -v --cov=fusion.modules.rl
```

## Migration Guide

The legacy `GeneralSimEnv` is deprecated. New code should use `UnifiedSimEnv`:

```python
# Old (deprecated)
from fusion.modules.rl.gymnasium_envs import SimEnv
env = SimEnv(sim_dict, engine_props, sdn_props)

# New (recommended)
from fusion.modules.rl.environments import UnifiedSimEnv
env = UnifiedSimEnv(engine_props=engine_props, sim_params=sim_params)
```

See `docs/migration/rl_to_unified_env.md` for detailed migration instructions.

## Error Handling

The module provides a custom exception hierarchy:

```python
from fusion.modules.rl.errors import (
    RLError,                  # Base exception
    RLConfigurationError,     # Invalid configuration
    AlgorithmNotFoundError,   # Unknown algorithm
    ModelLoadError,           # Model loading failed
    TrainingError,            # Training process failed
    RLEnvironmentError,       # Environment issues
    AgentError,               # Agent operation failed
    InvalidActionError,       # Invalid action attempted
    RouteSelectionError,      # Route selection failed
)
```

## Configuration

### Hyperparameters

Hyperparameters are loaded from YAML files in `fusion/configs/rl/`:

```yaml
# Example: ppo_hyperparams.yml
policy: MlpPolicy
n_steps: 2048
batch_size: 64
n_epochs: 10
learning_rate: 0.0003
clip_range: 0.2
```

### Environment Configuration

```python
from fusion.modules.rl.adapter import RLConfig

config = RLConfig(
    k_paths=3,                    # Number of candidate paths
    use_action_masking=True,      # Enable action masking
    reward_scale=1.0,             # Reward scaling factor
)
```
