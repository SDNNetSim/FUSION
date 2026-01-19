# Gymnasium Environments Module

## Purpose

This module provides Gymnasium-compatible environment implementations for reinforcement learning with FUSION network simulations. It enables RL agents to interact with network resource allocation problems through a standardized interface.

## Key Components

- **`general_sim_env.py`**: Main SimEnv class implementing the Gymnasium environment interface
- **`constants.py`**: Configuration constants and default values for the simulation environment
- **`__init__.py`**: Package exports and public API definition

## Usage

### Basic Environment Setup

```python
from fusion.modules.rl.gymnasium_envs import SimEnv

# Create environment with default configuration
env = SimEnv()

# Create environment with custom configuration
sim_config = {"s1": {"path_algorithm": "q_learning", "k_paths": 3}}
env = SimEnv(sim_dict=sim_config)
```

### Training Loop Example

```python
# Reset environment
obs, info = env.reset(seed=42)

for step in range(max_steps):
    # Select action (example: random action)
    action = env.action_space.sample()

    # Take step in environment
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```

### Integration with RL Libraries

```python
from stable_baselines3 import PPO
from fusion.modules.rl.gymnasium_envs import SimEnv

# Create environment
env = SimEnv(sim_dict=config)

# Train RL agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Environment Details

### Observation Space

The environment provides graph-structured observations including:
- Network topology information
- Current request details (bandwidth, holding time)
- Available resources and constraints
- Path and core allocation options

### Action Space

Actions represent resource allocation decisions:
- Path selection from available routes
- Core assignment within selected paths
- Resource allocation strategies

### Reward Structure

- **Positive reward**: Successful request allocation
- **Negative reward (penalty)**: Request blocking or failure

## Configuration

### Simulation Parameters

Key configuration parameters include:
- `path_algorithm`: RL algorithm for path selection
- `k_paths`: Number of candidate paths to consider
- `cores_per_link`: Available cores per network link
- `erlang_start/stop/step`: Traffic load parameters

### Supported Features

- **Spectral Bands**: Currently supports C-band (`"c"`)
- **Algorithms**: Q-learning, bandits, deep RL methods
- **Training/Testing**: Configurable mode switching
- **Model Persistence**: Save/load trained models

## Dependencies

### Internal Dependencies
- `fusion.modules.rl.utils.sim_env`: Environment utilities
- `fusion.modules.rl.utils.setup`: Simulation setup helpers
- `fusion.modules.rl.agents.path_agent`: Path selection agent
- `fusion.modules.rl.algorithms.algorithm_props`: RL properties

### External Dependencies
- `gymnasium`: Standard RL environment interface
- Compatible with Stable-Baselines3 and other RL libraries

## Architecture

The SimEnv class follows the standard Gymnasium interface:

1. **Initialization**: Set up simulation components and RL agents
2. **Reset**: Initialize new episode with random seed and options
3. **Step**: Process action, update environment state, return observation
4. **Render**: Visualization support (currently minimal)

## Notes

- The environment handles both training and inference modes
- Supports hyperparameter optimization workflows
- Integrates with FUSION's network simulation engine
- Provides extensive configuration flexibility for different scenarios
