# StableBaselines3 Integration Module

## Purpose

This module provides utilities for integrating custom FUSION environments with the StableBaselines3 reinforcement learning framework. It handles environment registration with Gymnasium and manages algorithm configuration files for seamless integration with RLZoo3 hyperparameter optimization.

## Key Components

- **`register_env.py`**: Environment registration utilities and configuration management
- **`__init__.py`**: Module exports and public API definition

## Core Functions

### copy_yml_file()

Copies algorithm configuration files from local storage to the RLZoo3 hyperparameters directory.

**Parameters:**
- `algorithm` (str): Algorithm name for configuration file lookup

**Features:**
- Robust error handling for file operations
- Comprehensive exception reporting
- Integration with RLZoo3 hyperparameter management
- Automatic configuration deployment

**Usage:**
```python
from fusion.modules.rl.sb3 import copy_yml_file

# Copy PPO configuration to RLZoo3
copy_yml_file("PPO")
```

**File Paths:**
- **Source**: `sb3_scripts/yml/{algorithm}.yml`
- **Destination**: `venvs/unity_venv/venv/lib/python3.11/site-packages/rl_zoo3/hyperparams/{algorithm}.yml`

### main()

Main entry point for environment registration script with command-line interface.

**Command-line Arguments:**
- `--algo`: Algorithm name for configuration file (e.g., 'PPO', 'DQN')
- `--env-name`: Environment class name to register (e.g., 'SimEnv')

**Features:**
- Gymnasium environment registration
- Algorithm configuration deployment
- Registry verification and display
- Comprehensive error handling

**Usage:**
```bash
# Register SimEnv with PPO algorithm
python register_env.py --algo PPO --env-name SimEnv

# Register custom environment with DQN
python register_env.py --algo DQN --env-name CustomEnv
```

## Integration Architecture

### Environment Registration Process

1. **Parse Arguments**: Extract algorithm and environment names from command line
2. **Register Environment**: Register custom environment with Gymnasium registry
3. **Verify Registration**: Display registered environments for confirmation
4. **Deploy Configuration**: Copy algorithm YAML to RLZoo3 hyperparams directory
5. **Confirm Success**: Report successful registration and configuration

### Gymnasium Integration

The module registers environments using the standard Gymnasium pattern:

```python
register(
    id="SimEnv",
    entry_point="reinforcement_learning.gymnasium_envs.general_sim_env:SimEnv"
)
```

**Critical Components:**
- **Environment ID**: Used by SB3 to identify the environment
- **Entry Point**: Python module path to environment class
- **Registry Integration**: Enables SB3 algorithms to find and instantiate environments

### RLZoo3 Configuration Management

RLZoo3 requires algorithm-specific YAML configuration files for hyperparameter optimization:

```yaml
# Example PPO.yml structure
SimEnv:
  policy: 'MlpPolicy'
  n_timesteps: !!float 1e6
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  # ... additional hyperparameters
```

## File System Requirements

### Directory Structure

```
project/
├── sb3_scripts/yml/          # Source configuration files
│   ├── PPO.yml
│   ├── DQN.yml
│   └── ...
└── venvs/unity_venv/venv/    # Virtual environment
    └── lib/python3.11/site-packages/rl_zoo3/hyperparams/
        ├── PPO.yml           # Deployed configurations
        ├── DQN.yml
        └── ...
```

### Configuration Files

Algorithm configurations must exist in `sb3_scripts/yml/` with proper YAML format:
- **Naming**: `{ALGORITHM}.yml` (e.g., `PPO.yml`, `DQN.yml`)
- **Format**: Valid YAML with environment-specific sections
- **Content**: SB3-compatible hyperparameters and policy configurations

## Error Handling

The module provides comprehensive error handling for common failure scenarios:

### FileNotFoundError
- **Cause**: Source configuration file doesn't exist
- **Solution**: Ensure algorithm YAML exists in `sb3_scripts/yml/`
- **Message**: Clear path to missing file and suggested resolution

### PermissionError  
- **Cause**: Insufficient permissions to write to RLZoo3 directory
- **Solution**: Check virtual environment permissions and file access
- **Message**: Specific destination path and permission requirements

### OSError
- **Cause**: General file system errors during copy operation
- **Solution**: Verify disk space, file locks, and system resources
- **Message**: Underlying system error with context

## Dependencies

### Internal Dependencies
- **FUSION Environments**: Requires properly implemented Gymnasium environments
- **Configuration Files**: Needs algorithm-specific YAML configurations
- **Virtual Environment**: Expects specific venv structure for RLZoo3

### External Dependencies
- **Gymnasium**: For environment registration and management
- **StableBaselines3**: Target framework for RL algorithms
- **RLZoo3**: Hyperparameter optimization and training utilities
- **PyYAML**: For configuration file parsing (via RLZoo3)

## Usage Patterns

### Development Workflow

1. **Implement Environment**: Create custom Gymnasium environment in FUSION
2. **Create Configuration**: Design algorithm-specific YAML configurations
3. **Register Environment**: Use registration script to enable SB3 integration
4. **Train Algorithms**: Use RLZoo3 for hyperparameter optimization and training

### Training Integration

```bash
# Step 1: Register environment
python register_env.py --algo PPO --env-name SimEnv

# Step 2: Train with RLZoo3
python -m rl_zoo3.train --algo ppo --env SimEnv

# Step 3: Evaluate trained model
python -m rl_zoo3.enjoy --algo ppo --env SimEnv --folder logs/
```

## Configuration Examples

### PPO Configuration
```yaml
SimEnv:
  policy: 'MlpPolicy'
  n_timesteps: !!float 2e6
  learning_rate: lin_3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
```

### DQN Configuration
```yaml
SimEnv:
  policy: 'MlpPolicy' 
  n_timesteps: !!float 1e6
  buffer_size: 1000000
  learning_rate: !!float 1e-4
  learning_starts: 50000
  batch_size: 32
  tau: 1.0
  gamma: 0.99
  train_freq: 4
  gradient_steps: 1
  target_update_interval: 10000
```

## Notes

- **Path Compatibility**: Hard-coded paths are preserved for SB3 integration requirements
- **Virtual Environment**: Assumes specific venv structure for RLZoo3 deployment
- **Algorithm Names**: Must match SB3 algorithm naming conventions
- **Configuration Format**: YAML files must follow RLZoo3 hyperparameter structure
- **Registry Management**: Environments persist in Gymnasium registry during Python session