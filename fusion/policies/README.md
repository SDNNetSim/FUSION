# Policies

## Purpose

This module provides control policy implementations for the FUSION simulation framework. Policies implement the `ControlPolicy` protocol and are responsible for selecting which path (action) to use when serving a network request. They decouple decision-making logic from the simulation engine, enabling pluggable strategies from simple heuristics to trained ML/RL models.

## Key Components

### Core Files

- `heuristic_policy.py`: Rule-based deterministic policies (first feasible, shortest, least congested, random, load balanced)
- `ml_policy.py`: Pre-trained ML model policies (PyTorch, sklearn, ONNX) with fallback support
- `rl_policy.py`: Pre-trained Stable-Baselines3 RL model policies
- `policy_factory.py`: Factory for instantiating policies from configuration

### Test Files

- `tests/test_heuristic_policy.py`: Tests for heuristic policy implementations
- `tests/test_ml_policy.py`: Tests for ML policy and model wrappers
- `tests/test_rl_policy.py`: Tests for RL policy wrapper

## Usage

### Heuristic Policies

```python
from fusion.policies import (
    FirstFeasiblePolicy,
    ShortestFeasiblePolicy,
    LeastCongestedPolicy,
    RandomFeasiblePolicy,
    LoadBalancedPolicy,
)

# Select first feasible path (K-shortest first fit)
policy = FirstFeasiblePolicy()
action = policy.select_action(request, options, network_state)

# Select shortest feasible path by distance
policy = ShortestFeasiblePolicy()
action = policy.select_action(request, options, network_state)

# Select least congested path
policy = LeastCongestedPolicy()
action = policy.select_action(request, options, network_state)

# Random selection among feasible paths
policy = RandomFeasiblePolicy(seed=42)
action = policy.select_action(request, options, network_state)

# Balance path length and congestion (alpha=0.5 means equal weight)
policy = LoadBalancedPolicy(alpha=0.5)
action = policy.select_action(request, options, network_state)
```

### ML Policies

```python
from fusion.policies import MLControlPolicy

# Load PyTorch model with fallback to first feasible
policy = MLControlPolicy(
    model_path="trained_model.pt",
    fallback_type="first_feasible",
    k_paths=5,
)
action = policy.select_action(request, options, network_state)

# Check fallback statistics
print(policy.get_stats())
# {'total_calls': 100, 'fallback_calls': 3, 'fallback_rate': 0.03, ...}
```

### RL Policies

```python
from fusion.policies import RLPolicy

# Load from file
policy = RLPolicy.from_file(
    model_path="trained_ppo.zip",
    algorithm="MaskablePPO",
    k_paths=5,
)
action = policy.select_action(request, options, network_state)

# Or wrap an existing SB3 model
from stable_baselines3 import PPO
model = PPO.load("model.zip")
policy = RLPolicy(model, k_paths=5)
```

### Policy Factory

```python
from fusion.policies import PolicyFactory, PolicyConfig

# Create from config object
config = PolicyConfig(
    policy_type="heuristic",
    policy_name="shortest_feasible",
)
policy = PolicyFactory.create(config)

# Create from dictionary (e.g., from config file)
policy = PolicyFactory.from_dict({
    "policy_type": "ml",
    "model_path": "model.pt",
    "fallback_policy": "first_feasible",
})

# Get default policy
policy = PolicyFactory.get_default_policy()  # FirstFeasiblePolicy
```

## Dependencies

### Internal Dependencies

- `fusion.interfaces.control_policy`: ControlPolicy protocol and PolicyAction
- `fusion.modules.rl.adapter`: PathOption dataclass for path information
- `fusion.domain.network_state`: NetworkState for network context
- `fusion.domain.request`: Request dataclass

### External Dependencies

- `numpy`: Array operations for ML inference
- `torch` (optional): PyTorch model support
- `joblib` (optional): sklearn model loading
- `onnxruntime` (optional): ONNX model support
- `stable_baselines3` (optional): RL model support
- `sb3_contrib` (optional): MaskablePPO support

## Configuration

### PolicyConfig Options

```python
from fusion.policies import PolicyConfig

config = PolicyConfig(
    policy_type="heuristic",      # "heuristic", "ml", or "rl"
    policy_name="first_feasible", # Policy variant name
    model_path=None,              # Path to model file (ml/rl)
    fallback_policy="first_feasible",  # Fallback for ml policies
    k_paths=3,                    # Number of candidate paths
    seed=None,                    # Random seed (for random policy)
    alpha=0.5,                    # Balance parameter (load_balanced)
    algorithm="PPO",              # RL algorithm name
)
```

### Available Heuristic Policies

| Name | Description |
|------|-------------|
| `first_feasible` | Select first feasible path in index order |
| `shortest` / `shortest_feasible` | Select shortest feasible path by distance |
| `least_congested` | Select path with lowest congestion |
| `random` / `random_feasible` | Randomly select among feasible paths |
| `load_balanced` | Balance path length and congestion |

## Testing

```bash
# Run policy tests
pytest fusion/policies/tests/

# Run with coverage
pytest --cov=fusion.policies fusion/policies/tests/

# Run specific test file
pytest fusion/policies/tests/test_heuristic_policy.py -v
```

## API Reference

### Base Classes and Protocols

- `ControlPolicy`: Protocol defining the policy interface
- `HeuristicPolicy`: Abstract base class for heuristic policies

### Heuristic Policies

- `FirstFeasiblePolicy`: Select first feasible path
- `ShortestFeasiblePolicy`: Select shortest by distance
- `LeastCongestedPolicy`: Select least congested
- `RandomFeasiblePolicy`: Random selection with optional seed
- `LoadBalancedPolicy`: Weighted balance of length and congestion

### ML/RL Policies

- `MLControlPolicy`: Pre-trained ML models with fallback
- `RLPolicy` / `RLControlPolicy`: Pre-trained SB3 models
- `FeatureBuilder`: Build feature vectors for ML inference

### Model Wrappers

- `ModelWrapper`: Protocol for model wrappers
- `TorchModelWrapper`: PyTorch model wrapper
- `SklearnModelWrapper`: sklearn model wrapper
- `OnnxModelWrapper`: ONNX model wrapper
- `CallableModelWrapper`: Generic callable wrapper

### Factory

- `PolicyFactory`: Create policies from configuration
- `PolicyConfig`: Configuration dataclass

## Notes

### Design Decisions

- All policies implement the `ControlPolicy` protocol for interchangeability
- Heuristic policies are stateless and deterministic (except RandomFeasiblePolicy)
- ML/RL policies are deployment-only; `update()` is a no-op
- MLControlPolicy includes robust fallback to heuristics on errors
- RLPolicy supports both standard SB3 and MaskablePPO action masking

### Performance Considerations

- Heuristic policies are O(n) where n is the number of path options
- ML inference time depends on model complexity and framework
- Feature building is O(k) where k is the number of paths

### Integration with Orchestrator

Policies are used by the SDNOrchestrator to select paths:

```python
# In orchestrator
policy = PolicyFactory.create(config.policy_config)
action = policy.select_action(request, path_options, network_state)
selected_path = path_options[action] if action >= 0 else None
```
