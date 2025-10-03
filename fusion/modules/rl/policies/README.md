# RL Policies Module

## Purpose

This module provides specialized policy implementations for reinforcement learning agents in the FUSION network simulation framework. It focuses on attention-based policies for path selection scenarios where traditional multi-layer perceptrons may not be optimal.

## Key Components

- **`pointer_policy.py`**: Pointer network-based policy implementation using attention mechanisms
- **`__init__.py`**: Module exports and public API definition

## Core Classes

### PointerHead

A torch.nn.Module that implements attention-based path selection using query-key-value attention mechanisms.

**Key Features:**
- Multi-head attention for path ranking
- Configurable attention heads and dimensions
- Robust input validation for tensor shapes
- Mathematical attention scoring with softmax normalization

**Usage:**
```python
from fusion.modules.rl.policies import PointerHead

# Initialize pointer head with feature dimension
head = PointerHead(dimension=64)

# Forward pass with path features (batch, paths, features)
path_features = torch.randn(32, 3, 64)
logits = head.forward(path_features)  # Returns (batch, paths) logits
```

### PointerPolicy

A Stable Baselines3 compatible ActorCriticPolicy that integrates PointerHead for attention-based policy networks.

**Key Features:**
- Inherits from ActorCriticPolicy for full SB3 compatibility
- Uses PointerHead for policy network (actor)
- Maintains standard linear layer for value network (critic)
- Custom MLP extractor building for specialized architectures

**Usage:**
```python
from fusion.modules.rl.policies import PointerPolicy
from stable_baselines3 import PPO

# Create environment and policy
policy = PointerPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=lambda _: 0.0003
)

# Use with PPO or other SB3 algorithms
model = PPO(policy, env)
```

## Architecture Details

### Attention Mechanism

The PointerHead uses scaled dot-product attention:

1. **Query-Key-Value Transformation**: Linear projection of input features
2. **Attention Scores**: Compute similarity between query and key vectors
3. **Softmax Normalization**: Convert scores to probability distribution
4. **Value Weighting**: Apply attention weights to value vectors
5. **Output Generation**: Sum weighted values to produce path logits

### Mathematical Formulation

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

Where:
- Q, K, V are query, key, value matrices
- d_k is the key dimension for scaling
- Softmax ensures attention weights sum to 1
```

## Configuration

### Constants

- `DEFAULT_ATTENTION_HEADS`: Number of attention heads (default: 3)
- `QUERY_KEY_VALUE_MULTIPLIER`: Multiplier for QKV transformation (default: 3)

### Customization

The policy can be customized by:
- Adjusting attention head dimensions
- Modifying the number of paths considered
- Changing the underlying feature extractor
- Tuning attention mechanisms

## Integration with FUSION

### Path Selection Scenarios

The pointer policy is designed for network routing scenarios where:
- Multiple candidate paths are available
- Path quality varies dynamically
- Attention-based selection improves performance
- Traditional MLP policies are insufficient

### Feature Requirements

Input features should be structured as:
- **Shape**: (batch_size, num_paths, feature_dim)
- **Content**: Path-specific features (latency, congestion, capacity)
- **Normalization**: Features should be appropriately scaled

## Dependencies

### Internal Dependencies
- Compatible with FUSION's RL framework
- Integrates with feature extractors in `feat_extrs` module
- Works with environment implementations in `gymnasium_envs`

### External Dependencies
- **PyTorch**: For neural network implementation
- **Stable Baselines3**: For RL policy framework
- **NumPy**: For mathematical operations (via PyTorch)

## Performance Considerations

### Memory Usage
- Attention mechanisms require O(n²) memory for n paths
- Suitable for small to medium path sets (≤ 10 paths)
- Consider efficiency for larger path spaces

### Computational Complexity
- Forward pass: O(batch_size × num_paths² × feature_dim)
- Training: Additional backpropagation through attention layers
- Inference: Efficient for real-time path selection

## Error Handling

The module includes comprehensive error handling:
- **Dimension Validation**: Ensures positive feature dimensions
- **Shape Checking**: Validates tensor shapes at runtime
- **Initialization Errors**: Clear error messages for configuration issues
- **Attribute Validation**: Checks for required components

## Notes

- The pointer policy is experimental and designed for specific routing scenarios
- Performance depends heavily on feature quality and path representation
- Consider traditional policies for simpler path selection problems
- Monitor attention patterns to understand model behavior
