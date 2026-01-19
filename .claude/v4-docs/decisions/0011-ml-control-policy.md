# ADR-0011: ML Control Policy

## Status

Proposed

## Context

While FUSION supports RL-based path selection through the `UnifiedSimEnv` and Stable-Baselines3, there is no clear integration path for:

1. **Pre-trained ML models**: Classifiers or regressors trained offline on historical data
2. **Imitation learning models**: Models trained to mimic expert (heuristic) behavior
3. **Offline RL policies**: Models trained using offline RL techniques (BC, IQL, CQL)

These approaches have advantages over online RL:

- **No training overhead**: Deploy immediately without simulation episodes
- **Consistent behavior**: Same decisions from the first request
- **Diverse architectures**: Support beyond RL algorithms (random forests, gradient boosting)
- **Interpretability**: Some ML models (decision trees) are more explainable

The challenge is integrating these models with FUSION's simulation pipeline while maintaining compatibility with the `ControlPolicy` protocol.

## Decision

We will implement an `MLControlPolicy` class that:

1. Implements the `ControlPolicy` protocol
2. Supports multiple ML frameworks (PyTorch, scikit-learn, ONNX)
3. Uses the same feature space as RL for consistency
4. Applies action masking to respect feasibility constraints
5. Provides fallback behavior when model predictions fail

### MLControlPolicy Design

```python
class MLControlPolicy(ControlPolicy):
    """
    ML-based control policy using pre-trained models.

    Supports:
    - PyTorch neural networks
    - Scikit-learn classifiers
    - ONNX runtime models
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        model_type: str = "pytorch",
        fallback_policy: Optional[ControlPolicy] = None,
    ):
        self.device = device
        self.model_type = model_type
        self.model = self._load_model(model_path)
        self.fallback = fallback_policy or FirstFeasiblePolicy()

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        try:
            features = self._build_features(request, options, network_state)
            raw_output = self._predict(features)
            action = self._apply_mask_and_select(raw_output, options)
            return action
        except Exception:
            # Fallback on any error
            return self.fallback.select_action(request, options, network_state)

    def update(self, request: Request, action: int, reward: float) -> None:
        pass  # Pre-trained, no online updates
```

### Key Design Decisions

1. **Multi-framework support**: Load models from PyTorch, scikit-learn, or ONNX
2. **Fallback policy**: Gracefully handle model failures with configurable fallback
3. **Feature compatibility**: Use same features as RL for model portability
4. **Action masking**: Always respect feasibility constraints
5. **No online learning**: `update()` is a no-op for pre-trained models

## Alternatives Considered

### Alternative 1: RL-Only Path Selection

Keep only RL (SB3) integration, train models online.

**Rejected because**:
- Training overhead not acceptable for all use cases
- Doesn't support offline RL techniques
- Limits model architecture choices

### Alternative 2: Separate ML Simulation Mode

Create entirely separate simulation path for ML models.

**Rejected because**:
- Code duplication with RL path
- Inconsistent feature computation
- Testing burden doubles

### Alternative 3: Feature Computation in Model

Let models compute their own features from raw network state.

**Rejected because**:
- Models would depend on network internals
- Feature mismatch between models
- Harder to validate model inputs

### Alternative 4: PyTorch Only

Support only PyTorch models for simplicity.

**Rejected because**:
- Excludes valuable model types (random forests, XGBoost)
- Forces users into specific framework
- Limits deployment options

## Consequences

### Positive

1. **Deployment flexibility**: Use best model for the task
2. **Framework agnostic**: PyTorch, sklearn, ONNX all supported
3. **Graceful degradation**: Fallback prevents simulation failures
4. **RL compatibility**: Same features enable model comparison
5. **Offline training**: Train on logs without simulation

### Negative

1. **Complexity**: Multiple model loading paths to maintain
2. **Feature coupling**: Changes to RL features affect ML models
3. **Testing surface**: Must test each framework integration
4. **Documentation**: Multiple model formats to document

### Neutral

1. **Model validation**: Users responsible for model quality
2. **Feature engineering**: Fixed feature set may limit some models

## Implementation Notes

### Feature Vector Format

```python
def _build_features(
    self,
    request: Request,
    options: list[PathOption],
    network_state: NetworkState,
) -> np.ndarray:
    """Build feature vector compatible with RL observations."""
    features = []

    # Request features
    features.extend([
        request.bandwidth_gbps / 1000.0,  # Normalized
    ])

    # Per-path features (for k_paths)
    for i, opt in enumerate(options):
        features.extend([
            opt.weight_km / 10000.0,           # Normalized path length
            opt.congestion,                     # 0-1 congestion
            1.0 if opt.is_feasible else 0.0,   # Feasibility flag
            (opt.slots_needed or 0) / 100.0,   # Normalized slots
        ])

    return np.array(features, dtype=np.float32)
```

### Action Masking

```python
def _apply_mask_and_select(
    self,
    raw_output: np.ndarray,
    options: list[PathOption],
) -> int:
    """Apply feasibility mask and select best action."""
    mask = np.array([opt.is_feasible for opt in options])

    # Set infeasible actions to -infinity
    masked_output = raw_output.copy()
    masked_output[~mask] = float('-inf')

    action = int(np.argmax(masked_output))

    # Validate action is actually feasible
    if action < len(options) and options[action].is_feasible:
        return action
    return -1
```

### Model Loading

```python
def _load_model(self, path: str) -> Any:
    if self.model_type == "pytorch":
        import torch
        model = torch.load(path, map_location=self.device)
        model.eval()
        return model

    elif self.model_type == "sklearn":
        import joblib
        return joblib.load(path)

    elif self.model_type == "onnx":
        import onnxruntime as ort
        return ort.InferenceSession(path)

    raise ValueError(f"Unknown model type: {self.model_type}")
```

### Training Data Generation

```python
def generate_training_data(
    config: SimulationConfig,
    heuristic: HeuristicPolicy,
    n_episodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training data by running heuristic policy."""
    features_list = []
    actions_list = []

    for episode in range(n_episodes):
        env = UnifiedSimEnv(config)
        obs, info = env.reset()

        while True:
            options = env._path_options
            request = env._requests[env._current_idx]

            # Get expert action
            action = heuristic.select_action(request, options, env._network_state)

            # Store features and action
            features = build_features(request, options, env._network_state)
            features_list.append(features)
            actions_list.append(action)

            # Step environment
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break

    return np.array(features_list), np.array(actions_list)
```

## Configuration

```ini
[policy]
type = ml
model_path = models/path_selection.pt
model_type = pytorch          ; pytorch, sklearn, onnx
device = cpu                  ; cpu, cuda
fallback_policy = first_feasible
```

## Related Decisions

- [ADR-0004: RL/ML Integration](./0004-rl-ml-integration.md)
- [ADR-0010: ControlPolicy Protocol](./0010-control-policy-protocol.md)

## References

- [PyTorch Save/Load](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Imitation Learning](https://imitation.readthedocs.io/)
- [Offline RL Algorithms](https://github.com/takuseno/d3rlpy)
