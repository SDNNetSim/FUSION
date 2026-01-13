# ML Policies Architecture

## Overview

ML Policies provide a mechanism for using pre-trained machine learning models to make path selection decisions in FUSION. Unlike RL policies that train online, ML policies use models trained offline using supervised learning, imitation learning, or offline RL techniques.

## Motivation

While RL policies excel at adaptive decision-making, they have limitations:

1. **Training overhead**: RL requires many simulation episodes to converge
2. **Sample efficiency**: Online RL may make poor decisions during early training
3. **Deployment simplicity**: Pre-trained models are easier to deploy and evaluate
4. **Interpretability**: Supervised models (e.g., decision trees) can be more interpretable

ML policies address these by:

- Using models trained on historical data or heuristic logs
- Providing consistent behavior from the first request
- Supporting diverse model architectures (neural networks, random forests, etc.)
- Enabling offline evaluation before deployment

## MLControlPolicy Class

```python
import torch
import numpy as np
from typing import Optional, Any

from fusion.interfaces.control_policy import ControlPolicy
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.modules.rl.env_adapter import PathOption


class MLControlPolicy(ControlPolicy):
    """
    ML-based control policy using a pre-trained model.

    Supports various model types:
    - PyTorch neural networks
    - Scikit-learn classifiers
    - Custom model wrappers
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        model_type: str = "pytorch",
    ):
        """
        Initialize ML policy.

        Args:
            model_path: Path to saved model file
            device: Device for inference ('cpu' or 'cuda')
            model_type: Type of model ('pytorch', 'sklearn', 'onnx')
        """
        self.device = torch.device(device) if model_type == "pytorch" else None
        self.model_type = model_type
        self.model = self._load_model(model_path)

    def _load_model(self, path: str) -> Any:
        """Load model from file based on type."""
        if self.model_type == "pytorch":
            model = torch.load(path, map_location=self.device)
            model.eval()
            return model
        elif self.model_type == "sklearn":
            import joblib
            return joblib.load(path)
        elif self.model_type == "onnx":
            import onnxruntime as ort
            return ort.InferenceSession(path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """Select action using the trained model."""
        # Build feature vector
        features = self._build_features(request, options, network_state)

        # Get model predictions
        if self.model_type == "pytorch":
            action = self._predict_pytorch(features, options)
        elif self.model_type == "sklearn":
            action = self._predict_sklearn(features, options)
        elif self.model_type == "onnx":
            action = self._predict_onnx(features, options)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Validate action is feasible
        if action < 0 or action >= len(options) or not options[action].is_feasible:
            # Fallback to first feasible
            for opt in options:
                if opt.is_feasible:
                    return opt.path_index
            return -1

        return action

    def _predict_pytorch(self, features: np.ndarray, options: list[PathOption]) -> int:
        """Predict using PyTorch model."""
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            logits = self.model(features_tensor)

            # Apply action mask
            mask = torch.tensor(
                [opt.is_feasible for opt in options],
                dtype=torch.bool,
                device=self.device,
            )
            logits[:, ~mask] = float('-inf')

            return logits.argmax(dim=1).item()

    def _predict_sklearn(self, features: np.ndarray, options: list[PathOption]) -> int:
        """Predict using scikit-learn model."""
        # Get class probabilities
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(features.reshape(1, -1))[0]

            # Apply action mask
            mask = np.array([opt.is_feasible for opt in options])
            probs[~mask] = 0

            return int(np.argmax(probs))
        else:
            # Model only supports hard predictions
            action = int(self.model.predict(features.reshape(1, -1))[0])
            if action < len(options) and options[action].is_feasible:
                return action
            return -1

    def _predict_onnx(self, features: np.ndarray, options: list[PathOption]) -> int:
        """Predict using ONNX runtime."""
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: features.reshape(1, -1).astype(np.float32)})
        logits = outputs[0][0]

        # Apply action mask
        mask = np.array([opt.is_feasible for opt in options])
        logits[~mask] = float('-inf')

        return int(np.argmax(logits))

    def _build_features(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> np.ndarray:
        """Build feature vector from request and path options."""
        features = []

        # Request features
        features.extend([
            request.bandwidth_gbps / 1000.0,  # Normalized
        ])

        # Path option features (fixed size for k_paths)
        for i in range(len(options)):
            opt = options[i]
            features.extend([
                opt.weight_km / 10000.0,       # Normalized path length
                opt.congestion,                 # Already 0-1
                1.0 if opt.is_feasible else 0.0,
                (opt.slots_needed or 0) / 100.0,  # Normalized
            ])

        # Pad if fewer than k_paths options
        # Assumes a fixed k_paths value
        return np.array(features, dtype=np.float32)

    def update(self, request: Request, action: int, reward: float) -> None:
        """ML policies are pre-trained - no online updates."""
        pass
```

## Feature Engineering

### Standard Feature Set

```python
def build_standard_features(
    request: Request,
    options: list[PathOption],
    network_state: NetworkState,
    k_paths: int,
) -> np.ndarray:
    """
    Build standard feature vector compatible with RL observations.

    Features (per path):
    - path_length_normalized: Path length / max_path_length
    - congestion: Average link congestion (0-1)
    - is_feasible: 1.0 if spectrum available, 0.0 otherwise
    - slots_needed_normalized: Required slots / max_slots
    - hop_count_normalized: Number of hops / max_hops

    Request features:
    - bandwidth_normalized: Request bandwidth / max_bandwidth
    - holding_time_normalized: Holding time / max_holding_time
    """
    MAX_PATH_LENGTH = 10000.0  # km
    MAX_SLOTS = 320
    MAX_HOPS = 15
    MAX_BANDWIDTH = 1000.0  # Gbps
    MAX_HOLDING_TIME = 100.0

    features = []

    # Request features
    features.extend([
        request.bandwidth_gbps / MAX_BANDWIDTH,
        request.holding_time / MAX_HOLDING_TIME,
    ])

    # Per-path features
    for i in range(k_paths):
        if i < len(options):
            opt = options[i]
            features.extend([
                opt.weight_km / MAX_PATH_LENGTH,
                opt.congestion,
                1.0 if opt.is_feasible else 0.0,
                (opt.slots_needed or 0) / MAX_SLOTS,
                len(opt.path) / MAX_HOPS,
            ])
        else:
            # Padding for missing paths
            features.extend([0.0, 1.0, 0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)
```

### Network State Features

```python
def build_network_state_features(network_state: NetworkState) -> np.ndarray:
    """
    Build global network state features.

    Features:
    - overall_utilization: Total spectrum used / total spectrum available
    - active_lightpaths: Number of active lightpaths / max_lightpaths
    - avg_link_congestion: Average congestion across all links
    """
    total_slots = 0
    used_slots = 0

    for link_spectrum in network_state._spectrum.values():
        for band, matrix in link_spectrum.cores_matrix.items():
            total_slots += matrix.size
            used_slots += np.count_nonzero(matrix)

    overall_utilization = used_slots / total_slots if total_slots > 0 else 0.0

    return np.array([
        overall_utilization,
        len(network_state._lightpaths) / 1000.0,  # Normalized
    ], dtype=np.float32)
```

## Model Architectures

### 1. Multi-Layer Perceptron (MLP)

```python
import torch.nn as nn


class PathSelectionMLP(nn.Module):
    """Simple MLP for path selection."""

    def __init__(self, input_dim: int, k_paths: int, hidden_dims: list[int] = [64, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, k_paths))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

### 2. Attention-Based Model

```python
class PathAttentionModel(nn.Module):
    """Attention model that attends over path options."""

    def __init__(self, path_feature_dim: int, k_paths: int, hidden_dim: int = 64):
        super().__init__()

        self.path_encoder = nn.Sequential(
            nn.Linear(path_feature_dim, hidden_dim),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, path_features: torch.Tensor) -> torch.Tensor:
        # path_features: (batch, k_paths, path_feature_dim)
        encoded = self.path_encoder(path_features)  # (batch, k_paths, hidden_dim)

        # Self-attention over paths
        attended, _ = self.attention(encoded, encoded, encoded)

        # Score each path
        scores = self.output_layer(attended).squeeze(-1)  # (batch, k_paths)

        return scores
```

## Training ML Policies

### Supervised Learning from Heuristic Logs

```python
def train_from_heuristic_logs(
    log_path: str,
    model: nn.Module,
    epochs: int = 100,
) -> nn.Module:
    """
    Train ML policy by imitating a heuristic policy.

    Log format: (features, action, success)
    """
    # Load logs
    data = np.load(log_path)
    features = torch.FloatTensor(data['features'])
    actions = torch.LongTensor(data['actions'])

    # Only train on successful allocations
    mask = torch.BoolTensor(data['success'])
    features = features[mask]
    actions = actions[mask]

    dataset = TensorDataset(features, actions)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_actions in dataloader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

    return model
```

### Offline RL (Behavior Cloning)

```python
def behavior_cloning(
    trajectories: list[dict],
    model: nn.Module,
    epochs: int = 100,
) -> nn.Module:
    """
    Train using behavior cloning from expert trajectories.

    Trajectory format: {
        'observations': [...],
        'actions': [...],
        'rewards': [...],
    }
    """
    # Collect (observation, action) pairs
    all_obs = []
    all_actions = []

    for traj in trajectories:
        all_obs.extend(traj['observations'])
        all_actions.extend(traj['actions'])

    features = torch.FloatTensor(np.array(all_obs))
    actions = torch.LongTensor(all_actions)

    # Weight by reward (optional)
    # Higher reward = more weight in training

    dataset = TensorDataset(features, actions)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_features, batch_actions in dataloader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()

    return model
```

## Model Serialization

```python
def save_ml_policy(model: nn.Module, path: str, model_type: str = "pytorch"):
    """Save model for deployment."""
    if model_type == "pytorch":
        torch.save(model, path)
    elif model_type == "onnx":
        # Export to ONNX for cross-platform deployment
        dummy_input = torch.randn(1, model.input_dim)
        torch.onnx.export(
            model,
            dummy_input,
            path,
            input_names=['features'],
            output_names=['logits'],
            dynamic_axes={'features': {0: 'batch_size'}},
        )
```

## Configuration

```ini
[policy]
type = ml
model_path = models/path_selection_mlp.pt
model_type = pytorch      ; pytorch, sklearn, onnx
device = cpu              ; cpu, cuda

[ml_policy]
feature_set = standard    ; standard, extended
normalize_features = true
fallback_policy = first_feasible
```

## Evaluation

```python
def evaluate_ml_policy(
    policy: MLControlPolicy,
    config: SimulationConfig,
    n_episodes: int = 100,
) -> dict:
    """Evaluate ML policy performance."""
    from fusion.modules.rl.gymnasium_envs.unified_sim_env import UnifiedSimEnv

    env = UnifiedSimEnv(config, topology_path=config.topology_path)
    results = {
        'blocking_rates': [],
        'avg_rewards': [],
    }

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        blocked = 0
        total = 0

        while not done:
            options = env._path_options
            request = env._requests[env._current_idx]
            action = policy.select_action(request, options, env._network_state)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            total += 1

            if info.get('allocation_result') and not info['allocation_result'].success:
                blocked += 1

            done = terminated or truncated

        results['blocking_rates'].append(blocked / total)
        results['avg_rewards'].append(episode_reward / total)

    return {
        'mean_blocking_rate': np.mean(results['blocking_rates']),
        'std_blocking_rate': np.std(results['blocking_rates']),
        'mean_reward': np.mean(results['avg_rewards']),
        'std_reward': np.std(results['avg_rewards']),
    }
```

## See Also

- [Control Policy Architecture](./control_policy.md)
- [Heuristic Policies Architecture](./heuristic_policies.md)
- [RL Integration](./rl_integration.md)
- [ADR-0011: ML Control Policy](../decisions/0011-ml-control-policy.md)
