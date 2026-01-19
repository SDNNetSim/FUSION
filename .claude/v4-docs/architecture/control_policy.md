# Control Policy Architecture

## Overview

The `ControlPolicy` protocol provides a unified abstraction for decision-making in FUSION's resource allocation process. It enables the orchestrator to delegate path selection decisions to pluggable policy implementations, supporting reinforcement learning (RL), machine learning (ML), and heuristic approaches through a common interface.

## Motivation

Prior to Phase 5, path selection logic was tightly coupled to specific implementations:

- **RL environments** had their own action selection mechanisms
- **Heuristic simulations** used hardcoded selection in SDNController
- **No support** for ML-based classifiers or hybrid approaches

The ControlPolicy abstraction addresses these issues by:

1. **Decoupling selection logic** from the orchestrator
2. **Enabling policy composition** (e.g., fallback from ML to heuristic)
3. **Providing a testable interface** for policy evaluation
4. **Supporting online and offline learning** through a consistent API

## Protocol Definition

```python
from typing import Protocol, runtime_checkable
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.modules.rl.env_adapter import PathOption


@runtime_checkable
class ControlPolicy(Protocol):
    """
    Protocol for control policies that select actions for resource allocation.

    Implementations include:
    - RL policies (wrapping SB3 models)
    - ML policies (pre-trained neural networks, classifiers)
    - Heuristic policies (first-fit, shortest-path, least-congested)
    """

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """
        Select an action (path index) for the given request.

        Args:
            request: The incoming request to serve
            options: Available path options with feasibility information
            network_state: Current state of the network (read-only access)

        Returns:
            Path index to use (0 to len(options)-1), or -1 if no valid action
        """
        ...

    def update(self, request: Request, action: int, reward: float) -> None:
        """
        Optional: Update policy based on experience.

        Called after action is executed and reward is computed.
        Heuristic policies typically ignore this method.

        Args:
            request: The request that was served
            action: The action that was taken
            reward: The reward received
        """
        ...
```

## PathOption Data Structure

```python
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PathOption:
    """Single path option presented to the policy for selection."""

    path_index: int                    # Index in the k-paths list
    path: list[str]                    # Node sequence
    weight_km: float                   # Path length in km
    is_feasible: bool                  # Whether spectrum is available
    modulation: Optional[str]          # Valid modulation if feasible
    slots_needed: Optional[int]        # Spectrum slots required
    congestion: float                  # Average link congestion (0.0-1.0)

    # Optional protection fields
    backup_path: Optional[list[str]] = None
    backup_feasible: Optional[bool] = None
```

## Policy Types

### 1. Heuristic Policies

Deterministic policies that do not learn from experience:

| Policy | Selection Criterion |
|--------|---------------------|
| `FirstFeasiblePolicy` | First path with `is_feasible=True` |
| `ShortestFeasiblePolicy` | Feasible path with minimum `weight_km` |
| `LeastCongestedPolicy` | Feasible path with minimum `congestion` |
| `RandomFeasiblePolicy` | Random selection among feasible paths |

### 2. RL Policies

Policies wrapping trained RL models (e.g., from Stable-Baselines3):

```python
class RLPolicy(ControlPolicy):
    """Wraps a trained SB3 model."""

    def __init__(self, model: BaseAlgorithm):
        self.model = model

    def select_action(self, request, options, network_state) -> int:
        obs = self._build_observation(request, options, network_state)
        action_mask = [opt.is_feasible for opt in options]
        action, _ = self.model.predict(obs, action_masks=action_mask)
        return action

    def update(self, request, action, reward) -> None:
        pass  # Model is pre-trained
```

### 3. ML Policies

Policies using pre-trained ML models (classifiers, neural networks):

```python
class MLControlPolicy(ControlPolicy):
    """Uses pre-trained neural network for path selection."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def select_action(self, request, options, network_state) -> int:
        features = self._extract_features(request, options, network_state)
        with torch.no_grad():
            logits = self.model(features)
            # Apply action mask
            mask = torch.tensor([opt.is_feasible for opt in options])
            logits[~mask] = float('-inf')
            return logits.argmax().item()
```

## Integration with SDNOrchestrator

```
                         ControlPolicy
                              |
                       select_action()
                              |
                              v
    +---------------------------------------------------------+
    |                    SDNOrchestrator                       |
    |  1. Get RouteResult from RoutingPipeline                |
    |  2. Convert to PathOptions via RLSimulationAdapter      |
    |  3. Call policy.select_action()                         |
    |  4. Execute chosen action via handle_arrival()          |
    |  5. Call policy.update() with reward                    |
    +---------------------------------------------------------+
```

### Orchestrator Integration Code

```python
class SDNOrchestrator:
    def __init__(
        self,
        config: SimulationConfig,
        pipelines: PipelineSet,
        policy: Optional[ControlPolicy] = None,
    ):
        self.config = config
        self.routing = pipelines.routing
        self.spectrum = pipelines.spectrum
        # ... other pipelines

        # Policy is optional - defaults to FirstFeasiblePolicy
        self.policy = policy or FirstFeasiblePolicy()

        # Adapter for converting routes to policy options
        self._adapter = RLSimulationAdapter(config, self)

    def handle_arrival_with_policy(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> AllocationResult:
        """Handle arrival using configured policy for path selection."""

        # Get path options (same as RL would see)
        options = self._adapter.get_path_options(
            request.source,
            request.destination,
            request.bandwidth_gbps,
            network_state,
        )

        if not any(opt.is_feasible for opt in options):
            return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)

        # Delegate to policy
        action = self.policy.select_action(request, options, network_state)

        if action < 0 or action >= len(options) or not options[action].is_feasible:
            return AllocationResult(success=False, block_reason=BlockReason.INVALID_ACTION)

        # Execute action
        result, reward = self._adapter.apply_action(
            action, request, options, network_state
        )

        # Update policy
        self.policy.update(request, action, reward)

        return result
```

## Policy Factory

```python
class PolicyFactory:
    """Factory for creating control policies from configuration."""

    @staticmethod
    def create(config: SimulationConfig) -> ControlPolicy:
        policy_type = config.policy_type

        if policy_type == "first_feasible":
            return FirstFeasiblePolicy()
        elif policy_type == "shortest_feasible":
            return ShortestFeasiblePolicy()
        elif policy_type == "least_congested":
            return LeastCongestedPolicy()
        elif policy_type == "rl":
            model = load_rl_model(config.policy_model_path)
            return RLPolicy(model)
        elif policy_type == "ml":
            return MLControlPolicy(config.policy_model_path, config.device)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
```

## Action Masking

All policies must respect feasibility constraints via action masking:

```python
def apply_action_mask(logits: torch.Tensor, options: list[PathOption]) -> torch.Tensor:
    """Apply action mask to model outputs."""
    mask = torch.tensor([opt.is_feasible for opt in options], dtype=torch.bool)
    masked_logits = logits.clone()
    masked_logits[~mask] = float('-inf')
    return masked_logits
```

## Configuration

```ini
[policy]
type = first_feasible  ; Options: first_feasible, shortest_feasible, least_congested, rl, ml
model_path =           ; Path to model file for rl/ml policies
device = cpu           ; Device for ML inference: cpu, cuda

[reward]
success = 1.0          ; Reward for successful allocation
failure = -1.0         ; Penalty for blocked request
efficiency_bonus = 0.1 ; Bonus multiplier for efficient allocations
```

## Thread Safety

Policies are **not thread-safe** by default. For parallel simulations:

1. Create one policy instance per simulation thread
2. Or use `ThreadSafePolicy` wrapper:

```python
class ThreadSafePolicy(ControlPolicy):
    def __init__(self, policy: ControlPolicy):
        self._policy = policy
        self._lock = threading.Lock()

    def select_action(self, request, options, network_state) -> int:
        with self._lock:
            return self._policy.select_action(request, options, network_state)
```

## Testing Policies

```python
def test_policy_respects_feasibility():
    """Policy should never select infeasible actions."""
    policy = FirstFeasiblePolicy()

    options = [
        PathOption(path_index=0, is_feasible=False, ...),
        PathOption(path_index=1, is_feasible=True, ...),
        PathOption(path_index=2, is_feasible=False, ...),
    ]

    action = policy.select_action(request, options, network_state)
    assert action == 1  # Only feasible option
    assert options[action].is_feasible
```

## See Also

- [ML Policies Architecture](./ml_policies.md)
- [Heuristic Policies Architecture](./heuristic_policies.md)
- [ADR-0010: ControlPolicy Protocol](../decisions/0010-control-policy-protocol.md)
- [Tutorial: Implementing a New Policy](../tutorials/implementing_a_new_policy.md)
