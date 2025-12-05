# Heuristic Policies Architecture

## Overview

Heuristic policies provide deterministic, rule-based path selection strategies for FUSION. They implement the `ControlPolicy` protocol and serve as baselines for evaluating RL and ML policies, as well as production-ready solutions for scenarios where learning-based approaches are not required.

## Policy Hierarchy

```
ControlPolicy (Protocol)
    |
    +-- HeuristicPolicy (Abstract Base)
            |
            +-- FirstFeasiblePolicy
            +-- ShortestFeasiblePolicy
            +-- LeastCongestedPolicy
            +-- RandomFeasiblePolicy
            +-- LoadBalancedPolicy
            +-- ModulationAwarePolicy
```

## Base Class

```python
from abc import ABC, abstractmethod
from fusion.interfaces.control_policy import ControlPolicy
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.modules.rl.env_adapter import PathOption


class HeuristicPolicy(ControlPolicy, ABC):
    """
    Abstract base class for heuristic policies.

    Heuristic policies:
    - Are deterministic (same input = same output)
    - Do not learn from experience
    - Provide baseline performance for comparison
    """

    @abstractmethod
    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """Select action based on heuristic rule."""
        ...

    def update(self, request: Request, action: int, reward: float) -> None:
        """Heuristics don't learn - this is a no-op."""
        pass

    def _get_feasible_options(self, options: list[PathOption]) -> list[PathOption]:
        """Filter to only feasible options."""
        return [opt for opt in options if opt.is_feasible]
```

## Standard Heuristic Policies

### 1. FirstFeasiblePolicy

Selects the first feasible path (index order from routing).

```python
class FirstFeasiblePolicy(HeuristicPolicy):
    """
    Select the first feasible path.

    This is equivalent to traditional k-shortest-path with first-fit spectrum:
    paths are tried in order until one succeeds.

    Characteristics:
    - Fastest decision time (O(k) worst case)
    - Tends to prefer shorter paths (since KSP returns paths by length)
    - May cause congestion on popular short paths
    """

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        for opt in options:
            if opt.is_feasible:
                return opt.path_index
        return -1
```

### 2. ShortestFeasiblePolicy

Selects the shortest (by distance) feasible path.

```python
class ShortestFeasiblePolicy(HeuristicPolicy):
    """
    Select the shortest feasible path by distance (km).

    Characteristics:
    - Minimizes signal degradation (shorter = less attenuation)
    - Good for SNR-sensitive scenarios
    - May concentrate traffic on few paths
    """

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        feasible = self._get_feasible_options(options)
        if not feasible:
            return -1

        shortest = min(feasible, key=lambda opt: opt.weight_km)
        return shortest.path_index
```

### 3. LeastCongestedPolicy

Selects the path with lowest average link congestion.

```python
class LeastCongestedPolicy(HeuristicPolicy):
    """
    Select the least congested feasible path.

    Congestion is measured as average spectrum utilization
    across all links in the path.

    Characteristics:
    - Spreads load across network
    - Reduces blocking probability under high load
    - May use longer paths, increasing spectrum usage per request
    """

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        feasible = self._get_feasible_options(options)
        if not feasible:
            return -1

        least_congested = min(feasible, key=lambda opt: opt.congestion)
        return least_congested.path_index
```

### 4. RandomFeasiblePolicy

Randomly selects among feasible paths.

```python
import random


class RandomFeasiblePolicy(HeuristicPolicy):
    """
    Randomly select among feasible paths.

    Characteristics:
    - Provides baseline for comparison
    - Naturally load-balances over many requests
    - Useful for exploring policy space
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        feasible = self._get_feasible_options(options)
        if not feasible:
            return -1

        return self._rng.choice(feasible).path_index
```

### 5. LoadBalancedPolicy

Balances load by considering both path length and congestion.

```python
class LoadBalancedPolicy(HeuristicPolicy):
    """
    Balance between path length and congestion.

    Score = alpha * normalized_length + (1 - alpha) * congestion

    Characteristics:
    - Tunable tradeoff between efficiency and load balancing
    - alpha=1.0 equivalent to ShortestFeasiblePolicy
    - alpha=0.0 equivalent to LeastCongestedPolicy
    """

    def __init__(self, alpha: float = 0.5, max_length_km: float = 10000.0):
        """
        Args:
            alpha: Weight for path length (0-1)
            max_length_km: Maximum expected path length for normalization
        """
        self.alpha = alpha
        self.max_length_km = max_length_km

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        feasible = self._get_feasible_options(options)
        if not feasible:
            return -1

        def score(opt: PathOption) -> float:
            normalized_length = opt.weight_km / self.max_length_km
            return self.alpha * normalized_length + (1 - self.alpha) * opt.congestion

        best = min(feasible, key=score)
        return best.path_index
```

### 6. ModulationAwarePolicy

Considers modulation format efficiency in selection.

```python
class ModulationAwarePolicy(HeuristicPolicy):
    """
    Select path based on modulation efficiency.

    Prefers paths that can use higher-order modulation formats,
    which require fewer spectrum slots.

    Characteristics:
    - Maximizes spectral efficiency
    - Good for bandwidth-constrained networks
    - May prefer shorter paths (support higher modulation)
    """

    # Modulation efficiency (bits per symbol)
    MODULATION_ORDER = {
        "BPSK": 1,
        "QPSK": 2,
        "8-QAM": 3,
        "16-QAM": 4,
        "32-QAM": 5,
        "64-QAM": 6,
    }

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        feasible = self._get_feasible_options(options)
        if not feasible:
            return -1

        def modulation_score(opt: PathOption) -> int:
            if opt.modulation is None:
                return 0
            return self.MODULATION_ORDER.get(opt.modulation, 0)

        # Higher modulation order = better (fewer slots needed)
        best = max(feasible, key=modulation_score)
        return best.path_index
```

## Composite Policies

### TiebreakingPolicy

Uses secondary criteria when primary criterion has ties.

```python
class TiebreakingPolicy(HeuristicPolicy):
    """
    Chain multiple policies for tiebreaking.

    Example: Shortest path, with ties broken by least congested.
    """

    def __init__(self, policies: list[HeuristicPolicy]):
        self.policies = policies

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        if not self.policies:
            return -1

        feasible = self._get_feasible_options(options)
        if not feasible:
            return -1

        # Apply first policy
        primary_action = self.policies[0].select_action(request, options, network_state)
        if len(self.policies) == 1:
            return primary_action

        # Find ties (same primary criterion value)
        primary_opt = next((o for o in options if o.path_index == primary_action), None)
        if primary_opt is None:
            return primary_action

        # This is simplified - real implementation would need
        # to track the actual criterion value for proper tiebreaking
        return primary_action
```

### FallbackPolicy

Tries policies in order until one succeeds.

```python
class FallbackPolicy(HeuristicPolicy):
    """
    Try policies in order until one returns a valid action.

    Useful for combining ML/RL policies with heuristic fallbacks.
    """

    def __init__(self, policies: list[ControlPolicy]):
        self.policies = policies

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        for policy in self.policies:
            action = policy.select_action(request, options, network_state)
            if action >= 0:
                return action
        return -1
```

## Policy Factory

```python
class HeuristicPolicyFactory:
    """Factory for creating heuristic policies from configuration."""

    REGISTRY: dict[str, type[HeuristicPolicy]] = {
        "first_feasible": FirstFeasiblePolicy,
        "shortest_feasible": ShortestFeasiblePolicy,
        "least_congested": LeastCongestedPolicy,
        "random_feasible": RandomFeasiblePolicy,
        "load_balanced": LoadBalancedPolicy,
        "modulation_aware": ModulationAwarePolicy,
    }

    @classmethod
    def create(cls, policy_name: str, **kwargs) -> HeuristicPolicy:
        """Create policy by name."""
        if policy_name not in cls.REGISTRY:
            raise ValueError(
                f"Unknown policy: {policy_name}. "
                f"Available: {list(cls.REGISTRY.keys())}"
            )

        policy_class = cls.REGISTRY[policy_name]
        return policy_class(**kwargs)

    @classmethod
    def register(cls, name: str, policy_class: type[HeuristicPolicy]) -> None:
        """Register a custom heuristic policy."""
        cls.REGISTRY[name] = policy_class
```

## Configuration

```ini
[policy]
type = heuristic
heuristic_name = load_balanced

[heuristic]
# LoadBalancedPolicy parameters
alpha = 0.6
max_length_km = 8000

# RandomFeasiblePolicy parameters
seed = 42
```

## Performance Characteristics

| Policy | Decision Time | Load Balancing | Spectral Efficiency |
|--------|---------------|----------------|---------------------|
| FirstFeasible | O(k) | Poor | Medium |
| ShortestFeasible | O(k) | Poor | High |
| LeastCongested | O(k) | Excellent | Low |
| RandomFeasible | O(k) | Good | Medium |
| LoadBalanced | O(k) | Good | Medium |
| ModulationAware | O(k) | Poor | Excellent |

## Use Cases

| Scenario | Recommended Policy |
|----------|-------------------|
| SNR-sensitive network | ShortestFeasible or ModulationAware |
| High-traffic network | LeastCongested or LoadBalanced |
| Baseline comparison | FirstFeasible |
| Initial exploration | RandomFeasible |
| Production default | LoadBalanced (alpha=0.5) |

## Testing Heuristic Policies

```python
import pytest
from fusion.policies.heuristic_policy import (
    FirstFeasiblePolicy,
    ShortestFeasiblePolicy,
    LeastCongestedPolicy,
)


class TestHeuristicPolicies:
    @pytest.fixture
    def sample_options(self):
        return [
            PathOption(path_index=0, weight_km=100, congestion=0.8, is_feasible=False, ...),
            PathOption(path_index=1, weight_km=200, congestion=0.2, is_feasible=True, ...),
            PathOption(path_index=2, weight_km=150, congestion=0.5, is_feasible=True, ...),
        ]

    def test_first_feasible_returns_first(self, sample_options, mock_request, mock_state):
        policy = FirstFeasiblePolicy()
        action = policy.select_action(mock_request, sample_options, mock_state)
        assert action == 1  # First feasible (index 0 is infeasible)

    def test_shortest_feasible_returns_shortest(self, sample_options, mock_request, mock_state):
        policy = ShortestFeasiblePolicy()
        action = policy.select_action(mock_request, sample_options, mock_state)
        assert action == 2  # 150km < 200km

    def test_least_congested_returns_least(self, sample_options, mock_request, mock_state):
        policy = LeastCongestedPolicy()
        action = policy.select_action(mock_request, sample_options, mock_state)
        assert action == 1  # 0.2 < 0.5

    def test_all_infeasible_returns_negative_one(self, mock_request, mock_state):
        options = [
            PathOption(path_index=0, is_feasible=False, ...),
            PathOption(path_index=1, is_feasible=False, ...),
        ]
        for policy in [FirstFeasiblePolicy(), ShortestFeasiblePolicy(), LeastCongestedPolicy()]:
            assert policy.select_action(mock_request, options, mock_state) == -1
```

## See Also

- [Control Policy Architecture](./control_policy.md)
- [ML Policies Architecture](./ml_policies.md)
- [Tutorial: Implementing a New Policy](../tutorials/implementing_a_new_policy.md)
