# Tutorial: Implementing a New Policy

This tutorial walks through implementing a custom control policy in FUSION. You will learn how to:

1. Implement the `ControlPolicy` protocol
2. Access path options and network state
3. Make selection decisions
4. Register your policy with the factory
5. Test your implementation

## Prerequisites

- Familiarity with Python dataclasses and protocols
- Understanding of FUSION's path selection concepts
- Phase 5 documentation read (ControlPolicy architecture)

## Goal

We will implement a `BalancedHopCountPolicy` that selects paths based on a weighted combination of hop count and congestion, preferring paths with fewer hops and lower congestion.

## Step 1: Understand the ControlPolicy Protocol

The `ControlPolicy` protocol requires two methods:

```python
from typing import Protocol

class ControlPolicy(Protocol):
    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """Select a path index, or -1 if no valid action."""
        ...

    def update(self, request: Request, action: int, reward: float) -> None:
        """Optional: Update based on experience."""
        ...
```

Key points:
- `select_action` receives the request, available path options, and network state
- Return a valid path index (0 to len(options)-1) or -1 if no action possible
- `options` contains `PathOption` objects with feasibility and metrics
- `update` is called after allocation with the reward (optional for heuristics)

## Step 2: Create the Policy Class

Create a new file `fusion/policies/balanced_hop_count_policy.py`:

```python
"""Balanced hop count policy for FUSION."""

from fusion.interfaces.control_policy import ControlPolicy
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.modules.rl.env_adapter import PathOption


class BalancedHopCountPolicy(ControlPolicy):
    """
    Select paths based on hop count and congestion balance.

    Score = alpha * normalized_hops + (1 - alpha) * congestion

    Lower score is better (fewer hops, less congestion).

    Attributes:
        alpha: Weight for hop count (0-1). Default 0.6.
        max_hops: Maximum expected hops for normalization. Default 15.
    """

    def __init__(self, alpha: float = 0.6, max_hops: int = 15):
        """
        Initialize the policy.

        Args:
            alpha: Weight for hop count in scoring (0-1)
            max_hops: Maximum expected hops for normalization
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if max_hops <= 0:
            raise ValueError(f"max_hops must be positive, got {max_hops}")

        self.alpha = alpha
        self.max_hops = max_hops

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """
        Select the path with best hop/congestion balance.

        Args:
            request: The incoming request
            options: Available path options
            network_state: Current network state (unused in this policy)

        Returns:
            Index of selected path, or -1 if no feasible option
        """
        # Filter to feasible options only
        feasible = [opt for opt in options if opt.is_feasible]

        if not feasible:
            return -1

        # Score each option (lower is better)
        def score(opt: PathOption) -> float:
            hop_count = len(opt.path) - 1  # Number of hops
            normalized_hops = min(hop_count / self.max_hops, 1.0)
            return self.alpha * normalized_hops + (1 - self.alpha) * opt.congestion

        # Select option with lowest score
        best = min(feasible, key=score)
        return best.path_index

    def update(self, request: Request, action: int, reward: float) -> None:
        """
        No learning for this heuristic policy.

        Args:
            request: The served request
            action: The action taken
            reward: The reward received
        """
        pass  # Heuristics don't learn
```

## Step 3: Understand PathOption

The `PathOption` dataclass provides information about each candidate path:

```python
@dataclass(frozen=True)
class PathOption:
    path_index: int           # Index in the k-paths list (0 to k-1)
    path: list[str]           # Node sequence, e.g., ["A", "B", "C"]
    weight_km: float          # Total path length in km
    is_feasible: bool         # Whether spectrum is available
    modulation: str | None    # Valid modulation format (if feasible)
    slots_needed: int | None  # Spectrum slots required (if feasible)
    congestion: float         # Average link congestion (0.0 to 1.0)
```

Your policy should:
- Only consider options where `is_feasible=True`
- Return the `path_index` of the selected option
- Use available metrics (weight_km, congestion, slots_needed) for decisions

## Step 4: Add Tests

Create `fusion/tests/policies/test_balanced_hop_count_policy.py`:

```python
"""Tests for BalancedHopCountPolicy."""

import pytest
from fusion.policies.balanced_hop_count_policy import BalancedHopCountPolicy
from fusion.modules.rl.env_adapter import PathOption


@pytest.fixture
def sample_options():
    """Create sample path options."""
    return [
        PathOption(
            path_index=0,
            path=["A", "B", "C"],  # 2 hops
            weight_km=200.0,
            is_feasible=True,
            modulation="QPSK",
            slots_needed=10,
            congestion=0.8,  # High congestion
        ),
        PathOption(
            path_index=1,
            path=["A", "D", "E", "C"],  # 3 hops
            weight_km=300.0,
            is_feasible=True,
            modulation="QPSK",
            slots_needed=10,
            congestion=0.2,  # Low congestion
        ),
        PathOption(
            path_index=2,
            path=["A", "F", "C"],  # 2 hops
            weight_km=250.0,
            is_feasible=False,  # Infeasible
            modulation=None,
            slots_needed=None,
            congestion=0.1,
        ),
    ]


@pytest.fixture
def mock_request():
    """Create mock request."""
    from fusion.domain.request import Request, RequestStatus
    return Request(
        request_id=1,
        source="A",
        destination="C",
        bandwidth_gbps=100,
        holding_time=1.0,
        arrival_time=0.0,
        status=RequestStatus.PENDING,
    )


@pytest.fixture
def mock_network_state(mocker):
    """Create mock network state."""
    return mocker.MagicMock()


class TestBalancedHopCountPolicy:
    """Tests for BalancedHopCountPolicy."""

    def test_initialization(self):
        """Test policy initialization."""
        policy = BalancedHopCountPolicy(alpha=0.5, max_hops=10)
        assert policy.alpha == 0.5
        assert policy.max_hops == 10

    def test_invalid_alpha_raises(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError):
            BalancedHopCountPolicy(alpha=1.5)
        with pytest.raises(ValueError):
            BalancedHopCountPolicy(alpha=-0.1)

    def test_invalid_max_hops_raises(self):
        """Test that invalid max_hops raises ValueError."""
        with pytest.raises(ValueError):
            BalancedHopCountPolicy(max_hops=0)
        with pytest.raises(ValueError):
            BalancedHopCountPolicy(max_hops=-1)

    def test_selects_feasible_option(self, sample_options, mock_request, mock_network_state):
        """Test that policy only selects feasible options."""
        policy = BalancedHopCountPolicy()
        action = policy.select_action(mock_request, sample_options, mock_network_state)
        assert action in [0, 1]  # Not 2 (infeasible)
        assert sample_options[action].is_feasible

    def test_alpha_one_prefers_fewer_hops(self, sample_options, mock_request, mock_network_state):
        """Test that alpha=1 prefers fewer hops regardless of congestion."""
        policy = BalancedHopCountPolicy(alpha=1.0)
        action = policy.select_action(mock_request, sample_options, mock_network_state)
        assert action == 0  # 2 hops vs 3 hops

    def test_alpha_zero_prefers_lower_congestion(self, sample_options, mock_request, mock_network_state):
        """Test that alpha=0 prefers lower congestion regardless of hops."""
        policy = BalancedHopCountPolicy(alpha=0.0)
        action = policy.select_action(mock_request, sample_options, mock_network_state)
        assert action == 1  # 0.2 congestion vs 0.8

    def test_returns_negative_one_when_all_infeasible(self, mock_request, mock_network_state):
        """Test that -1 is returned when no feasible options."""
        options = [
            PathOption(path_index=0, path=["A", "B"], weight_km=100, is_feasible=False,
                       modulation=None, slots_needed=None, congestion=0.5),
        ]
        policy = BalancedHopCountPolicy()
        action = policy.select_action(mock_request, options, mock_network_state)
        assert action == -1

    def test_returns_negative_one_for_empty_options(self, mock_request, mock_network_state):
        """Test that -1 is returned for empty options list."""
        policy = BalancedHopCountPolicy()
        action = policy.select_action(mock_request, [], mock_network_state)
        assert action == -1

    def test_update_is_noop(self, mock_request):
        """Test that update does nothing (heuristic policy)."""
        policy = BalancedHopCountPolicy()
        # Should not raise
        policy.update(mock_request, 0, 1.0)

    def test_balanced_selection(self, mock_request, mock_network_state):
        """Test balanced selection with alpha=0.5."""
        # Create options where balance matters
        options = [
            PathOption(path_index=0, path=["A", "B"], weight_km=100, is_feasible=True,
                       modulation="QPSK", slots_needed=10, congestion=0.9),  # 1 hop, high cong
            PathOption(path_index=1, path=["A", "C", "D"], weight_km=200, is_feasible=True,
                       modulation="QPSK", slots_needed=10, congestion=0.1),  # 2 hops, low cong
        ]
        policy = BalancedHopCountPolicy(alpha=0.5, max_hops=10)
        action = policy.select_action(mock_request, options, mock_network_state)

        # Calculate scores
        # Option 0: 0.5 * (1/10) + 0.5 * 0.9 = 0.05 + 0.45 = 0.50
        # Option 1: 0.5 * (2/10) + 0.5 * 0.1 = 0.10 + 0.05 = 0.15
        # Option 1 should win (lower score)
        assert action == 1
```

Run the tests:

```bash
pytest fusion/tests/policies/test_balanced_hop_count_policy.py -v
```

## Step 5: Register with Policy Factory

Add your policy to the factory in `fusion/policies/policy_factory.py`:

```python
from fusion.policies.balanced_hop_count_policy import BalancedHopCountPolicy

# In the REGISTRY
POLICY_REGISTRY: dict[str, type] = {
    "first_feasible": FirstFeasiblePolicy,
    "shortest_feasible": ShortestFeasiblePolicy,
    "least_congested": LeastCongestedPolicy,
    "random_feasible": RandomFeasiblePolicy,
    "load_balanced": LoadBalancedPolicy,
    "balanced_hop_count": BalancedHopCountPolicy,  # NEW
}
```

Or use the registration method:

```python
from fusion.policies.policy_factory import HeuristicPolicyFactory
from fusion.policies.balanced_hop_count_policy import BalancedHopCountPolicy

HeuristicPolicyFactory.register("balanced_hop_count", BalancedHopCountPolicy)
```

## Step 6: Configure and Use

### Via Configuration

```ini
[policy]
type = balanced_hop_count

[heuristic]
alpha = 0.6
max_hops = 15
```

### Via Code

```python
from fusion.policies.balanced_hop_count_policy import BalancedHopCountPolicy
from fusion.core.orchestrator import SDNOrchestrator

# Create policy
policy = BalancedHopCountPolicy(alpha=0.7, max_hops=12)

# Use with orchestrator
orchestrator = SDNOrchestrator(config, pipelines, policy=policy)
```

## Step 7: Run Integration Test

Verify your policy works in simulation:

```python
from fusion.domain.config import SimulationConfig
from fusion.core.simulation import SimulationEngine

config = SimulationConfig.from_ini("config.ini")
engine = SimulationEngine(config)

# Run simulation
results = engine.run()
print(f"Blocking rate: {results['blocking_probability']:.4f}")
```

## Advanced: Stateful Policy

If your policy needs to learn or maintain state:

```python
class AdaptivePolicy(ControlPolicy):
    """Policy that adapts based on observed rewards."""

    def __init__(self):
        self.path_rewards: dict[int, list[float]] = {}

    def select_action(self, request, options, network_state) -> int:
        feasible = [o for o in options if o.is_feasible]
        if not feasible:
            return -1

        # Use historical rewards to inform selection
        best_score = float('-inf')
        best_action = feasible[0].path_index

        for opt in feasible:
            idx = opt.path_index
            avg_reward = (
                sum(self.path_rewards.get(idx, [0])) /
                len(self.path_rewards.get(idx, [1]))
            )
            if avg_reward > best_score:
                best_score = avg_reward
                best_action = idx

        return best_action

    def update(self, request, action, reward) -> None:
        """Track rewards per path index."""
        if action not in self.path_rewards:
            self.path_rewards[action] = []
        self.path_rewards[action].append(reward)

        # Keep only recent history
        if len(self.path_rewards[action]) > 100:
            self.path_rewards[action] = self.path_rewards[action][-100:]
```

## Summary

To implement a new policy:

1. Create a class implementing `ControlPolicy`
2. Implement `select_action` to return a path index (or -1)
3. Implement `update` (can be no-op for heuristics)
4. Filter for feasible options only
5. Add comprehensive tests
6. Register with the policy factory
7. Test in simulation

## See Also

- [Control Policy Architecture](../architecture/control_policy.md)
- [Heuristic Policies Architecture](../architecture/heuristic_policies.md)
- [ADR-0010: ControlPolicy Protocol](../decisions/0010-control-policy-protocol.md)
