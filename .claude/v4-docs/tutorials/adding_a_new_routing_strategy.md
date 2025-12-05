# Adding a New Routing Strategy

This tutorial shows how to add a new routing strategy that plugs into the `RoutingStrategy` protocol and `RoutingPipeline` without breaking other parts of the system.

## Prerequisites

- [Getting Started with Domain Model](./getting_started_with_domain_model.md)
- [Working with Requests and Results](./working_with_requests_and_results.md)
- Understanding of optical network routing concepts

## Related Documentation

- [Architecture: Routing Strategies](../architecture/routing_strategies.md)
- [Architecture: Result Objects](../architecture/result_objects.md)
- [ADR-0003: Result Object Design](../decisions/0003-result-object-design.md)

---

## Overview: The RoutingStrategy Protocol

All routing strategies implement the `RoutingStrategy` protocol:

```python
from typing import Protocol
from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState
from fusion.domain.results import RouteResult

class RoutingStrategy(Protocol):
    """Protocol for routing algorithm implementations."""

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        """
        Find candidate routes between source and destination.

        Args:
            source: Source node ID
            destination: Destination node ID
            bandwidth_gbps: Required bandwidth
            network_state: Current network state (read-only access)
            forced_path: Optional path to use (e.g., for grooming continuation)

        Returns:
            RouteResult with candidate paths, modulations, and weights
        """
        ...
```

---

## Step 1: Define the Strategy Class

Let's create a "Least Congested Path" routing strategy that prioritizes paths with lower link utilization.

### File Location

Create: `fusion/routing/strategies/least_congested.py`

```python
"""Least Congested Path routing strategy."""

from __future__ import annotations

import networkx as nx

from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState
from fusion.domain.results import RouteResult


class LeastCongestedStrategy:
    """
    Routing strategy that prioritizes paths with lower congestion.

    This strategy:
    1. Computes k-shortest paths (like KSP)
    2. Scores each path by average link congestion
    3. Returns paths sorted by congestion (lowest first)
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the strategy.

        Args:
            config: Simulation configuration with routing parameters
        """
        self.config = config
        self.k = config.k_paths
        self.mod_formats = config.modulation_formats
        self.mod_per_bw = config.mod_per_bw

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        """
        Find routes prioritized by congestion level.

        Args:
            source: Source node ID
            destination: Destination node ID
            bandwidth_gbps: Required bandwidth in Gbps
            network_state: Current network state
            forced_path: If provided, return only this path

        Returns:
            RouteResult with paths sorted by congestion (least congested first)
        """
        # Handle forced path case (e.g., partial grooming)
        if forced_path is not None:
            return self._single_path_result(
                forced_path, bandwidth_gbps, network_state
            )

        topology = network_state.topology

        # Step 1: Get candidate paths using k-shortest paths
        try:
            candidates = list(
                nx.shortest_simple_paths(topology, source, destination, weight="weight")
            )[:self.k * 2]  # Get extra candidates for scoring
        except nx.NetworkXNoPath:
            return RouteResult(paths=[], weights_km=[], modulations=[])

        if not candidates:
            return RouteResult(paths=[], weights_km=[], modulations=[])

        # Step 2: Score paths by congestion
        scored_paths = []
        for path in candidates:
            congestion = self._compute_path_congestion(path, network_state)
            weight = self._compute_path_weight(path, topology)
            modulations = self._get_valid_modulations(weight, bandwidth_gbps)

            # Only include paths with valid modulations
            if any(m is not None for m in modulations):
                scored_paths.append((congestion, path, weight, modulations))

        # Step 3: Sort by congestion and take top k
        scored_paths.sort(key=lambda x: x[0])  # Sort by congestion (ascending)
        top_k = scored_paths[:self.k]

        if not top_k:
            return RouteResult(paths=[], weights_km=[], modulations=[])

        # Step 4: Build result
        paths = [p[1] for p in top_k]
        weights = [p[2] for p in top_k]
        modulations = [p[3] for p in top_k]

        return RouteResult(
            paths=paths,
            weights_km=weights,
            modulations=modulations,
            strategy_name="least_congested",
            metadata={"congestion_scores": [p[0] for p in top_k]},
        )

    def _single_path_result(
        self,
        path: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> RouteResult:
        """Create result for a single forced path."""
        topology = network_state.topology
        weight = self._compute_path_weight(path, topology)
        modulations = self._get_valid_modulations(weight, bandwidth_gbps)

        return RouteResult(
            paths=[path],
            weights_km=[weight],
            modulations=[modulations],
            strategy_name="least_congested",
        )

    def _compute_path_weight(
        self, path: list[str], topology: nx.Graph
    ) -> float:
        """Compute total path length in km."""
        total = 0.0
        for i in range(len(path) - 1):
            edge_data = topology[path[i]][path[i + 1]]
            total += edge_data.get("weight", 0.0)
        return total

    def _compute_path_congestion(
        self, path: list[str], network_state: NetworkState
    ) -> float:
        """
        Compute average congestion across path links.

        Returns value between 0.0 (empty) and 1.0 (fully congested).
        """
        total_slots = 0
        used_slots = 0

        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            link_spectrum = network_state.get_link_spectrum(link)

            for band in self.config.band_list:
                matrix = link_spectrum.cores_matrix[band]
                total_slots += matrix.size
                # Count non-zero entries as used
                used_slots += (matrix != 0).sum()

        if total_slots == 0:
            return 0.0

        return used_slots / total_slots

    def _get_valid_modulations(
        self, path_length_km: float, bandwidth_gbps: int
    ) -> list[str | None]:
        """
        Get valid modulation formats for given path length and bandwidth.

        Returns list of modulations (most efficient first) or [None] if none valid.
        """
        # Get candidate modulations for this bandwidth
        candidates = self.mod_per_bw.get(bandwidth_gbps, [])
        if not candidates:
            # Try default modulation list
            candidates = list(self.mod_formats.keys())

        valid = []
        for mod in candidates:
            mod_info = self.mod_formats.get(mod, {})
            max_reach = mod_info.get("max_reach_km", float("inf"))

            if path_length_km <= max_reach:
                valid.append(mod)

        if not valid:
            return [None]

        # Sort by spectral efficiency (fewer slots = more efficient)
        valid.sort(
            key=lambda m: self.mod_formats.get(m, {}).get("slots_per_100g", 999)
        )

        return valid
```

---

## Step 2: Register the Strategy

Add the new strategy to the registry so it can be instantiated by the factory.

### Update: `fusion/routing/strategies/__init__.py`

```python
"""Routing strategy implementations."""

from fusion.routing.strategies.ksp import KShortestPathStrategy
from fusion.routing.strategies.load_balanced import LoadBalancedStrategy
from fusion.routing.strategies.least_congested import LeastCongestedStrategy
from fusion.routing.strategies.protection import OnePlusOneProtectionStrategy

__all__ = [
    "KShortestPathStrategy",
    "LoadBalancedStrategy",
    "LeastCongestedStrategy",
    "OnePlusOneProtectionStrategy",
]

# Strategy registry - maps config values to strategy classes
ROUTING_STRATEGIES = {
    "k_shortest_path": KShortestPathStrategy,
    "load_balanced": LoadBalancedStrategy,
    "least_congested": LeastCongestedStrategy,  # NEW
    "1plus1_protection": OnePlusOneProtectionStrategy,
}
```

### Update: `fusion/core/pipeline_factory.py`

```python
from fusion.routing.strategies import ROUTING_STRATEGIES

class PipelineFactory:
    @staticmethod
    def create_routing(config: SimulationConfig) -> RoutingStrategy:
        """Create routing strategy based on configuration."""
        strategy_cls = ROUTING_STRATEGIES.get(config.route_method)

        if strategy_cls is None:
            raise ValueError(
                f"Unknown routing method: {config.route_method}. "
                f"Available: {list(ROUTING_STRATEGIES.keys())}"
            )

        return strategy_cls(config)
```

---

## Step 3: Write Unit Tests

Create comprehensive tests for your strategy.

### File: `fusion/tests/routing/strategies/test_least_congested.py`

```python
"""Tests for LeastCongestedStrategy."""

import pytest
import networkx as nx
import numpy as np

from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState
from fusion.routing.strategies.least_congested import LeastCongestedStrategy


@pytest.fixture
def simple_config():
    """Create a simple test configuration."""
    return SimulationConfig(
        network_name="test",
        cores_per_link=1,
        band_list=("c",),
        band_slots={"c": 320},
        guard_slots=1,
        num_requests=100,
        erlang=100.0,
        holding_time=1.0,
        route_method="least_congested",
        k_paths=3,
        allocation_method="first_fit",
        grooming_enabled=False,
        slicing_enabled=False,
        max_slices=1,
        snr_enabled=False,
        snr_type=None,
        snr_recheck=False,
        can_partially_serve=False,
        modulation_formats={
            "BPSK": {"max_reach_km": 5000, "slots_per_100g": 8},
            "QPSK": {"max_reach_km": 2500, "slots_per_100g": 4},
            "16-QAM": {"max_reach_km": 1000, "slots_per_100g": 2},
        },
        mod_per_bw={100: ["BPSK", "QPSK", "16-QAM"]},
        snr_thresholds={},
    )


@pytest.fixture
def diamond_topology():
    """
    Create a diamond topology for testing:

        B
       / \
      A   D
       \ /
        C

    Weights: A-B=100, A-C=100, B-D=100, C-D=100
    """
    G = nx.Graph()
    G.add_edge("A", "B", weight=100.0)
    G.add_edge("A", "C", weight=100.0)
    G.add_edge("B", "D", weight=100.0)
    G.add_edge("C", "D", weight=100.0)
    return G


@pytest.fixture
def network_state(diamond_topology, simple_config):
    """Create network state with diamond topology."""
    return NetworkState(diamond_topology, simple_config)


class TestLeastCongestedStrategy:
    """Tests for LeastCongestedStrategy."""

    def test_find_routes_returns_valid_paths(
        self, simple_config, network_state
    ):
        """Strategy returns valid paths between endpoints."""
        strategy = LeastCongestedStrategy(simple_config)

        result = strategy.find_routes("A", "D", 100, network_state)

        assert not result.is_empty
        assert len(result.paths) <= simple_config.k_paths
        # All paths should start at A and end at D
        for path in result.paths:
            assert path[0] == "A"
            assert path[-1] == "D"

    def test_find_routes_empty_when_no_path(
        self, simple_config, network_state
    ):
        """Strategy returns empty result when no path exists."""
        strategy = LeastCongestedStrategy(simple_config)

        # X and Y don't exist in the topology
        result = strategy.find_routes("X", "Y", 100, network_state)

        assert result.is_empty

    def test_paths_sorted_by_congestion(
        self, simple_config, network_state
    ):
        """Strategy returns paths sorted by congestion (lowest first)."""
        strategy = LeastCongestedStrategy(simple_config)

        # Congest the A-B link
        network_state._spectrum[("A", "B")].cores_matrix["c"][0][:160] = 1

        result = strategy.find_routes("A", "D", 100, network_state)

        assert not result.is_empty
        # The A-C-D path should come first (less congested)
        assert result.paths[0] == ["A", "C", "D"]

        # Verify congestion scores in metadata
        scores = result.metadata.get("congestion_scores", [])
        assert scores == sorted(scores)  # Should be ascending

    def test_forced_path_honored(self, simple_config, network_state):
        """Forced path is returned without modification."""
        strategy = LeastCongestedStrategy(simple_config)

        forced = ["A", "B", "D"]
        result = strategy.find_routes("A", "D", 100, network_state, forced_path=forced)

        assert result.num_paths == 1
        assert result.paths[0] == forced

    def test_modulations_respect_reach(self, simple_config, network_state):
        """Modulations are filtered based on path reach."""
        # Create long topology
        G = nx.Graph()
        G.add_edge("A", "B", weight=3000.0)  # 3000 km - too long for 16-QAM
        long_state = NetworkState(G, simple_config)

        strategy = LeastCongestedStrategy(simple_config)
        result = strategy.find_routes("A", "B", 100, long_state)

        assert not result.is_empty
        # 16-QAM has max_reach=1000, should not be included
        assert "16-QAM" not in result.modulations[0]
        # BPSK has max_reach=5000, should be included
        assert "BPSK" in result.modulations[0]

    def test_invariant_path_modulation_weight_alignment(
        self, simple_config, network_state
    ):
        """paths, weights_km, and modulations arrays have same length."""
        strategy = LeastCongestedStrategy(simple_config)

        result = strategy.find_routes("A", "D", 100, network_state)

        assert len(result.paths) == len(result.weights_km)
        assert len(result.paths) == len(result.modulations)

    def test_strategy_name_set_correctly(
        self, simple_config, network_state
    ):
        """Strategy name is set in result metadata."""
        strategy = LeastCongestedStrategy(simple_config)

        result = strategy.find_routes("A", "D", 100, network_state)

        assert result.strategy_name == "least_congested"
```

---

## Step 4: Add Configuration Support

Allow users to select the new strategy via configuration.

### Update INI Template

Add to `fusion/configs/templates/simulation.ini`:

```ini
[routing]
# Routing algorithm: k_shortest_path, load_balanced, least_congested, 1plus1_protection
route_method = k_shortest_path
```

### Update CLI Parameters

Add to `fusion/cli/parameters/network.py`:

```python
ROUTE_METHOD_CHOICES = [
    "k_shortest_path",
    "load_balanced",
    "least_congested",  # NEW
    "1plus1_protection",
]
```

---

## Step 5: Document the Strategy

Add documentation for users.

### Update: `docs/routing_strategies.md`

```markdown
## Least Congested Path

**Name**: `least_congested`

**Description**: Prioritizes paths with lower current congestion to spread
load across the network.

**Algorithm**:
1. Compute k-shortest paths (like KSP)
2. For each path, calculate average link utilization
3. Sort paths by congestion (lowest first)
4. Return top k paths

**Use Cases**:
- Load balancing under high traffic
- Reducing fragmentation
- Improving acceptance rate at high Erlang

**Configuration**:
```ini
[routing]
route_method = least_congested
k_paths = 3
```

**Trade-offs**:
- May select longer paths to avoid congestion
- Slightly higher computational cost than KSP
- More dynamic - results change as network fills
```

---

## Integration Checklist

Use this checklist when adding a new routing strategy:

### Implementation

- [ ] Strategy class implements `RoutingStrategy` protocol
- [ ] `find_routes()` returns valid `RouteResult`
- [ ] Handles `forced_path` parameter correctly
- [ ] Handles "no path found" case (returns empty result)
- [ ] Modulations respect reach constraints
- [ ] Strategy name set in result metadata

### Registration

- [ ] Strategy added to `ROUTING_STRATEGIES` registry
- [ ] Strategy class exported from `__init__.py`
- [ ] `PipelineFactory.create_routing()` can instantiate it

### Testing

- [ ] Unit tests for happy path
- [ ] Test for no-path case
- [ ] Test for forced_path case
- [ ] Test invariants (array lengths match)
- [ ] Test modulation reach filtering
- [ ] Integration test with full simulation

### Configuration

- [ ] Config value added to route_method choices
- [ ] INI template updated
- [ ] CLI parameters updated

### Documentation

- [ ] Strategy documented in routing_strategies.md
- [ ] Algorithm description included
- [ ] Use cases and trade-offs documented
- [ ] Configuration example provided

---

## Advanced: Protection Strategy

For strategies that provide 1+1 protection, also populate backup paths:

```python
def find_routes(self, source, destination, bandwidth_gbps, network_state, forced_path=None):
    # Find edge-disjoint path pairs
    try:
        disjoint = list(nx.edge_disjoint_paths(
            network_state.topology, source, destination
        ))
    except nx.NetworkXNoPath:
        return RouteResult(paths=[], weights_km=[], modulations=[])

    if len(disjoint) < 2:
        return RouteResult(paths=[], weights_km=[], modulations=[])

    # Pair primary and backup paths
    paths = []
    backup_paths = []
    weights = []
    backup_weights = []
    modulations = []
    backup_modulations = []

    for i in range(0, len(disjoint) - 1, 2):
        primary = disjoint[i]
        backup = disjoint[i + 1]

        primary_mods = self._get_valid_modulations(...)
        backup_mods = self._get_valid_modulations(...)

        if any(primary_mods) and any(backup_mods):
            paths.append(primary)
            backup_paths.append(backup)
            # ... populate weights and modulations

    return RouteResult(
        paths=paths,
        weights_km=weights,
        modulations=modulations,
        backup_paths=backup_paths,
        backup_weights_km=backup_weights,
        backup_modulations=backup_modulations,
        strategy_name="my_protection_strategy",
    )
```

---

## Next Steps

- [Migrating Legacy Code to V4](./migrating_legacy_code_to_v4_domain_model.md) - Migration guide
- [Architecture: Routing Strategies](../architecture/routing_strategies.md) - Full specification
