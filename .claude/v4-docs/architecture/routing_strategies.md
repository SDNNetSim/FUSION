# Routing Strategies

This document defines the routing strategy pattern and `RouteResult` specification for V4.

## Overview

Routing in V4 follows the Strategy pattern, where different routing algorithms implement a common interface. This replaces the current approach of switching on `route_method` strings inside `sdn_controller.py`.

---

## RoutingStrategy Protocol

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

## RouteResult Specification

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class RouteResult:
    """Output of routing strategy - candidate paths with metadata."""

    # Primary paths (required)
    paths: list[list[str]]              # [[node1, node2, ...], ...]
    weights_km: list[float]             # Path lengths in km
    modulations: list[list[str | None]] # Valid modulations per path

    # Backup paths (for protection, optional)
    backup_paths: list[list[str]] | None = None
    backup_weights_km: list[float] | None = None
    backup_modulations: list[list[str | None]] | None = None

    # Metadata
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """True if no routes were found."""
        return len(self.paths) == 0

    @property
    def has_backup(self) -> bool:
        """True if backup paths are present."""
        return self.backup_paths is not None and len(self.backup_paths) > 0

    @property
    def num_paths(self) -> int:
        """Number of candidate paths."""
        return len(self.paths)

    def get_path(self, index: int) -> list[str]:
        """Get path at index."""
        return self.paths[index]

    def get_modulations(self, index: int) -> list[str | None]:
        """Get valid modulations for path at index."""
        return self.modulations[index]

    def get_weight(self, index: int) -> float:
        """Get weight (length) for path at index."""
        return self.weights_km[index]
```

### Invariants

1. `len(paths) == len(weights_km) == len(modulations)`
2. If `backup_paths` is not None: `len(backup_paths) == len(paths)`
3. Each path has at least 2 nodes (source and destination)
4. Each modulation list has at least one non-None entry (or path is invalid)

---

## Available Routing Strategies

### 1. K-Shortest Path (KSP)

**Name**: `k_shortest_path`

**Description**: Finds the k shortest paths using Yen's algorithm.

```python
class KShortestPathStrategy:
    """K-shortest path routing strategy."""

    def __init__(self, config: SimulationConfig):
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
        if forced_path:
            return self._single_path_result(forced_path, bandwidth_gbps, network_state)

        topology = network_state.topology
        paths = list(nx.shortest_simple_paths(
            topology, source, destination, weight="weight"
        ))[:self.k]

        weights = []
        modulations = []

        for path in paths:
            weight = self._compute_path_weight(path, topology)
            weights.append(weight)
            mods = self._get_valid_modulations(weight, bandwidth_gbps)
            modulations.append(mods)

        return RouteResult(
            paths=paths,
            weights_km=weights,
            modulations=modulations,
            strategy_name="k_shortest_path",
        )
```

### 2. Load-Balanced Routing

**Name**: `load_balanced`

**Description**: Selects paths based on current link utilization to spread load.

```python
class LoadBalancedStrategy:
    """Load-balanced routing considering link utilization."""

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        # Get k candidates
        candidates = self._get_candidates(source, destination, network_state)

        # Score by utilization (lower is better)
        scored = []
        for path in candidates:
            utilization = self._compute_path_utilization(path, network_state)
            scored.append((utilization, path))

        # Sort by utilization, return top k
        scored.sort(key=lambda x: x[0])
        paths = [p for _, p in scored[:self.k]]

        # ... compute weights and modulations
```

### 3. 1+1 Protection (Disjoint Paths)

**Name**: `1plus1_protection`

**Description**: Finds edge-disjoint primary and backup path pairs.

```python
class OnePlusOneProtectionStrategy:
    """1+1 protection with disjoint path pairs."""

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        topology = network_state.topology

        # Find edge-disjoint paths
        try:
            disjoint = list(nx.edge_disjoint_paths(topology, source, destination))
        except nx.NetworkXNoPath:
            return RouteResult(paths=[], weights_km=[], modulations=[])

        if len(disjoint) < 2:
            return RouteResult(paths=[], weights_km=[], modulations=[])

        # Pair primary + backup
        paths = []
        backup_paths = []
        weights = []
        backup_weights = []
        modulations = []
        backup_modulations = []

        for i in range(0, len(disjoint) - 1, 2):
            primary = disjoint[i]
            backup = disjoint[i + 1]

            primary_weight = self._compute_path_weight(primary, topology)
            backup_weight = self._compute_path_weight(backup, topology)

            primary_mods = self._get_valid_modulations(primary_weight, bandwidth_gbps)
            backup_mods = self._get_valid_modulations(backup_weight, bandwidth_gbps)

            if any(primary_mods) and any(backup_mods):
                paths.append(primary)
                backup_paths.append(backup)
                weights.append(primary_weight)
                backup_weights.append(backup_weight)
                modulations.append(primary_mods)
                backup_modulations.append(backup_mods)

        return RouteResult(
            paths=paths,
            weights_km=weights,
            modulations=modulations,
            backup_paths=backup_paths,
            backup_weights_km=backup_weights,
            backup_modulations=backup_modulations,
            strategy_name="1plus1_protection",
        )
```

### 4. Segment Routing (Future)

**Name**: `segment_routing`

**Description**: For multi-segment lightpaths with intermediate regeneration.

---

## Strategy Registry

Strategies are registered and instantiated via factory:

```python
ROUTING_STRATEGIES = {
    "k_shortest_path": KShortestPathStrategy,
    "load_balanced": LoadBalancedStrategy,
    "1plus1_protection": OnePlusOneProtectionStrategy,
    "segment_routing": SegmentRoutingStrategy,
}

def create_routing_strategy(config: SimulationConfig) -> RoutingStrategy:
    """Create routing strategy from config."""
    strategy_cls = ROUTING_STRATEGIES.get(config.route_method)
    if strategy_cls is None:
        raise ValueError(f"Unknown routing method: {config.route_method}")
    return strategy_cls(config)
```

---

## Modulation Selection

Each strategy must compute valid modulations for each path:

```python
def _get_valid_modulations(
    self,
    path_length_km: float,
    bandwidth_gbps: int,
) -> list[str | None]:
    """
    Determine valid modulation formats for path.

    Returns list of modulations (most efficient first), or [None] if none valid.
    """
    valid = []

    # Get candidate modulations for this bandwidth
    candidates = self.mod_per_bw.get(bandwidth_gbps, [])

    for mod in candidates:
        mod_info = self.mod_formats.get(mod, {})
        max_reach = mod_info.get("max_reach_km", float("inf"))

        if path_length_km <= max_reach:
            valid.append(mod)

    # Sort by spectral efficiency (slots needed, ascending)
    valid.sort(key=lambda m: self.mod_formats[m].get("slots_per_100g", 999))

    return valid if valid else [None]
```

---

## Integration with Orchestrator

The orchestrator uses routing strategies through the pipeline interface:

```python
class SDNOrchestrator:
    def __init__(self, config: SimulationConfig, pipelines: PipelineSet):
        self.routing = pipelines.routing  # RoutingStrategy

    def handle_arrival(self, request: Request, network_state: NetworkState) -> AllocationResult:
        # Get routes
        route_result = self.routing.find_routes(
            request.source,
            request.destination,
            request.bandwidth_gbps,
            network_state,
        )

        if route_result.is_empty:
            return AllocationResult(success=False, block_reason=BlockReason.NO_ROUTE)

        # Try each path
        for i in range(route_result.num_paths):
            path = route_result.get_path(i)
            mods = route_result.get_modulations(i)
            weight = route_result.get_weight(i)

            result = self._try_allocate_on_path(request, path, mods, weight, network_state)
            if result.success:
                return result

        return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)
```

---

## Legacy Adapter

During migration, wrap existing routing code:

```python
class RoutingAdapter(RoutingStrategy):
    """Adapter wrapping legacy Routing class."""

    def __init__(self, config: SimulationConfig):
        self._engine_props = config.to_engine_props()

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        # Create minimal props objects for legacy code
        from fusion.core.routing import Routing
        from fusion.core.properties import RoutingProps

        route_props = RoutingProps()

        class MinimalSDNProps:
            def __init__(self):
                self.topology = network_state.topology
                self.source = source
                self.destination = destination

        sdn_props = MinimalSDNProps()
        routing = Routing(self._engine_props, sdn_props, route_props)
        routing.get_route()

        return RouteResult(
            paths=route_props.paths_matrix,
            modulations=route_props.modulation_formats_matrix,
            weights_km=route_props.weights_list,
            strategy_name="legacy_adapter",
        )
```

---

## Testing Strategies

### Unit Tests

```python
def test_ksp_finds_k_paths():
    """KSP returns exactly k paths when available."""
    config = SimulationConfig(..., k_paths=3)
    strategy = KShortestPathStrategy(config)

    result = strategy.find_routes("A", "Z", 100, network_state)

    assert len(result.paths) <= 3
    assert len(result.paths) == len(result.weights_km)

def test_protection_returns_disjoint_pairs():
    """1+1 protection returns edge-disjoint primary and backup."""
    strategy = OnePlusOneProtectionStrategy(config)

    result = strategy.find_routes("A", "Z", 100, network_state)

    assert result.has_backup
    for i, (primary, backup) in enumerate(zip(result.paths, result.backup_paths)):
        primary_edges = set(zip(primary[:-1], primary[1:]))
        backup_edges = set(zip(backup[:-1], backup[1:]))
        assert primary_edges.isdisjoint(backup_edges)

def test_modulations_respect_reach():
    """Modulations are only valid within reach."""
    # Long path should exclude high-order modulations
    result = strategy.find_routes("A", "Z", 100, network_state)

    for i, mods in enumerate(result.modulations):
        weight = result.weights_km[i]
        for mod in mods:
            if mod is not None:
                max_reach = config.modulation_formats[mod]["max_reach_km"]
                assert weight <= max_reach
```

---

## Configuration Reference

| Config Key | Type | Description |
|------------|------|-------------|
| `route_method` | str | Strategy name |
| `k_paths` | int | Number of candidate paths |
| `modulation_formats` | dict | Modulation format specifications |
| `mod_per_bw` | dict | Bandwidth to modulation mapping |
