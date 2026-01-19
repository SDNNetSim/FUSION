# Protection Pipeline Architecture

## Overview

The Protection Pipeline provides 1+1 dedicated path protection for survivability in FUSION. It integrates with the routing and spectrum pipelines to find disjoint path pairs and allocate common spectrum on both primary and backup paths.

## 1+1 Protection Concept

In 1+1 protection, traffic is simultaneously transmitted on two link-disjoint paths:

```
                    Primary Path
         +---------[B]----------+
         |                      |
    [A]--+                      +--[D]
         |                      |
         +---------[C]----------+
                    Backup Path
```

- **Primary path**: Active path carrying traffic
- **Backup path**: Standby path with pre-allocated spectrum
- **Disjoint requirement**: No shared links or nodes (configurable)
- **Switchover**: Instantaneous failover when primary fails

## Architecture

```
                     ProtectionPipeline
                            |
        +-------------------+-------------------+
        |                                       |
   find_protected_routes()              allocate_protected()
        |                                       |
        v                                       v
  +-------------+                      +------------------+
  | NetworkX    |                      | NetworkState     |
  | disjoint    |                      | create_lightpath |
  | path finder |                      | (both paths)     |
  +-------------+                      +------------------+
```

## Key Components

### 1. ProtectionPipeline Class

```python
class ProtectionPipeline:
    """
    Pipeline for 1+1 dedicated path protection.

    Responsibilities:
    - Find disjoint primary/backup path pairs
    - Coordinate spectrum allocation on both paths
    - Handle failure events and switchover
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.disjointness = config.protection_disjointness  # "link" or "node"
        self.switchover_time_ms = config.protection_switchover_ms

    def find_protected_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> RouteResult:
        """Find disjoint primary+backup path pairs."""
        ...

    def allocate_protected(
        self,
        request: Request,
        primary_path: list[str],
        backup_path: list[str],
        spectrum_result: SpectrumResult,
        path_weight_km: float,
        network_state: NetworkState,
        snr_pipeline: Optional[SNRPipeline] = None,
    ) -> AllocationResult:
        """Allocate spectrum on both primary and backup paths."""
        ...

    def handle_failure(
        self,
        failed_link: tuple[str, str],
        network_state: NetworkState,
    ) -> list[SwitchoverAction]:
        """Handle link failure - switch affected lightpaths to backup."""
        ...
```

### 2. Disjoint Path Finding

```python
def find_protected_routes(
    self,
    source: str,
    destination: str,
    bandwidth_gbps: int,
    network_state: NetworkState,
) -> RouteResult:
    """Find disjoint primary+backup path pairs."""
    import networkx as nx

    topology = network_state.topology

    # Find disjoint paths based on configuration
    try:
        if self.disjointness == "link":
            disjoint_paths = list(nx.edge_disjoint_paths(
                topology, source, destination
            ))
        else:  # node disjoint
            disjoint_paths = list(nx.node_disjoint_paths(
                topology, source, destination
            ))
    except nx.NetworkXNoPath:
        return RouteResult(paths=[], modulations=[], weights_km=[])

    if len(disjoint_paths) < 2:
        # Cannot provide protection without 2 disjoint paths
        return RouteResult(paths=[], modulations=[], weights_km=[])

    # Pair up paths: (primary, backup)
    paths = []
    backup_paths = []
    modulations = []
    backup_modulations = []
    weights = []
    backup_weights = []

    for i in range(0, len(disjoint_paths) - 1, 2):
        primary = list(disjoint_paths[i])
        backup = list(disjoint_paths[i + 1])

        # Get valid modulations for each path
        primary_mods = self._get_valid_modulations(primary, bandwidth_gbps, network_state)
        backup_mods = self._get_valid_modulations(backup, bandwidth_gbps, network_state)

        # Both paths must have valid modulation formats
        if not any(primary_mods) or not any(backup_mods):
            continue

        primary_weight = self._compute_path_weight(primary, network_state)
        backup_weight = self._compute_path_weight(backup, network_state)

        paths.append(primary)
        backup_paths.append(backup)
        modulations.append(primary_mods)
        backup_modulations.append(backup_mods)
        weights.append(primary_weight)
        backup_weights.append(backup_weight)

    return RouteResult(
        paths=paths,
        modulations=modulations,
        weights_km=weights,
        backup_paths=backup_paths,
        backup_modulations=backup_modulations,
        backup_weights_km=backup_weights,
        strategy_name="1plus1_protection",
    )
```

### 3. Protected Spectrum Allocation

Finding spectrum available on BOTH paths:

```python
class ProtectedSpectrumPipeline(SpectrumPipeline):
    """Spectrum allocation for protection - finds common slots."""

    def find_protected_spectrum(
        self,
        primary_path: list[str],
        backup_path: list[str],
        modulations: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """Find spectrum available on BOTH paths."""

        for modulation in modulations:
            if modulation is None:
                continue

            slots_needed = self._calculate_slots(bandwidth_gbps, modulation)

            for band in self.config.band_list:
                for core in range(self.config.cores_per_link):
                    # Find slots free on BOTH paths
                    common_starts = self._find_common_slots(
                        [primary_path, backup_path],
                        band,
                        core,
                        slots_needed,
                        network_state,
                    )

                    if common_starts:
                        start = common_starts[0]
                        return SpectrumResult(
                            is_free=True,
                            start_slot=start,
                            end_slot=start + slots_needed,
                            core=core,
                            band=band,
                            modulation=modulation,
                            slots_needed=slots_needed,
                            is_protected=True,
                        )

        return SpectrumResult(is_free=False)

    def _find_common_slots(
        self,
        paths: list[list[str]],
        band: str,
        core: int,
        slots_needed: int,
        network_state: NetworkState,
    ) -> list[int]:
        """Find slot indices available on ALL paths."""
        common_starts: Optional[set[int]] = None

        for path in paths:
            path_starts: set[int] = set()
            max_start = self.config.band_slots[band] - slots_needed

            for start in range(max_start + 1):
                if network_state.is_spectrum_available(
                    path, start, start + slots_needed, core, band
                ):
                    path_starts.add(start)

            if common_starts is None:
                common_starts = path_starts
            else:
                common_starts = common_starts.intersection(path_starts)

            if not common_starts:
                return []

        return sorted(common_starts) if common_starts else []
```

### 4. NetworkState Protected Lightpath

```python
# In NetworkState

def create_protected_lightpath(
    self,
    primary_path: list[str],
    backup_path: list[str],
    spectrum_result: SpectrumResult,
    bandwidth_gbps: int,
    path_weight_km: float,
) -> Lightpath:
    """Create lightpath with protection - allocates on both paths."""

    lp = Lightpath(
        lightpath_id=self._next_lightpath_id,
        path=primary_path,
        start_slot=spectrum_result.start_slot,
        end_slot=spectrum_result.end_slot,
        core=spectrum_result.core,
        band=spectrum_result.band,
        modulation=spectrum_result.modulation,
        total_bandwidth_gbps=bandwidth_gbps,
        remaining_bandwidth_gbps=bandwidth_gbps,
        path_weight_km=path_weight_km,
        # Protection-specific fields
        backup_path=backup_path,
        is_protected=True,
        active_path="primary",
    )
    self._next_lightpath_id += 1
    self._lightpaths[lp.lightpath_id] = lp

    # Allocate SAME spectrum on BOTH paths
    self._allocate_spectrum_on_path(
        primary_path,
        spectrum_result.start_slot,
        spectrum_result.end_slot,
        spectrum_result.core,
        spectrum_result.band,
        lp.lightpath_id,
    )
    self._allocate_spectrum_on_path(
        backup_path,
        spectrum_result.start_slot,
        spectrum_result.end_slot,
        spectrum_result.core,
        spectrum_result.band,
        lp.lightpath_id,
    )

    return lp
```

### 5. Failure Handling

```python
@dataclass
class SwitchoverAction:
    """Records a protection switchover event."""
    lightpath_id: int
    action: str  # "switchover" or "affected_unprotected"
    from_path: str  # "primary" or "backup"
    to_path: str
    latency_ms: float
    failed_link: tuple[str, str]


def handle_failure(
    self,
    failed_link: tuple[str, str],
    network_state: NetworkState,
) -> list[SwitchoverAction]:
    """Handle link failure - switch affected lightpaths to backup."""

    affected = network_state.get_lightpaths_on_link(failed_link)
    actions: list[SwitchoverAction] = []

    for lp in affected:
        if lp.is_protected:
            if lp.active_path == "primary" and self._path_contains_link(lp.path, failed_link):
                # Primary path affected - switch to backup
                lp.active_path = "backup"
                actions.append(SwitchoverAction(
                    lightpath_id=lp.lightpath_id,
                    action="switchover",
                    from_path="primary",
                    to_path="backup",
                    latency_ms=self.switchover_time_ms,
                    failed_link=failed_link,
                ))
            elif lp.active_path == "backup" and self._path_contains_link(lp.backup_path, failed_link):
                # Backup affected while using backup - dual failure
                actions.append(SwitchoverAction(
                    lightpath_id=lp.lightpath_id,
                    action="dual_failure",
                    from_path="backup",
                    to_path="none",
                    latency_ms=0,
                    failed_link=failed_link,
                ))
        else:
            # Unprotected lightpath affected
            actions.append(SwitchoverAction(
                lightpath_id=lp.lightpath_id,
                action="affected_unprotected",
                from_path="primary",
                to_path="none",
                latency_ms=0,
                failed_link=failed_link,
            ))

    return actions

def _path_contains_link(self, path: list[str], link: tuple[str, str]) -> bool:
    """Check if path traverses the given link."""
    for i in range(len(path) - 1):
        path_link = (path[i], path[i + 1])
        # Check both directions
        if path_link == link or path_link == (link[1], link[0]):
            return True
    return False
```

## Integration with SDNOrchestrator

```python
# In PipelineFactory

@staticmethod
def create_routing(config: SimulationConfig) -> RoutingPipeline:
    if config.route_method == "1plus1_protection":
        return ProtectionPipeline(config)
    elif config.route_method == "k_shortest_path":
        return KSPRoutingPipeline(config)
    # ... other routing strategies

@staticmethod
def create_spectrum(config: SimulationConfig) -> SpectrumPipeline:
    if config.route_method == "1plus1_protection":
        return ProtectedSpectrumPipeline(config)
    elif config.allocation_method == "first_fit":
        return FirstFitSpectrumPipeline(config)
    # ... other allocation strategies
```

## Lightpath Data Model Extension

```python
@dataclass
class Lightpath:
    # ... existing fields ...

    # Protection fields
    backup_path: Optional[list[str]] = None
    is_protected: bool = False
    active_path: str = "primary"  # "primary" or "backup"

    @property
    def current_path(self) -> list[str]:
        """Return currently active path."""
        if self.is_protected and self.active_path == "backup":
            return self.backup_path
        return self.path
```

## Configuration

```ini
[protection]
enabled = true
disjointness = link           ; Options: link, node
switchover_time_ms = 50       ; Protection switchover latency

[route]
method = 1plus1_protection    ; Enables protection pipeline
```

## Metrics and Statistics

```python
@dataclass
class ProtectionStats:
    """Statistics specific to protection."""
    total_protected_lightpaths: int = 0
    switchover_events: int = 0
    dual_failures: int = 0
    unprotected_affected: int = 0
    total_switchover_latency_ms: float = 0.0

    @property
    def avg_switchover_latency_ms(self) -> float:
        if self.switchover_events == 0:
            return 0.0
        return self.total_switchover_latency_ms / self.switchover_events
```

## Blocking Analysis

Protected requests may be blocked due to:

| Block Reason | Description |
|--------------|-------------|
| `NO_DISJOINT_PATHS` | Fewer than 2 disjoint paths exist |
| `NO_COMMON_SPECTRUM` | No spectrum available on both paths |
| `SNR_FAILURE_PRIMARY` | SNR validation failed on primary |
| `SNR_FAILURE_BACKUP` | SNR validation failed on backup |

## See Also

- [Routing Strategies](./routing_strategies.md)
- [Pipeline Walkthroughs - Scenario E](./pipeline_walkthroughs.md)
- [ADR-0012: Protection Pipeline](../decisions/0012-protection-pipeline.md)
