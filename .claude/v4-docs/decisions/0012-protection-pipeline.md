# ADR-0012: Protection Pipeline

## Status

Proposed

## Context

Network survivability is critical for production optical networks. The most common protection scheme is 1+1 dedicated path protection, where:

- Traffic is transmitted on both primary and backup paths simultaneously
- Backup path is link-disjoint (or node-disjoint) from primary
- Failure on primary causes immediate switchover to backup
- Both paths consume spectrum resources

FUSION has existing 1+1 protection logic in `fusion/modules/routing/one_plus_one_protection.py`, but it:

1. Is tightly coupled to the old `SDNController`
2. Doesn't integrate with the new pipeline architecture
3. Has no clear interface with `NetworkState`
4. Lacks configurable disjointness (link vs node)

We need to integrate protection into the V4 architecture while maintaining compatibility with existing protection tests.

## Decision

We will implement protection as a specialized `RoutingPipeline` that:

1. Finds disjoint path pairs using NetworkX algorithms
2. Returns `RouteResult` with both primary and backup paths
3. Integrates with spectrum pipeline for common slot allocation
4. Handles failure events through a dedicated method

### Protection Pipeline Design

```python
class ProtectionPipeline(RoutingPipeline):
    """
    Pipeline for 1+1 dedicated path protection.

    Finds disjoint primary/backup path pairs and coordinates
    spectrum allocation on both paths.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.disjointness = config.protection_disjointness  # "link" or "node"
        self.switchover_time_ms = config.protection_switchover_ms

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> RouteResult:
        """Find disjoint primary+backup path pairs."""
        # Uses NetworkX edge_disjoint_paths or node_disjoint_paths
        ...

    def handle_failure(
        self,
        failed_link: tuple[str, str],
        network_state: NetworkState,
    ) -> list[SwitchoverAction]:
        """Handle link failure by switching affected lightpaths."""
        ...
```

### Key Design Decisions

1. **Protection as Routing**: Protection is a specialized routing strategy, not a separate pipeline stage. This keeps the orchestrator simple.

2. **Dual allocation**: Both primary and backup paths consume spectrum. This is true 1+1 dedicated protection (not shared protection).

3. **Same spectrum**: Both paths use the same spectrum slots for simplicity. This is consistent with the existing implementation.

4. **Configurable disjointness**: Support both link-disjoint and node-disjoint modes via configuration.

5. **NetworkX integration**: Use proven NetworkX algorithms (`edge_disjoint_paths`, `node_disjoint_paths`) rather than custom implementations.

6. **Lightpath model extension**: Add protection fields to `Lightpath` dataclass (backup_path, is_protected, active_path).

## Alternatives Considered

### Alternative 1: Separate Protection Stage

Add protection as a distinct pipeline stage after routing:

```
Routing -> Protection -> Spectrum -> SNR
```

**Rejected because**:
- Protection is fundamentally a routing decision
- Would complicate orchestrator logic
- Unclear when protection stage runs vs routing

### Alternative 2: Shared Protection (1:1)

Implement 1:1 protection where backup capacity is shared:

```
Working Path: Dedicated spectrum
Backup Path: Shared spectrum pool
```

**Rejected because**:
- More complex resource management
- Existing codebase uses 1+1 dedicated
- Can be added as future enhancement

### Alternative 3: Different Spectrum on Paths

Allow primary and backup to use different spectrum slots:

```
Primary: Slots 0-10 on Path A
Backup: Slots 50-60 on Path B
```

**Rejected because**:
- Complicates spectrum search
- Existing implementation uses same spectrum
- Marginal benefit in elastic networks

### Alternative 4: Inline in SDNOrchestrator

Handle protection logic directly in orchestrator:

```python
# In SDNOrchestrator
if self.config.protection_enabled:
    paths = find_disjoint_paths(...)
    # Protection logic inline
```

**Rejected because**:
- Violates orchestrator design rules
- Would grow orchestrator size significantly
- Not testable in isolation

## Consequences

### Positive

1. **Clean integration**: Protection fits naturally as routing strategy
2. **Testable**: Pipeline can be tested independently
3. **Configurable**: Link vs node disjointness via config
4. **Compatible**: Uses same RouteResult as other routing strategies
5. **NetworkX leverage**: Reliable disjoint path algorithms

### Negative

1. **Double spectrum usage**: 1+1 consumes 2x resources
2. **Model extension**: Lightpath class grows
3. **Blocking increase**: Harder to find common spectrum on two paths

### Neutral

1. **Existing test adaptation**: Some tests may need updates
2. **Configuration growth**: New protection section in config

## Implementation Notes

### RouteResult Extension

```python
@dataclass(frozen=True)
class RouteResult:
    paths: list[list[str]]
    weights_km: list[float]
    modulations: list[list[str | None]]

    # Protection fields
    backup_paths: Optional[list[list[str]]] = None
    backup_weights_km: Optional[list[float]] = None
    backup_modulations: Optional[list[list[str | None]]] = None
    strategy_name: str = ""
```

### Lightpath Extension

```python
@dataclass
class Lightpath:
    # Existing fields...

    # Protection fields
    backup_path: Optional[list[str]] = None
    is_protected: bool = False
    active_path: str = "primary"  # "primary" or "backup"

    @property
    def current_path(self) -> list[str]:
        if self.is_protected and self.active_path == "backup":
            return self.backup_path
        return self.path
```

### NetworkState Extension

```python
class NetworkState:
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
            # ... standard fields
            backup_path=backup_path,
            is_protected=True,
        )

        # Allocate on BOTH paths
        self._allocate_spectrum_on_path(primary_path, ...)
        self._allocate_spectrum_on_path(backup_path, ...)

        return lp

    def get_lightpaths_on_link(
        self,
        link: tuple[str, str],
    ) -> list[Lightpath]:
        """Find all lightpaths traversing the given link."""
        # Used for failure handling
        ...
```

### Common Spectrum Finding

```python
class ProtectedSpectrumPipeline(SpectrumPipeline):
    def find_protected_spectrum(
        self,
        primary_path: list[str],
        backup_path: list[str],
        modulations: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """Find spectrum available on BOTH paths."""
        for mod in modulations:
            slots_needed = self._calculate_slots(bandwidth_gbps, mod)

            for band in self.config.band_list:
                for core in range(self.config.cores_per_link):
                    # Find intersection of available slots
                    common = self._find_common_slots(
                        [primary_path, backup_path],
                        band, core, slots_needed,
                        network_state,
                    )
                    if common:
                        return SpectrumResult(is_free=True, start_slot=common[0], ...)

        return SpectrumResult(is_free=False)
```

### Failure Handling

```python
def handle_failure(
    self,
    failed_link: tuple[str, str],
    network_state: NetworkState,
) -> list[SwitchoverAction]:
    """Handle link failure."""
    affected = network_state.get_lightpaths_on_link(failed_link)
    actions = []

    for lp in affected:
        if lp.is_protected and lp.active_path == "primary":
            if self._path_contains_link(lp.path, failed_link):
                lp.active_path = "backup"
                actions.append(SwitchoverAction(
                    lightpath_id=lp.lightpath_id,
                    action="switchover",
                    latency_ms=self.switchover_time_ms,
                ))
        elif not lp.is_protected:
            # Unprotected lightpath affected - needs restoration
            actions.append(SwitchoverAction(
                lightpath_id=lp.lightpath_id,
                action="affected_unprotected",
                latency_ms=0,
            ))

    return actions
```

## Configuration

```ini
[protection]
enabled = true
disjointness = link           ; link or node
switchover_time_ms = 50       ; Switchover latency

[route]
method = 1plus1_protection    ; Triggers protection pipeline
```

## Related Decisions

- [ADR-0007: Orchestrator Design](./0007-orchestrator-design.md)
- [ADR-0008: Routing Strategy Pattern](./0008-routing-strategy-pattern.md)

## References

- [NetworkX Disjoint Paths](https://networkx.org/documentation/stable/reference/algorithms/connectivity.html)
- [1+1 Protection in Optical Networks](https://en.wikipedia.org/wiki/1%2B1_protection)
- Existing FUSION: `fusion/modules/routing/one_plus_one_protection.py`
