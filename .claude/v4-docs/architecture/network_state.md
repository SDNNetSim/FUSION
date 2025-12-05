# NetworkState: Centralized State Authority

## Overview

`NetworkState` is the **single source of truth** for all mutable network state during simulation. It owns spectrum allocation, lightpath management, and topology access. No other component stores or caches network state.

## Design Principles

### Single Instance Policy

Exactly **one** `NetworkState` instance exists per simulation run:

| Component | Relationship |
|-----------|-------------|
| `SimulationEngine` | **Owns** the single instance |
| `SDNOrchestrator` | Receives by reference per call |
| `RoutingPipeline` | Receives by reference (reads only) |
| `SpectrumPipeline` | Receives by reference |
| `GroomingPipeline` | Receives by reference |
| `SNRPipeline` | Receives by reference |
| `SlicingPipeline` | Receives by reference |
| `StatsCollector` | Receives by reference for snapshots |
| `RLSimulationAdapter` | Receives by reference |

### Pass-by-Reference Model

```
SimulationEngine
    |
    v
NetworkState (single instance)
    |
    +-- passed to SDNOrchestrator.handle_arrival(request, network_state)
    |       |
    |       +-- passed to RoutingPipeline.find_routes(..., network_state)
    |       +-- passed to SpectrumPipeline.find_spectrum(..., network_state)
    |       +-- passed to GroomingPipeline.try_groom(..., network_state)
    |
    +-- passed to StatsCollector.record_result(..., network_state)
```

**Critical Rule**: Components receive `network_state` as a method parameter. They do NOT store it as an instance variable or cache any data from it.

## Class Structure

```python
class NetworkState:
    """Single source of truth for network state."""

    def __init__(self, topology: nx.Graph, config: SimulationConfig):
        self._topology = topology
        self._config = config
        self._spectrum: dict[tuple[str, str], LinkSpectrum] = {}
        self._lightpaths: dict[int, Lightpath] = {}
        self._next_lightpath_id = 1
```

### Owned Data

| Data | Type | Description |
|------|------|-------------|
| `_topology` | `nx.Graph` | Network topology (read-only after init) |
| `_config` | `SimulationConfig` | Simulation configuration |
| `_spectrum` | `dict[link, LinkSpectrum]` | Spectrum state per link |
| `_lightpaths` | `dict[int, Lightpath]` | Active lightpaths by ID |
| `_next_lightpath_id` | `int` | Counter for unique IDs |

## Object Relationships

### LinkSpectrum

Each link has a `LinkSpectrum` object managing its spectrum:

```python
@dataclass
class LinkSpectrum:
    """Spectrum state for a single link."""
    link: tuple[str, str]
    cores_matrix: dict[str, np.ndarray]  # band -> (cores, slots)
    usage_count: int = 0
    throughput: float = 0.0
    link_num: int = 0

    def is_range_free(self, band: str, core: int, start: int, end: int) -> bool:
        return np.all(self.cores_matrix[band][core][start:end] == 0)

    def allocate(self, band: str, core: int, start: int, end: int, lp_id: int) -> None:
        self.cores_matrix[band][core][start:end] = lp_id

    def release(self, lp_id: int) -> None:
        for band, matrix in self.cores_matrix.items():
            matrix[matrix == lp_id] = 0
            matrix[matrix == -lp_id] = 0
```

### Lightpath

Each active lightpath is tracked:

```python
@dataclass
class Lightpath:
    lightpath_id: int
    path: list[str]
    start_slot: int
    end_slot: int
    core: int
    band: str
    modulation: str
    total_bandwidth_gbps: int
    remaining_bandwidth_gbps: int
    path_weight_km: float
    request_allocations: dict[int, int]  # request_id -> bandwidth
    # Protection fields
    backup_path: list[str] | None = None
    is_protected: bool = False
```

### Relationship Diagram

```
NetworkState
    |
    +-- _topology: nx.Graph
    |       |
    |       +-- nodes: {node_id -> attributes}
    |       +-- edges: {(u,v) -> attributes}
    |
    +-- _spectrum: dict[link, LinkSpectrum]
    |       |
    |       +-- (A,B): LinkSpectrum
    |       |       +-- cores_matrix["c"]: ndarray (7, 320)
    |       |       +-- cores_matrix["l"]: ndarray (7, 320)
    |       |
    |       +-- (B,C): LinkSpectrum
    |               ...
    |
    +-- _lightpaths: dict[int, Lightpath]
            |
            +-- 1: Lightpath(path=[A,B,C], slots=10-18, ...)
            +-- 2: Lightpath(path=[A,D,C], slots=0-8, ...)
```

## Object Lifecycle

### Lightpath Lifecycle

```
                    create_lightpath()
    [Not Exists] -----------------------> [Active]
                                             |
                                             | release_lightpath()
                                             v
                                        [Released]
```

### State Transitions During Request Processing

```
1. Request arrives
2. Orchestrator receives NetworkState reference
3. RoutingPipeline reads topology (no state change)
4. SpectrumPipeline queries spectrum availability (no state change)
5. NetworkState.create_lightpath() called:
   - Lightpath added to _lightpaths
   - Spectrum allocated on all path links
6. Request mapped to lightpath
7. StatsCollector records result

On release:
1. NetworkState.release_lightpath() called:
   - Spectrum freed on all path links
   - Lightpath removed from _lightpaths
```

## Public API

### Read Methods (No State Change)

```python
@property
def topology(self) -> nx.Graph:
    """Get network topology (read-only reference)."""

def get_lightpath(self, lightpath_id: int) -> Lightpath | None:
    """Get lightpath by ID."""

def get_lightpaths_between(self, source: str, dest: str) -> list[Lightpath]:
    """Find all lightpaths between two endpoints."""

def get_lightpaths_with_capacity(self, source: str, dest: str, min_bw: int) -> list[Lightpath]:
    """Find lightpaths with available bandwidth for grooming."""

def is_spectrum_available(self, path: list[str], start: int, end: int, core: int, band: str) -> bool:
    """Check if spectrum range is free on all links of path."""

def get_link_spectrum(self, link: tuple[str, str]) -> LinkSpectrum:
    """Get spectrum state for a link."""

def find_first_fit(self, path: list[str], band: str, core: int, slots_needed: int) -> int | None:
    """Find first available slot range on path."""
```

### Write Methods (State Change)

```python
def create_lightpath(
    self,
    path: list[str],
    start_slot: int,
    end_slot: int,
    core: int,
    band: str,
    modulation: str,
    bandwidth_gbps: int,
    path_weight_km: float,
    backup_path: list[str] | None = None,
) -> Lightpath:
    """Create lightpath and allocate spectrum."""

def release_lightpath(self, lightpath_id: int) -> None:
    """Release lightpath and free spectrum."""

def create_protected_lightpath(
    self,
    primary_path: list[str],
    backup_path: list[str],
    spectrum_result: SpectrumResult,
    bandwidth_gbps: int,
    path_weight_km: float,
) -> Lightpath:
    """Create 1+1 protected lightpath (allocates on both paths)."""
```

## Preventing Mutability Issues

### Anti-Pattern: Local Caching (FORBIDDEN)

```python
# BAD: Caching internal state
class BadPipeline:
    def __init__(self, network_state: NetworkState):
        # WRONG: Storing reference
        self._network_state = network_state
        # VERY WRONG: Copying internal dict
        self._local_spectrum = network_state.network_spectrum_dict.copy()

    def find_spectrum(self, path, ...):
        # WRONG: Reading from stale cache
        for link in path:
            link_spectrum = self._local_spectrum[link]  # May be outdated!
```

**Why this is dangerous:**
1. **Stale reads**: Other pipelines may have allocated spectrum
2. **Lost writes**: Mutations to local copy invisible elsewhere
3. **Inconsistent state**: Partial state in cache vs NetworkState

### Correct Pattern: Pass-by-Reference

```python
# GOOD: No caching, receive per call
class GoodPipeline:
    def __init__(self, config: SimulationConfig):
        self.config = config
        # No network_state stored

    def find_spectrum(
        self,
        path: list[str],
        network_state: NetworkState,  # Fresh reference each call
    ) -> SpectrumResult:
        # CORRECT: Always query authoritative source
        for band in self.config.band_list:
            start = network_state.find_first_fit(path, band, 0, slots_needed)
            if start is not None:
                return SpectrumResult(is_free=True, ...)
```

### Enforcement Mechanism

```python
# In NetworkState.__init__
self._creation_id = id(self)

# In sensitive methods
def is_spectrum_available(self, ...) -> bool:
    assert id(self) == self._creation_id, "NetworkState appears to have been copied!"
    # ... implementation
```

## State Sharing Across Pipelines

### Sequence: Multiple Pipelines Reading Same State

```
Time    Pipeline            NetworkState Read       State Change
----    --------            ----------------        ------------
t1      Routing             topology.edges          None
t2      Routing             (completes)             None
t3      Spectrum            is_spectrum_available   None
t4      Spectrum            find_first_fit          None
t5      Orchestrator        create_lightpath        Lightpath added
t6      SNR                 get_lightpath           None (reads new LP)
t7      SNR                 (validates)             None
```

All pipelines see the same state because they share the same `NetworkState` reference.

### Sequence: Rollback on Failure

```
Time    Action                          State Change
----    ------                          ------------
t1      create_lightpath(path_A)        LP#1 added, spectrum allocated
t2      snr_pipeline.validate(LP#1)     None
t3      SNR fails threshold             None
t4      release_lightpath(1)            LP#1 removed, spectrum freed
t5      Try next path...                (state back to t0)
```

## Legacy Compatibility Properties

During migration, legacy properties provide backward compatibility:

```python
@property
def network_spectrum_dict(self) -> dict:
    """DEPRECATED: Legacy format for backward compatibility.
    Will be removed in Phase 5.
    """
    return {
        link: {
            'cores_matrix': ls.cores_matrix,
            'usage_count': ls.usage_count,
            'throughput': ls.throughput,
        }
        for link, ls in self._spectrum.items()
    }

@property
def lightpath_status_dict(self) -> dict:
    """DEPRECATED: Legacy format for backward compatibility.
    Will be removed in Phase 5.
    """
    result = {}
    for lp in self._lightpaths.values():
        key = lp.endpoint_key
        if key not in result:
            result[key] = {}
        result[key][lp.lightpath_id] = {
            'path': lp.path,
            'core': lp.core,
            # ... other fields
        }
    return result
```

**Important**: These properties:
- Return **copies**, not references
- Are O(n) per access
- Will be **removed** after migration completes
- Should trigger deprecation warnings in Phase 4

## Thread Safety Considerations

FUSION simulations are single-threaded, so `NetworkState` does not need locking. However, the design principles support future thread safety:

1. **Single owner**: Only `SimulationEngine` owns the instance
2. **Explicit mutations**: All changes go through defined methods
3. **No hidden state**: No caching means no cache invalidation issues

## Testing Considerations

Key invariants to test:

1. **Spectrum consistency**: After `create_lightpath`, spectrum is marked used; after `release_lightpath`, spectrum is free
2. **Lightpath tracking**: `get_lightpath` returns correct lightpath; `release_lightpath` removes it
3. **No orphaned spectrum**: If lightpath is released, all spectrum on all links is freed
4. **Legacy parity**: Legacy properties return equivalent data to old `sdn_props` format

See `phase_2_testing.md` for detailed test specifications.
