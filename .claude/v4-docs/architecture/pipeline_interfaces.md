# Pipeline Interfaces

## Overview

Pipelines are modular, composable components that handle specific aspects of request processing. Each pipeline:

- Defines a clear **protocol** (interface)
- Receives `NetworkState` by reference per call
- Returns a typed **result object**
- Does NOT store or cache network state

## Pipeline Types

| Pipeline | Responsibility | Input | Output |
|----------|---------------|-------|--------|
| `RoutingPipeline` | Find candidate paths | src, dst, bw, network_state | `RouteResult` |
| `SpectrumPipeline` | Assign spectrum slots | path, mods, bw, network_state | `SpectrumResult` |
| `GroomingPipeline` | Reuse existing lightpaths | request, network_state | `GroomingResult` |
| `SNRPipeline` | Validate signal quality | lightpath, network_state | `SNRResult` |
| `SlicingPipeline` | Split request into slices | request, path, network_state | `AllocationResult` |

## Protocol Definitions

### RoutingPipeline

```python
from typing import Protocol

class RoutingPipeline(Protocol):
    """Protocol for routing pipeline implementations."""

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        """Find candidate paths between endpoints.

        Args:
            source: Source node ID
            destination: Destination node ID
            bandwidth_gbps: Requested bandwidth
            network_state: Current network state (read-only)
            forced_path: If provided, only consider this path

        Returns:
            RouteResult with candidate paths and modulation options
        """
        ...
```

**Implementations:**
- `KSPRoutingPipeline`: K-shortest paths
- `ProtectedRoutingPipeline`: 1+1 disjoint paths
- `LoadBalancedRoutingPipeline`: Congestion-aware routing
- `RoutingAdapter`: Wraps legacy `Routing` class

### SpectrumPipeline

```python
class SpectrumPipeline(Protocol):
    """Protocol for spectrum assignment implementations."""

    def find_spectrum(
        self,
        path: list[str],
        modulations: list[str | None],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """Find available spectrum on path.

        Args:
            path: Node sequence
            modulations: Valid modulation formats (ordered by preference)
            bandwidth_gbps: Required bandwidth
            network_state: Current network state (read-only)

        Returns:
            SpectrumResult indicating availability and assignment
        """
        ...

    def find_protected_spectrum(
        self,
        primary_path: list[str],
        backup_path: list[str],
        modulations: list[str | None],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """Find common spectrum on both primary and backup paths.

        Returns spectrum available on BOTH paths for 1+1 protection.
        """
        ...
```

**Implementations:**
- `FirstFitSpectrumPipeline`: First available slot range
- `BestFitSpectrumPipeline`: Smallest sufficient slot range
- `SpectrumAdapter`: Wraps legacy spectrum assignment

### GroomingPipeline

```python
class GroomingPipeline(Protocol):
    """Protocol for grooming pipeline implementations."""

    def try_groom(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> GroomingResult:
        """Attempt to serve request using existing lightpaths.

        Args:
            request: Incoming request
            network_state: Current network state

        Returns:
            GroomingResult indicating grooming outcome
        """
        ...

    def rollback(
        self,
        request: Request,
        lightpath_ids: list[int],
        network_state: NetworkState,
    ) -> None:
        """Rollback grooming allocations on failure.

        Called when request cannot be fully served after partial grooming.
        """
        ...
```

**Implementations:**
- `StandardGroomingPipeline`: Greedy bandwidth allocation
- `GroomingAdapter`: Wraps legacy grooming logic

### SNRPipeline

```python
class SNRPipeline(Protocol):
    """Protocol for SNR validation implementations."""

    def validate(
        self,
        lightpath: Lightpath,
        network_state: NetworkState,
    ) -> SNRResult:
        """Validate SNR for a lightpath.

        Args:
            lightpath: Lightpath to validate
            network_state: Current network state (for neighbor interference)

        Returns:
            SNRResult with pass/fail and computed SNR value
        """
        ...

    def validate_protected(
        self,
        lightpath: Lightpath,
        network_state: NetworkState,
    ) -> SNRResult:
        """Validate SNR on both primary and backup paths."""
        ...
```

**Implementations:**
- `GSNRPipeline`: Gaussian SNR model
- `MultiBandSNRPipeline`: Multi-band GSNR
- `SNRAdapter`: Wraps legacy SNR calculation

### SlicingPipeline

```python
class SlicingPipeline(Protocol):
    """Protocol for request slicing implementations."""

    def try_slice(
        self,
        request: Request,
        path: list[str],
        modulations: list[str | None],
        bandwidth_gbps: int,
        network_state: NetworkState,
        spectrum_pipeline: SpectrumPipeline,
        snr_pipeline: SNRPipeline | None,
    ) -> AllocationResult:
        """Attempt to serve request via multiple smaller lightpaths.

        Args:
            request: Request to slice
            path: Path to use for all slices
            modulations: Valid modulations
            bandwidth_gbps: Total bandwidth to allocate
            network_state: Current network state
            spectrum_pipeline: For finding spectrum per slice
            snr_pipeline: For validating each slice (optional)

        Returns:
            AllocationResult indicating success/failure
        """
        ...
```

**Implementations:**
- `StandardSlicingPipeline`: Equal bandwidth slices
- `AdaptiveSlicingPipeline`: Variable slice sizes

## Result Objects

### RouteResult

```python
@dataclass(frozen=True)
class RouteResult:
    """Result from routing pipeline."""
    paths: list[list[str]]                    # [[node1, node2, ...], ...]
    weights_km: list[float]                   # Path lengths
    modulations: list[list[str | None]]       # Valid mods per path

    # For 1+1 protection
    backup_paths: list[list[str]] | None = None
    backup_weights_km: list[float] | None = None
    backup_modulations: list[list[str | None]] | None = None

    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.paths) == 0
```

### SpectrumResult

```python
@dataclass(frozen=True)
class SpectrumResult:
    """Result from spectrum pipeline."""
    is_free: bool
    start_slot: int = -1
    end_slot: int = -1
    core: int = -1
    band: str = ""
    modulation: str = ""
    slots_needed: int = 0

    # For protection
    backup_start_slot: int | None = None
    backup_end_slot: int | None = None
    backup_core: int | None = None
    backup_band: str | None = None
```

### GroomingResult

```python
@dataclass(frozen=True)
class GroomingResult:
    """Result from grooming pipeline."""
    fully_groomed: bool
    partially_groomed: bool
    bandwidth_groomed_gbps: int
    remaining_bandwidth_gbps: int
    lightpaths_used: list[int]
    forced_path: list[str] | None = None
```

### SNRResult

```python
@dataclass(frozen=True)
class SNRResult:
    """Result from SNR pipeline."""
    passed: bool
    snr_db: float
    required_snr_db: float
    xt_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

### AllocationResult

```python
@dataclass(frozen=True)
class AllocationResult:
    """Final result from orchestrator."""
    success: bool
    block_reason: BlockReason | None = None

    lightpaths_created: list[int] = field(default_factory=list)
    lightpaths_groomed: list[int] = field(default_factory=list)

    is_groomed: bool = False
    is_partially_groomed: bool = False
    is_sliced: bool = False
    is_protected: bool = False

    total_bandwidth_allocated_gbps: int = 0
```

## Pipeline Chaining

### Standard Flow

```
Request
    |
    v
GroomingPipeline.try_groom()
    |
    +-- fully_groomed=True --> Return success
    |
    +-- partially_groomed or no_groom
            |
            v
    RoutingPipeline.find_routes()
            |
            v
    For each path:
        |
        v
    SpectrumPipeline.find_spectrum()
        |
        +-- is_free=False --> Try next path
        |
        +-- is_free=True
                |
                v
        NetworkState.create_lightpath()
                |
                v
        SNRPipeline.validate() [if enabled]
            |
            +-- passed=False --> release_lightpath(), try next
            |
            +-- passed=True --> Return success
```

### Slicing Flow

When standard allocation fails but slicing is enabled:

```
SpectrumPipeline.find_spectrum() -> is_free=False
    |
    v
SlicingPipeline.try_slice()
    |
    +-- For num_slices in [2, 3, 4, ...]:
            |
            +-- For each slice:
                    |
                    v
                SpectrumPipeline.find_spectrum(slice_bw)
                    |
                    v
                NetworkState.create_lightpath()
                    |
                    v
                SNRPipeline.validate() [if enabled]
                    |
                    +-- fail --> rollback all slices, try next num_slices
                    |
                    +-- pass --> continue to next slice
            |
            +-- All slices pass --> Return success
    |
    v
All slice counts failed --> Return failure
```

### Protection Flow

```
ProtectedRoutingPipeline.find_routes()
    |
    v
RouteResult with paths AND backup_paths
    |
    v
SpectrumPipeline.find_protected_spectrum()
    |
    v
Returns spectrum available on BOTH paths
    |
    v
NetworkState.create_protected_lightpath()
    |
    v
Allocates same slots on primary AND backup
    |
    v
SNRPipeline.validate_protected()
    |
    v
Validates SNR on BOTH paths
```

## How Wrappers Add Functionality

### Wrapper Pattern

Wrappers implement the same protocol but add behavior:

```python
class LoggingRoutingPipeline:
    """Wrapper that adds logging to any routing pipeline."""

    def __init__(self, inner: RoutingPipeline):
        self._inner = inner

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        logger.debug(f"Finding routes: {source} -> {destination}, {bandwidth_gbps} Gbps")

        result = self._inner.find_routes(
            source, destination, bandwidth_gbps, network_state, forced_path
        )

        logger.debug(f"Found {len(result.paths)} paths")
        return result
```

### Metrics Wrapper

```python
class MetricsRoutingPipeline:
    """Wrapper that collects timing metrics."""

    def __init__(self, inner: RoutingPipeline, stats: StatsCollector):
        self._inner = inner
        self._stats = stats

    def find_routes(self, ...) -> RouteResult:
        start = time.perf_counter()
        result = self._inner.find_routes(...)
        elapsed = time.perf_counter() - start

        self._stats.record_routing_time(elapsed)
        self._stats.record_paths_found(len(result.paths))

        return result
```

### Caching Wrapper (Use Carefully)

```python
class CachedRoutingPipeline:
    """Wrapper that caches route computations.

    IMPORTANT: Only caches topology-dependent results.
    Does NOT cache anything from NetworkState (spectrum, lightpaths).
    """

    def __init__(self, inner: RoutingPipeline):
        self._inner = inner
        # Cache key: (source, dest) -> paths only (not spectrum-dependent)
        self._path_cache: dict[tuple[str, str], list[list[str]]] = {}

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        if forced_path:
            # Don't cache forced paths
            return self._inner.find_routes(...)

        cache_key = (source, destination)
        if cache_key in self._path_cache:
            # Reuse cached paths but recompute modulations
            cached_paths = self._path_cache[cache_key]
            return self._build_result_from_cached_paths(
                cached_paths, bandwidth_gbps, network_state
            )

        result = self._inner.find_routes(...)
        self._path_cache[cache_key] = result.paths
        return result
```

**Warning**: Never cache spectrum availability or lightpath state!

## Key Design Rules

### 1. Pipelines Are Stateless

```python
# GOOD: Stateless pipeline
class FirstFitSpectrumPipeline:
    def __init__(self, config: SimulationConfig):
        self.config = config  # Immutable config only

    def find_spectrum(self, ..., network_state: NetworkState) -> SpectrumResult:
        # Reads from network_state, no internal state
        ...

# BAD: Stateful pipeline
class BadPipeline:
    def __init__(self, network_state: NetworkState):
        self._state = network_state  # FORBIDDEN: storing state
        self._cache = {}              # FORBIDDEN: caching
```

### 2. Results Are Immutable

All result dataclasses use `frozen=True`:

```python
@dataclass(frozen=True)
class RouteResult:
    paths: list[list[str]]  # Immutable after creation
```

### 3. Single Responsibility

Each pipeline does ONE thing:
- `RoutingPipeline`: Find paths
- `SpectrumPipeline`: Find slots
- `SNRPipeline`: Validate quality

The `SDNOrchestrator` coordinates them, but never duplicates their logic.

### 4. No Side Effects on Read

Read methods (`find_routes`, `find_spectrum`, `validate`) must NOT modify `NetworkState`.

Only `SDNOrchestrator` calls write methods (`create_lightpath`, `release_lightpath`).

## File Layout

```
fusion/interfaces/
    __init__.py
    pipelines.py          # Protocol definitions

fusion/pipelines/
    __init__.py
    routing_pipeline.py   # KSPRoutingPipeline, ProtectedRoutingPipeline
    spectrum_pipeline.py  # FirstFitSpectrumPipeline
    grooming_pipeline.py  # StandardGroomingPipeline
    snr_pipeline.py       # GSNRPipeline
    slicing_pipeline.py   # StandardSlicingPipeline

fusion/core/adapters/
    __init__.py
    routing_adapter.py    # Wraps legacy Routing
    spectrum_adapter.py   # Wraps legacy spectrum assignment
    grooming_adapter.py   # Wraps legacy grooming
    snr_adapter.py        # Wraps legacy SNR calculation
```

## Integration with SDNOrchestrator

The orchestrator holds pipeline references but NOT `NetworkState`:

```python
class SDNOrchestrator:
    def __init__(self, config: SimulationConfig, pipelines: PipelineSet):
        self.config = config
        self.routing = pipelines.routing
        self.spectrum = pipelines.spectrum
        self.grooming = pipelines.grooming
        self.snr = pipelines.snr
        self.slicing = pipelines.slicing
        # NOTE: No self._network_state

    def handle_arrival(
        self,
        request: Request,
        network_state: NetworkState,  # Received per call
        forced_path: list[str] | None = None,
    ) -> AllocationResult:
        # Pass network_state to each pipeline
        route_result = self.routing.find_routes(
            request.source, request.destination,
            request.bandwidth_gbps, network_state, forced_path
        )
        # ...
```

See `architecture/orchestration.md` (Phase 3) for full orchestrator documentation.
