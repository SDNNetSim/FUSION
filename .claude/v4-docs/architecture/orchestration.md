# SDNOrchestrator Design

## Overview

The `SDNOrchestrator` is a **thin coordination layer** that routes requests through pipelines. It does not implement algorithms or contain feature-specific logic. Its sole responsibility is sequencing pipeline calls and combining their results.

## Core Responsibilities

| Responsibility | Description |
|---------------|-------------|
| **Stage Sequencing** | Deciding which pipeline to call next based on config |
| **Feature Checking** | `if self.grooming and self.config.grooming_enabled` |
| **Result Combination** | Merging groomed + newly allocated lightpaths into AllocationResult |
| **Rollback Coordination** | Calling `release_lightpath()` on failure |
| **Error Handling** | Catching exceptions, returning appropriate BlockReason |

## Non-Responsibilities

The orchestrator **MUST NOT**:

| Forbidden | Why |
|-----------|-----|
| Algorithm implementation | K-shortest-path, first-fit, SNR calculation belong in pipelines |
| Feature-specific logic | Grooming bandwidth calculations go in GroomingPipeline |
| Data structure access | No direct access to `cores_matrix` or numpy arrays |
| Value-based branching | No `if modulation == "QPSK"` type checks |
| Caching or state storage | Receives `NetworkState` per call, never stores it |

## Pipeline Delegation Pattern

```
SDNOrchestrator
    |
    +-- receives Request + NetworkState per call
    |
    +-- calls pipelines in sequence
    |
    +-- pipelines return result objects
    |
    +-- orchestrator combines results
    |
    +-- returns AllocationResult
```

The orchestrator holds **references to pipelines** but NOT to `NetworkState`:

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
        network_state: NetworkState,  # Received per call, never stored
        forced_path: list[str] | None = None,
    ) -> AllocationResult:
        ...
```

## Request Routing Through Pipelines

### Decision Points

The orchestrator routes requests through pipelines based on:

1. **Configuration flags**: `grooming_enabled`, `slicing_enabled`, `snr_enabled`
2. **Pipeline availability**: `if self.grooming is not None`
3. **Previous stage results**: `if groom_result.fully_groomed`

It does NOT make decisions based on:
- Algorithm-specific data (modulation formats, SNR values)
- Internal data structures (numpy arrays, spectrum matrices)
- Hardcoded thresholds or magic numbers

---

## Scenario Walkthroughs

### Scenario A: Plain KSP

**Config**: `grooming_enabled=False, slicing_enabled=False, snr_enabled=False`

**Flow**:
```
1. Request arrives
2. [Grooming: SKIPPED - not enabled]
3. RoutingPipeline.find_routes(src, dst, bw, network_state)
   -> RouteResult(paths=[[A,B,C], [A,D,C]], ...)
4. For each path:
   a. SpectrumPipeline.find_spectrum(path, mods, bw, network_state)
   b. If is_free: NetworkState.create_lightpath(...)
   c. Return AllocationResult(success=True)
5. If all paths fail: Return AllocationResult(success=False)
```

**Sequence Diagram**:
```
SimulationEngine         SDNOrchestrator         RoutingPipeline         SpectrumPipeline         NetworkState
       |                        |                       |                        |                      |
       |-- handle_arrival() --->|                       |                        |                      |
       |                        |-- find_routes() ----->|                        |                      |
       |                        |<-- RouteResult -------|                        |                      |
       |                        |                       |                        |                      |
       |                        |-- find_spectrum() ----|----------------------->|                      |
       |                        |<-- SpectrumResult ----|------------------------|                      |
       |                        |                       |                        |                      |
       |                        |-- create_lightpath() -|------------------------|--------------------->|
       |                        |<-- Lightpath ---------|------------------------|----------------------|
       |                        |                       |                        |                      |
       |<-- AllocationResult ---|                       |                        |                      |
```

### Scenario B: Grooming Only

**Config**: `grooming_enabled=True, slicing_enabled=False, snr_enabled=False`

**Flow (Fully Groomed)**:
```
1. Request arrives (100 Gbps)
2. GroomingPipeline.try_groom(request, network_state)
   -> GroomingResult(fully_groomed=True, lightpaths_used=[1])
3. Return AllocationResult(success=True, is_groomed=True)
```

**Flow (Partial Grooming)**:
```
1. Request arrives (100 Gbps)
2. GroomingPipeline.try_groom(request, network_state)
   -> GroomingResult(partially_groomed=True, bw_groomed=50, remaining=50, forced_path=[A,B,C])
3. RoutingPipeline.find_routes(src, dst, 50, network_state, forced_path=[A,B,C])
   -> RouteResult(paths=[[A,B,C]], ...)
4. SpectrumPipeline.find_spectrum([A,B,C], mods, 50, network_state)
   -> SpectrumResult(is_free=True, ...)
5. NetworkState.create_lightpath(...)
6. Return AllocationResult(success=True, is_groomed=True, is_partially_groomed=True)
```

**Sequence Diagram**:
```
SimulationEngine      SDNOrchestrator      GroomingPipeline      RoutingPipeline      SpectrumPipeline      NetworkState
       |                    |                     |                    |                     |                    |
       |-- handle_arrival-->|                     |                    |                     |                    |
       |                    |-- try_groom() ----->|                    |                     |                    |
       |                    |<-- GroomingResult --|                    |                     |                    |
       |                    |  (partial, bw=50)   |                    |                     |                    |
       |                    |                     |                    |                     |                    |
       |                    |-- find_routes() ----|------------------>|                     |                    |
       |                    |<-- RouteResult -----|------------------|                     |                    |
       |                    |                     |                    |                     |                    |
       |                    |-- find_spectrum() --|--------------------|------------------->|                    |
       |                    |<-- SpectrumResult --|--------------------|--------------------|                    |
       |                    |                     |                    |                     |                    |
       |                    |-- create_lightpath()|--------------------|--------------------|------------------>|
       |                    |<-- Lightpath -------|--------------------|--------------------|-------------------|
       |                    |                     |                    |                     |                    |
       |<-- AllocationResult|                     |                    |                     |                    |
```

### Scenario C: Slicing Only

**Config**: `grooming_enabled=False, slicing_enabled=True, max_slices=4, snr_enabled=False`

**Flow**:
```
1. Request arrives (400 Gbps)
2. [Grooming: SKIPPED]
3. RoutingPipeline.find_routes(src, dst, 400, network_state)
   -> RouteResult(paths=[[A,B,C]], modulations=[[None]])  # 400 Gbps too high
4. For path [A,B,C]:
   a. SpectrumPipeline.find_spectrum([A,B,C], [None], 400, network_state)
      -> SpectrumResult(is_free=False)  # No valid modulation
   b. SlicingPipeline.try_slice(request, [A,B,C], 400, network_state, ...)
      - Tries 4x100 Gbps slices
      - Each slice: find_spectrum -> create_lightpath
      -> AllocationResult(success=True, is_sliced=True, lightpaths=[1,2,3,4])
5. Return success
```

**Bullet-Step Flow**:
- Request: 400 Gbps, A -> C
- Routing finds path [A,B,C], no modulation supports 400 Gbps at this distance
- Standard spectrum fails (no valid modulation)
- Slicing takes over:
  - Split into 4 x 100 Gbps
  - Each 100 Gbps slice gets QPSK modulation
  - Allocate 4 separate lightpaths on same path
- Result: is_sliced=True, lightpaths_created=[1,2,3,4]

### Scenario D: Grooming + Slicing

**Config**: `grooming_enabled=True, slicing_enabled=True, max_slices=4, snr_enabled=False`

**Flow (Partial Groom + Slice)**:
```
1. Request arrives (500 Gbps)
2. GroomingPipeline.try_groom(request, network_state)
   -> GroomingResult(partially_groomed=True, bw_groomed=100, remaining=400, forced_path=[A,B,C])
3. RoutingPipeline.find_routes(src, dst, 400, network_state, forced_path=[A,B,C])
   -> RouteResult(paths=[[A,B,C]], modulations=[[None]])
4. For path [A,B,C]:
   a. SpectrumPipeline.find_spectrum([A,B,C], [None], 400, network_state)
      -> SpectrumResult(is_free=False)
   b. SlicingPipeline.try_slice(request, [A,B,C], 400, ...)
      -> AllocationResult(success=True, is_sliced=True, lightpaths=[2,3,4,5])
5. Combine: groomed=[1], created=[2,3,4,5]
6. Return AllocationResult(success=True, is_groomed=True, is_sliced=True)
```

**State Changes**:
- Existing lightpath 1: remaining_bandwidth reduced by 100 Gbps
- New lightpaths 2,3,4,5 created with 100 Gbps each
- Request linked to all 5 lightpaths

### Scenario E: Grooming + Slicing + SNR

**Config**: `grooming_enabled=True, slicing_enabled=True, max_slices=4, snr_enabled=True`

**Flow (With SNR Validation)**:
```
1. Request arrives (500 Gbps)
2. GroomingPipeline.try_groom(request, network_state)
   -> GroomingResult(partially_groomed=True, bw_groomed=100, remaining=400, forced_path=[A,B,C])
3. RoutingPipeline.find_routes(...)
   -> RouteResult(paths=[[A,B,C]], ...)
4. For path [A,B,C]:
   a. SpectrumPipeline.find_spectrum(...) -> is_free=False
   b. SlicingPipeline.try_slice(...)
      - For each slice:
        - find_spectrum -> SpectrumResult
        - create_lightpath -> Lightpath
        - SNRPipeline.validate(lightpath, network_state)
          - If fails: release_lightpath, try different slot
          - If passes: continue
      - All 4 slices pass SNR
      -> AllocationResult(success=True, is_sliced=True)
5. Return success with groomed + sliced
```

**SNR Failure Handling**:
```
SlicingPipeline (inside try_slice):
  For slice 1: create_lightpath -> SNR.validate -> PASS
  For slice 2: create_lightpath -> SNR.validate -> PASS
  For slice 3: create_lightpath -> SNR.validate -> FAIL
    -> release_lightpath(slice_3_id)
    -> Try next slot range
    -> create_lightpath -> SNR.validate -> PASS
  For slice 4: create_lightpath -> SNR.validate -> PASS
  -> Return success
```

---

## Orchestrator Implementation Pattern

```python
class SDNOrchestrator:
    def __init__(self, config: SimulationConfig, pipelines: PipelineSet):
        self.config = config
        self.routing = pipelines.routing
        self.spectrum = pipelines.spectrum
        self.grooming = pipelines.grooming
        self.snr = pipelines.snr
        self.slicing = pipelines.slicing
        # NOTE: No network_state storage

    def handle_arrival(
        self,
        request: Request,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> AllocationResult:
        groomed_lightpaths = []
        remaining_bw = request.bandwidth_gbps

        # STAGE 1: Try grooming
        if self.grooming and self.config.grooming_enabled:
            groom_result = self.grooming.try_groom(request, network_state)

            if groom_result.fully_groomed:
                return AllocationResult(success=True, is_groomed=True, ...)

            if groom_result.partially_groomed:
                groomed_lightpaths = groom_result.lightpaths_used
                remaining_bw = groom_result.remaining_bandwidth_gbps
                forced_path = groom_result.forced_path

        # STAGE 2: Routing
        route_result = self.routing.find_routes(
            request.source, request.destination,
            remaining_bw, network_state, forced_path
        )

        if route_result.is_empty:
            return self._handle_failure(request, groomed_lightpaths, BlockReason.NO_ROUTE, network_state)

        # STAGE 3: Try each path
        for path_idx, path in enumerate(route_result.paths):
            result = self._try_allocate_on_path(
                request, path, route_result.modulations[path_idx],
                route_result.weights_km[path_idx], remaining_bw, network_state
            )

            if result is not None:
                return self._combine_results(groomed_lightpaths, result)

        # STAGE 4: All paths failed
        return self._handle_failure(request, groomed_lightpaths, BlockReason.NO_SPECTRUM, network_state)

    def _try_allocate_on_path(
        self, request, path, mods, weight, bw, network_state
    ) -> AllocationResult | None:
        # Standard allocation attempt
        spectrum_result = self.spectrum.find_spectrum(path, mods, bw, network_state)

        if spectrum_result.is_free:
            alloc_result = self._allocate_and_validate(
                request, path, spectrum_result, weight, bw, network_state
            )
            if alloc_result is not None:
                return alloc_result

        # Fallback to slicing
        if self.slicing and self.config.slicing_enabled:
            return self.slicing.try_slice(
                request, path, mods, bw, network_state, self.spectrum, self.snr
            )

        return None

    def _handle_failure(
        self, request, groomed_lps, reason, network_state
    ) -> AllocationResult:
        if groomed_lps and self.config.can_partially_serve:
            # Accept partial grooming
            return AllocationResult(
                success=True,
                lightpaths_groomed=groomed_lps,
                is_partially_groomed=True,
            )

        if groomed_lps:
            # Rollback grooming
            self.grooming.rollback(request, groomed_lps, network_state)

        return AllocationResult(success=False, block_reason=reason)
```

---

## Adding New Features

When adding a new feature (e.g., QoS prioritization):

**CORRECT Approach**:
```python
# 1. Create QoSPipeline with its own logic
class QoSPipeline:
    def prioritize(self, route_result: RouteResult, request: Request) -> RouteResult:
        # All QoS logic here
        ...

# 2. Add to PipelineSet
@dataclass
class PipelineSet:
    routing: RoutingPipeline
    spectrum: SpectrumPipeline
    qos: QoSPipeline | None = None  # NEW

# 3. Add 3 lines to orchestrator
def handle_arrival(self, ...):
    route_result = self.routing.find_routes(...)

    if self.qos and self.config.qos_enabled:
        route_result = self.qos.prioritize(route_result, request)

    # ... rest unchanged ...
```

**INCORRECT Approach**:
```python
# BAD: QoS logic directly in orchestrator
def handle_arrival(self, ...):
    route_result = self.routing.find_routes(...)

    # BAD: Algorithm logic in orchestrator
    if self.config.qos_enabled:
        priority = request.qos_class
        if priority == "gold":
            route_result.paths = [route_result.paths[0]]
        elif priority == "silver":
            route_result.paths.sort(key=lambda p: self._congestion(p))
        # ... more algorithm logic ...
```

---

## File Location

```
fusion/core/orchestrator.py    # SDNOrchestrator class
fusion/core/pipeline_factory.py # PipelineFactory, PipelineSet
```

## Related Documentation

- `architecture/pipelines.md` - Pipeline implementations
- `architecture/pipeline_interfaces.md` - Protocol definitions
- `decisions/0007-orchestrator-design.md` - Design rationale
