# FUSION Architecture Migration Plan - V3

## Document Purpose

This document builds on V2 to provide:
1. A phase-by-phase migration plan with concrete before/after code
2. Clarified stories for legacy compatibility, state consistency, RL/SB3, ML control, and 1+1 protection
3. Detailed pipeline walkthroughs for all scenarios
4. SDNOrchestrator design rules to prevent future monolith growth

---

# Section 1: Legacy Compatibility - Transitional or Permanent?

## 1.1 Recommendation: Transitional Only

**Recommendation**: Legacy compatibility views (`network_spectrum_dict`, `lightpath_status_dict` properties on `NetworkState`) should be **temporary migration helpers** that we delete after full migration.

**Reasoning**:

1. **Dual maintenance burden**: Keeping two APIs (new methods + legacy properties) doubles the surface area for bugs. Every change to `NetworkState` internals requires updating both.

2. **Performance overhead**: The legacy properties construct new dicts on every access. For example, `lightpath_status_dict` iterates all lightpaths and builds nested dicts - this is O(n) per access vs O(1) for direct `Lightpath` lookup.

3. **Inconsistency risk**: If someone reads the legacy property, mutates it, and expects changes to persist, they'll get silent bugs. Properties return copies, not references.

4. **Clear migration signal**: Having a deadline for removal forces all code paths to migrate, preventing "legacy forever" drift.

---

## 1.2 Policy and Timeline for Legacy Removal

### Phase-by-Phase Legacy Support

| Phase | Legacy Properties | Status |
|-------|------------------|--------|
| Phase 1 (Domain Model) | Not yet created | N/A |
| Phase 2 (NetworkState) | Created with full compatibility | Both old and new APIs work |
| Phase 3 (Orchestrator) | Maintained | New code uses new APIs; old code still works |
| Phase 4 (RL Integration) | Maintained | RL migrates to new APIs |
| Phase 5 (Legacy Removal) | **Deprecated then deleted** | Only new APIs remain |

### Removal Criteria Checklist

Before removing `network_spectrum_dict` property:
- [ ] All spectrum reads go through `NetworkState.is_spectrum_available()` or `NetworkState.get_link_spectrum()`
- [ ] All spectrum writes go through `NetworkState.create_lightpath()` / `release_lightpath()`
- [ ] No direct numpy array access outside `NetworkState`
- [ ] `run_comparison.py` passes without legacy property access
- [ ] Grep for `network_spectrum_dict` returns only the property definition itself

Before removing `lightpath_status_dict` property:
- [ ] Grooming uses `NetworkState.get_lightpaths_with_capacity()`
- [ ] Statistics use `NetworkState.get_lightpath()` or iteration
- [ ] No code constructs the `(src, dst)` sorted tuple key manually
- [ ] `run_comparison.py` passes without legacy property access

### Migration Tracking

Add deprecation warnings in Phase 4:

```python
@property
def network_spectrum_dict(self) -> dict:
    """DEPRECATED: Use get_link_spectrum() or is_spectrum_available() instead."""
    warnings.warn(
        "network_spectrum_dict is deprecated and will be removed in Phase 5. "
        "Use NetworkState methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self._build_legacy_spectrum_dict()
```

---

## 1.3 Migration Example: Grooming Lookup

### BEFORE: Reading lightpath_status_dict

```python
# In fusion/core/grooming.py, _find_path_max_bw() around line 180

def _find_path_max_bw(self, light_id: tuple[str, str]) -> dict | None:
    """Find lightpath group with maximum remaining bandwidth."""

    # OLD: Direct dict access with sorted tuple key
    if light_id not in self.sdn_props.lightpath_status_dict:
        return None

    path_group = self.sdn_props.lightpath_status_dict[light_id]

    max_bw = 0
    best_lp_id = None
    for lp_id, lp_info in path_group.items():
        if lp_info["remaining_bandwidth"] > max_bw:
            max_bw = lp_info["remaining_bandwidth"]
            best_lp_id = lp_id

    if best_lp_id is None:
        return None

    return path_group[best_lp_id]
```

### AFTER: Using NetworkState API

```python
# In fusion/core/grooming.py, using new NetworkState methods

def _find_path_max_bw(self, source: str, destination: str, network_state: NetworkState) -> Lightpath | None:
    """Find lightpath with maximum remaining bandwidth between endpoints."""

    # NEW: Use NetworkState method - no manual tuple construction
    lightpaths = network_state.get_lightpaths_between(source, destination)

    if not lightpaths:
        return None

    # Find max remaining bandwidth
    best_lp = max(lightpaths, key=lambda lp: lp.remaining_bandwidth_gbps, default=None)

    if best_lp is None or best_lp.remaining_bandwidth_gbps <= 0:
        return None

    return best_lp
```

### Transition Period: Supporting Both

During Phase 3-4, we can support both by having the new API delegate to the old implementation:

```python
# In NetworkState during transition

def get_lightpaths_between(self, source: str, destination: str) -> list[Lightpath]:
    """Find all lightpaths between two endpoints."""

    # During transition: build from legacy dict if _lightpaths not yet populated
    if not self._lightpaths and self._legacy_lightpath_status_dict:
        return self._build_lightpaths_from_legacy(source, destination)

    # After transition: direct lookup
    key = tuple(sorted([source, destination]))
    return [lp for lp in self._lightpaths.values() if lp.endpoint_key == key]

def _build_lightpaths_from_legacy(self, source: str, destination: str) -> list[Lightpath]:
    """Temporary: Convert legacy dict to Lightpath objects."""
    key = tuple(sorted([source, destination]))
    if key not in self._legacy_lightpath_status_dict:
        return []

    result = []
    for lp_id, lp_info in self._legacy_lightpath_status_dict[key].items():
        result.append(Lightpath(
            lightpath_id=lp_id,
            path=lp_info["path"],
            start_slot=lp_info["start_slot"],
            end_slot=lp_info["end_slot"],
            core=lp_info["core"],
            band=lp_info["band"],
            modulation=lp_info["mod_format"],
            total_bandwidth_gbps=lp_info["lightpath_bandwidth"],
            remaining_bandwidth_gbps=lp_info["remaining_bandwidth"],
            path_weight_km=lp_info["path_weight"],
            snr_db=lp_info.get("snr_cost"),
            xt_cost=lp_info.get("xt_cost"),
            is_degraded=lp_info.get("is_degraded", False),
            request_allocations=lp_info.get("requests_dict", {}).copy(),
        ))
    return result
```

---

# Section 2: NetworkState Sharing and State Consistency

## 2.1 Single Instance Policy

**Answer**: Exactly **one** `NetworkState` instance exists per simulation run.

| Component | Relationship to NetworkState |
|-----------|------------------------------|
| `SimulationEngine` | **Owns** the single instance |
| `SDNOrchestrator` | Receives by reference per call |
| `RoutingPipeline` | Receives by reference per call (reads only) |
| `SpectrumPipeline` | Receives by reference per call |
| `GroomingPipeline` | Receives by reference per call |
| `SNRPipeline` | Receives by reference per call |
| `SlicingPipeline` | Receives by reference per call |
| `StatsCollector` | Receives by reference for snapshots |
| `RLSimulationAdapter` | Receives by reference |

---

## 2.2 Ownership Diagram and Code

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SimulationEngine                                │
│  ┌─────────────────┐                                                   │
│  │  NetworkState   │ ◄── SINGLE INSTANCE, owned here                   │
│  │  (self._state)  │                                                   │
│  └────────┬────────┘                                                   │
│           │                                                            │
│           │ passed by reference (never copied)                         │
│           ▼                                                            │
│  ┌─────────────────┐      ┌─────────────────┐                         │
│  │ SDNOrchestrator │      │  StatsCollector │                         │
│  │ (self._orch)    │      │  (self._stats)  │                         │
│  └────────┬────────┘      └─────────────────┘                         │
│           │                                                            │
│           │ passes reference to pipelines                              │
│           ▼                                                            │
│  ┌────────┴────────┬────────────┬────────────┬────────────┐           │
│  │ RoutingPipeline │ SpectrumP. │ GroomingP. │   SNRP.    │           │
│  │ (reads only)    │ (reads)    │ (mutates)  │ (reads)    │           │
│  └─────────────────┴────────────┴────────────┴────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Code: SimulationEngine owns NetworkState

```python
# fusion/core/simulation.py

class SimulationEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config

        # Create topology (unchanged from current)
        self._topology = self._create_topology()

        # SINGLE NetworkState instance - owned by SimulationEngine
        self._network_state = NetworkState(self._topology, config)

        # Stats collector - receives reference for snapshots
        self._stats = StatsCollector(config)

        # Create orchestrator - does NOT own network state
        self._orchestrator = PipelineFactory.create_orchestrator(config)

        # Request storage
        self._requests: dict[int, Request] = {}

    def handle_request(self, request: Request) -> None:
        """Process a request arrival or release."""
        if request.is_arrival:
            # Pass network_state by reference - orchestrator does not store it
            result = self._orchestrator.handle_arrival(request, self._network_state)
            self._stats.record_result(request, result, self._network_state)
        else:
            self._orchestrator.handle_release(request, self._network_state)
            self._stats.record_release(request)

    @property
    def network_state(self) -> NetworkState:
        """Read-only access for external tools (plotting, analysis)."""
        return self._network_state
```

### Code: Orchestrator receives, does not store

```python
# fusion/core/orchestrator.py

class SDNOrchestrator:
    def __init__(self, config: SimulationConfig, pipelines: PipelineSet):
        self.config = config
        self.routing = pipelines.routing
        self.spectrum = pipelines.spectrum
        self.grooming = pipelines.grooming
        self.snr = pipelines.snr
        self.slicing = pipelines.slicing
        # NOTE: No self._network_state - we receive it per call

    def handle_arrival(
        self,
        request: Request,
        network_state: NetworkState,  # Received, not stored
        forced_path: list[str] | None = None,
    ) -> AllocationResult:
        """Process arrival - network_state passed through to pipelines."""

        # Pass network_state to each pipeline - they also don't store it
        if self.grooming and self.config.grooming_enabled:
            groom_result = self.grooming.try_groom(request, network_state)
            # ...

        route_result = self.routing.find_routes(
            request.source, request.destination,
            request.bandwidth_gbps, network_state
        )
        # ... etc
```

### Code: Pipeline receives, reads/mutates, does not cache

```python
# fusion/pipelines/spectrum_pipeline.py

class FirstFitSpectrumPipeline:
    def __init__(self, config: SimulationConfig):
        self.config = config
        # NOTE: No self._network_state or self._spectrum_cache

    def find_spectrum(
        self,
        path: list[str],
        modulations: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,  # Received per call
    ) -> SpectrumResult:
        """Find spectrum - reads from network_state, does not cache."""

        for modulation in modulations:
            if modulation is None:
                continue

            slots_needed = self._calculate_slots(bandwidth_gbps, modulation)

            for band in self.config.band_list:
                for core in range(self.config.cores_per_link):
                    # CORRECT: Always read fresh from network_state
                    if network_state.is_spectrum_available(path, 0, slots_needed, core, band):
                        start = self._find_first_fit(path, band, core, slots_needed, network_state)
                        if start is not None:
                            return SpectrumResult(
                                is_free=True,
                                start_slot=start,
                                end_slot=start + slots_needed,
                                core=core,
                                band=band,
                                modulation=modulation,
                                slots_needed=slots_needed,
                            )

        return SpectrumResult(is_free=False)
```

---

## 2.3 Anti-Pattern: Local Caching

### FORBIDDEN Pattern

```python
# BAD: Caching internal state

class BadSpectrumPipeline:
    def __init__(self, config: SimulationConfig, network_state: NetworkState):
        self.config = config
        # WRONG: Storing reference that may go stale
        self._network_state = network_state
        # VERY WRONG: Copying internal dict
        self._local_spectrum = network_state.network_spectrum_dict.copy()

    def find_spectrum(self, path, modulations, bandwidth_gbps):
        # WRONG: Reading from stale cache
        for link in path:
            link_spectrum = self._local_spectrum[link]  # May be outdated!
            # ...

    def allocate(self, path, start_slot, end_slot, core, band, lp_id):
        # VERY WRONG: Mutating local copy - changes not visible elsewhere!
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            self._local_spectrum[link]["cores_matrix"][band][core][start_slot:end_slot] = lp_id
        # The actual NetworkState is unchanged - silent corruption!
```

### Why This Is Dangerous

1. **Stale reads**: Other pipelines may have allocated spectrum since this pipeline cached its copy. The cache doesn't see those allocations, leading to double-booking.

2. **Lost writes**: Mutations to the local copy are invisible to other components. The simulation thinks spectrum is free when it's not.

3. **Inconsistent state**: If the simulation crashes mid-request, some state is in the cache, some in NetworkState - recovery is impossible.

### CORRECT Pattern

```python
# GOOD: No caching, always pass by reference

class GoodSpectrumPipeline:
    def __init__(self, config: SimulationConfig):
        self.config = config
        # No network_state stored - received per call

    def find_spectrum(
        self,
        path: list[str],
        modulations: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,  # Fresh reference each call
    ) -> SpectrumResult:
        # CORRECT: Always query the authoritative source
        for band in self.config.band_list:
            for core in range(self.config.cores_per_link):
                # Each call goes to NetworkState - always fresh
                start = network_state.find_first_fit(path, band, core, slots_needed)
                if start is not None:
                    return SpectrumResult(is_free=True, start_slot=start, ...)

        return SpectrumResult(is_free=False)
```

### Enforcement

Add a check in code review:

```python
# In NetworkState.__init__
def __init__(self, topology: nx.Graph, config: SimulationConfig):
    self._topology = topology
    self._config = config
    self._spectrum: dict[tuple[str, str], LinkSpectrum] = {}
    self._lightpaths: dict[int, Lightpath] = {}

    # Marker to detect improper caching
    self._creation_id = id(self)

# In sensitive methods
def is_spectrum_available(self, path, start, end, core, band) -> bool:
    assert id(self) == self._creation_id, "NetworkState appears to have been copied!"
    # ... implementation
```

---

# Section 3: Full Pipeline Walkthroughs

## Scenario A: Plain KSP (No Grooming, No Slicing, No SNR)

**Config**: `grooming_enabled=False, slicing_enabled=False, snr_enabled=False`

### Call Chain

```
1. SimulationEngine.handle_request(request)
2. SDNOrchestrator.handle_arrival(request, network_state)
3. [Grooming: SKIPPED - not enabled]
4. RoutingPipeline.find_routes(src, dst, bw, network_state)
5. For each path in routes:
   5a. SpectrumPipeline.find_spectrum(path, mods, bw, network_state)
   5b. If is_free: NetworkState.create_lightpath(...)
   5c. Return AllocationResult(success=True)
6. If all paths fail: Return AllocationResult(success=False)
7. StatsCollector.record_result(request, result, network_state)
```

### Step-by-Step with Code

#### Step 1: SimulationEngine.handle_request

```python
def handle_request(self, request: Request) -> None:
    # request.status = PENDING at this point

    if request.is_arrival:
        result = self._orchestrator.handle_arrival(request, self._network_state)
        self._stats.record_result(request, result, self._network_state)
    else:
        self._orchestrator.handle_release(request, self._network_state)
```

**State change**: None yet

#### Step 2: SDNOrchestrator.handle_arrival

```python
def handle_arrival(self, request: Request, network_state: NetworkState) -> AllocationResult:
    # Grooming check - SKIPPED because self.grooming is None or not enabled

    # Routing
    route_result = self.routing.find_routes(
        request.source, request.destination,
        request.bandwidth_gbps, network_state
    )
    # route_result.paths = [["A", "B", "C"], ["A", "D", "C"]]
    # route_result.modulations = [["QPSK", "16-QAM"], ["QPSK"]]
    # route_result.weights_km = [200.0, 350.0]
```

**State change**: None - routing is read-only

#### Step 4: RoutingPipeline.find_routes

```python
def find_routes(
    self, source: str, destination: str,
    bandwidth_gbps: int, network_state: NetworkState
) -> RouteResult:
    # Compute k-shortest paths using NetworkX
    topology = network_state.topology
    paths = list(nx.shortest_simple_paths(topology, source, destination, weight="weight"))[:self.config.k_paths]

    # For each path, determine valid modulation formats based on distance
    modulations = []
    weights = []
    for path in paths:
        path_length = sum(topology[path[i]][path[i+1]]["weight"] for i in range(len(path)-1))
        weights.append(path_length)
        mods = self._get_valid_modulations(path_length, bandwidth_gbps)
        modulations.append(mods)

    return RouteResult(paths=paths, modulations=modulations, weights_km=weights)
```

**State change**: None - pure computation

#### Step 5a: SpectrumPipeline.find_spectrum

```python
def find_spectrum(
    self, path: list[str], modulations: list[str],
    bandwidth_gbps: int, network_state: NetworkState
) -> SpectrumResult:
    for modulation in modulations:
        if modulation is None:
            continue

        slots_needed = self._calculate_slots(bandwidth_gbps, modulation)

        for band in self.config.band_list:  # ["c", "l"]
            for core in range(self.config.cores_per_link):  # 0..6
                start = self._find_first_fit(path, band, core, slots_needed, network_state)
                if start is not None:
                    return SpectrumResult(
                        is_free=True,
                        start_slot=start,
                        end_slot=start + slots_needed,
                        core=core,
                        band=band,
                        modulation=modulation,
                        slots_needed=slots_needed,
                    )

    return SpectrumResult(is_free=False)

def _find_first_fit(self, path, band, core, slots_needed, network_state) -> int | None:
    """Find first contiguous free slot range on all links of path."""
    for start in range(self.config.band_slots[band] - slots_needed):
        if network_state.is_spectrum_available(path, start, start + slots_needed, core, band):
            return start
    return None
```

**State change**: None - read-only query

#### Step 5b: NetworkState.create_lightpath

```python
def create_lightpath(
    self, path: list[str], start_slot: int, end_slot: int,
    core: int, band: str, modulation: str,
    bandwidth_gbps: int, path_weight_km: float,
) -> Lightpath:
    lp = Lightpath(
        lightpath_id=self._next_lightpath_id,
        path=path,
        start_slot=start_slot,
        end_slot=end_slot,
        core=core,
        band=band,
        modulation=modulation,
        total_bandwidth_gbps=bandwidth_gbps,
        remaining_bandwidth_gbps=bandwidth_gbps,
        path_weight_km=path_weight_km,
    )
    self._next_lightpath_id += 1
    self._lightpaths[lp.lightpath_id] = lp

    # Allocate spectrum on all links
    self._allocate_spectrum(path, start_slot, end_slot, core, band, lp.lightpath_id)

    return lp

def _allocate_spectrum(self, path, start, end, core, band, lp_id):
    for i in range(len(path) - 1):
        link = (path[i], path[i+1])
        self._spectrum[link].allocate(band, core, start, end, lp_id)
        # Also allocate reverse direction
        reverse_link = (path[i+1], path[i])
        self._spectrum[reverse_link].allocate(band, core, start, end, lp_id)
```

**State change**:
- `network_state._lightpaths[1] = Lightpath(id=1, path=["A","B","C"], ...)`
- `network_state._spectrum[("A","B")].cores_matrix["c"][0][10:18] = 1`
- `network_state._spectrum[("B","C")].cores_matrix["c"][0][10:18] = 1`

#### Step 5c: Return AllocationResult

```python
# Back in SDNOrchestrator.handle_arrival
lightpath.request_allocations[request.request_id] = request.bandwidth_gbps
request.lightpath_ids.append(lightpath.lightpath_id)
request.status = RequestStatus.ROUTED

return AllocationResult(
    success=True,
    lightpaths_created=[1],
    total_bandwidth_allocated_gbps=100,
)
```

**State change**:
- `lightpath.request_allocations = {42: 100}`
- `request.lightpath_ids = [1]`
- `request.status = ROUTED`

#### Step 7: StatsCollector.record_result

```python
def record_result(self, request: Request, result: AllocationResult, network_state: NetworkState):
    if result.success:
        self._successful_requests += 1
        self._total_bandwidth_allocated += result.total_bandwidth_allocated_gbps
    else:
        self._blocked_requests += 1
        self._block_reasons[result.block_reason.value] += 1
```

**State change**: Statistics updated

---

## Scenario B: Grooming + SNR (No Slicing)

**Config**: `grooming_enabled=True, slicing_enabled=False, snr_enabled=True`

### Call Chain (Fully Groomed Case)

```
1. SimulationEngine.handle_request(request)
2. SDNOrchestrator.handle_arrival(request, network_state)
3. GroomingPipeline.try_groom(request, network_state)
   -> GroomingResult(fully_groomed=True, lightpaths_used=[1])
4. Return AllocationResult(success=True, is_groomed=True)
5. StatsCollector.record_result(...)
```

### Call Chain (Partial Grooming Case)

```
1. SimulationEngine.handle_request(request)
2. SDNOrchestrator.handle_arrival(request, network_state)
3. GroomingPipeline.try_groom(request, network_state)
   -> GroomingResult(partially_groomed=True, remaining_bw=50, forced_path=[...])
4. RoutingPipeline.find_routes(src, dst, 50, network_state, forced_path)
5. For each path:
   5a. SpectrumPipeline.find_spectrum(path, mods, 50, network_state)
   5b. NetworkState.create_lightpath(...)
   5c. SNRPipeline.validate(lightpath, network_state)
       - If fails: NetworkState.release_lightpath(lp_id), continue
       - If passes: Return AllocationResult(success=True)
6. StatsCollector.record_result(...)
```

### Step-by-Step: Partial Grooming with SNR

#### Step 3: GroomingPipeline.try_groom

```python
def try_groom(self, request: Request, network_state: NetworkState) -> GroomingResult:
    # Find existing lightpaths between endpoints
    lightpaths = network_state.get_lightpaths_with_capacity(
        request.source, request.destination, min_bandwidth=1
    )

    if not lightpaths:
        return GroomingResult(fully_groomed=False, partially_groomed=False,
                              bandwidth_groomed_gbps=0, remaining_bandwidth_gbps=request.bandwidth_gbps,
                              lightpaths_used=[])

    remaining_bw = request.bandwidth_gbps
    used_lightpaths = []
    forced_path = None

    for lp in sorted(lightpaths, key=lambda x: -x.remaining_bandwidth_gbps):
        if remaining_bw <= 0:
            break

        allocate_bw = min(remaining_bw, lp.remaining_bandwidth_gbps)
        lp.request_allocations[request.request_id] = allocate_bw
        lp.remaining_bandwidth_gbps -= allocate_bw
        remaining_bw -= allocate_bw
        used_lightpaths.append(lp.lightpath_id)
        forced_path = lp.path  # Must allocate new lightpath on same path

    fully_groomed = remaining_bw <= 0
    partially_groomed = not fully_groomed and len(used_lightpaths) > 0

    return GroomingResult(
        fully_groomed=fully_groomed,
        partially_groomed=partially_groomed,
        bandwidth_groomed_gbps=request.bandwidth_gbps - remaining_bw,
        remaining_bandwidth_gbps=remaining_bw,
        lightpaths_used=used_lightpaths,
        forced_path=forced_path if partially_groomed else None,
    )
```

**State change** (if grooming allocates bandwidth):
- `lightpath.request_allocations[42] = 50`
- `lightpath.remaining_bandwidth_gbps = 0` (was 50)

#### Step 5c: SNRPipeline.validate

```python
def validate(self, lightpath: Lightpath, network_state: NetworkState) -> bool:
    """Validate SNR for a newly allocated lightpath."""

    # Calculate SNR based on path, modulation, and neighboring lightpaths
    snr_db = self._calculate_gsnr(
        path=lightpath.path,
        modulation=lightpath.modulation,
        band=lightpath.band,
        core=lightpath.core,
        start_slot=lightpath.start_slot,
        end_slot=lightpath.end_slot,
        network_state=network_state,
    )

    # Get required SNR for this modulation format
    required_snr = self.config.snr_thresholds[lightpath.modulation]

    if snr_db < required_snr:
        logger.debug(f"SNR validation failed: {snr_db:.2f} < {required_snr:.2f}")
        return False

    # Store SNR in lightpath for monitoring
    lightpath.snr_db = snr_db
    return True
```

**State change** (if validation passes):
- `lightpath.snr_db = 18.5`

**If validation fails**:
```python
# Back in SDNOrchestrator
if not snr_ok:
    network_state.release_lightpath(lightpath.lightpath_id)
    continue  # Try next path
```

**State change** (on failure):
- Lightpath removed from `network_state._lightpaths`
- Spectrum deallocated from all links

---

## Scenario C: Slicing + SNR (No Grooming)

**Config**: `grooming_enabled=False, slicing_enabled=True, max_slices=4, snr_enabled=True`

### Call Chain

```
1. SimulationEngine.handle_request(request)  # bw=400 Gbps
2. SDNOrchestrator.handle_arrival(request, network_state)
3. [Grooming: SKIPPED]
4. RoutingPipeline.find_routes(src, dst, 400, network_state)
   -> paths with modulations=[[None], [QPSK]]  # 400Gbps too high for short path
5. For path[0] with mod=None:
   5a. SpectrumPipeline.find_spectrum -> is_free=False (no valid mod)
   5b. SlicingPipeline.try_slice(request, path, network_state)
       - Split 400 into 4x100 Gbps
       - For each slice:
         - find_spectrum(bw=100) -> success
         - create_lightpath -> lp_id=N
         - SNR.validate(lp) -> pass/fail
       -> AllocationResult(success=True, is_sliced=True, lightpaths=[1,2,3,4])
6. StatsCollector.record_result(...)
```

### Step-by-Step: Slicing with SNR

#### Step 5b: SlicingPipeline.try_slice

```python
def try_slice(
    self,
    request: Request,
    path: list[str],
    modulations: list[str],
    bandwidth_gbps: int,
    network_state: NetworkState,
    spectrum_pipeline: SpectrumPipeline,
    snr_pipeline: SNRPipeline | None,
) -> AllocationResult:
    """Attempt to serve request through multiple smaller lightpaths."""

    # Determine slice configurations: try splitting into 2, 3, 4, ... slices
    for num_slices in range(2, self.config.max_slices + 1):
        slice_bw = bandwidth_gbps // num_slices

        # Check if this slice bandwidth has a valid modulation for this path
        slice_mods = self._get_modulations_for_bandwidth(slice_bw, path)
        if not any(slice_mods):
            continue

        # Try to allocate all slices
        created_lightpaths = []
        success = True

        for slice_idx in range(num_slices):
            spectrum_result = spectrum_pipeline.find_spectrum(
                path, slice_mods, slice_bw, network_state
            )

            if not spectrum_result.is_free:
                success = False
                break

            lightpath = network_state.create_lightpath(
                path=path,
                start_slot=spectrum_result.start_slot,
                end_slot=spectrum_result.end_slot,
                core=spectrum_result.core,
                band=spectrum_result.band,
                modulation=spectrum_result.modulation,
                bandwidth_gbps=slice_bw,
                path_weight_km=self._compute_path_weight(path, network_state),
            )

            # SNR validation for this slice
            if snr_pipeline:
                if not snr_pipeline.validate(lightpath, network_state):
                    # Rollback this slice
                    network_state.release_lightpath(lightpath.lightpath_id)
                    success = False
                    break

            created_lightpaths.append(lightpath.lightpath_id)
            lightpath.request_allocations[request.request_id] = slice_bw

        if success:
            request.is_sliced = True
            request.lightpath_ids.extend(created_lightpaths)
            return AllocationResult(
                success=True,
                lightpaths_created=created_lightpaths,
                is_sliced=True,
                total_bandwidth_allocated_gbps=bandwidth_gbps,
            )
        else:
            # Rollback all slices created so far
            for lp_id in created_lightpaths:
                network_state.release_lightpath(lp_id)

    return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)
```

**State change** (on success with 4 slices):
- `network_state._lightpaths = {1: Lightpath(...), 2: Lightpath(...), 3: Lightpath(...), 4: Lightpath(...)}`
- `request.lightpath_ids = [1, 2, 3, 4]`
- `request.is_sliced = True`
- Spectrum allocated for 4 separate ranges

---

## Scenario D: Grooming + Slicing + SNR (Hard Case)

**Config**: `grooming_enabled=True, slicing_enabled=True, max_slices=4, snr_enabled=True`

### Call Chain

```
1. SimulationEngine.handle_request(request)  # bw=500 Gbps
2. SDNOrchestrator.handle_arrival(request, network_state)
3. GroomingPipeline.try_groom(request, network_state)
   -> GroomingResult(partially_groomed=True, bw_groomed=100, remaining=400, forced_path=P)
4. RoutingPipeline.find_routes(src, dst, 400, network_state, forced_path=P)
5. For path P:
   5a. SpectrumPipeline.find_spectrum(P, mods, 400, network_state) -> is_free=False
   5b. SlicingPipeline.try_slice(request, P, 400, network_state, ...)
       - 4x100 Gbps slices
       - Slices 1,2,3 pass SNR
       - Slice 4 FAILS SNR -> rollback slice 4
       - Try 2x200 Gbps -> also fails
       -> AllocationResult(success=False)
6. SDNOrchestrator._handle_failure(request, groomed_lps, NO_SPECTRUM, network_state)
   - If can_partially_serve: accept partial (100 Gbps groomed)
   - Else: rollback grooming too -> fully blocked
```

### Branching Logic in SDNOrchestrator

```python
def handle_arrival(self, request: Request, network_state: NetworkState, ...) -> AllocationResult:
    groomed_lightpaths = []
    remaining_bw = request.bandwidth_gbps

    # DECISION POINT 1: Try grooming?
    if self.grooming and self.config.grooming_enabled:
        groom_result = self.grooming.try_groom(request, network_state)

        if groom_result.fully_groomed:
            # EARLY EXIT: Done, no routing/spectrum needed
            return AllocationResult(success=True, is_groomed=True, ...)

        if groom_result.partially_groomed:
            groomed_lightpaths = groom_result.lightpaths_used
            remaining_bw = groom_result.remaining_bandwidth_gbps
            forced_path = groom_result.forced_path

    # Routing
    route_result = self.routing.find_routes(src, dst, remaining_bw, network_state, forced_path)

    if not route_result.paths:
        # DECISION POINT 2: No routes found
        return self._handle_failure(request, groomed_lightpaths, BlockReason.NO_ROUTE, network_state)

    for path_idx, path in enumerate(route_result.paths):
        result = self._try_allocate_on_path(request, path, ..., remaining_bw, network_state)

        if result is not None:
            # Success - combine with any groomed lightpaths
            return AllocationResult(
                success=True,
                lightpaths_created=result.lightpaths_created,
                lightpaths_groomed=groomed_lightpaths,
                is_groomed=len(groomed_lightpaths) > 0,
                is_partially_groomed=len(groomed_lightpaths) > 0 and len(result.lightpaths_created) > 0,
                is_sliced=result.is_sliced,
            )

    # DECISION POINT 3: All paths failed
    return self._handle_failure(request, groomed_lightpaths, BlockReason.NO_SPECTRUM, network_state)

def _try_allocate_on_path(self, request, path, mods, weight, bw, network_state) -> AllocationResult | None:
    # Standard allocation
    spectrum_result = self.spectrum.find_spectrum(path, mods, bw, network_state)

    if spectrum_result.is_free:
        alloc_result = self._allocate_and_validate(request, path, spectrum_result, weight, bw, network_state)
        if alloc_result is not None:
            return alloc_result

    # DECISION POINT 4: Try slicing as fallback?
    if self.slicing and self.config.slicing_enabled:
        return self.slicing.try_slice(request, path, mods, bw, network_state, self.spectrum, self.snr)

    return None

def _handle_failure(self, request, groomed_lps, reason, network_state) -> AllocationResult:
    # DECISION POINT 5: Accept partial or rollback?
    if groomed_lps and self.config.can_partially_serve:
        request.status = RequestStatus.ROUTED
        request.is_partially_groomed = True
        return AllocationResult(
            success=True,
            lightpaths_groomed=groomed_lps,
            is_partially_groomed=True,
            block_reason=reason,  # Record why full allocation failed
        )

    if groomed_lps:
        # DECISION POINT 6: Must rollback grooming
        self.grooming.rollback(request, groomed_lps, network_state)

    request.status = RequestStatus.BLOCKED
    request.block_reason = reason.value
    return AllocationResult(success=False, block_reason=reason)
```

---

## Scenario E: 1+1 Protection + SNR

**Config**: `route_method="1plus1_protection", snr_enabled=True`

### Call Chain

```
1. SimulationEngine.handle_request(request)
2. SDNOrchestrator.handle_arrival(request, network_state)
3. [Grooming: typically disabled for protected requests]
4. ProtectedRoutingPipeline.find_routes(src, dst, bw, network_state)
   -> RouteResult with paths AND backup_paths (disjoint)
5. For each (primary, backup) pair:
   5a. SpectrumPipeline.find_protected_spectrum(primary, backup, mods, bw, network_state)
       -> Finds COMMON spectrum slots available on BOTH paths
   5b. NetworkState.create_protected_lightpath(primary, backup, spectrum_result)
       -> Allocates same slots on both paths
   5c. SNRPipeline.validate_protected(lightpath, network_state)
       -> Validates SNR on both primary and backup paths
   5d. Return AllocationResult(success=True, is_protected=True)
6. FailureManager integration for switchover
```

### Step 4: ProtectedRoutingPipeline.find_routes

```python
class ProtectedRoutingPipeline:
    def find_routes(
        self, source: str, destination: str,
        bandwidth_gbps: int, network_state: NetworkState
    ) -> RouteResult:
        """Find disjoint primary+backup path pairs."""
        topology = network_state.topology

        # Use Suurballe's algorithm for edge-disjoint paths
        try:
            disjoint_paths = list(nx.edge_disjoint_paths(topology, source, destination))
        except nx.NetworkXNoPath:
            return RouteResult(paths=[], modulations=[], weights_km=[])

        if len(disjoint_paths) < 2:
            # Not enough disjoint paths for protection
            return RouteResult(paths=[], modulations=[], weights_km=[])

        # Select best pair(s)
        paths = []
        backup_paths = []
        modulations = []
        backup_modulations = []
        weights = []

        for i in range(0, len(disjoint_paths) - 1, 2):
            primary = disjoint_paths[i]
            backup = disjoint_paths[i + 1]

            primary_mods = self._get_valid_modulations(primary, bandwidth_gbps)
            backup_mods = self._get_valid_modulations(backup, bandwidth_gbps)

            # Both paths must have at least one valid modulation
            if not any(primary_mods) or not any(backup_mods):
                continue

            paths.append(primary)
            backup_paths.append(backup)
            modulations.append(primary_mods)
            backup_modulations.append(backup_mods)
            weights.append(self._compute_path_weight(primary, network_state))

        return RouteResult(
            paths=paths,
            modulations=modulations,
            weights_km=weights,
            backup_paths=backup_paths,
            backup_modulations=backup_modulations,
        )
```

### Step 5a: SpectrumPipeline.find_protected_spectrum

```python
def find_protected_spectrum(
    self,
    primary_path: list[str],
    backup_path: list[str],
    modulations: list[str],
    bandwidth_gbps: int,
    network_state: NetworkState,
) -> SpectrumResult:
    """Find spectrum available on BOTH primary and backup paths."""

    for modulation in modulations:
        if modulation is None:
            continue

        slots_needed = self._calculate_slots(bandwidth_gbps, modulation)

        for band in self.config.band_list:
            for core in range(self.config.cores_per_link):
                # Find common available slots on both paths
                common_starts = self._find_common_slots(
                    [primary_path, backup_path], band, core, slots_needed, network_state
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
                        # Protection-specific fields
                        backup_start_slot=start,
                        backup_end_slot=start + slots_needed,
                        backup_core=core,
                        backup_band=band,
                    )

    return SpectrumResult(is_free=False)

def _find_common_slots(
    self, paths: list[list[str]], band: str, core: int,
    slots_needed: int, network_state: NetworkState
) -> list[int]:
    """Find slot indices available on ALL paths."""
    common_starts = None

    for path in paths:
        path_starts = set()
        for start in range(self.config.band_slots[band] - slots_needed):
            if network_state.is_spectrum_available(path, start, start + slots_needed, core, band):
                path_starts.add(start)

        if common_starts is None:
            common_starts = path_starts
        else:
            common_starts = common_starts.intersection(path_starts)

    return sorted(common_starts) if common_starts else []
```

### Step 5b: NetworkState.create_protected_lightpath

```python
def create_protected_lightpath(
    self,
    primary_path: list[str],
    backup_path: list[str],
    spectrum_result: SpectrumResult,
    bandwidth_gbps: int,
    path_weight_km: float,
) -> Lightpath:
    """Create lightpath with protection, allocating on both paths."""

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
        # Protection fields
        backup_path=backup_path,
        is_protected=True,
        active_path="primary",
    )
    self._next_lightpath_id += 1
    self._lightpaths[lp.lightpath_id] = lp

    # Allocate SAME spectrum on BOTH paths
    self._allocate_spectrum(primary_path, spectrum_result.start_slot,
                           spectrum_result.end_slot, spectrum_result.core,
                           spectrum_result.band, lp.lightpath_id)
    self._allocate_spectrum(backup_path, spectrum_result.start_slot,
                           spectrum_result.end_slot, spectrum_result.core,
                           spectrum_result.band, lp.lightpath_id)

    return lp
```

### Failure Handling Integration

```python
# In SDNOrchestrator or a dedicated FailureHandler

def handle_link_failure(self, failed_link: tuple[str, str], network_state: NetworkState):
    """Handle link failure - switch protected lightpaths to backup."""

    affected_lightpaths = network_state.get_lightpaths_on_link(failed_link)

    for lp in affected_lightpaths:
        if lp.is_protected and lp.active_path == "primary":
            # Check if primary path uses the failed link
            if self._path_uses_link(lp.path, failed_link):
                # Switch to backup
                lp.active_path = "backup"
                self._stats.record_protection_switchover(lp.lightpath_id)
                logger.info(f"Lightpath {lp.lightpath_id} switched to backup due to failure on {failed_link}")
        elif not lp.is_protected:
            # Unprotected lightpath - must be rerouted or blocked
            self._handle_unprotected_failure(lp, network_state)
```

---

# Section 4: Phase-by-Phase Migration Plan

## Phase 1: Core Domain Model Extraction

### Objectives
- Introduce `SimulationConfig`, `Request`, `Lightpath` dataclasses
- No changes to existing SDNController or RL code
- New code is additive only

### Constraints
- `run_comparison.py` must pass unchanged
- No modifications to existing function signatures
- All new code is in new files

### Entry Criteria
- V2 architecture document approved
- Test baseline established

### Exit Criteria
- Domain classes created with full test coverage
- `from_legacy_dict()` and `to_legacy_dict()` work correctly
- No regressions in existing tests

### Files Created

```
fusion/domain/
├── __init__.py
├── config.py          # SimulationConfig
├── request.py         # Request, RequestStatus
├── lightpath.py       # Lightpath
└── results.py         # RouteResult, SpectrumResult, GroomingResult, AllocationResult
```

### Files Modified
None in Phase 1 - purely additive.

### Code Example: SimulationConfig

**NEW FILE: fusion/domain/config.py**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class SimulationConfig:
    """Immutable simulation configuration."""

    # Network
    network_name: str
    cores_per_link: int
    band_list: tuple[str, ...]
    band_slots: dict[str, int]
    guard_slots: int

    # Traffic
    num_requests: int
    erlang: float
    holding_time: float

    # Routing
    route_method: str
    k_paths: int
    allocation_method: str

    # Features
    grooming_enabled: bool
    slicing_enabled: bool
    max_slices: int
    snr_enabled: bool
    snr_type: Optional[str]
    snr_recheck: bool
    can_partially_serve: bool

    # Modulation
    modulation_formats: dict
    mod_per_bw: dict
    snr_thresholds: dict[str, float]

    @classmethod
    def from_engine_props(cls, engine_props: dict) -> 'SimulationConfig':
        """Create from legacy engine_props dict."""
        return cls(
            network_name=engine_props.get('network', ''),
            cores_per_link=engine_props.get('cores_per_link', 1),
            band_list=tuple(engine_props.get('band_list', ['c'])),
            band_slots={
                'c': engine_props.get('c_band', 320),
                'l': engine_props.get('l_band', 0),
                's': engine_props.get('s_band', 0),
            },
            guard_slots=engine_props.get('guard_slots', 1),
            num_requests=engine_props.get('num_requests', 1000),
            erlang=engine_props.get('erlang', 100),
            holding_time=engine_props.get('holding_time', 1),
            route_method=engine_props.get('route_method', 'k_shortest_path'),
            k_paths=engine_props.get('k_paths', 3),
            allocation_method=engine_props.get('allocation_method', 'first_fit'),
            grooming_enabled=engine_props.get('is_grooming_enabled', False),
            slicing_enabled=engine_props.get('max_segments', 1) > 1,
            max_slices=engine_props.get('max_segments', 1),
            snr_enabled=engine_props.get('snr_type') not in (None, 'None', ''),
            snr_type=engine_props.get('snr_type'),
            snr_recheck=engine_props.get('snr_recheck', False),
            can_partially_serve=engine_props.get('can_partially_serve', False),
            modulation_formats=engine_props.get('modulation_formats_dict', {}),
            mod_per_bw=engine_props.get('mod_per_bw', {}),
            snr_thresholds=engine_props.get('req_snr', {}),
        )

    def to_engine_props(self) -> dict:
        """Convert back to legacy format for compatibility."""
        return {
            'network': self.network_name,
            'cores_per_link': self.cores_per_link,
            'band_list': list(self.band_list),
            'c_band': self.band_slots.get('c', 0),
            'l_band': self.band_slots.get('l', 0),
            's_band': self.band_slots.get('s', 0),
            'guard_slots': self.guard_slots,
            'num_requests': self.num_requests,
            'erlang': self.erlang,
            'holding_time': self.holding_time,
            'route_method': self.route_method,
            'k_paths': self.k_paths,
            'allocation_method': self.allocation_method,
            'is_grooming_enabled': self.grooming_enabled,
            'max_segments': self.max_slices,
            'snr_type': self.snr_type,
            'snr_recheck': self.snr_recheck,
            'can_partially_serve': self.can_partially_serve,
            'modulation_formats_dict': self.modulation_formats,
            'mod_per_bw': self.mod_per_bw,
            'req_snr': self.snr_thresholds,
        }
```

---

## Phase 2: NetworkState + Pipeline Interfaces

### Objectives
- Introduce `NetworkState` with legacy compatibility properties
- Define pipeline protocols (interfaces)
- Create adapter classes wrapping existing implementations

### Constraints
- Existing SDNController continues to work
- Adapters delegate to existing classes
- No behavior changes

### Entry Criteria
- Phase 1 complete
- Domain classes tested

### Exit Criteria
- NetworkState manages spectrum and lightpaths
- Legacy properties return correct data
- Pipeline adapters pass same inputs/outputs as originals
- `run_comparison.py` passes

### Files Created

```
fusion/domain/
└── network_state.py   # NetworkState, LinkSpectrum

fusion/interfaces/
├── __init__.py
└── pipelines.py       # Protocol definitions

fusion/core/adapters/
├── __init__.py
├── routing_adapter.py
├── spectrum_adapter.py
├── grooming_adapter.py
├── snr_adapter.py
└── slicing_adapter.py
```

### Files Modified

```
fusion/core/simulation.py     # Add NetworkState creation alongside existing
fusion/core/sdn_controller.py # Read from NetworkState where possible
```

### Code Example: NetworkState with Legacy Compatibility

**NEW FILE: fusion/domain/network_state.py**

```python
from dataclasses import dataclass, field
import numpy as np
import networkx as nx
from typing import Any, Optional

from .lightpath import Lightpath
from .config import SimulationConfig


@dataclass
class LinkSpectrum:
    """Spectrum state for a single link."""
    link: tuple[str, str]
    cores_matrix: dict[str, np.ndarray]
    usage_count: int = 0
    throughput: float = 0.0
    link_num: int = 0

    def is_range_free(self, band: str, core: int, start: int, end: int) -> bool:
        return np.all(self.cores_matrix[band][core][start:end] == 0)

    def allocate(self, band: str, core: int, start: int, end: int, lp_id: int) -> None:
        self.cores_matrix[band][core][start:end] = lp_id
        # Guard band
        if end < self.cores_matrix[band].shape[1]:
            self.cores_matrix[band][core][end] = -lp_id
        self.usage_count += 1

    def release(self, lp_id: int) -> None:
        for band, matrix in self.cores_matrix.items():
            matrix[matrix == lp_id] = 0
            matrix[matrix == -lp_id] = 0


class NetworkState:
    """Single source of truth for network state."""

    def __init__(self, topology: nx.Graph, config: SimulationConfig):
        self._topology = topology
        self._config = config
        self._spectrum: dict[tuple[str, str], LinkSpectrum] = {}
        self._lightpaths: dict[int, Lightpath] = {}
        self._next_lightpath_id = 1

        self._initialize_spectrum()

    def _initialize_spectrum(self) -> None:
        """Initialize spectrum matrices for all links."""
        for u, v in self._topology.edges():
            for link in [(u, v), (v, u)]:
                cores_matrix = {}
                for band in self._config.band_list:
                    slots = self._config.band_slots.get(band, 0)
                    cores_matrix[band] = np.zeros(
                        (self._config.cores_per_link, slots), dtype=np.float64
                    )
                self._spectrum[link] = LinkSpectrum(link=link, cores_matrix=cores_matrix)

    # === Public Read Methods ===

    @property
    def topology(self) -> nx.Graph:
        return self._topology

    def get_lightpath(self, lightpath_id: int) -> Optional[Lightpath]:
        return self._lightpaths.get(lightpath_id)

    def get_lightpaths_between(self, source: str, destination: str) -> list[Lightpath]:
        key = tuple(sorted([source, destination]))
        return [lp for lp in self._lightpaths.values() if lp.endpoint_key == key]

    def get_lightpaths_with_capacity(self, source: str, dest: str, min_bw: int) -> list[Lightpath]:
        return [lp for lp in self.get_lightpaths_between(source, dest)
                if lp.remaining_bandwidth_gbps >= min_bw]

    def is_spectrum_available(
        self, path: list[str], start: int, end: int, core: int, band: str
    ) -> bool:
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            if not self._spectrum[link].is_range_free(band, core, start, end):
                return False
        return True

    def get_link_spectrum(self, link: tuple[str, str]) -> LinkSpectrum:
        return self._spectrum[link]

    # === Public Write Methods ===

    def create_lightpath(
        self, path: list[str], start_slot: int, end_slot: int,
        core: int, band: str, modulation: str,
        bandwidth_gbps: int, path_weight_km: float,
        backup_path: Optional[list[str]] = None,
    ) -> Lightpath:
        lp = Lightpath(
            lightpath_id=self._next_lightpath_id,
            path=path,
            start_slot=start_slot,
            end_slot=end_slot,
            core=core,
            band=band,
            modulation=modulation,
            total_bandwidth_gbps=bandwidth_gbps,
            remaining_bandwidth_gbps=bandwidth_gbps,
            path_weight_km=path_weight_km,
            backup_path=backup_path,
            is_protected=backup_path is not None,
        )
        self._next_lightpath_id += 1
        self._lightpaths[lp.lightpath_id] = lp

        # Allocate spectrum
        self._allocate_spectrum_on_path(path, start_slot, end_slot, core, band, lp.lightpath_id)
        if backup_path:
            self._allocate_spectrum_on_path(backup_path, start_slot, end_slot, core, band, lp.lightpath_id)

        return lp

    def release_lightpath(self, lightpath_id: int) -> None:
        lp = self._lightpaths.pop(lightpath_id, None)
        if lp is None:
            return

        # Release spectrum on primary path
        self._release_spectrum_on_path(lp.path, lightpath_id)

        # Release on backup if protected
        if lp.backup_path:
            self._release_spectrum_on_path(lp.backup_path, lightpath_id)

    def _allocate_spectrum_on_path(
        self, path: list[str], start: int, end: int, core: int, band: str, lp_id: int
    ) -> None:
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            self._spectrum[link].allocate(band, core, start, end, lp_id)
            reverse = (path[i+1], path[i])
            self._spectrum[reverse].allocate(band, core, start, end, lp_id)

    def _release_spectrum_on_path(self, path: list[str], lp_id: int) -> None:
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            self._spectrum[link].release(lp_id)
            reverse = (path[i+1], path[i])
            self._spectrum[reverse].release(lp_id)

    # === Legacy Compatibility (TRANSITIONAL - remove in Phase 5) ===

    @property
    def network_spectrum_dict(self) -> dict:
        """DEPRECATED: Legacy format for backward compatibility."""
        return {
            link: {
                'cores_matrix': ls.cores_matrix,
                'usage_count': ls.usage_count,
                'throughput': ls.throughput,
                'link_num': ls.link_num,
            }
            for link, ls in self._spectrum.items()
        }

    @property
    def lightpath_status_dict(self) -> dict:
        """DEPRECATED: Legacy format for backward compatibility."""
        result: dict = {}
        for lp in self._lightpaths.values():
            key = lp.endpoint_key
            if key not in result:
                result[key] = {}
            result[key][lp.lightpath_id] = {
                'path': lp.path,
                'core': lp.core,
                'band': lp.band,
                'start_slot': lp.start_slot,
                'end_slot': lp.end_slot,
                'mod_format': lp.modulation,
                'lightpath_bandwidth': lp.total_bandwidth_gbps,
                'remaining_bandwidth': lp.remaining_bandwidth_gbps,
                'snr_cost': lp.snr_db,
                'xt_cost': lp.xt_cost,
                'path_weight': lp.path_weight_km,
                'is_degraded': lp.is_degraded,
                'requests_dict': lp.request_allocations.copy(),
            }
        return result
```

### Code Example: Routing Adapter

**NEW FILE: fusion/core/adapters/routing_adapter.py**

```python
from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState
from fusion.domain.results import RouteResult
from fusion.interfaces.pipelines import RoutingPipeline

# Import existing routing class
from fusion.core.routing import Routing
from fusion.core.properties import RoutingProps


class RoutingAdapter(RoutingPipeline):
    """Adapter wrapping existing Routing class to new interface."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        # We'll create the legacy routing object when needed
        self._engine_props = config.to_engine_props()

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: Optional[list[str]] = None,
    ) -> RouteResult:
        # Create legacy routing object
        route_props = RoutingProps()

        # Build minimal sdn_props for routing
        class MinimalSDNProps:
            def __init__(self):
                self.topology = network_state.topology
                self.source = source
                self.destination = destination

        sdn_props = MinimalSDNProps()

        # Use existing routing
        routing_obj = Routing(self._engine_props, sdn_props, route_props)
        routing_obj.get_route()

        # Convert to new format
        return RouteResult(
            paths=route_props.paths_matrix,
            modulations=route_props.modulation_formats_matrix,
            weights_km=route_props.weights_list,
            backup_paths=route_props.backup_paths_matrix if route_props.backup_paths_matrix else None,
            backup_modulations=route_props.backup_modulation_formats_matrix if route_props.backup_modulation_formats_matrix else None,
        )
```

---

## Phase 3: SDNOrchestrator Integration

### Objectives
- Create `SDNOrchestrator` with pipeline coordination
- Create `PipelineFactory`
- Add feature flag to switch between old and new paths
- Run both in parallel to verify identical results

### Constraints
- Old path must remain functional
- New path behind feature flag
- A/B testing infrastructure for comparison

### Entry Criteria
- Phase 2 complete
- All adapters tested

### Exit Criteria
- Orchestrator coordinates all pipelines correctly
- Feature flag switches cleanly
- `run_comparison.py` passes with new path enabled
- No performance regression

### Files Created

```
fusion/core/
├── orchestrator.py        # SDNOrchestrator
└── pipeline_factory.py    # PipelineFactory, PipelineSet

fusion/pipelines/
├── __init__.py
├── routing_pipeline.py    # KSPRoutingPipeline, ProtectedRoutingPipeline
├── spectrum_pipeline.py   # FirstFitSpectrumPipeline, BestFitSpectrumPipeline
├── grooming_pipeline.py   # GroomingPipelineImpl
├── snr_pipeline.py        # SNRPipelineImpl
└── slicing_pipeline.py    # SlicingPipelineImpl
```

### Files Modified

```
fusion/core/simulation.py  # Add orchestrator creation, feature flag
```

### Code Example: SimulationEngine with Feature Flag

**MODIFIED FILE: fusion/core/simulation.py**

```python
# BEFORE (existing code, simplified)
class SimulationEngine:
    def __init__(self, engine_props: dict):
        self.engine_props = engine_props
        self.topology = self.create_topology()
        self.sdn_obj = SDNController(engine_props)
        # ... etc

# AFTER (with feature flag)
class SimulationEngine:
    def __init__(self, engine_props: dict):
        self.engine_props = engine_props
        self._config = SimulationConfig.from_engine_props(engine_props)

        self.topology = self.create_topology()
        self._network_state = NetworkState(self.topology, self._config)

        # Feature flag for new architecture
        self._use_orchestrator = engine_props.get('use_orchestrator', False)

        if self._use_orchestrator:
            # New path: orchestrator with pipelines
            self._orchestrator = PipelineFactory.create_orchestrator(self._config)
        else:
            # Old path: existing SDNController
            self.sdn_obj = SDNController(engine_props)

        self._stats = StatsCollector(self._config)

    def handle_arrival(self, current_time: tuple[int, float]) -> None:
        request_dict = self.reqs_dict[current_time]

        if self._use_orchestrator:
            # New path
            request = Request.from_legacy_dict(current_time, request_dict, self._config)
            result = self._orchestrator.handle_arrival(request, self._network_state)
            self._update_stats_from_result(request, result)
        else:
            # Old path
            self.sdn_obj.handle_event(request_dict, 'arrival')
            self._update_stats_from_sdn_props()
```

---

## Phase 4: RL/SB3 Integration

### Objectives
- Create `RLSimulationAdapter`
- Create `UnifiedSimEnv` using new architecture
- Migrate RL to use same pipelines as simulation
- Remove `mock_handle_arrival()` and duplicated logic

### Constraints
- RL training must produce comparable results
- SB3 API compatibility
- No changes to trained model format

### Entry Criteria
- Phase 3 complete
- Orchestrator verified

### Exit Criteria
- RL uses same pipelines as simulation
- No duplicated feasibility checking
- Training runs successfully
- Evaluation matches baseline

### Files Created

```
fusion/modules/rl/
├── env_adapter.py              # RLSimulationAdapter
└── gymnasium_envs/
    └── unified_sim_env.py      # UnifiedSimEnv
```

### Files Modified

```
fusion/modules/rl/utils/general_utils.py  # Remove mock_handle_arrival
fusion/modules/rl/utils/sim_env.py        # Deprecate old helpers
```

### Code Example: Full RL Adapter

**NEW FILE: fusion/modules/rl/env_adapter.py**

```python
from dataclasses import dataclass
from typing import Optional, Any

from fusion.domain.config import SimulationConfig
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.domain.results import AllocationResult
from fusion.interfaces.pipelines import RoutingPipeline, SpectrumPipeline
from fusion.core.orchestrator import SDNOrchestrator


@dataclass
class PathOption:
    """Single path option for RL action selection."""
    path_index: int
    path: list[str]
    weight_km: float
    is_feasible: bool
    modulation: Optional[str]
    slots_needed: Optional[int]
    congestion: float  # 0.0 = empty, 1.0 = fully congested


class RLSimulationAdapter:
    """
    RL-friendly interface that uses the SAME pipelines as simulation.

    This adapter eliminates the need for mock_handle_arrival() by
    directly using the routing and spectrum pipelines.
    """

    def __init__(
        self,
        config: SimulationConfig,
        orchestrator: SDNOrchestrator,
    ):
        self.config = config
        self.orchestrator = orchestrator
        # Reuse the SAME pipeline instances
        self.routing = orchestrator.routing
        self.spectrum = orchestrator.spectrum

    def get_path_options(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> list[PathOption]:
        """
        Get candidate paths with feasibility info.

        Uses the SAME routing and spectrum pipelines as simulation.
        No duplicated logic.
        """
        # Use real routing pipeline
        route_result = self.routing.find_routes(
            source, destination, bandwidth_gbps, network_state
        )

        options = []
        for i, path in enumerate(route_result.paths):
            # Use real spectrum pipeline for feasibility check
            spectrum_result = self.spectrum.find_spectrum(
                path, route_result.modulations[i], bandwidth_gbps, network_state
            )

            # Compute congestion metric
            congestion = self._compute_path_congestion(path, network_state)

            options.append(PathOption(
                path_index=i,
                path=path,
                weight_km=route_result.weights_km[i],
                is_feasible=spectrum_result.is_free,
                modulation=spectrum_result.modulation if spectrum_result.is_free else None,
                slots_needed=spectrum_result.slots_needed if spectrum_result.is_free else None,
                congestion=congestion,
            ))

        return options

    def get_action_mask(self, options: list[PathOption]) -> list[bool]:
        """Get mask indicating which actions (path indices) are valid."""
        return [opt.is_feasible for opt in options]

    def apply_action(
        self,
        action: int,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> tuple[AllocationResult, float]:
        """
        Apply RL action using the real orchestrator.

        Returns (result, reward).
        """
        if action < 0 or action >= len(options):
            return AllocationResult(success=False), self.config.penalty

        chosen = options[action]

        if not chosen.is_feasible:
            return AllocationResult(success=False), self.config.penalty

        # Use the REAL orchestrator with forced path
        result = self.orchestrator.handle_arrival(
            request,
            network_state,
            forced_path=chosen.path,
        )

        # Compute reward
        reward = self.config.reward if result.success else self.config.penalty

        return result, reward

    def _compute_path_congestion(self, path: list[str], network_state: NetworkState) -> float:
        """Compute average congestion level across path links."""
        total_slots = 0
        used_slots = 0

        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            link_spectrum = network_state.get_link_spectrum(link)

            for band in self.config.band_list:
                matrix = link_spectrum.cores_matrix[band]
                total_slots += matrix.size
                used_slots += np.count_nonzero(matrix)

        return used_slots / total_slots if total_slots > 0 else 0.0
```

### Code Example: Unified Gymnasium Environment

**NEW FILE: fusion/modules/rl/gymnasium_envs/unified_sim_env.py**

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Optional

from fusion.domain.config import SimulationConfig
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.core.pipeline_factory import PipelineFactory
from fusion.modules.rl.env_adapter import RLSimulationAdapter, PathOption


class UnifiedSimEnv(gym.Env):
    """
    Gymnasium environment using unified simulation architecture.

    Key difference from old SimEnv:
    - Uses RLSimulationAdapter which delegates to real pipelines
    - No mock_handle_arrival() or duplicated feasibility checking
    - Same code path as non-RL simulation
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: SimulationConfig, topology_path: str):
        super().__init__()

        self.config = config
        self._topology = self._load_topology(topology_path)

        # Action space: select one of k paths
        self.action_space = spaces.Discrete(config.k_paths)

        # Observation space: dictionary of features
        self.observation_space = self._build_observation_space()

        # Will be initialized in reset()
        self._network_state: Optional[NetworkState] = None
        self._orchestrator = None
        self._adapter: Optional[RLSimulationAdapter] = None
        self._requests: list[Request] = []
        self._current_idx = 0
        self._path_options: list[PathOption] = []

    def _build_observation_space(self) -> spaces.Dict:
        """Build observation space matching old format for compatibility."""
        num_nodes = len(self._topology.nodes())
        k_paths = self.config.k_paths

        return spaces.Dict({
            "source": spaces.MultiBinary(num_nodes),
            "destination": spaces.MultiBinary(num_nodes),
            "slots_needed": spaces.Box(low=-1, high=1000, shape=(k_paths,), dtype=np.int32),
            "path_lengths": spaces.Box(low=0, high=10, shape=(k_paths,), dtype=np.float32),
            "paths_cong": spaces.Box(low=0, high=1, shape=(k_paths,), dtype=np.float32),
            "available_slots": spaces.Box(low=0, high=1, shape=(k_paths,), dtype=np.float32),
        })

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        # Create fresh network state
        self._network_state = NetworkState(self._topology, self.config)

        # Create orchestrator and adapter
        self._orchestrator = PipelineFactory.create_orchestrator(self.config)
        self._adapter = RLSimulationAdapter(self.config, self._orchestrator)

        # Generate requests
        self._requests = self._generate_requests(seed)
        self._current_idx = 0

        # Get options for first request
        self._path_options = self._get_current_options()

        obs = self._build_observation()
        info = self._build_info()

        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        request = self._requests[self._current_idx]

        # Apply action using adapter (which uses real orchestrator)
        result, reward = self._adapter.apply_action(
            action, request, self._path_options, self._network_state
        )

        # Handle any releases that should occur at this time
        self._process_releases()

        # Move to next request
        self._current_idx += 1
        terminated = self._current_idx >= len(self._requests)

        if not terminated:
            self._path_options = self._get_current_options()

        obs = self._build_observation()
        info = self._build_info()
        info["allocation_result"] = result

        return obs, reward, terminated, False, info

    def _get_current_options(self) -> list[PathOption]:
        """Get path options for current request."""
        request = self._requests[self._current_idx]
        return self._adapter.get_path_options(
            request.source, request.destination,
            request.bandwidth_gbps, self._network_state
        )

    def _build_observation(self) -> dict:
        """Build observation dict from current state and path options."""
        if self._current_idx >= len(self._requests):
            return self._empty_observation()

        request = self._requests[self._current_idx]

        # Source/destination one-hot
        source_oh = np.zeros(len(self._topology.nodes()), dtype=np.int8)
        dest_oh = np.zeros(len(self._topology.nodes()), dtype=np.int8)
        source_idx = list(self._topology.nodes()).index(request.source)
        dest_idx = list(self._topology.nodes()).index(request.destination)
        source_oh[source_idx] = 1
        dest_oh[dest_idx] = 1

        # Path-level features
        slots_needed = np.full(self.config.k_paths, -1, dtype=np.int32)
        path_lengths = np.zeros(self.config.k_paths, dtype=np.float32)
        paths_cong = np.zeros(self.config.k_paths, dtype=np.float32)
        available = np.zeros(self.config.k_paths, dtype=np.float32)

        for opt in self._path_options:
            i = opt.path_index
            if i < self.config.k_paths:
                slots_needed[i] = opt.slots_needed if opt.slots_needed else -1
                path_lengths[i] = opt.weight_km / 1000.0  # Normalize
                paths_cong[i] = opt.congestion
                available[i] = 1.0 if opt.is_feasible else 0.0

        return {
            "source": source_oh,
            "destination": dest_oh,
            "slots_needed": slots_needed,
            "path_lengths": path_lengths,
            "paths_cong": paths_cong,
            "available_slots": available,
        }

    def _build_info(self) -> dict:
        """Build info dict including action mask."""
        mask = self._adapter.get_action_mask(self._path_options) if self._path_options else []
        # Pad to k_paths
        while len(mask) < self.config.k_paths:
            mask.append(False)
        return {"action_mask": np.array(mask, dtype=bool)}
```

---

## Phase 5: ML Control + Protection Integration

### Objectives
- Introduce `ControlPolicy` abstraction
- Integrate 1+1 protection into pipeline architecture
- Support ML-based control policies

### Constraints
- RL continues to work
- Protection logic reuses existing algorithms
- ML policies have same interface as RL

### Files Created

```
fusion/interfaces/
└── control_policy.py    # ControlPolicy protocol

fusion/pipelines/
└── protection_pipeline.py  # ProtectionPipeline

fusion/policies/
├── __init__.py
├── heuristic_policy.py   # FirstFeasiblePolicy, ShortestPathPolicy
└── ml_policy.py          # MLControlPolicy
```

### Code: ControlPolicy Abstraction

**NEW FILE: fusion/interfaces/control_policy.py**

```python
from typing import Protocol, Any
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.modules.rl.env_adapter import PathOption


class ControlPolicy(Protocol):
    """
    Abstraction for control policies.

    Can be implemented by:
    - RL agents (PPO, DQN, etc.)
    - ML-based policies (classifiers, neural networks)
    - Heuristic policies (first-fit, shortest-path)
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
            request: The request to serve
            options: Available path options with feasibility
            network_state: Current network state

        Returns:
            Action index (path to use), or -1 if no valid action
        """
        ...

    def update(self, request: Request, action: int, reward: float) -> None:
        """Optional: Update policy based on experience."""
        ...


class HeuristicPolicy(ControlPolicy):
    """Base class for heuristic policies."""

    def update(self, request: Request, action: int, reward: float) -> None:
        pass  # Heuristics don't learn


class FirstFeasiblePolicy(HeuristicPolicy):
    """Select first feasible path."""

    def select_action(
        self, request: Request, options: list[PathOption], network_state: NetworkState
    ) -> int:
        for opt in options:
            if opt.is_feasible:
                return opt.path_index
        return -1


class ShortestFeasiblePolicy(HeuristicPolicy):
    """Select shortest feasible path."""

    def select_action(
        self, request: Request, options: list[PathOption], network_state: NetworkState
    ) -> int:
        feasible = [opt for opt in options if opt.is_feasible]
        if not feasible:
            return -1
        return min(feasible, key=lambda x: x.weight_km).path_index


class LeastCongestedPolicy(HeuristicPolicy):
    """Select least congested feasible path."""

    def select_action(
        self, request: Request, options: list[PathOption], network_state: NetworkState
    ) -> int:
        feasible = [opt for opt in options if opt.is_feasible]
        if not feasible:
            return -1
        return min(feasible, key=lambda x: x.congestion).path_index
```

### Code: ML Control Policy

**NEW FILE: fusion/policies/ml_policy.py**

```python
import torch
import numpy as np
from typing import Optional

from fusion.interfaces.control_policy import ControlPolicy
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.modules.rl.env_adapter import PathOption


class MLControlPolicy(ControlPolicy):
    """
    ML-based control policy using a trained neural network.

    The network sees the same features as RL (paths, feasibility, congestion)
    and outputs action probabilities or Q-values.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, path: str) -> torch.nn.Module:
        """Load trained model from path."""
        model = torch.load(path, map_location=self.device)
        return model

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        # Build feature vector (same format as RL observations)
        features = self._build_features(request, options, network_state)

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Get model output (logits or Q-values)
            output = self.model(features_tensor)

            # Apply action mask
            mask = torch.BoolTensor([opt.is_feasible for opt in options])
            output[:, ~mask] = float('-inf')

            # Select best action
            action = output.argmax(dim=1).item()

        return action if options[action].is_feasible else -1

    def _build_features(
        self, request: Request, options: list[PathOption], network_state: NetworkState
    ) -> np.ndarray:
        """Build feature vector from options."""
        features = []

        for opt in options:
            features.extend([
                opt.weight_km / 1000.0,
                opt.congestion,
                1.0 if opt.is_feasible else 0.0,
                opt.slots_needed / 100.0 if opt.slots_needed else 0.0,
            ])

        return np.array(features, dtype=np.float32)

    def update(self, request: Request, action: int, reward: float) -> None:
        # ML policy is typically pre-trained; no online updates
        pass
```

### Code: Protection as Pipeline

**NEW FILE: fusion/pipelines/protection_pipeline.py**

```python
from typing import Optional
from fusion.domain.config import SimulationConfig
from fusion.domain.request import Request
from fusion.domain.network_state import NetworkState
from fusion.domain.results import RouteResult, AllocationResult
from fusion.interfaces.pipelines import RoutingPipeline, SpectrumPipeline, SNRPipeline


class ProtectionPipeline:
    """
    Pipeline for 1+1 protection handling.

    Integrates with routing (for disjoint paths) and spectrum
    (for common spectrum allocation on both paths).
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.switchover_time_ms = 50.0  # Default protection switchover time

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

        try:
            disjoint_paths = list(nx.edge_disjoint_paths(topology, source, destination))
        except nx.NetworkXNoPath:
            return RouteResult(paths=[], modulations=[], weights_km=[])

        if len(disjoint_paths) < 2:
            return RouteResult(paths=[], modulations=[], weights_km=[])

        # Pair up disjoint paths
        paths = []
        backup_paths = []
        modulations = []
        backup_modulations = []
        weights = []

        for i in range(0, len(disjoint_paths) - 1, 2):
            primary = disjoint_paths[i]
            backup = disjoint_paths[i + 1]

            primary_mods = self._get_valid_modulations(primary, bandwidth_gbps, network_state)
            backup_mods = self._get_valid_modulations(backup, bandwidth_gbps, network_state)

            if not any(primary_mods) or not any(backup_mods):
                continue

            paths.append(primary)
            backup_paths.append(backup)
            modulations.append(primary_mods)
            backup_modulations.append(backup_mods)
            weights.append(self._compute_path_weight(primary, network_state))

        return RouteResult(
            paths=paths,
            modulations=modulations,
            weights_km=weights,
            backup_paths=backup_paths,
            backup_modulations=backup_modulations,
        )

    def allocate_protected(
        self,
        request: Request,
        primary_path: list[str],
        backup_path: list[str],
        spectrum_result: 'SpectrumResult',
        path_weight_km: float,
        network_state: NetworkState,
        snr_pipeline: Optional[SNRPipeline],
    ) -> AllocationResult:
        """Allocate spectrum on both primary and backup paths."""

        # Create protected lightpath
        lightpath = network_state.create_lightpath(
            path=primary_path,
            start_slot=spectrum_result.start_slot,
            end_slot=spectrum_result.end_slot,
            core=spectrum_result.core,
            band=spectrum_result.band,
            modulation=spectrum_result.modulation,
            bandwidth_gbps=request.bandwidth_gbps,
            path_weight_km=path_weight_km,
            backup_path=backup_path,  # This triggers allocation on both paths
        )

        # SNR validation on both paths
        if snr_pipeline:
            if not snr_pipeline.validate(lightpath, network_state):
                network_state.release_lightpath(lightpath.lightpath_id)
                return AllocationResult(success=False, block_reason=BlockReason.SNR_FAILURE)

        # Link to request
        lightpath.request_allocations[request.request_id] = request.bandwidth_gbps
        request.lightpath_ids.append(lightpath.lightpath_id)
        request.status = RequestStatus.ROUTED

        return AllocationResult(
            success=True,
            lightpaths_created=[lightpath.lightpath_id],
            total_bandwidth_allocated_gbps=request.bandwidth_gbps,
            is_protected=True,
        )

    def handle_failure(
        self,
        failed_link: tuple[str, str],
        network_state: NetworkState,
    ) -> list[dict]:
        """Handle link failure - switch affected lightpaths to backup."""

        affected = network_state.get_lightpaths_on_link(failed_link)
        switchover_actions = []

        for lp in affected:
            if lp.is_protected and lp.active_path == "primary":
                if self._path_contains_link(lp.path, failed_link):
                    lp.active_path = "backup"
                    switchover_actions.append({
                        "lightpath_id": lp.lightpath_id,
                        "action": "switchover",
                        "from": "primary",
                        "to": "backup",
                        "latency_ms": self.switchover_time_ms,
                    })

        return switchover_actions
```

---

## Phase 6: Legacy Removal

### Objectives
- Remove legacy compatibility properties
- Remove old SDNController (or mark deprecated)
- Clean up unused Props classes
- Final documentation update

### Constraints
- All tests must pass
- No functionality regression
- Clear upgrade guide for users

### Entry Criteria
- All previous phases complete
- No code using legacy properties (verified by grep)
- `run_comparison.py` passes with new architecture

### Exit Criteria
- Legacy properties removed
- Old code paths removed or clearly deprecated
- Documentation updated
- Clean codebase

### Files Modified

```
fusion/domain/network_state.py  # Remove legacy properties
fusion/core/properties.py       # Mark deprecated or remove
fusion/core/sdn_controller.py   # Mark deprecated
```

### Removal Checklist

```markdown
## Legacy Removal Checklist

### network_spectrum_dict property
- [ ] Grep returns no usages outside NetworkState
- [ ] All spectrum reads use NetworkState methods
- [ ] All spectrum writes use NetworkState methods
- [ ] Property removed from NetworkState

### lightpath_status_dict property
- [ ] Grep returns no usages outside NetworkState
- [ ] Grooming uses get_lightpaths_between()
- [ ] Stats use iteration over _lightpaths
- [ ] Property removed from NetworkState

### SDNProps class
- [ ] No code instantiates SDNProps
- [ ] All request state uses Request class
- [ ] All result state uses AllocationResult
- [ ] Class removed or marked deprecated

### engine_props dict
- [ ] SimulationConfig used everywhere
- [ ] to_engine_props() only for legacy CLI
- [ ] Document migration path for users

### Old SDNController
- [ ] All simulation runs use orchestrator
- [ ] Feature flag removed
- [ ] Class removed or moved to legacy/
```

---

# Section 5: RL + SB3 Integration Details

## 5.1 Complete RL Adapter API

```python
class RLSimulationAdapter:
    """RL-friendly interface using real simulation pipelines."""

    def __init__(self, config: SimulationConfig, orchestrator: SDNOrchestrator):
        """
        Initialize adapter.

        Args:
            config: Simulation configuration
            orchestrator: The real orchestrator (shares pipelines)
        """

    def get_path_options(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> list[PathOption]:
        """
        Get candidate paths with feasibility.

        Uses real routing and spectrum pipelines.
        No duplication with simulation.

        Returns:
            List of PathOption with path, feasibility, metrics
        """

    def get_action_mask(self, options: list[PathOption]) -> list[bool]:
        """
        Get boolean mask for valid actions.

        Used for action masking in SB3 policies.
        """

    def apply_action(
        self,
        action: int,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> tuple[AllocationResult, float]:
        """
        Apply selected action using real orchestrator.

        Returns:
            (result, reward) tuple
        """

    def compute_observation(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> dict[str, np.ndarray]:
        """
        Build observation dict for SB3.

        Compatible with existing observation spaces.
        """
```

## 5.2 SB3 Training Integration

```python
# Example SB3 training script using new architecture

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from fusion.domain.config import SimulationConfig
from fusion.modules.rl.gymnasium_envs.unified_sim_env import UnifiedSimEnv


def train_ppo(config_path: str, total_timesteps: int = 100000):
    # Load config
    config = SimulationConfig.from_yaml(config_path)

    # Create environment
    env = UnifiedSimEnv(config, topology_path="networks/USbackbone60.json")

    # Create model with action masking
    policy_kwargs = {
        "net_arch": [64, 64],
    }

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/ppo/",
    )

    # Evaluation callback
    eval_env = UnifiedSimEnv(config, topology_path="networks/USbackbone60.json")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=5000,
    )

    # Train
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save
    model.save("models/ppo_usbackbone60")

    return model


def evaluate_model(model_path: str, config_path: str, n_episodes: int = 10):
    config = SimulationConfig.from_yaml(config_path)
    env = UnifiedSimEnv(config, topology_path="networks/USbackbone60.json")
    model = PPO.load(model_path)

    total_reward = 0
    total_blocked = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Get action with masking
            action_masks = info.get("action_mask", None)
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if info.get("allocation_result") and not info["allocation_result"].success:
                total_blocked += 1

        total_reward += episode_reward

    avg_reward = total_reward / n_episodes
    blocking_rate = total_blocked / (n_episodes * config.num_requests)

    return {"avg_reward": avg_reward, "blocking_rate": blocking_rate}
```

## 5.3 Observation and Action Spaces

```python
# Observation space structure (matching existing format)
observation_space = spaces.Dict({
    # Node encodings
    "source": spaces.MultiBinary(num_nodes),
    "destination": spaces.MultiBinary(num_nodes),

    # Path-level features (for k_paths)
    "slots_needed": spaces.Box(low=-1, high=1000, shape=(k_paths,), dtype=np.int32),
    "path_lengths": spaces.Box(low=0, high=10, shape=(k_paths,), dtype=np.float32),
    "paths_cong": spaces.Box(low=0, high=1, shape=(k_paths,), dtype=np.float32),
    "available_slots": spaces.Box(low=0, high=1, shape=(k_paths,), dtype=np.float32),
})

# Action space
action_space = spaces.Discrete(k_paths)

# Reward computation (from AllocationResult)
def compute_reward(result: AllocationResult, config: SimulationConfig) -> float:
    if result.success:
        base_reward = config.reward  # e.g., 1.0
        # Optional: bonus for efficiency
        if result.is_groomed:
            base_reward *= 1.1  # 10% bonus for reusing lightpaths
        return base_reward
    else:
        return config.penalty  # e.g., -1.0
```

## 5.4 Configuration for Different Modes

```python
# RL with plain KSP (no grooming/slicing)
config_ksp = SimulationConfig(
    route_method="k_shortest_path",
    grooming_enabled=False,
    slicing_enabled=False,
    snr_enabled=False,
    # ... other settings
)

# RL with grooming + SNR
config_grooming = SimulationConfig(
    route_method="k_shortest_path",
    grooming_enabled=True,
    slicing_enabled=False,
    snr_enabled=True,
    # ... other settings
)

# RL with all features
config_full = SimulationConfig(
    route_method="k_shortest_path",
    grooming_enabled=True,
    slicing_enabled=True,
    max_slices=4,
    snr_enabled=True,
    snr_recheck=True,
    # ... other settings
)

# Same UnifiedSimEnv works for all configurations
# The orchestrator automatically enables/disables features
env_ksp = UnifiedSimEnv(config_ksp, ...)
env_grooming = UnifiedSimEnv(config_grooming, ...)
env_full = UnifiedSimEnv(config_full, ...)
```

---

# Section 6: ML Control and Protection Integration

## 6.1 ControlPolicy Interface

The `ControlPolicy` protocol unifies RL, ML, and heuristic policies:

```python
from typing import Protocol

class ControlPolicy(Protocol):
    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """Select action. Returns path index or -1."""
        ...

    def update(self, request: Request, action: int, reward: float) -> None:
        """Optional learning update."""
        ...
```

**Implementations**:
- `RLPolicy`: Wraps SB3 model, calls `model.predict()`
- `MLControlPolicy`: Uses pre-trained neural network
- `FirstFeasiblePolicy`: Heuristic - first valid path
- `ShortestFeasiblePolicy`: Heuristic - shortest valid path
- `LeastCongestedPolicy`: Heuristic - least congested valid path

## 6.2 Protection Pipeline Placement

**Protection is a specialized RoutingPipeline**, not a separate stage:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SDNOrchestrator                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ If route_method == "1plus1_protection":                                │
│   routing = ProtectedRoutingPipeline (finds disjoint pairs)            │
│   spectrum = ProtectedSpectrumPipeline (finds common slots)            │
│ Else:                                                                   │
│   routing = KSPRoutingPipeline                                         │
│   spectrum = FirstFitSpectrumPipeline                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

The `PipelineFactory` selects the appropriate pipeline based on config:

```python
class PipelineFactory:
    @staticmethod
    def create_routing(config: SimulationConfig) -> RoutingPipeline:
        if config.route_method == "1plus1_protection":
            return ProtectedRoutingPipeline(config)
        elif config.route_method == "k_shortest_path":
            return KSPRoutingPipeline(config)
        # ... etc

    @staticmethod
    def create_spectrum(config: SimulationConfig) -> SpectrumPipeline:
        if config.route_method == "1plus1_protection":
            # Protection requires finding common spectrum on two paths
            return ProtectedSpectrumPipeline(config)
        elif config.allocation_method == "first_fit":
            return FirstFitSpectrumPipeline(config)
        # ... etc
```

## 6.3 ML Policy Example

```python
class MLControlPolicy(ControlPolicy):
    """Use trained classifier/regressor for path selection."""

    def __init__(self, model_path: str):
        self.model = torch.load(model_path)
        self.model.eval()

    def select_action(
        self, request: Request, options: list[PathOption], network_state: NetworkState
    ) -> int:
        features = self._extract_features(request, options, network_state)

        with torch.no_grad():
            logits = self.model(features)

            # Apply action mask
            mask = torch.tensor([opt.is_feasible for opt in options])
            logits[~mask] = float('-inf')

            return logits.argmax().item()

    def _extract_features(self, request, options, network_state):
        # Same features as RL observation
        return torch.tensor([
            [opt.weight_km, opt.congestion, opt.slots_needed or 0]
            for opt in options
        ])
```

---

# Section 7: SDNOrchestrator Design Rules

## 7.1 Allowed Logic in SDNOrchestrator

**ALLOWED**:
1. **Stage sequencing**: Deciding which pipeline to call next
2. **Feature checking**: `if self.grooming and self.config.grooming_enabled`
3. **Result combination**: Merging groomed + newly allocated lightpaths
4. **Rollback coordination**: Calling `release_lightpath()` on failure
5. **Global error handling**: Catching exceptions, returning BlockReason

**FORBIDDEN**:
1. **Algorithm implementation**: No K-shortest-path code, no first-fit code
2. **Feature-specific logic**: No grooming bandwidth calculations inside orchestrator
3. **Data structure knowledge**: No direct access to `cores_matrix` or `lightpath_status_dict` internals
4. **Conditional branching on data values**: No `if modulation == "QPSK"` type checks

## 7.2 Examples of Good and Bad Changes

### GOOD CHANGE: Adding QoS Pipeline

```python
# In SDNOrchestrator.__init__
def __init__(self, config: SimulationConfig, pipelines: PipelineSet):
    self.config = config
    self.routing = pipelines.routing
    self.spectrum = pipelines.spectrum
    self.grooming = pipelines.grooming
    self.snr = pipelines.snr
    self.slicing = pipelines.slicing
    self.qos = pipelines.qos  # NEW: QoS pipeline

# In handle_arrival - 3 lines added
def handle_arrival(self, request, network_state, forced_path=None):
    # ... existing grooming/routing code ...

    # NEW: Apply QoS prioritization after routing
    if self.qos and self.config.qos_enabled:
        route_result = self.qos.prioritize(route_result, request)

    # ... rest of existing code unchanged ...
```

**Why this is GOOD**:
- Change is localized to orchestrator + new pipeline
- No algorithm details in orchestrator
- Other pipelines unchanged
- Clear enable/disable via config

### BAD CHANGE: Slicing Logic in Orchestrator

```python
# BAD: This puts slicing algorithm logic inside orchestrator
def handle_arrival(self, request, network_state, forced_path=None):
    # ... routing ...

    for path in route_result.paths:
        spectrum_result = self.spectrum.find_spectrum(path, mods, bw, network_state)

        if not spectrum_result.is_free:
            # BAD: Slicing algorithm logic directly in orchestrator!
            slice_bandwidth = request.bandwidth_gbps // 4
            if slice_bandwidth >= 50:
                for slice_idx in range(4):
                    slice_mods = self._get_mods_for_bandwidth(slice_bandwidth, path)  # BAD
                    slice_result = self.spectrum.find_spectrum(path, slice_mods, slice_bandwidth, network_state)
                    if slice_result.is_free:
                        # BAD: More slicing logic here
                        lp = network_state.create_lightpath(...)
                        if self.snr and not self.snr.validate(lp, network_state):
                            network_state.release_lightpath(lp.lightpath_id)
                            break
                        # ... etc
```

**Why this is BAD**:
- Slicing algorithm (4 slices, 50 Gbps minimum) is hardcoded
- `_get_mods_for_bandwidth` is algorithm logic, not orchestration
- Changes to slicing require editing orchestrator
- Hard to test slicing independently

**CORRECT approach**:
```python
# GOOD: Delegate to SlicingPipeline
def handle_arrival(self, request, network_state, forced_path=None):
    for path in route_result.paths:
        spectrum_result = self.spectrum.find_spectrum(path, mods, bw, network_state)

        if spectrum_result.is_free:
            # ... allocate ...
        elif self.slicing and self.config.slicing_enabled:
            # GOOD: All slicing logic in the pipeline
            result = self.slicing.try_slice(request, path, mods, bw, network_state,
                                            self.spectrum, self.snr)
            if result.success:
                return result
```

## 7.3 PR Review Checklist for SDNOrchestrator

When reviewing PRs that touch `orchestrator.py`:

```markdown
## SDNOrchestrator PR Checklist

### Size Check
- [ ] Total lines in orchestrator.py < 200
- [ ] No single method > 50 lines
- [ ] Change adds < 20 lines to orchestrator

### Logic Check
- [ ] No algorithm implementations (sorting, searching, calculations)
- [ ] No direct access to numpy arrays or internal data structures
- [ ] No hardcoded thresholds or magic numbers
- [ ] No feature-specific conditionals beyond "is this feature enabled?"

### Delegation Check
- [ ] New functionality is in a pipeline, not inline
- [ ] Pipeline is created by PipelineFactory
- [ ] Pipeline has its own tests

### Interface Check
- [ ] Uses standard result types (RouteResult, SpectrumResult, etc.)
- [ ] Passes Request and NetworkState, not dicts
- [ ] Returns AllocationResult

### Rollback Check
- [ ] Failure paths call release_lightpath() appropriately
- [ ] Partial allocation is handled correctly
- [ ] No resource leaks on exceptions
```

---

# Summary

This V3 document provides:

1. **Clear legacy policy**: Transitional compatibility with removal timeline and criteria
2. **State consistency guarantees**: Single NetworkState instance, no caching anti-patterns
3. **Complete pipeline walkthroughs**: 5 scenarios with step-by-step code
4. **Phased migration plan**: 6 phases with files, code examples, entry/exit criteria
5. **RL/SB3 integration**: Full adapter API, environment implementation, training examples
6. **ML/Protection integration**: ControlPolicy abstraction, protection pipeline, ML policy example
7. **Orchestrator rules**: Allowed/forbidden logic, good/bad examples, PR checklist

Each phase is designed to be implementable independently with verifiable exit criteria. The plan maintains backward compatibility until Phase 6, allowing gradual migration without breaking existing functionality.
