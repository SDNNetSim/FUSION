# Pipeline Implementations

## Overview

This document describes concrete pipeline implementations. For protocol definitions and general patterns, see `architecture/pipeline_interfaces.md`.

Pipelines are:
- **Composable**: Multiple pipelines can be combined without special cases
- **Stateless**: Receive `NetworkState` per call, never store it
- **Single-purpose**: Each pipeline handles one aspect of allocation

---

## Concrete Implementations

### RoutingPipeline Implementations

#### KSPRoutingPipeline

K-shortest paths routing using NetworkX.

```python
class KSPRoutingPipeline:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        if forced_path:
            return self._build_result_for_path(forced_path, bandwidth_gbps, network_state)

        topology = network_state.topology
        paths = list(nx.shortest_simple_paths(
            topology, source, destination, weight="weight"
        ))[:self.config.k_paths]

        modulations = []
        weights = []
        for path in paths:
            weight = self._compute_weight(path, topology)
            weights.append(weight)
            mods = self._get_valid_modulations(weight, bandwidth_gbps)
            modulations.append(mods)

        return RouteResult(paths=paths, modulations=modulations, weights_km=weights)
```

#### ProtectedRoutingPipeline

Finds disjoint primary+backup path pairs for 1+1 protection.

```python
class ProtectedRoutingPipeline:
    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        topology = network_state.topology

        try:
            disjoint_paths = list(nx.edge_disjoint_paths(topology, source, destination))
        except nx.NetworkXNoPath:
            return RouteResult(paths=[], modulations=[], weights_km=[])

        if len(disjoint_paths) < 2:
            return RouteResult(paths=[], modulations=[], weights_km=[])

        # Pair disjoint paths
        paths, backup_paths = [], []
        modulations, backup_modulations = [], []
        weights = []

        for i in range(0, len(disjoint_paths) - 1, 2):
            primary, backup = disjoint_paths[i], disjoint_paths[i + 1]

            primary_mods = self._get_valid_modulations(primary, bandwidth_gbps)
            backup_mods = self._get_valid_modulations(backup, bandwidth_gbps)

            if any(primary_mods) and any(backup_mods):
                paths.append(primary)
                backup_paths.append(backup)
                modulations.append(primary_mods)
                backup_modulations.append(backup_mods)
                weights.append(self._compute_weight(primary, network_state.topology))

        return RouteResult(
            paths=paths,
            modulations=modulations,
            weights_km=weights,
            backup_paths=backup_paths,
            backup_modulations=backup_modulations,
        )
```

#### LoadBalancedRoutingPipeline

Congestion-aware routing that considers current network load.

```python
class LoadBalancedRoutingPipeline:
    def find_routes(self, ...) -> RouteResult:
        # Get k-shortest paths
        paths = self._get_ksp(source, destination, network_state)

        # Score by congestion
        scored = []
        for path in paths:
            congestion = self._compute_path_congestion(path, network_state)
            weight = self._compute_weight(path, network_state.topology)
            score = weight * (1 + congestion)  # Penalize congested paths
            scored.append((path, score))

        # Sort by score
        scored.sort(key=lambda x: x[1])
        sorted_paths = [p for p, _ in scored]

        return self._build_route_result(sorted_paths, bandwidth_gbps, network_state)
```

### SpectrumPipeline Implementations

#### FirstFitSpectrumPipeline

First-fit spectrum assignment algorithm.

```python
class FirstFitSpectrumPipeline:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def find_spectrum(
        self,
        path: list[str],
        modulations: list[str | None],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        for modulation in modulations:
            if modulation is None:
                continue

            slots_needed = self._calculate_slots(bandwidth_gbps, modulation)

            for band in self.config.band_list:
                for core in range(self.config.cores_per_link):
                    for start in range(self.config.band_slots[band] - slots_needed):
                        if network_state.is_spectrum_available(
                            path, start, start + slots_needed, core, band
                        ):
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

    def find_protected_spectrum(
        self,
        primary_path: list[str],
        backup_path: list[str],
        modulations: list[str | None],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """Find common spectrum available on BOTH paths."""
        for modulation in modulations:
            if modulation is None:
                continue

            slots_needed = self._calculate_slots(bandwidth_gbps, modulation)

            for band in self.config.band_list:
                for core in range(self.config.cores_per_link):
                    for start in range(self.config.band_slots[band] - slots_needed):
                        primary_free = network_state.is_spectrum_available(
                            primary_path, start, start + slots_needed, core, band
                        )
                        backup_free = network_state.is_spectrum_available(
                            backup_path, start, start + slots_needed, core, band
                        )

                        if primary_free and backup_free:
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

#### BestFitSpectrumPipeline

Best-fit spectrum assignment (smallest sufficient gap).

```python
class BestFitSpectrumPipeline:
    def find_spectrum(self, ...) -> SpectrumResult:
        candidates = []

        for modulation in modulations:
            if modulation is None:
                continue

            slots_needed = self._calculate_slots(bandwidth_gbps, modulation)

            for band in self.config.band_list:
                for core in range(self.config.cores_per_link):
                    # Find all gaps on this band/core
                    gaps = self._find_gaps(path, band, core, network_state)

                    for gap_start, gap_end in gaps:
                        gap_size = gap_end - gap_start
                        if gap_size >= slots_needed:
                            candidates.append((
                                gap_start,
                                gap_size,
                                band,
                                core,
                                modulation,
                                slots_needed,
                            ))

        if not candidates:
            return SpectrumResult(is_free=False)

        # Select smallest sufficient gap
        best = min(candidates, key=lambda x: x[1])
        return SpectrumResult(
            is_free=True,
            start_slot=best[0],
            end_slot=best[0] + best[5],
            core=best[3],
            band=best[2],
            modulation=best[4],
            slots_needed=best[5],
        )
```

### GroomingPipeline Implementation

#### StandardGroomingPipeline

Greedy bandwidth allocation to existing lightpaths.

```python
class StandardGroomingPipeline:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def try_groom(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> GroomingResult:
        lightpaths = network_state.get_lightpaths_with_capacity(
            request.source, request.destination, min_bandwidth=1
        )

        if not lightpaths:
            return GroomingResult(
                fully_groomed=False,
                partially_groomed=False,
                bandwidth_groomed_gbps=0,
                remaining_bandwidth_gbps=request.bandwidth_gbps,
                lightpaths_used=[],
            )

        remaining_bw = request.bandwidth_gbps
        used_lightpaths = []
        forced_path = None

        # Greedy: use lightpaths with most available bandwidth first
        for lp in sorted(lightpaths, key=lambda x: -x.remaining_bandwidth_gbps):
            if remaining_bw <= 0:
                break

            allocate_bw = min(remaining_bw, lp.remaining_bandwidth_gbps)
            lp.request_allocations[request.request_id] = allocate_bw
            lp.remaining_bandwidth_gbps -= allocate_bw
            remaining_bw -= allocate_bw
            used_lightpaths.append(lp.lightpath_id)
            forced_path = lp.path

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

    def rollback(
        self,
        request: Request,
        lightpath_ids: list[int],
        network_state: NetworkState,
    ) -> None:
        for lp_id in lightpath_ids:
            lp = network_state.get_lightpath(lp_id)
            if lp and request.request_id in lp.request_allocations:
                bw = lp.request_allocations.pop(request.request_id)
                lp.remaining_bandwidth_gbps += bw
```

### SlicingPipeline Implementation

#### StandardSlicingPipeline

Equal-bandwidth slicing.

```python
class StandardSlicingPipeline:
    def __init__(self, config: SimulationConfig):
        self.config = config

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
        for num_slices in range(2, self.config.max_slices + 1):
            slice_bw = bandwidth_gbps // num_slices

            slice_mods = self._get_modulations_for_bandwidth(slice_bw, path)
            if not any(slice_mods):
                continue

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

                if snr_pipeline:
                    snr_result = snr_pipeline.validate(lightpath, network_state)
                    if not snr_result.passed:
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
                # Rollback slices created so far
                for lp_id in created_lightpaths:
                    network_state.release_lightpath(lp_id)

        return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)
```

### SNRPipeline Implementation

#### GSNRPipeline

Gaussian SNR validation.

```python
class GSNRPipeline:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def validate(
        self,
        lightpath: Lightpath,
        network_state: NetworkState,
    ) -> SNRResult:
        snr_db = self._calculate_gsnr(
            path=lightpath.path,
            modulation=lightpath.modulation,
            band=lightpath.band,
            core=lightpath.core,
            start_slot=lightpath.start_slot,
            end_slot=lightpath.end_slot,
            network_state=network_state,
        )

        required_snr = self.config.snr_thresholds[lightpath.modulation]
        passed = snr_db >= required_snr

        if passed:
            lightpath.snr_db = snr_db

        return SNRResult(
            passed=passed,
            snr_db=snr_db,
            required_snr_db=required_snr,
        )

    def validate_protected(
        self,
        lightpath: Lightpath,
        network_state: NetworkState,
    ) -> SNRResult:
        # Validate primary path
        primary_result = self.validate(lightpath, network_state)
        if not primary_result.passed:
            return primary_result

        # Validate backup path
        if lightpath.backup_path:
            backup_snr = self._calculate_gsnr(
                path=lightpath.backup_path,
                modulation=lightpath.modulation,
                band=lightpath.band,
                core=lightpath.core,
                start_slot=lightpath.start_slot,
                end_slot=lightpath.end_slot,
                network_state=network_state,
            )
            required = self.config.snr_thresholds[lightpath.modulation]
            if backup_snr < required:
                return SNRResult(passed=False, snr_db=backup_snr, required_snr_db=required)

        return primary_result
```

### ProtectionPipeline Implementation

#### ProtectionPipeline

Handles 1+1 protection allocation and failure switchover.

```python
class ProtectionPipeline:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.switchover_time_ms = 50.0

    def allocate_protected(
        self,
        request: Request,
        primary_path: list[str],
        backup_path: list[str],
        spectrum_result: SpectrumResult,
        path_weight_km: float,
        network_state: NetworkState,
        snr_pipeline: SNRPipeline | None,
    ) -> AllocationResult:
        lightpath = network_state.create_lightpath(
            path=primary_path,
            start_slot=spectrum_result.start_slot,
            end_slot=spectrum_result.end_slot,
            core=spectrum_result.core,
            band=spectrum_result.band,
            modulation=spectrum_result.modulation,
            bandwidth_gbps=request.bandwidth_gbps,
            path_weight_km=path_weight_km,
            backup_path=backup_path,
        )

        if snr_pipeline:
            snr_result = snr_pipeline.validate_protected(lightpath, network_state)
            if not snr_result.passed:
                network_state.release_lightpath(lightpath.lightpath_id)
                return AllocationResult(success=False, block_reason=BlockReason.SNR_FAILURE)

        lightpath.request_allocations[request.request_id] = request.bandwidth_gbps
        request.lightpath_ids.append(lightpath.lightpath_id)

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
        affected = network_state.get_lightpaths_on_link(failed_link)
        switchover_actions = []

        for lp in affected:
            if lp.is_protected and lp.active_path == "primary":
                if self._path_contains_link(lp.path, failed_link):
                    lp.active_path = "backup"
                    switchover_actions.append({
                        "lightpath_id": lp.lightpath_id,
                        "action": "switchover",
                        "latency_ms": self.switchover_time_ms,
                    })

        return switchover_actions
```

---

## How Wrappers Work

Wrappers implement the same protocol and add cross-cutting concerns:

### Logging Wrapper

```python
class LoggingRoutingPipeline:
    def __init__(self, inner: RoutingPipeline):
        self._inner = inner

    def find_routes(self, source, destination, bandwidth_gbps, network_state, forced_path=None):
        logger.debug(f"Finding routes: {source} -> {destination}, {bandwidth_gbps} Gbps")
        result = self._inner.find_routes(source, destination, bandwidth_gbps, network_state, forced_path)
        logger.debug(f"Found {len(result.paths)} paths")
        return result
```

### Metrics Wrapper

```python
class MetricsSpectrumPipeline:
    def __init__(self, inner: SpectrumPipeline, stats: StatsCollector):
        self._inner = inner
        self._stats = stats

    def find_spectrum(self, path, modulations, bandwidth_gbps, network_state):
        start = time.perf_counter()
        result = self._inner.find_spectrum(path, modulations, bandwidth_gbps, network_state)
        elapsed = time.perf_counter() - start

        self._stats.record_spectrum_search_time(elapsed)
        self._stats.record_spectrum_result(result.is_free)

        return result
```

---

## Feature Composition

Features combine without special cases because:

1. **Pipelines are independent**: Each pipeline handles its own logic
2. **Orchestrator is thin**: Only calls pipelines, doesn't implement features
3. **Results are typed**: Each pipeline returns a specific result type
4. **No duplicated logic**: One implementation per algorithm

### Composition Examples

| Features Enabled | Pipelines Used |
|-----------------|----------------|
| KSP only | Routing + Spectrum |
| KSP + SNR | Routing + Spectrum + SNR |
| KSP + Grooming | Routing + Spectrum + Grooming |
| KSP + Grooming + SNR | Routing + Spectrum + Grooming + SNR |
| KSP + Slicing | Routing + Spectrum + Slicing |
| KSP + Grooming + Slicing + SNR | All pipelines |
| 1+1 Protection | Protected Routing + Protected Spectrum + Protection |

### No Explosion of Special Cases

Wrong approach (what we avoid):
```python
# BAD: N^2 special cases
if grooming and slicing:
    do_grooming_with_slicing()
elif grooming and snr:
    do_grooming_with_snr()
elif slicing and snr:
    do_slicing_with_snr()
# ... etc
```

Correct approach (what we do):
```python
# GOOD: Linear composition
if grooming_enabled:
    groom_result = grooming.try_groom(...)
if need_new_allocation:
    route_result = routing.find_routes(...)
    spectrum_result = spectrum.find_spectrum(...)
    if snr_enabled:
        snr.validate(...)
    if slicing_enabled and not spectrum_result.is_free:
        slicing.try_slice(...)
```

Each feature is an independent stage that can be enabled/disabled without affecting others.

---

## Avoiding RL-Style Fork

The old architecture had separate code paths for RL and simulation:
- `handle_event()` for simulation
- `mock_handle_arrival()` for RL feasibility checks

This led to duplicated logic and divergent behavior.

### New Architecture: Unified Pipeline

```
SimulationEngine uses:            RL Adapter uses:
       |                                |
       v                                v
SDNOrchestrator  <-- same -->  SDNOrchestrator
       |                                |
       v                                v
   Pipelines     <-- same -->     Pipelines
```

The `RLSimulationAdapter` uses the SAME pipelines:

```python
class RLSimulationAdapter:
    def __init__(self, config: SimulationConfig, orchestrator: SDNOrchestrator):
        self.orchestrator = orchestrator
        # Reuse the SAME pipeline instances
        self.routing = orchestrator.routing
        self.spectrum = orchestrator.spectrum

    def get_path_options(self, source, destination, bandwidth_gbps, network_state):
        # Uses real routing pipeline
        route_result = self.routing.find_routes(source, destination, bandwidth_gbps, network_state)

        # Uses real spectrum pipeline for feasibility
        options = []
        for path in route_result.paths:
            spectrum_result = self.spectrum.find_spectrum(path, ...)
            options.append(PathOption(
                is_feasible=spectrum_result.is_free,
                ...
            ))

        return options

    def apply_action(self, action, request, options, network_state):
        # Uses real orchestrator
        return self.orchestrator.handle_arrival(
            request, network_state, forced_path=options[action].path
        )
```

**Key Benefits**:
- No duplicated feasibility logic
- RL sees exactly what simulation sees
- Single source of truth for all algorithms
- Changes to pipelines automatically apply to RL

---

## File Layout

```
fusion/pipelines/
    __init__.py
    routing_pipeline.py      # KSPRoutingPipeline, ProtectedRoutingPipeline, LoadBalancedRoutingPipeline
    spectrum_pipeline.py     # FirstFitSpectrumPipeline, BestFitSpectrumPipeline
    grooming_pipeline.py     # StandardGroomingPipeline
    slicing_pipeline.py      # StandardSlicingPipeline
    snr_pipeline.py          # GSNRPipeline, MultiBandSNRPipeline
    protection_pipeline.py   # ProtectionPipeline

fusion/core/adapters/
    __init__.py
    routing_adapter.py       # Wraps legacy Routing class
    spectrum_adapter.py      # Wraps legacy spectrum assignment
    grooming_adapter.py      # Wraps legacy grooming
    snr_adapter.py           # Wraps legacy SNR calculation
```

## Related Documentation

- `architecture/pipeline_interfaces.md` - Protocol definitions
- `architecture/orchestration.md` - SDNOrchestrator design
- `architecture/network_state.md` - NetworkState single source of truth
