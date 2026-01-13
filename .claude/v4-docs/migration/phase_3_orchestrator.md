# Phase 3: SDNOrchestrator Integration

## Overview

Phase 3 introduces the `SDNOrchestrator` and `PipelineFactory`, creating a thin coordination layer that routes requests through pipelines. This phase addresses the core "wrappers" concern: SDN as a router of pipelines, not a place where slicing/grooming logic lives.

## Prerequisites

- Phase 1 complete: Domain objects (`SimulationConfig`, `Request`, `Lightpath`, result types)
- Phase 2 complete: `NetworkState` with legacy compatibility, pipeline protocols, adapters

## Objectives

1. Create `PipelineFactory` and `PipelineSet`
2. Create `SDNOrchestrator` with pipeline coordination
3. Add feature flag to switch between old and new paths
4. Integrate `StatsCollector` with orchestrator
5. Verify with `run_comparison.py`

---

## Micro-Phases

### P3.1: PipelineFactory Scaffolding

**Goal**: Create factory that produces pipelines based on configuration.

**Files Created**:
- `fusion/core/pipeline_factory.py`
- `fusion/pipelines/__init__.py`

**Files Touched**:
- None (purely additive)

**Implementation**:

```python
# fusion/core/pipeline_factory.py

from dataclasses import dataclass
from typing import Optional

from fusion.domain.config import SimulationConfig
from fusion.interfaces.pipelines import (
    RoutingPipeline, SpectrumPipeline, GroomingPipeline,
    SNRPipeline, SlicingPipeline
)


@dataclass
class PipelineSet:
    """Container for all pipelines used by orchestrator."""
    routing: RoutingPipeline
    spectrum: SpectrumPipeline
    grooming: Optional[GroomingPipeline] = None
    snr: Optional[SNRPipeline] = None
    slicing: Optional[SlicingPipeline] = None


class PipelineFactory:
    """Factory for creating pipelines based on configuration."""

    @staticmethod
    def create_routing(config: SimulationConfig) -> RoutingPipeline:
        if config.route_method == "1plus1_protection":
            from fusion.pipelines.routing_pipeline import ProtectedRoutingPipeline
            return ProtectedRoutingPipeline(config)
        else:
            # Default: use adapter wrapping legacy routing
            from fusion.core.adapters.routing_adapter import RoutingAdapter
            return RoutingAdapter(config)

    @staticmethod
    def create_spectrum(config: SimulationConfig) -> SpectrumPipeline:
        if config.allocation_method == "best_fit":
            from fusion.pipelines.spectrum_pipeline import BestFitSpectrumPipeline
            return BestFitSpectrumPipeline(config)
        else:
            # Default: use adapter wrapping legacy spectrum assignment
            from fusion.core.adapters.spectrum_adapter import SpectrumAdapter
            return SpectrumAdapter(config)

    @staticmethod
    def create_grooming(config: SimulationConfig) -> Optional[GroomingPipeline]:
        if not config.grooming_enabled:
            return None
        from fusion.core.adapters.grooming_adapter import GroomingAdapter
        return GroomingAdapter(config)

    @staticmethod
    def create_snr(config: SimulationConfig) -> Optional[SNRPipeline]:
        if not config.snr_enabled:
            return None
        from fusion.core.adapters.snr_adapter import SNRAdapter
        return SNRAdapter(config)

    @staticmethod
    def create_slicing(config: SimulationConfig) -> Optional[SlicingPipeline]:
        if not config.slicing_enabled:
            return None
        from fusion.pipelines.slicing_pipeline import StandardSlicingPipeline
        return StandardSlicingPipeline(config)

    @classmethod
    def create_pipeline_set(cls, config: SimulationConfig) -> PipelineSet:
        return PipelineSet(
            routing=cls.create_routing(config),
            spectrum=cls.create_spectrum(config),
            grooming=cls.create_grooming(config),
            snr=cls.create_snr(config),
            slicing=cls.create_slicing(config),
        )

    @classmethod
    def create_orchestrator(cls, config: SimulationConfig) -> 'SDNOrchestrator':
        from fusion.core.orchestrator import SDNOrchestrator
        pipelines = cls.create_pipeline_set(config)
        return SDNOrchestrator(config, pipelines)
```

**Legacy Path**: Unchanged. Factory is purely additive.

**Verification**:
```bash
pytest fusion/tests/core/test_pipeline_factory.py -v
ruff check fusion/core/pipeline_factory.py
mypy fusion/core/pipeline_factory.py
```

---

### P3.2: SDNOrchestrator Creation

**Goal**: Create orchestrator that coordinates pipelines.

**Files Created**:
- `fusion/core/orchestrator.py`

**Files Touched**:
- None (purely additive)

**Implementation**:

```python
# fusion/core/orchestrator.py

from fusion.domain.config import SimulationConfig
from fusion.domain.request import Request, RequestStatus
from fusion.domain.network_state import NetworkState
from fusion.domain.results import AllocationResult, BlockReason
from fusion.core.pipeline_factory import PipelineSet


class SDNOrchestrator:
    """
    Thin coordination layer for request handling.

    RULES:
    - No algorithm logic (K-shortest-path, first-fit, SNR calculation)
    - No direct numpy access
    - No hardcoded slicing/grooming logic
    - Receives NetworkState per call, never stores it
    """

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
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> AllocationResult:
        """Handle request arrival by coordinating pipelines."""
        groomed_lightpaths = []
        remaining_bw = request.bandwidth_gbps

        # Stage 1: Grooming
        if self.grooming and self.config.grooming_enabled:
            groom_result = self.grooming.try_groom(request, network_state)

            if groom_result.fully_groomed:
                request.status = RequestStatus.ROUTED
                return AllocationResult(
                    success=True,
                    is_groomed=True,
                    lightpaths_groomed=groom_result.lightpaths_used,
                    total_bandwidth_allocated_gbps=request.bandwidth_gbps,
                )

            if groom_result.partially_groomed:
                groomed_lightpaths = groom_result.lightpaths_used
                remaining_bw = groom_result.remaining_bandwidth_gbps
                forced_path = groom_result.forced_path

        # Stage 2: Routing
        route_result = self.routing.find_routes(
            request.source, request.destination,
            remaining_bw, network_state, forced_path
        )

        if route_result.is_empty:
            return self._handle_failure(
                request, groomed_lightpaths, BlockReason.NO_ROUTE, network_state
            )

        # Stage 3: Try each path
        for path_idx, path in enumerate(route_result.paths):
            result = self._try_allocate_on_path(
                request, path,
                route_result.modulations[path_idx],
                route_result.weights_km[path_idx],
                remaining_bw, network_state
            )

            if result is not None:
                return self._combine_results(request, groomed_lightpaths, result)

        # Stage 4: All paths failed
        return self._handle_failure(
            request, groomed_lightpaths, BlockReason.NO_SPECTRUM, network_state
        )

    def handle_release(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> None:
        """Handle request release."""
        for lp_id in request.lightpath_ids:
            lp = network_state.get_lightpath(lp_id)
            if lp is None:
                continue

            if request.request_id in lp.request_allocations:
                bw = lp.request_allocations.pop(request.request_id)
                lp.remaining_bandwidth_gbps += bw

            if not lp.request_allocations:
                network_state.release_lightpath(lp_id)

        request.lightpath_ids.clear()
        request.status = RequestStatus.RELEASED

    def _try_allocate_on_path(
        self, request, path, mods, weight, bw, network_state
    ) -> AllocationResult | None:
        """Try to allocate on a single path."""
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

    def _allocate_and_validate(
        self, request, path, spectrum_result, weight, bw, network_state
    ) -> AllocationResult | None:
        """Allocate lightpath and validate SNR."""
        lightpath = network_state.create_lightpath(
            path=path,
            start_slot=spectrum_result.start_slot,
            end_slot=spectrum_result.end_slot,
            core=spectrum_result.core,
            band=spectrum_result.band,
            modulation=spectrum_result.modulation,
            bandwidth_gbps=bw,
            path_weight_km=weight,
        )

        if self.snr and self.config.snr_enabled:
            snr_result = self.snr.validate(lightpath, network_state)
            if not snr_result.passed:
                network_state.release_lightpath(lightpath.lightpath_id)
                return None

        lightpath.request_allocations[request.request_id] = bw
        request.lightpath_ids.append(lightpath.lightpath_id)

        return AllocationResult(
            success=True,
            lightpaths_created=[lightpath.lightpath_id],
            total_bandwidth_allocated_gbps=bw,
        )

    def _handle_failure(
        self, request, groomed_lps, reason, network_state
    ) -> AllocationResult:
        """Handle allocation failure."""
        if groomed_lps and self.config.can_partially_serve:
            request.status = RequestStatus.ROUTED
            return AllocationResult(
                success=True,
                lightpaths_groomed=groomed_lps,
                is_partially_groomed=True,
            )

        if groomed_lps:
            self.grooming.rollback(request, groomed_lps, network_state)

        request.status = RequestStatus.BLOCKED
        request.block_reason = reason.value
        return AllocationResult(success=False, block_reason=reason)

    def _combine_results(
        self, request, groomed_lps, alloc_result
    ) -> AllocationResult:
        """Combine groomed and allocated results."""
        request.status = RequestStatus.ROUTED
        return AllocationResult(
            success=True,
            lightpaths_created=alloc_result.lightpaths_created,
            lightpaths_groomed=groomed_lps,
            is_groomed=len(groomed_lps) > 0,
            is_partially_groomed=len(groomed_lps) > 0 and len(alloc_result.lightpaths_created) > 0,
            is_sliced=alloc_result.is_sliced,
            total_bandwidth_allocated_gbps=alloc_result.total_bandwidth_allocated_gbps,
        )
```

**Legacy Path**: Unchanged. Orchestrator is purely additive.

**Verification**:
```bash
pytest fusion/tests/core/test_orchestrator.py -v
ruff check fusion/core/orchestrator.py
mypy fusion/core/orchestrator.py
```

---

### P3.3: Feature Flag in Simulation Core

**Goal**: Add feature flag to switch between old and new paths.

**Files Touched**:
- `fusion/core/simulation.py`

**Implementation**:

```python
# In fusion/core/simulation.py

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

        # Stats (used by both paths)
        self._stats = StatsCollector(self._config)

    def handle_arrival(self, current_time: tuple[int, float]) -> None:
        request_dict = self.reqs_dict[current_time]

        if self._use_orchestrator:
            # New path
            request = Request.from_legacy_dict(current_time, request_dict, self._config)
            result = self._orchestrator.handle_arrival(request, self._network_state)
            self._update_stats_from_result(request, result)
        else:
            # Old path (unchanged)
            self.sdn_obj.handle_event(request_dict, 'arrival')
            self._update_stats_from_sdn_props()

    def handle_release(self, current_time: tuple[int, float]) -> None:
        request_dict = self.reqs_dict[current_time]

        if self._use_orchestrator:
            # New path
            request = self._requests.get(current_time)
            if request:
                self._orchestrator.handle_release(request, self._network_state)
        else:
            # Old path (unchanged)
            self.sdn_obj.handle_event(request_dict, 'release')
```

**Legacy Path**: Fully preserved. `use_orchestrator=False` (default) uses old path.

**Verification**:
```bash
# Test old path still works
pytest tests/integration/ -v

# Test new path
USE_ORCHESTRATOR=1 pytest tests/integration/ -v
```

---

### P3.4: StatsCollector Integration

**Goal**: Integrate `StatsCollector` with orchestrator results.

**Files Touched**:
- `fusion/core/simulation.py`
- `fusion/stats/collector.py`

**Implementation**:

```python
# In fusion/stats/collector.py

from fusion.domain.results import AllocationResult

class StatsCollector:
    def record_arrival(
        self,
        request: Request,
        result: AllocationResult,
        network_state: NetworkState,
    ) -> None:
        """Record stats from orchestrator result."""
        self.total_requests += 1

        if result.success:
            self.successful_requests += 1
            self.total_bandwidth_allocated += result.total_bandwidth_allocated_gbps

            if result.is_groomed:
                self.groomed_requests += 1
            if result.is_sliced:
                self.sliced_requests += 1
            if result.is_protected:
                self.protected_requests += 1
        else:
            self.blocked_requests += 1
            if result.block_reason:
                self.block_reasons[result.block_reason.value] += 1

    def record_release(self, request: Request) -> None:
        """Record release event."""
        self.released_requests += 1
```

```python
# In fusion/core/simulation.py

def _update_stats_from_result(self, request: Request, result: AllocationResult) -> None:
    """Update stats from orchestrator result."""
    self._stats.record_arrival(request, result, self._network_state)
```

**Legacy Path**: Old path uses existing stats update via `_update_stats_from_sdn_props()`.

**Verification**:
```bash
pytest fusion/tests/stats/test_collector.py -v
```

---

### P3.5: run_comparison Verification

**Goal**: Verify new path produces identical results to old path.

**Files Touched**:
- `tests/run_comparison.py` (if needed for orchestrator support)

**Implementation**:

No changes needed if `run_comparison.py` already supports engine_props configuration. Otherwise:

```python
# In tests/run_comparison.py

def run_comparison_test(config_name: str, use_orchestrator: bool = False):
    """Run comparison with optional orchestrator mode."""
    engine_props = load_config(config_name)

    # Enable orchestrator if requested
    if use_orchestrator:
        engine_props['use_orchestrator'] = True

    # Run simulation
    results = run_simulation(engine_props)

    # Compare against baseline
    baseline = load_baseline(config_name)
    compare_results(results, baseline, abs_tol=0.02)
```

**Verification**:
```bash
# Run comparison with old path
python tests/run_comparison.py

# Run comparison with new path
USE_ORCHESTRATOR=1 python tests/run_comparison.py

# Both should pass with same tolerance
```

---

## Addressing the "Wrappers" Concern

### The Original Problem

In the old architecture:
- `SDNController` contained algorithm logic (grooming calculations, slicing decisions)
- Adding features required modifying SDNController
- RL had separate code paths (`mock_handle_arrival`)
- Feature combinations led to explosion of special cases

### The Solution: SDN as Pipeline Router

The new `SDNOrchestrator`:
- **Routes** requests through pipelines
- **Does NOT** implement algorithms
- **Does NOT** contain feature-specific logic
- **Delegates** all work to pipelines

```
OLD: SDNController
     |
     +-- grooming logic (inline)
     +-- slicing logic (inline)
     +-- routing logic (inline)
     +-- spectrum logic (inline)
     +-- SNR logic (inline)
     +-- many if/else branches

NEW: SDNOrchestrator
     |
     +-- calls GroomingPipeline
     +-- calls RoutingPipeline
     +-- calls SpectrumPipeline
     +-- calls SNRPipeline
     +-- calls SlicingPipeline
     +-- thin coordination only
```

### Rules Enforced

| Rule | Enforcement |
|------|-------------|
| No algorithm logic | PR checklist, code review |
| No numpy access | Type checking (NetworkState methods only) |
| No hardcoded thresholds | Config-only values |
| < 200 lines total | CI size check |
| < 50 lines per method | CI size check |

---

## Rollback Plan

If Phase 3 causes issues:

1. Set `use_orchestrator=False` (default) - immediate rollback
2. Old SDNController path remains fully functional
3. New code is isolated in separate files

---

## Exit Criteria

- [ ] `PipelineFactory` creates correct pipelines for all config combinations
- [ ] `SDNOrchestrator` coordinates pipelines correctly
- [ ] Feature flag switches cleanly between paths
- [ ] `StatsCollector` records all metrics from orchestrator
- [ ] `run_comparison.py` passes with both paths
- [ ] No performance regression (< 5% slower)
- [ ] Code passes ruff, mypy, and tests

## File Summary

### New Files Created

| File | Purpose |
|------|---------|
| `fusion/core/pipeline_factory.py` | Factory for creating pipelines |
| `fusion/core/orchestrator.py` | Thin coordination layer |
| `fusion/pipelines/__init__.py` | Pipeline package |

### Existing Files Modified

| File | Changes |
|------|---------|
| `fusion/core/simulation.py` | Feature flag, orchestrator integration |
| `fusion/stats/collector.py` | Record from AllocationResult |

## Related Documentation

- `architecture/orchestration.md` - Orchestrator design and rules
- `architecture/pipelines.md` - Pipeline implementations
- `testing/phase_3_testing.md` - Test plan for Phase 3
- `decisions/0007-orchestrator-design.md` - Design rationale
