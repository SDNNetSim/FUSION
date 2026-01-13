# Phase 2: State Management Migration

## Overview

Phase 2 migrates state management from scattered `engine_props`, `sdn_props`, and `stats_props` dictionaries into the unified `NetworkState` class. This is a critical phase that establishes the foundation for all subsequent work.

## Prerequisites

- Phase 1 complete: Domain model classes (`SimulationConfig`, `Request`, `Lightpath`, result objects) tested and working
- All Phase 1 tests passing
- `run_comparison.py` baseline established

## Goals

1. Create `NetworkState` as the single source of truth for network state
2. Introduce `LinkSpectrum` for per-link spectrum management
3. Define pipeline protocols (interfaces)
4. Create adapter classes wrapping existing implementations
5. Maintain full backward compatibility via legacy properties

## Micro-Phases

### P2.1: NetworkState Core

**Files Created:**
- `fusion/domain/network_state.py`

**Scope:**
- `NetworkState` class with `_topology`, `_spectrum`, `_lightpaths`
- `LinkSpectrum` dataclass for per-link spectrum
- Initialization from topology and config
- Basic read methods: `get_lightpath()`, `is_spectrum_available()`

**Entry Criteria:**
- Phase 1 complete
- `SimulationConfig` tested

**Exit Criteria:**
- `NetworkState` can be instantiated with topology
- Spectrum matrices initialized correctly
- Unit tests pass

**Verification:**
```bash
pytest fusion/tests/domain/test_network_state.py -v
ruff check fusion/domain/network_state.py
mypy fusion/domain/network_state.py
```

### P2.2: NetworkState Write Methods + Legacy Compat

**Files Modified:**
- `fusion/domain/network_state.py`

**Scope:**
- Write methods: `create_lightpath()`, `release_lightpath()`
- Legacy compatibility properties: `network_spectrum_dict`, `lightpath_status_dict`
- Helper methods: `get_lightpaths_between()`, `get_lightpaths_with_capacity()`

**Entry Criteria:**
- P2.1 complete

**Exit Criteria:**
- Can create and release lightpaths
- Legacy properties return correct format
- Round-trip: create -> legacy prop -> verify data matches

**Verification:**
```bash
pytest fusion/tests/domain/test_network_state.py -v -k "create or release or legacy"
```

### P2.3: Pipeline Protocols

**Files Created:**
- `fusion/interfaces/__init__.py`
- `fusion/interfaces/pipelines.py`

**Scope:**
- `RoutingPipeline` protocol
- `SpectrumPipeline` protocol
- `GroomingPipeline` protocol
- `SNRPipeline` protocol
- `SlicingPipeline` protocol

**Entry Criteria:**
- P2.2 complete

**Exit Criteria:**
- All protocols defined with type hints
- Protocols pass mypy
- No runtime imports (protocols are just type definitions)

**Verification:**
```bash
mypy fusion/interfaces/pipelines.py --strict
```

### P2.4: Legacy Adapters

**Files Created:**
- `fusion/core/adapters/__init__.py`
- `fusion/core/adapters/routing_adapter.py`
- `fusion/core/adapters/spectrum_adapter.py`
- `fusion/core/adapters/grooming_adapter.py`
- `fusion/core/adapters/snr_adapter.py`

**Scope:**
- Adapters wrap existing implementations
- Convert legacy inputs/outputs to new formats
- Implement pipeline protocols

**Entry Criteria:**
- P2.3 complete
- Protocols defined

**Exit Criteria:**
- Each adapter implements its protocol
- Adapters produce same results as legacy code
- Integration tests pass

**Verification:**
```bash
pytest fusion/tests/adapters/ -v
```

## Before/After Examples

### Example 1: Spectrum Access

**BEFORE (Legacy):**
```python
# In sdn_controller.py
def check_spectrum_available(self, path, start, end, core, band):
    for i in range(len(path) - 1):
        link = (path[i], path[i+1])
        link_data = self.sdn_props.network_spectrum_dict[link]
        spectrum = link_data['cores_matrix'][band][core][start:end]
        if not np.all(spectrum == 0):
            return False
    return True
```

**AFTER (NetworkState):**
```python
# Direct NetworkState method
def check_spectrum_available(self, path, start, end, core, band, network_state):
    return network_state.is_spectrum_available(path, start, end, core, band)

# Inside NetworkState
def is_spectrum_available(self, path, start, end, core, band) -> bool:
    for i in range(len(path) - 1):
        link = (path[i], path[i+1])
        if not self._spectrum[link].is_range_free(band, core, start, end):
            return False
    return True
```

### Example 2: Lightpath Creation

**BEFORE (Legacy):**
```python
# In sdn_controller.py
def create_lightpath(self, path, start_slot, end_slot, core, band, mod, bw):
    lp_id = self.sdn_props.next_lightpath_id
    self.sdn_props.next_lightpath_id += 1

    # Store in dict with sorted tuple key
    key = tuple(sorted([path[0], path[-1]]))
    if key not in self.sdn_props.lightpath_status_dict:
        self.sdn_props.lightpath_status_dict[key] = {}

    self.sdn_props.lightpath_status_dict[key][lp_id] = {
        'path': path,
        'core': core,
        'band': band,
        'start_slot': start_slot,
        'end_slot': end_slot,
        'mod_format': mod,
        'lightpath_bandwidth': bw,
        'remaining_bandwidth': bw,
        'requests_dict': {},
    }

    # Allocate spectrum
    for i in range(len(path) - 1):
        link = (path[i], path[i+1])
        self.sdn_props.network_spectrum_dict[link]['cores_matrix'][band][core][start_slot:end_slot] = lp_id
        # reverse direction too
        reverse = (path[i+1], path[i])
        self.sdn_props.network_spectrum_dict[reverse]['cores_matrix'][band][core][start_slot:end_slot] = lp_id

    return lp_id
```

**AFTER (NetworkState):**
```python
# Call site
lightpath = network_state.create_lightpath(
    path=path,
    start_slot=start_slot,
    end_slot=end_slot,
    core=core,
    band=band,
    modulation=mod,
    bandwidth_gbps=bw,
    path_weight_km=weight,
)

# Inside NetworkState
def create_lightpath(self, ...) -> Lightpath:
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
    self._allocate_spectrum_on_path(path, start_slot, end_slot, core, band, lp.lightpath_id)
    return lp
```

### Example 3: Grooming Lookup

**BEFORE (Legacy):**
```python
# In grooming.py
def _find_path_max_bw(self, light_id):
    if light_id not in self.sdn_props.lightpath_status_dict:
        return None

    path_group = self.sdn_props.lightpath_status_dict[light_id]

    max_bw = 0
    best_lp_id = None
    for lp_id, lp_info in path_group.items():
        if lp_info["remaining_bandwidth"] > max_bw:
            max_bw = lp_info["remaining_bandwidth"]
            best_lp_id = lp_id

    return path_group[best_lp_id] if best_lp_id else None
```

**AFTER (NetworkState):**
```python
# In grooming pipeline
def _find_path_max_bw(self, source, destination, network_state) -> Lightpath | None:
    lightpaths = network_state.get_lightpaths_with_capacity(source, destination, min_bw=1)

    if not lightpaths:
        return None

    return max(lightpaths, key=lambda lp: lp.remaining_bandwidth_gbps, default=None)
```

### Example 4: Adapter Pattern

**Routing Adapter:**
```python
class RoutingAdapter(RoutingPipeline):
    """Adapter wrapping existing Routing class to new interface."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._engine_props = config.to_engine_props()

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        # Create legacy objects
        route_props = RoutingProps()

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
        )
```

## Migration Steps

### Step 1: Create NetworkState (P2.1)

1. Create `fusion/domain/network_state.py`
2. Implement `LinkSpectrum` dataclass
3. Implement `NetworkState.__init__` with spectrum initialization
4. Add read methods: `get_lightpath`, `is_spectrum_available`, `get_link_spectrum`
5. Write unit tests

### Step 2: Add Write Methods and Legacy Props (P2.2)

1. Add `create_lightpath()` method
2. Add `release_lightpath()` method
3. Add `_allocate_spectrum_on_path()` helper
4. Add `_release_spectrum_on_path()` helper
5. Add `network_spectrum_dict` property
6. Add `lightpath_status_dict` property
7. Write tests verifying legacy format matches expected

### Step 3: Define Protocols (P2.3)

1. Create `fusion/interfaces/__init__.py`
2. Create `fusion/interfaces/pipelines.py`
3. Define each protocol with full type hints
4. Verify with mypy --strict

### Step 4: Create Adapters (P2.4)

1. Create adapter directory structure
2. Implement `RoutingAdapter`
3. Implement `SpectrumAdapter`
4. Implement `GroomingAdapter`
5. Implement `SNRAdapter`
6. Write integration tests comparing adapter output to legacy

### Step 5: Integration Test

```bash
# Run full test suite
pytest fusion/tests/ -v

# Verify no regressions
python tests/run_comparison.py

# Check types
mypy fusion/domain/ fusion/interfaces/ fusion/core/adapters/
```

## Legacy Property Removal Checklist

These properties are TEMPORARY. Track usage for removal in Phase 5:

### `network_spectrum_dict`

Before removal, verify:
- [ ] All spectrum reads use `NetworkState.is_spectrum_available()` or `get_link_spectrum()`
- [ ] All spectrum writes use `NetworkState.create_lightpath()` / `release_lightpath()`
- [ ] No direct numpy array access outside NetworkState
- [ ] `run_comparison.py` passes without legacy property access
- [ ] `grep 'network_spectrum_dict'` returns only the property definition

### `lightpath_status_dict`

Before removal, verify:
- [ ] Grooming uses `NetworkState.get_lightpaths_with_capacity()`
- [ ] Statistics use `NetworkState.get_lightpath()` or iteration
- [ ] No code constructs `(src, dst)` sorted tuple key manually
- [ ] `run_comparison.py` passes without legacy property access
- [ ] `grep 'lightpath_status_dict'` returns only the property definition

## Rollback Plan

If Phase 2 causes issues:

1. **P2.1-P2.2 Rollback**: Delete `fusion/domain/network_state.py`. No other code depends on it yet.

2. **P2.3 Rollback**: Delete `fusion/interfaces/`. Protocols are just type definitions, no runtime impact.

3. **P2.4 Rollback**: Delete `fusion/core/adapters/`. Adapters are not used by production code until Phase 3.

## Verification Checklist

After Phase 2 completion:

- [ ] All P2.x micro-phases pass their exit criteria
- [ ] `pytest fusion/tests/domain/test_network_state.py` passes
- [ ] `pytest fusion/tests/adapters/` passes
- [ ] `mypy fusion/domain/ fusion/interfaces/` passes
- [ ] `ruff check fusion/domain/ fusion/interfaces/ fusion/core/adapters/` passes
- [ ] `run_comparison.py` still passes (no functional changes)
- [ ] Legacy properties return data matching current `sdn_props` format
- [ ] Documentation updated in `.claude/v4-docs/`

## Timeline Dependencies

```
Phase 1 Complete
       |
       v
    P2.1 NetworkState Core
       |
       v
    P2.2 Write Methods + Legacy Compat
       |
       v
    P2.3 Pipeline Protocols
       |
       v
    P2.4 Legacy Adapters
       |
       v
Phase 2 Complete --> Phase 3 (Orchestrator)
```

## Common Issues and Solutions

### Issue: Legacy format mismatch

**Symptom**: `lightpath_status_dict` returns different structure than expected.

**Solution**: Compare field by field with existing `sdn_props.lightpath_status_dict`. Ensure:
- Keys are sorted tuples: `tuple(sorted([src, dst]))`
- Nested dict keys are lightpath IDs (int)
- All expected fields present

### Issue: Spectrum not being freed

**Symptom**: After `release_lightpath`, spectrum still shows allocated.

**Solution**: Verify:
- Both directions freed: `(u,v)` and `(v,u)`
- Guard bands also cleared: values of `-lp_id`
- All links in path processed

### Issue: Adapter returns different routes

**Symptom**: `RoutingAdapter` returns different paths than legacy code.

**Solution**:
- Ensure topology reference is the same
- Check that edge weights match
- Verify k_paths parameter passed correctly
