# Phase 2 Migration Checklist

This document provides step-by-step checklists for completing Phase 2: NetworkState as Single Source of Truth.

## Overview

Phase 2 establishes `NetworkState` as the authoritative source for all mutable network state. This phase is divided into four micro-phases:

| Micro-Phase | Focus | Key Deliverable |
|-------------|-------|-----------------|
| P2.1 | NetworkState class | Core state management |
| P2.2 | Pipeline adapters | Legacy code bridging |
| P2.3 | Legacy property migration | Gradual deprecation |
| P2.4 | Cleanup and verification | Remove deprecated code |

---

## P2.1: NetworkState Core Implementation

### Objective
Create `NetworkState` class that owns all mutable network data.

### Implementation Checklist

#### Data Structures
- [ ] Create `fusion/domain/network_state.py`
- [ ] Create `fusion/domain/link_spectrum.py`
- [ ] Implement `NetworkState.__init__(topology, config)`
- [ ] Implement `LinkSpectrum` with `cores_matrix` per band
- [ ] Initialize spectrum for all links (both directions)

#### Core Methods
- [ ] `is_spectrum_available(path, start, end, core, band) -> bool`
- [ ] `create_lightpath(...) -> Lightpath`
- [ ] `release_lightpath(lightpath_id) -> None`
- [ ] `get_lightpath(lightpath_id) -> Lightpath | None`
- [ ] `get_lightpaths_between(src, dst) -> list[Lightpath]`
- [ ] `get_lightpaths_with_capacity(src, dst, min_bw) -> list[Lightpath]`
- [ ] `get_link_spectrum(link) -> LinkSpectrum`

#### State Invariants
- [ ] Lightpath IDs are unique and incrementing
- [ ] Spectrum allocation/release is atomic per path
- [ ] Both link directions updated together
- [ ] No spectrum double-allocation

### Verification Commands

```bash
# Run NetworkState unit tests
pytest fusion/tests/domain/test_network_state.py -v

# Run LinkSpectrum unit tests
pytest fusion/tests/domain/test_link_spectrum.py -v

# Type checking
mypy fusion/domain/network_state.py fusion/domain/link_spectrum.py
```

### Completion Criteria
- [ ] All unit tests pass
- [ ] mypy reports no errors
- [ ] Snapshot tests verify create/release restores state
- [ ] Code coverage >= 90%

---

## P2.2: Pipeline Adapters

### Objective
Create adapter classes that wrap legacy code with new protocol interfaces.

### Implementation Checklist

#### Routing Adapter
- [ ] Create `fusion/core/adapters/routing_adapter.py`
- [ ] `RoutingAdapter` implements `RoutingPipeline` protocol
- [ ] `find_routes()` wraps legacy `Routing.get_route()`
- [ ] Returns `RouteResult` with paths, weights, modulations
- [ ] Handles `forced_path` parameter

#### Spectrum Adapter
- [ ] Create `fusion/core/adapters/spectrum_adapter.py`
- [ ] `SpectrumAdapter` implements `SpectrumPipeline` protocol
- [ ] `find_spectrum()` wraps legacy spectrum assignment
- [ ] Returns `SpectrumResult` with slot range, core, band
- [ ] `find_protected_spectrum()` for 1+1 protection

#### Grooming Adapter
- [ ] Create `fusion/core/adapters/grooming_adapter.py`
- [ ] `GroomingAdapter` implements `GroomingPipeline` protocol
- [ ] `try_groom()` wraps legacy grooming logic
- [ ] Returns `GroomingResult` with groomed bandwidth, forced path
- [ ] `rollback()` reverses partial grooming

#### SNR Adapter
- [ ] Create `fusion/core/adapters/snr_adapter.py`
- [ ] `SNRAdapter` implements `SNRPipeline` protocol
- [ ] `validate()` wraps legacy SNR calculation
- [ ] Returns `SNRResult` with pass/fail and SNR value

### Adapter Design Rules
- [ ] Adapters receive `NetworkState` per call, never store it
- [ ] Adapters translate between legacy dict format and new types
- [ ] Adapters do NOT modify algorithm behavior
- [ ] Legacy code path unchanged when adapter used

### Verification Commands

```bash
# Run adapter tests
pytest fusion/tests/adapters/ -v

# Verify protocol compliance
python -c "
from fusion.core.adapters.routing_adapter import RoutingAdapter
from fusion.interfaces.pipelines import RoutingPipeline
assert isinstance(RoutingAdapter(...), RoutingPipeline)
"

# Legacy parity test
pytest fusion/tests/integration/test_legacy_parity.py -v
```

### Completion Criteria
- [ ] All adapters implement respective protocols
- [ ] Adapter outputs match legacy outputs exactly
- [ ] Integration tests pass with adapters in place
- [ ] No behavior change in simulation results

---

## P2.3: Legacy Property Migration

### Objective
Add legacy compatibility properties to `NetworkState` and migrate consumers.

### Implementation Checklist

#### Legacy Properties
- [ ] `NetworkState.network_spectrum_dict` property (deprecated)
- [ ] `NetworkState.lightpath_status_dict` property (deprecated)
- [ ] Properties return copies, not references
- [ ] Add deprecation warnings to properties

#### Consumer Migration (by module)

**SDNController**
- [ ] Replace `sdn_props.network_spectrum_dict` reads with `NetworkState.get_link_spectrum()`
- [ ] Replace `sdn_props.lightpath_status_dict` reads with `NetworkState.get_lightpath()`
- [ ] Replace direct spectrum allocation with `NetworkState.create_lightpath()`
- [ ] Replace direct spectrum release with `NetworkState.release_lightpath()`

**Grooming Module**
- [ ] Replace lightpath lookups with `NetworkState.get_lightpaths_between()`
- [ ] Replace capacity checks with `NetworkState.get_lightpaths_with_capacity()`
- [ ] Use `Lightpath.remaining_bandwidth_gbps` instead of dict access

**Spectrum Assignment**
- [ ] Replace `cores_matrix` direct access with `NetworkState.get_link_spectrum()`
- [ ] Use `LinkSpectrum.cores_matrix` for read-only access
- [ ] Allocation via `NetworkState.create_lightpath()` only

**SNR Calculation**
- [ ] Replace neighbor lightpath lookup with `NetworkState` queries
- [ ] Use typed `Lightpath` objects instead of dicts

### Migration Pattern

```python
# BEFORE (legacy)
key = tuple(sorted([src, dst]))
if key in sdn_props.lightpath_status_dict:
    for lp_id, lp_info in sdn_props.lightpath_status_dict[key].items():
        if lp_info["remaining_bandwidth"] >= bw:
            ...

# AFTER (new)
for lp in network_state.get_lightpaths_with_capacity(src, dst, min_bw=bw):
    ...
```

### Verification Commands

```bash
# Run integration tests
pytest fusion/tests/integration/test_state_consistency.py -v

# Run full simulation comparison
python tests/run_comparison.py --config plain_ksp

# Check for deprecated property usage
grep -r "network_spectrum_dict" fusion/ --include="*.py" | grep -v "def network_spectrum_dict"
grep -r "lightpath_status_dict" fusion/ --include="*.py" | grep -v "def lightpath_status_dict"
```

### Completion Criteria
- [ ] All modules use `NetworkState` methods for reads
- [ ] All mutations go through `NetworkState.create/release_lightpath()`
- [ ] Legacy properties only called by deprecated code paths
- [ ] Full simulation produces identical results

---

## P2.4: Cleanup and Verification

### Objective
Remove deprecated code and verify system stability.

### Cleanup Checklist

#### Remove Deprecated Code
- [ ] Remove `sdn_props.network_spectrum_dict` initialization
- [ ] Remove `sdn_props.lightpath_status_dict` initialization
- [ ] Remove direct `cores_matrix` allocations outside `NetworkState`
- [ ] Remove duplicated state synchronization code

#### Remove Legacy Properties (after all consumers migrated)
- [ ] Remove `NetworkState.network_spectrum_dict` property
- [ ] Remove `NetworkState.lightpath_status_dict` property
- [ ] Remove compatibility shims in adapters

#### Code Quality
- [ ] Run `make lint-new` - no errors
- [ ] Run `make format` - all files formatted
- [ ] Run `mypy fusion/domain/ fusion/core/adapters/` - no errors
- [ ] Update docstrings to remove deprecated references

### Final Verification

```bash
# Full quality check
make check-all

# Run all tests
pytest fusion/tests/ -v --cov=fusion/domain --cov=fusion/core/adapters

# Run comparison across multiple configs
python tests/run_comparison.py --all-configs

# Verify no deprecated property usage remains
grep -r "DEPRECATED" fusion/domain/network_state.py
# Should return empty or only historical comments
```

### Completion Criteria
- [ ] All deprecated code removed
- [ ] All tests pass (unit, integration, regression)
- [ ] Code coverage meets targets (domain: 90%, adapters: 80%)
- [ ] No regressions in simulation results
- [ ] Documentation updated

---

## Rollback Procedures

### If P2.1 Fails
```bash
git checkout -- fusion/domain/network_state.py fusion/domain/link_spectrum.py
```

### If P2.2 Fails
```bash
git checkout -- fusion/core/adapters/
# Revert to direct legacy code usage
```

### If P2.3 Migration Breaks Behavior
```bash
# Re-enable legacy code path
export FUSION_USE_LEGACY_STATE=1
python run_simulation.py ...
```

### If P2.4 Cleanup Causes Issues
```bash
# Restore legacy properties
git checkout HEAD~1 -- fusion/domain/network_state.py
```

---

## Progress Tracking

| Task | Status | Verified By | Date |
|------|--------|-------------|------|
| P2.1: NetworkState implementation | | | |
| P2.1: Unit tests pass | | | |
| P2.2: RoutingAdapter complete | | | |
| P2.2: SpectrumAdapter complete | | | |
| P2.2: GroomingAdapter complete | | | |
| P2.2: SNRAdapter complete | | | |
| P2.2: Legacy parity verified | | | |
| P2.3: SDNController migrated | | | |
| P2.3: Grooming migrated | | | |
| P2.3: Spectrum migrated | | | |
| P2.3: SNR migrated | | | |
| P2.4: Deprecated code removed | | | |
| P2.4: Full regression pass | | | |

---

## Related Documentation

- [Architecture: NetworkState](../architecture/network_state.md)
- [Architecture: Adapter Pattern](../architecture/adapter_pattern.md)
- [Testing: Phase 2 Testing](../testing/phase_2_testing.md)
- [ADR-0006: NetworkState Authority](../decisions/0006-networkstate-authority.md)
