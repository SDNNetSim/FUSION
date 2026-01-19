# Phase 1 Migration Checklist

Complete checklist for Phase 1: Core Domain Model implementation.

## Pre-Flight Checks

Before starting Phase 1:

- [ ] V2/V3 architecture documents reviewed and approved
- [ ] `run_comparison.py` baseline established (save current output)
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] Quality checks pass (`make check-all`)
- [ ] Branch created: `feature/v4-phase-1-domain-model`

---

## P1.1: Domain Scaffolding

### Files to Create

- [ ] `fusion/domain/__init__.py`
- [ ] `fusion/domain/config.py`

### Implementation Tasks

- [ ] Create `fusion/domain/` directory
- [ ] Implement `SimulationConfig` dataclass with all fields
- [ ] Implement `from_engine_props()` classmethod
- [ ] Implement `to_engine_props()` method
- [ ] Add docstrings with Sphinx-style formatting
- [ ] Export in `__init__.py`

### Verification

```bash
# Create test file first
touch fusion/tests/domain/__init__.py
touch fusion/tests/domain/test_config.py

# Run checks
pytest fusion/tests/domain/test_config.py -v
ruff check fusion/domain/
mypy fusion/domain/
```

### Exit Criteria

- [ ] `SimulationConfig` instantiates from sample `engine_props`
- [ ] Roundtrip: `from_engine_props` -> `to_engine_props` preserves data
- [ ] Frozen: mutation raises `FrozenInstanceError`
- [ ] All type hints present
- [ ] ruff passes
- [ ] mypy passes

---

## P1.2: Request Wrapper

### Files to Create

- [ ] `fusion/domain/request.py`

### Files to Modify

- [ ] `fusion/domain/__init__.py` (add exports)

### Implementation Tasks

- [ ] Create `RequestStatus` enum (PENDING, ROUTED, BLOCKED, RELEASED)
- [ ] Create `BlockReason` enum with all blocking reasons
- [ ] Implement `Request` dataclass with:
  - Identity fields (request_id, source, destination, bandwidth_gbps, times)
  - Allocation state (status, lightpath_ids, block_reason)
  - Feature flags (is_sliced, is_groomed, is_protected)
- [ ] Implement computed properties (is_arrival, endpoint_key, holding_time)
- [ ] Implement `from_legacy_dict()` classmethod
- [ ] Implement `to_legacy_dict()` method
- [ ] Add docstrings
- [ ] Update `__init__.py` exports

### Verification

```bash
pytest fusion/tests/domain/test_request.py -v
ruff check fusion/domain/request.py
mypy fusion/domain/request.py
```

### Exit Criteria

- [ ] All status transitions work correctly
- [ ] `endpoint_key` returns sorted tuple
- [ ] `from_legacy_dict` handles all request dict fields
- [ ] `to_legacy_dict` produces compatible output
- [ ] Enum values match legacy string constants
- [ ] ruff passes
- [ ] mypy passes

---

## P1.3: Lightpath Wrapper

### Files to Create

- [ ] `fusion/domain/lightpath.py`

### Files to Modify

- [ ] `fusion/domain/__init__.py` (add exports)

### Implementation Tasks

- [ ] Implement `Lightpath` dataclass with:
  - Identity fields (lightpath_id, path)
  - Spectrum assignment (start_slot, end_slot, core, band, modulation)
  - Capacity (total_bandwidth_gbps, remaining_bandwidth_gbps, path_weight_km)
  - Quality metrics (snr_db, xt_cost, is_degraded)
  - Protection (backup_path, is_protected, active_path)
  - Request tracking (request_allocations dict)
- [ ] Implement computed properties (endpoint_key, num_slots, num_hops, utilization)
- [ ] Implement `from_legacy_dict()` classmethod
- [ ] Implement `to_legacy_dict()` method
- [ ] Add docstrings
- [ ] Update `__init__.py` exports

### Verification

```bash
pytest fusion/tests/domain/test_lightpath.py -v
ruff check fusion/domain/lightpath.py
mypy fusion/domain/lightpath.py
```

### Exit Criteria

- [ ] Lightpath tracks request allocations correctly
- [ ] Utilization calculation is accurate
- [ ] `endpoint_key` returns sorted tuple from path endpoints
- [ ] Legacy conversion handles all fields including optional ones
- [ ] Protection fields default correctly
- [ ] ruff passes
- [ ] mypy passes

---

## P1.4: Result Objects

### Files to Create

- [ ] `fusion/domain/results.py`

### Files to Modify

- [ ] `fusion/domain/__init__.py` (add exports)

### Implementation Tasks

- [ ] Implement `RouteResult` frozen dataclass
- [ ] Implement `SpectrumResult` frozen dataclass
- [ ] Implement `GroomingResult` frozen dataclass
- [ ] Implement `SlicingResult` frozen dataclass
- [ ] Implement `SNRResult` frozen dataclass
- [ ] Implement `AllocationResult` frozen dataclass
- [ ] Add factory methods for common cases (e.g., `AllocationResult.blocked()`)
- [ ] Add docstrings
- [ ] Update `__init__.py` exports

### Result Types Summary

| Type | Key Fields | Created By |
|------|------------|------------|
| `RouteResult` | paths, modulations, weights_km, backup_paths | RoutingPipeline |
| `SpectrumResult` | is_free, start_slot, end_slot, core, band, modulation | SpectrumPipeline |
| `GroomingResult` | fully_groomed, partially_groomed, lightpaths_used, remaining_bw | GroomingPipeline |
| `SlicingResult` | success, num_slices, lightpath_ids | SlicingPipeline |
| `SNRResult` | passed, snr_db, required_snr | SNRPipeline |
| `AllocationResult` | success, lightpaths_created, block_reason, is_groomed, is_sliced | Orchestrator |

### Verification

```bash
pytest fusion/tests/domain/test_results.py -v
ruff check fusion/domain/results.py
mypy fusion/domain/results.py
```

### Exit Criteria

- [ ] All result types are frozen dataclasses
- [ ] `RouteResult.is_empty` property works
- [ ] `SpectrumResult.is_free` indicates availability
- [ ] `AllocationResult.success` is the final authority
- [ ] Factory methods simplify common patterns
- [ ] ruff passes
- [ ] mypy passes

---

## P1.5: StatsCollector Skeleton

### Files to Create

- [ ] `fusion/stats/__init__.py`
- [ ] `fusion/stats/collector.py`

### Implementation Tasks

- [ ] Create `fusion/stats/` directory
- [ ] Implement `StatsCollector` dataclass with:
  - Request counters (total, successful, blocked)
  - Block reason tracking
  - Feature counters (groomed, sliced, protected)
  - SNR value tracking
  - Modulation tracking
- [ ] Implement `record_arrival()` method
- [ ] Implement `record_snr()` method
- [ ] Implement `blocking_probability` property
- [ ] Implement `to_comparison_format()` for run_comparison.py
- [ ] Add docstrings
- [ ] Export in `__init__.py`

### Verification

```bash
# Create test file
touch fusion/tests/stats/__init__.py
touch fusion/tests/stats/test_collector.py

pytest fusion/tests/stats/test_collector.py -v
ruff check fusion/stats/
mypy fusion/stats/
```

### Exit Criteria

- [ ] `record_arrival()` updates all relevant counters
- [ ] `blocking_probability` calculates correctly
- [ ] `to_comparison_format()` matches run_comparison.py expectations
- [ ] Counters use defaultdict for safety
- [ ] ruff passes
- [ ] mypy passes

---

## Final Phase 1 Verification

### Quality Checks

```bash
# All domain tests
pytest fusion/tests/domain/ fusion/tests/stats/ -v --cov=fusion/domain --cov=fusion/stats

# Linting
ruff check fusion/domain/ fusion/stats/
ruff format --check fusion/domain/ fusion/stats/

# Type checking
mypy fusion/domain/ fusion/stats/

# Existing tests still pass
pytest tests/ -v --ignore=fusion/tests/
```

### Regression Check

```bash
# run_comparison.py must still work (no changes to existing code)
python tests/run_comparison.py --config configs/test_config.ini
```

### Documentation Build

```bash
cd docs && make html
# Verify new modules appear in API docs
```

### Coverage Requirements

| Module | Target Coverage |
|--------|-----------------|
| `fusion/domain/config.py` | 95% |
| `fusion/domain/request.py` | 95% |
| `fusion/domain/lightpath.py` | 95% |
| `fusion/domain/results.py` | 90% |
| `fusion/stats/collector.py` | 90% |

---

## Phase 1 Complete Checklist

Before moving to Phase 2:

- [ ] All micro-phases complete (P1.1 - P1.5)
- [ ] All unit tests pass with target coverage
- [ ] All quality checks pass (ruff, mypy)
- [ ] No regressions in existing tests
- [ ] `run_comparison.py` passes unchanged
- [ ] Documentation builds without warnings
- [ ] PR reviewed and approved
- [ ] Branch merged to main

---

## Rollback Plan

If Phase 1 causes issues:

1. Domain classes are additive only - no existing code modified
2. Simply delete `fusion/domain/` and `fusion/stats/` directories
3. Remove test files from `fusion/tests/domain/` and `fusion/tests/stats/`
4. Revert any `__init__.py` changes

No rollback should affect existing functionality since Phase 1 is purely additive.
