# Migrating Legacy Code to V4 Domain Model

This guide helps existing contributors migrate code from the legacy engine-props-based patterns to the new V4 domain model and result objects.

## Prerequisites

- [Getting Started with Domain Model](./getting_started_with_domain_model.md)
- [Working with Requests and Results](./working_with_requests_and_results.md)
- Familiarity with existing FUSION codebase

## Related Documentation

- [Architecture: Domain Model](../architecture/domain_model.md)
- [Migration: Before/After Examples](../migration/before_after_examples.md)
- [ADR-0005: Legacy Compatibility](../decisions/0005-legacy-compatibility.md)

---

## Migration Philosophy

The V4 migration follows these principles:

1. **Additive first**: New code is added alongside legacy, not replacing it
2. **Transitional compatibility**: Legacy patterns work during migration
3. **Gradual adoption**: Migrate one module at a time
4. **Verification**: `run_comparison.py` validates equivalence
5. **Final removal**: Delete legacy code after full migration

---

## Quick Reference: Legacy to V4 Mapping

| Legacy Pattern | V4 Replacement |
|----------------|----------------|
| `engine_props` dict | `SimulationConfig` |
| `engine_props.get("key")` | `config.key` (typed attribute) |
| `engine_props.get("snr_type") != "None"` | `config.snr_enabled` (boolean) |
| `request_dict` in `reqs_dict` | `Request` dataclass |
| `sdn_props.source` | `request.source` |
| `sdn_props.block_reason = "x"` | `request.block_reason = "x"` |
| `sdn_props.lightpath_status_dict` | `NetworkState.get_lightpath()` |
| `sdn_props.network_spectrum_dict` | `NetworkState.get_link_spectrum()` |
| `route_props.paths_matrix` | `RouteResult.paths` |
| Mutating Props objects | Return result objects |

---

## Migration Pattern 1: Configuration Access

### Before: Dictionary Access

```python
def process_request(engine_props: dict, sdn_props):
    # String keys, no IDE support
    k_paths = engine_props.get("k_paths", 3)
    grooming_enabled = engine_props.get("is_grooming_enabled", False)

    # Fragile type checking
    snr_type = engine_props.get("snr_type")
    if snr_type and snr_type != "None" and snr_type != "":
        check_snr = True

    # Easy to typo
    bandwidth = engine_props.get("bandwith")  # Bug: misspelled
```

### After: Typed Configuration

```python
def process_request(config: SimulationConfig, request: Request):
    # IDE autocomplete, type checking
    k_paths = config.k_paths
    grooming_enabled = config.grooming_enabled

    # Explicit boolean flag
    if config.snr_enabled:
        check_snr = True

    # Typos caught at compile time
    bandwidth = config.bandwith  # Error: AttributeError
```

### Migration Steps

1. Add `from fusion.domain.config import SimulationConfig`
2. At function entry, convert: `config = SimulationConfig.from_engine_props(engine_props)`
3. Replace `engine_props.get("key")` with `config.key`
4. Replace string checks with boolean flags
5. Pass `config` to called functions instead of `engine_props`

---

## Migration Pattern 2: Request Handling

### Before: Scattered State

```python
def handle_arrival(reqs_dict, time_key, sdn_props, stats_props):
    request_dict = reqs_dict[time_key]

    # Data spread across dicts
    source = request_dict["source"]
    destination = request_dict["destination"]
    bandwidth = request_dict["bandwidth"]

    # State in sdn_props
    sdn_props.source = source
    sdn_props.destination = destination

    # Process...

    # Status tracked via multiple variables
    if blocked:
        sdn_props.block_reason = "no_spectrum"
        stats_props.blocked_requests += 1
        return False
    else:
        sdn_props.routed = True
        stats_props.successful_requests += 1
        return True
```

### After: Request Object

```python
from fusion.domain.request import Request, RequestStatus
from fusion.domain.results import AllocationResult

def handle_arrival(
    request: Request,
    network_state: NetworkState
) -> AllocationResult:
    # All data in request object
    source = request.source
    destination = request.destination
    bandwidth = request.bandwidth_gbps

    # Process...

    # Status tracked on request
    if blocked:
        request.status = RequestStatus.BLOCKED
        request.block_reason = "no_spectrum"
        return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)
    else:
        request.status = RequestStatus.ROUTED
        request.lightpath_ids.append(lp.lightpath_id)
        return AllocationResult(success=True, lightpaths_created=[lp.lightpath_id])
```

### Migration Steps

1. Add `from fusion.domain.request import Request, RequestStatus`
2. Convert at entry: `request = Request.from_legacy_dict(time_key, request_dict)`
3. Replace `request_dict["field"]` with `request.field`
4. Replace `sdn_props.block_reason = "x"` with `request.block_reason = "x"`
5. Return `AllocationResult` instead of boolean
6. Update caller to handle `AllocationResult`

---

## Migration Pattern 3: Lightpath Access

### Before: Nested Dictionary

```python
def find_grooming_candidate(sdn_props, source, dest, needed_bw):
    # Manual key construction
    key = tuple(sorted([source, dest]))

    # Dict-of-dict access
    if key not in sdn_props.lightpath_status_dict:
        return None

    lp_group = sdn_props.lightpath_status_dict[key]

    # Iterate and access nested dict
    best_lp_id = None
    max_remaining = 0
    for lp_id, lp_info in lp_group.items():
        remaining = lp_info["remaining_bandwidth"]
        if remaining >= needed_bw and remaining > max_remaining:
            max_remaining = remaining
            best_lp_id = lp_id

    return lp_group[best_lp_id] if best_lp_id else None
```

### After: NetworkState Methods

```python
from fusion.domain.lightpath import Lightpath
from fusion.domain.network_state import NetworkState

def find_grooming_candidate(
    network_state: NetworkState,
    source: str,
    dest: str,
    needed_bw: int
) -> Lightpath | None:
    # Direct method call - no manual key
    lightpaths = network_state.get_lightpaths_with_capacity(
        source, dest, min_bandwidth=needed_bw
    )

    if not lightpaths:
        return None

    # Typed objects with properties
    return max(lightpaths, key=lambda lp: lp.remaining_bandwidth_gbps)
```

### Migration Steps

1. Replace `sdn_props.lightpath_status_dict` access with `NetworkState` methods
2. Use `network_state.get_lightpath(id)` for single lookup
3. Use `network_state.get_lightpaths_between(src, dst)` for endpoint lookup
4. Use `network_state.get_lightpaths_with_capacity(src, dst, min_bw)` for grooming
5. Access `Lightpath` attributes instead of dict keys

---

## Migration Pattern 4: Spectrum Operations

### Before: Direct Array Access

```python
def is_spectrum_free(sdn_props, path, start, end, core, band):
    for i in range(len(path) - 1):
        link = (path[i], path[i + 1])

        # Deep dict access
        if link not in sdn_props.network_spectrum_dict:
            return False

        link_dict = sdn_props.network_spectrum_dict[link]
        cores_matrix = link_dict["cores_matrix"]

        # Direct numpy access
        band_matrix = cores_matrix[band]
        if not np.all(band_matrix[core][start:end] == 0):
            return False

    return True

def allocate_spectrum(sdn_props, path, start, end, core, band, lp_id):
    for i in range(len(path) - 1):
        link = (path[i], path[i + 1])
        # Direct mutation
        sdn_props.network_spectrum_dict[link]["cores_matrix"][band][core][start:end] = lp_id
```

### After: NetworkState Methods

```python
def is_spectrum_free(
    network_state: NetworkState,
    path: list[str],
    start: int,
    end: int,
    core: int,
    band: str
) -> bool:
    # Single method call - encapsulated logic
    return network_state.is_spectrum_available(path, start, end, core, band)

# Allocation is now part of create_lightpath
lightpath = network_state.create_lightpath(
    path=path,
    start_slot=start,
    end_slot=end,
    core=core,
    band=band,
    modulation=modulation,
    bandwidth_gbps=bandwidth,
    path_weight_km=weight,
)
# Spectrum allocation happens automatically
```

### Migration Steps

1. Replace spectrum checks with `network_state.is_spectrum_available()`
2. Replace spectrum allocation with `network_state.create_lightpath()`
3. Replace spectrum deallocation with `network_state.release_lightpath()`
4. Never access `cores_matrix` directly outside NetworkState

---

## Migration Pattern 5: Routing Results

### Before: Mutating Props

```python
def get_routes(engine_props, sdn_props, route_props):
    routing = Routing(engine_props, sdn_props, route_props)
    routing.get_route()  # Side effect: mutates route_props

    # Check results in mutated object
    if not route_props.paths_matrix:
        sdn_props.block_reason = "no_route"
        return False

    # Use mutated data
    for path_idx, path in enumerate(route_props.paths_matrix):
        weight = route_props.weights_list[path_idx]
        mods = route_props.modulation_formats_matrix[path_idx]
        # ...

    return True
```

### After: Returning Results

```python
from fusion.domain.results import RouteResult

def get_routes(
    routing_pipeline: RoutingPipeline,
    request: Request,
    network_state: NetworkState
) -> RouteResult:
    # Pure function - returns result
    result = routing_pipeline.find_routes(
        request.source,
        request.destination,
        request.bandwidth_gbps,
        network_state
    )

    # Typed access
    if result.is_empty:
        # Caller handles blocking
        pass

    # Use typed result
    for i in range(result.num_paths):
        path = result.get_path(i)
        weight = result.get_weight(i)
        mods = result.get_modulations(i)
        # ...

    return result
```

### Migration Steps

1. Replace routing class instantiation with pipeline call
2. Capture `RouteResult` instead of checking mutated props
3. Use `result.is_empty` instead of checking `if not paths_matrix`
4. Access via `result.paths`, `result.weights_km`, `result.modulations`
5. Handle blocking by returning `AllocationResult` with `block_reason`

---

## Migration Pattern 6: Feature Flag Checking

### Before: String Comparisons

```python
def should_check_snr(engine_props):
    snr_type = engine_props.get("snr_type")
    # Multiple ways to disable: None, "None", ""
    return snr_type is not None and snr_type != "None" and snr_type != ""

def should_groom(engine_props):
    return engine_props.get("is_grooming_enabled", False)

def should_slice(engine_props):
    max_segments = engine_props.get("max_segments", 1)
    return max_segments > 1
```

### After: Boolean Flags

```python
def process_request(config: SimulationConfig):
    # Clear boolean checks
    if config.snr_enabled:
        check_snr()

    if config.grooming_enabled:
        try_grooming()

    if config.slicing_enabled:
        try_slicing()
```

### Migration Steps

1. Replace string-based checks with boolean flags
2. `config.snr_enabled` replaces `snr_type != "None"` checks
3. `config.grooming_enabled` replaces `is_grooming_enabled` checks
4. `config.slicing_enabled` replaces `max_segments > 1` checks

---

## Migration Pattern 7: Statistics Collection

### Before: Scattered Updates

```python
def record_result(stats_props, sdn_props, request_dict):
    stats_props.total_arrivals += 1

    if sdn_props.routed:
        stats_props.successful_arrivals += 1
        if sdn_props.is_groomed:
            stats_props.groomed_count += 1
    else:
        stats_props.blocked_arrivals += 1
        reason = sdn_props.block_reason or "unknown"
        if reason not in stats_props.block_reasons_dict:
            stats_props.block_reasons_dict[reason] = 0
        stats_props.block_reasons_dict[reason] += 1
```

### After: StatsCollector

```python
from fusion.stats.collector import StatsCollector

def record_result(
    stats: StatsCollector,
    request: Request,
    result: AllocationResult
) -> None:
    # Single method call with typed objects
    stats.record_arrival(request, result)

    # StatsCollector automatically:
    # - Tracks total
    # - Tracks success/blocked
    # - Tracks features (groomed, sliced, protected)
    # - Tracks block reasons
```

### Migration Steps

1. Replace `StatsProps` with `StatsCollector`
2. Call `stats.record_arrival(request, result)` instead of manual updates
3. Use `stats.record_release(request)` for departures
4. Access computed metrics via `StatsCollector` properties

---

## Transitional: Dual-Mode Function

During migration, you may need functions that work with both old and new patterns:

```python
def process_request_transitional(
    # Accept both old and new
    engine_props: dict | None = None,
    config: SimulationConfig | None = None,
    request_dict: dict | None = None,
    request: Request | None = None,
    sdn_props=None,
    network_state: NetworkState | None = None,
):
    # Convert to new format internally
    if config is None and engine_props is not None:
        config = SimulationConfig.from_engine_props(engine_props)

    if request is None and request_dict is not None:
        request = Request.from_legacy_dict((0, 0.0), request_dict)

    if network_state is None and sdn_props is not None:
        # Use sdn_props during transition
        pass

    # Use new-style processing
    result = _process_internal(config, request, network_state)

    # Update legacy objects if provided (for backward compat)
    if sdn_props is not None:
        sdn_props.routed = result.success
        if not result.success:
            sdn_props.block_reason = result.block_reason.value

    return result
```

---

## Testing Migration

### Step 1: Add Conversion Tests

```python
def test_config_roundtrip():
    """from_engine_props -> to_engine_props preserves data."""
    original = {
        "k_paths": 5,
        "is_grooming_enabled": True,
        "snr_type": "snr_e2e",
        # ... all fields
    }

    config = SimulationConfig.from_engine_props(original)
    result = config.to_engine_props()

    for key in original:
        assert result[key] == original[key], f"Mismatch for {key}"
```

### Step 2: Add Equivalence Tests

```python
def test_old_new_equivalence():
    """Old and new paths produce same results."""
    # Run with old code
    old_result = old_handle_arrival(engine_props, sdn_props, ...)

    # Run with new code
    config = SimulationConfig.from_engine_props(engine_props)
    request = Request.from_legacy_dict(...)
    new_result = new_handle_arrival(config, request, network_state)

    # Compare outcomes
    assert old_result.blocked == (not new_result.success)
    if not old_result.blocked:
        assert old_result.lightpath_ids == new_result.lightpaths_created
```

### Step 3: Use run_comparison.py

```bash
# Run comparison test
python tests/run_comparison.py --config test_config.ini --tolerance 0.02
```

---

## Migration Checklist

Use this checklist when migrating a module:

### Preparation

- [ ] Identify all `engine_props` accesses
- [ ] Identify all `sdn_props` accesses
- [ ] Identify all `lightpath_status_dict` accesses
- [ ] Identify all `network_spectrum_dict` accesses
- [ ] Map each to V4 equivalent

### Implementation

- [ ] Add V4 imports
- [ ] Convert configuration access
- [ ] Convert request handling
- [ ] Convert lightpath operations
- [ ] Convert spectrum operations
- [ ] Convert routing to return `RouteResult`
- [ ] Return `AllocationResult` instead of boolean

### Verification

- [ ] Add roundtrip conversion tests
- [ ] Add equivalence tests
- [ ] Run `make test-new`
- [ ] Run `run_comparison.py`
- [ ] Verify no regressions

### Cleanup (Phase 6)

- [ ] Remove legacy parameters from function signatures
- [ ] Remove conversion code at boundaries
- [ ] Delete unused imports
- [ ] Update docstrings

---

## Common Pitfalls

### Pitfall 1: Caching NetworkState Data

```python
# WRONG: Caching will cause stale data bugs
class BadPipeline:
    def __init__(self, network_state):
        self._cached = network_state.network_spectrum_dict.copy()  # WRONG

# CORRECT: Always query fresh
class GoodPipeline:
    def process(self, network_state):
        if network_state.is_spectrum_available(...):  # Fresh query
            ...
```

### Pitfall 2: Modifying Frozen Config

```python
# WRONG: SimulationConfig is frozen
config = SimulationConfig.from_engine_props(...)
config.k_paths = 5  # FrozenInstanceError!

# CORRECT: Create new config if needed
from dataclasses import replace
new_config = replace(config, k_paths=5)
```

### Pitfall 3: Forgetting Rollback

```python
# WRONG: Lightpath created but not released on failure
lightpath = network_state.create_lightpath(...)
if not snr_pipeline.validate(lightpath, network_state):
    return AllocationResult(success=False)  # Lightpath still exists!

# CORRECT: Release on failure
lightpath = network_state.create_lightpath(...)
if not snr_pipeline.validate(lightpath, network_state):
    network_state.release_lightpath(lightpath.lightpath_id)  # Cleanup
    return AllocationResult(success=False)
```

### Pitfall 4: String Block Reasons

```python
# WRONG: Using string instead of enum
request.block_reason = "no_spectrum"

# CORRECT: Use BlockReason enum value
request.block_reason = BlockReason.NO_SPECTRUM.value
# Or return AllocationResult with enum:
return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)
```

---

## Getting Help

If you encounter migration issues:

1. Check [Migration: Before/After Examples](../migration/before_after_examples.md)
2. Review [ADR-0005: Legacy Compatibility](../decisions/0005-legacy-compatibility.md)
3. Search for similar patterns in already-migrated code
4. Ask in project discussions

---

## Next Steps

After migrating your module:

1. Run full test suite: `make test-new`
2. Run comparison tests: `python tests/run_comparison.py`
3. Remove legacy parameters once all callers are updated
4. Update module documentation
