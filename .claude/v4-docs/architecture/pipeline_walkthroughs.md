# Pipeline Walkthroughs

This document provides detailed step-by-step walkthroughs of request processing through the V4 pipeline architecture. Each scenario shows the exact flow, state changes, and component interactions.

## Overview

The SDNOrchestrator routes requests through pipelines based on configuration. These walkthroughs cover the five canonical scenarios:

| Scenario | Features Enabled | Complexity |
|----------|-----------------|------------|
| A | Plain KSP | Basic |
| B | Grooming + SNR | Medium |
| C | Slicing + SNR | Medium |
| D | Grooming + Slicing + SNR | High |
| E | 1+1 Protection + SNR | High |

---

## Scenario A: Plain K-Shortest-Path

### Configuration
```python
SimulationConfig(
    route_method="k_shortest_path",
    k_paths=3,
    allocation_method="first_fit",
    grooming_enabled=False,
    slicing_enabled=False,
    snr_enabled=False,
)
```

### Request
```python
Request(
    request_id=1,
    source="A",
    destination="D",
    bandwidth_gbps=100,
    arrival_time=0.0,
    holding_time=10.0,
)
```

### Network Topology
```
    B
   / \
  A   D
   \ /
    C
```

### Step-by-Step Flow

#### Step 1: Request Arrives
```
SimulationEngine.handle_event(arrival_event)
    |
    v
SDNOrchestrator.handle_arrival(request, network_state)
```

#### Step 2: Grooming Check (Skipped)
```python
# Orchestrator checks grooming
if self.grooming and self.config.grooming_enabled:
    # SKIPPED: grooming_enabled=False
    ...
```

#### Step 3: Routing
```python
route_result = self.routing.find_routes(
    source="A",
    destination="D",
    bandwidth_gbps=100,
    network_state=network_state,
    forced_path=None,
)

# Returns:
RouteResult(
    paths=[["A", "B", "D"], ["A", "C", "D"]],
    weights_km=[200.0, 200.0],
    modulations=[["QPSK", "16-QAM"], ["QPSK", "16-QAM"]],
    strategy_name="k_shortest_path",
)
```

#### Step 4: Spectrum Assignment (First Path)
```python
path = ["A", "B", "D"]
mods = ["QPSK", "16-QAM"]

spectrum_result = self.spectrum.find_spectrum(
    path=path,
    modulations=mods,
    bandwidth_gbps=100,
    network_state=network_state,
)

# Returns:
SpectrumResult(
    is_free=True,
    start_slot=0,
    end_slot=8,
    core=0,
    band="c",
    modulation="QPSK",
    slots_needed=8,
)
```

#### Step 5: Lightpath Creation
```python
lightpath = network_state.create_lightpath(
    path=["A", "B", "D"],
    start_slot=0,
    end_slot=8,
    core=0,
    band="c",
    modulation="QPSK",
    bandwidth_gbps=100,
    path_weight_km=200.0,
)

# Returns:
Lightpath(
    lightpath_id=1,
    path=["A", "B", "D"],
    start_slot=0,
    end_slot=8,
    core=0,
    band="c",
    modulation="QPSK",
    bandwidth_gbps=100,
    remaining_bandwidth_gbps=100,
    path_weight_km=200.0,
)
```

#### Step 6: Return Result
```python
return AllocationResult(
    success=True,
    lightpaths_created=[1],
    total_bandwidth_allocated_gbps=100,
)
```

### State Changes

| State | Before | After |
|-------|--------|-------|
| `network_state._lightpaths[1]` | (none) | Lightpath object |
| `spectrum[("A","B")]["c"][0][0:8]` | 0 | 1 |
| `spectrum[("B","D")]["c"][0][0:8]` | 0 | 1 |
| `network_state._next_lightpath_id` | 1 | 2 |

### Sequence Diagram

```
SimEngine      Orchestrator      RoutingPipeline     SpectrumPipeline    NetworkState
    |               |                   |                   |                 |
    |--arrival----->|                   |                   |                 |
    |               |--find_routes----->|                   |                 |
    |               |<--RouteResult-----|                   |                 |
    |               |                   |                   |                 |
    |               |--find_spectrum----|------------------>|                 |
    |               |<--SpectrumResult--|-------------------|                 |
    |               |                   |                   |                 |
    |               |--create_lightpath-|-------------------|---------------->|
    |               |<--Lightpath-------|-------------------|-----------------|
    |               |                   |                   |                 |
    |<--AllocationResult                |                   |                 |
```

---

## Scenario B: Grooming + SNR

### Configuration
```python
SimulationConfig(
    route_method="k_shortest_path",
    k_paths=3,
    grooming_enabled=True,
    snr_enabled=True,
    snr_type="gsnr",
    slicing_enabled=False,
)
```

### Pre-existing State
```python
# Existing lightpath from A to D with spare capacity
Lightpath(
    lightpath_id=1,
    path=["A", "B", "D"],
    bandwidth_gbps=200,
    remaining_bandwidth_gbps=100,  # 100 Gbps available
)
```

### Request
```python
Request(
    request_id=2,
    source="A",
    destination="D",
    bandwidth_gbps=50,  # Can be groomed onto existing LP
)
```

### Step-by-Step Flow

#### Step 1: Grooming Attempt
```python
groom_result = self.grooming.try_groom(
    request=request,
    network_state=network_state,
)

# Returns (fully groomed):
GroomingResult(
    fully_groomed=True,
    partially_groomed=False,
    bandwidth_groomed_gbps=50,
    remaining_bandwidth_gbps=0,
    lightpaths_used=[1],
    forced_path=None,
)
```

#### Step 2: Early Return (Fully Groomed)
```python
if groom_result.fully_groomed:
    return AllocationResult(
        success=True,
        lightpaths_groomed=[1],
        is_groomed=True,
        total_bandwidth_allocated_gbps=50,
    )
```

### Partial Grooming Variant

If request bandwidth is 150 Gbps (only 100 available):

#### Step 1b: Partial Grooming
```python
groom_result = self.grooming.try_groom(request, network_state)

# Returns:
GroomingResult(
    fully_groomed=False,
    partially_groomed=True,
    bandwidth_groomed_gbps=100,
    remaining_bandwidth_gbps=50,
    lightpaths_used=[1],
    forced_path=["A", "B", "D"],  # Must use same path
)
```

#### Step 2b: Routing with Forced Path
```python
route_result = self.routing.find_routes(
    source="A",
    destination="D",
    bandwidth_gbps=50,  # Remaining bandwidth
    network_state=network_state,
    forced_path=["A", "B", "D"],  # From grooming
)
```

#### Step 3b: Spectrum Assignment
```python
spectrum_result = self.spectrum.find_spectrum(
    path=["A", "B", "D"],
    modulations=["QPSK"],
    bandwidth_gbps=50,
    network_state=network_state,
)
# Returns: is_free=True, slots 8-12
```

#### Step 4b: Lightpath Creation
```python
lightpath = network_state.create_lightpath(
    path=["A", "B", "D"],
    start_slot=8,
    end_slot=12,
    ...
)
# Returns: Lightpath(lightpath_id=2, ...)
```

#### Step 5b: SNR Validation
```python
snr_result = self.snr.validate(
    lightpath=lightpath,
    network_state=network_state,
)

# Returns:
SNRResult(
    passed=True,
    snr_db=18.5,
    required_snr_db=12.0,
)
```

#### Step 6b: Combined Result
```python
return AllocationResult(
    success=True,
    lightpaths_created=[2],
    lightpaths_groomed=[1],
    is_groomed=True,
    is_partially_groomed=True,
    total_bandwidth_allocated_gbps=150,
)
```

### SNR Failure Handling

If SNR validation fails:

```python
snr_result = self.snr.validate(lightpath, network_state)
# Returns: SNRResult(passed=False, snr_db=10.0, required_snr_db=12.0)

# Release the just-created lightpath
network_state.release_lightpath(lightpath.lightpath_id)

# Try next slot range or next path
# If all paths fail, rollback grooming too
```

### Grooming Rollback on Total Failure

```python
if all_paths_failed:
    # Must undo the partial grooming
    self.grooming.rollback(request, groom_result.lightpaths_used, network_state)

    return AllocationResult(
        success=False,
        block_reason=BlockReason.NO_SPECTRUM,
    )
```

### State Changes (Partial Grooming Success)

| State | Before | After |
|-------|--------|-------|
| `lightpath[1].remaining_bandwidth_gbps` | 100 | 0 |
| `lightpath[1].request_allocations` | {} | {2: 100} |
| `network_state._lightpaths[2]` | (none) | Lightpath |
| `spectrum[("A","B")]["c"][0][8:12]` | 0 | 2 |

---

## Scenario C: Slicing + SNR

### Configuration
```python
SimulationConfig(
    route_method="k_shortest_path",
    slicing_enabled=True,
    max_slices=4,
    snr_enabled=True,
    grooming_enabled=False,
)
```

### Request
```python
Request(
    request_id=1,
    source="A",
    destination="D",
    bandwidth_gbps=400,  # Too high for single modulation at distance
)
```

### Step-by-Step Flow

#### Step 1: Routing
```python
route_result = self.routing.find_routes(
    source="A",
    destination="D",
    bandwidth_gbps=400,
    network_state=network_state,
)

# Returns:
RouteResult(
    paths=[["A", "B", "D"]],
    modulations=[[None]],  # No modulation supports 400 Gbps at this distance
    weights_km=[200.0],
)
```

#### Step 2: Spectrum Assignment (Fails)
```python
spectrum_result = self.spectrum.find_spectrum(
    path=["A", "B", "D"],
    modulations=[None],  # No valid modulation
    bandwidth_gbps=400,
    network_state=network_state,
)

# Returns:
SpectrumResult(is_free=False)  # Cannot allocate
```

#### Step 3: Slicing Fallback
```python
slice_result = self.slicing.try_slice(
    request=request,
    path=["A", "B", "D"],
    modulations=["QPSK"],  # Valid for 100 Gbps slices
    bandwidth_gbps=400,
    network_state=network_state,
    spectrum_pipeline=self.spectrum,
    snr_pipeline=self.snr,
)
```

#### Inside Slicing Pipeline

```python
# SlicingPipeline.try_slice() implementation:

num_slices = 4
slice_bw = 400 // 4  # = 100 Gbps each
created_lightpaths = []

for i in range(num_slices):
    # Find spectrum for this slice
    spectrum_result = spectrum_pipeline.find_spectrum(
        path=path,
        modulations=["QPSK"],
        bandwidth_gbps=100,
        network_state=network_state,
    )

    if not spectrum_result.is_free:
        # Rollback all created slices
        for lp_id in created_lightpaths:
            network_state.release_lightpath(lp_id)
        return AllocationResult(success=False)

    # Create lightpath for slice
    lightpath = network_state.create_lightpath(
        path=path,
        start_slot=spectrum_result.start_slot,
        end_slot=spectrum_result.end_slot,
        ...
    )

    # SNR validation
    if snr_pipeline:
        snr_result = snr_pipeline.validate(lightpath, network_state)
        if not snr_result.passed:
            network_state.release_lightpath(lightpath.lightpath_id)
            # Try next slot range
            continue

    created_lightpaths.append(lightpath.lightpath_id)

return AllocationResult(
    success=True,
    lightpaths_created=created_lightpaths,
    is_sliced=True,
    total_bandwidth_allocated_gbps=400,
)
```

#### Step 4: Slice Creation (4 iterations)

| Slice | Slots | Lightpath ID | SNR Check |
|-------|-------|--------------|-----------|
| 1 | 0-8 | 1 | PASS (17.5 dB) |
| 2 | 8-16 | 2 | PASS (17.2 dB) |
| 3 | 16-24 | 3 | PASS (16.8 dB) |
| 4 | 24-32 | 4 | PASS (16.5 dB) |

#### Step 5: Return Combined Result
```python
return AllocationResult(
    success=True,
    lightpaths_created=[1, 2, 3, 4],
    is_sliced=True,
    total_bandwidth_allocated_gbps=400,
)
```

### State Changes

| State | Before | After |
|-------|--------|-------|
| `network_state._lightpaths` | {} | {1: LP, 2: LP, 3: LP, 4: LP} |
| `spectrum[("A","B")]["c"][0][0:32]` | 0 | [1,1,1,1,1,1,1,1,2,2,...,4,4] |
| `request.lightpath_ids` | [] | [1, 2, 3, 4] |

---

## Scenario D: Grooming + Slicing + SNR

### Configuration
```python
SimulationConfig(
    grooming_enabled=True,
    slicing_enabled=True,
    max_slices=4,
    snr_enabled=True,
)
```

### Pre-existing State
```python
Lightpath(
    lightpath_id=1,
    path=["A", "B", "D"],
    bandwidth_gbps=200,
    remaining_bandwidth_gbps=100,
)
```

### Request
```python
Request(
    request_id=2,
    source="A",
    destination="D",
    bandwidth_gbps=500,  # Needs grooming + slicing
)
```

### Step-by-Step Flow

#### Step 1: Grooming (Partial)
```python
groom_result = self.grooming.try_groom(request, network_state)

# Returns:
GroomingResult(
    fully_groomed=False,
    partially_groomed=True,
    bandwidth_groomed_gbps=100,
    remaining_bandwidth_gbps=400,  # Still need 400 Gbps
    lightpaths_used=[1],
    forced_path=["A", "B", "D"],
)
```

#### Step 2: Routing (with forced path)
```python
route_result = self.routing.find_routes(
    source="A",
    destination="D",
    bandwidth_gbps=400,
    network_state=network_state,
    forced_path=["A", "B", "D"],
)
```

#### Step 3: Spectrum (Fails for 400 Gbps)
```python
spectrum_result = self.spectrum.find_spectrum(...)
# Returns: is_free=False (no modulation for 400 Gbps)
```

#### Step 4: Slicing (4 x 100 Gbps)
```python
slice_result = self.slicing.try_slice(
    request=request,
    path=["A", "B", "D"],
    bandwidth_gbps=400,
    network_state=network_state,
    spectrum_pipeline=self.spectrum,
    snr_pipeline=self.snr,
)
```

| Slice | Slots | Lightpath ID | SNR |
|-------|-------|--------------|-----|
| 1 | 8-16 | 2 | PASS |
| 2 | 16-24 | 3 | PASS |
| 3 | 24-32 | 4 | PASS |
| 4 | 32-40 | 5 | PASS |

#### Step 5: Combine Results
```python
return AllocationResult(
    success=True,
    lightpaths_created=[2, 3, 4, 5],
    lightpaths_groomed=[1],
    is_groomed=True,
    is_partially_groomed=True,
    is_sliced=True,
    total_bandwidth_allocated_gbps=500,
)
```

### Request-to-Lightpath Mapping

```
Request 2 (500 Gbps) --> [LP 1 (groomed, 100 Gbps)]
                     --> [LP 2 (created, 100 Gbps)]
                     --> [LP 3 (created, 100 Gbps)]
                     --> [LP 4 (created, 100 Gbps)]
                     --> [LP 5 (created, 100 Gbps)]
```

---

## Scenario E: 1+1 Protection + SNR

### Configuration
```python
SimulationConfig(
    route_method="1plus1_protection",
    snr_enabled=True,
    protection_type="dedicated",
)
```

### Request
```python
Request(
    request_id=1,
    source="A",
    destination="D",
    bandwidth_gbps=100,
    protection_required=True,
)
```

### Topology
```
      B
     / \
    A   D
     \ /
      C
```

### Step-by-Step Flow

#### Step 1: Protected Routing
```python
route_result = self.routing.find_routes(
    source="A",
    destination="D",
    bandwidth_gbps=100,
    network_state=network_state,
)

# Returns (with backup paths):
RouteResult(
    paths=[["A", "B", "D"]],
    backup_paths=[["A", "C", "D"]],
    weights_km=[200.0],
    backup_weights_km=[200.0],
    modulations=[["QPSK"]],
    backup_modulations=[["QPSK"]],
)
```

#### Step 2: Protected Spectrum Assignment
```python
spectrum_result = self.spectrum.find_protected_spectrum(
    primary_path=["A", "B", "D"],
    backup_path=["A", "C", "D"],
    modulations=["QPSK"],
    bandwidth_gbps=100,
    network_state=network_state,
)

# Returns spectrum available on BOTH paths:
SpectrumResult(
    is_free=True,
    start_slot=0,
    end_slot=8,
    core=0,
    band="c",
    modulation="QPSK",
    # Backup uses same slots:
    backup_start_slot=0,
    backup_end_slot=8,
    backup_core=0,
    backup_band="c",
)
```

#### Step 3: Protected Lightpath Creation
```python
lightpath = network_state.create_protected_lightpath(
    primary_path=["A", "B", "D"],
    backup_path=["A", "C", "D"],
    start_slot=0,
    end_slot=8,
    core=0,
    band="c",
    modulation="QPSK",
    bandwidth_gbps=100,
    primary_weight_km=200.0,
    backup_weight_km=200.0,
)

# Returns:
Lightpath(
    lightpath_id=1,
    path=["A", "B", "D"],
    backup_path=["A", "C", "D"],
    is_protected=True,
    ...
)
```

#### Step 4: Protected SNR Validation
```python
snr_result = self.snr.validate_protected(
    lightpath=lightpath,
    network_state=network_state,
)

# Validates SNR on BOTH primary and backup:
SNRResult(
    passed=True,
    snr_db=17.0,         # Primary path SNR
    required_snr_db=12.0,
    metadata={
        "backup_snr_db": 16.5,  # Backup path SNR
    },
)
```

#### Step 5: Return Result
```python
return AllocationResult(
    success=True,
    lightpaths_created=[1],
    is_protected=True,
    total_bandwidth_allocated_gbps=100,
)
```

### State Changes

| State | Before | After |
|-------|--------|-------|
| `spectrum[("A","B")]["c"][0][0:8]` | 0 | 1 |
| `spectrum[("B","D")]["c"][0][0:8]` | 0 | 1 |
| `spectrum[("A","C")]["c"][0][0:8]` | 0 | 1 |
| `spectrum[("C","D")]["c"][0][0:8]` | 0 | 1 |

Note: Backup path spectrum is allocated but marked differently for capacity planning.

---

## Summary: Flow Decision Tree

```
Request Arrives
    |
    +--[grooming_enabled?]
    |       |
    |      yes --> try_groom()
    |       |         |
    |       |         +--fully_groomed --> RETURN SUCCESS
    |       |         |
    |       |         +--partially_groomed --> continue with remaining_bw, forced_path
    |       |
    |      no
    |       |
    +-------+
    |
    v
find_routes()
    |
    v
For each path:
    |
    +--[protection?]
    |       |
    |      yes --> find_protected_spectrum()
    |       |
    |      no --> find_spectrum()
    |
    +--[is_free?]
            |
           no --> [slicing_enabled?]
            |           |
            |          yes --> try_slice()
            |           |
            |          no --> try next path
            |
           yes --> create_lightpath()
                        |
                        +--[snr_enabled?]
                                |
                               yes --> validate()
                                |         |
                                |         +--fail --> release, try next
                                |         |
                                |         +--pass --> RETURN SUCCESS
                                |
                               no --> RETURN SUCCESS
```

---

## Related Documentation

- [Architecture: Orchestration](./orchestration.md)
- [Architecture: Pipeline Interfaces](./pipeline_interfaces.md)
- [Testing: Phase 3 Testing](../testing/phase_3_testing.md)
