# State Transitions

This document describes how state changes flow through the V4 domain model during simulation events.

## Overview

State transitions in V4 follow a clear pattern:
1. **Events** trigger state changes (arrivals, departures, failures)
2. **Pipelines** produce immutable results
3. **NetworkState** is the single mutable source of truth
4. **Request** and **Lightpath** objects track allocation state

---

## Event Types

### Arrival Event

Triggered when a new request enters the system.

```
Input: Request (status=PENDING)
Output: Request (status=ROUTED or BLOCKED)
```

### Departure Event

Triggered when a routed request's holding time expires.

```
Input: Request (status=ROUTED)
Output: Request (status=RELEASED), Lightpath capacity restored
```

### Failure Event

Triggered when a network component fails.

```
Input: Failed component (link, node, SRLG)
Output: Affected lightpaths switched to backup or marked degraded
```

---

## Arrival Event Flow

### State Before

```python
request = Request(
    request_id=42,
    source="A",
    destination="B",
    bandwidth_gbps=100,
    status=RequestStatus.PENDING,
    lightpath_ids=[],
)

network_state = NetworkState(...)
# network_state._lightpaths = {}
# network_state._spectrum = {all links: empty}
```

### Processing Steps

```
1. Orchestrator receives (request, network_state)
   |
   v
2. RoutingPipeline.find_routes() -> RouteResult
   - Pure computation, no state change
   - Returns candidate paths with modulations
   |
   v
3. For each path in RouteResult:
   |
   v
4. SpectrumPipeline.find_spectrum() -> SpectrumResult
   - Reads network_state (no mutation)
   - Returns slot assignment if available
   |
   v
5. If spectrum found:
   |
   v
6. NetworkState.create_lightpath()
   - MUTATION: Creates Lightpath object
   - MUTATION: Marks spectrum as allocated
   |
   v
7. SNRPipeline.validate() -> SNRResult (if SNR enabled)
   - Reads network_state
   - Returns pass/fail
   |
   v
8. If SNR fails:
   - NetworkState.release_lightpath()
   - Try next path
   |
   v
9. If SNR passes (or not enabled):
   - Update request.status = ROUTED
   - Update request.lightpath_ids
   - Return AllocationResult(success=True)
```

### State After (Success)

```python
request = Request(
    request_id=42,
    source="A",
    destination="B",
    bandwidth_gbps=100,
    status=RequestStatus.ROUTED,  # Changed
    lightpath_ids=[1],             # Added
)

lightpath = Lightpath(
    lightpath_id=1,
    path=["A", "C", "B"],
    start_slot=10,
    end_slot=18,
    total_bandwidth_gbps=100,
    remaining_bandwidth_gbps=100,
    request_allocations={42: 100},  # Tracks which request
)

# network_state._lightpaths = {1: lightpath}
# network_state._spectrum[("A","C")] slots 10-18 = 1
# network_state._spectrum[("C","B")] slots 10-18 = 1
```

### State After (Blocked)

```python
request = Request(
    request_id=42,
    source="A",
    destination="B",
    bandwidth_gbps=100,
    status=RequestStatus.BLOCKED,  # Changed
    block_reason="no_spectrum",     # Set
    lightpath_ids=[],               # Empty
)

# network_state unchanged
```

---

## Grooming Flow

When grooming is enabled, requests may reuse existing lightpaths.

### Full Grooming (Request Fits in Existing Lightpath)

```
1. GroomingPipeline.try_groom()
   - Finds lightpath with remaining capacity
   - MUTATION: Updates lightpath.request_allocations
   - MUTATION: Updates lightpath.remaining_bandwidth_gbps
   - Returns GroomingResult(fully_groomed=True)
   |
   v
2. No routing/spectrum needed
   - Update request.status = ROUTED
   - Update request.is_groomed = True
   - Return AllocationResult(success=True, is_groomed=True)
```

### State Changes

```python
# Before
lightpath = Lightpath(
    lightpath_id=1,
    remaining_bandwidth_gbps=100,
    request_allocations={41: 50},
)

# After grooming request 42 for 30 Gbps
lightpath = Lightpath(
    lightpath_id=1,
    remaining_bandwidth_gbps=70,  # 100 - 30
    request_allocations={41: 50, 42: 30},  # Added
)

request = Request(
    request_id=42,
    status=RequestStatus.ROUTED,
    lightpath_ids=[1],
    is_groomed=True,
)
```

### Partial Grooming

```
1. GroomingPipeline.try_groom()
   - Uses some capacity from existing lightpath
   - Returns GroomingResult(partially_groomed=True, remaining_bw=X)
   |
   v
2. RoutingPipeline.find_routes() for remaining bandwidth
   - forced_path = existing lightpath's path
   |
   v
3. SpectrumPipeline + NetworkState.create_lightpath()
   - Creates new lightpath for remaining bandwidth
   |
   v
4. Both lightpaths linked to request
```

---

## Departure Event Flow

### State Before

```python
request = Request(
    request_id=42,
    status=RequestStatus.ROUTED,
    lightpath_ids=[1],
)

lightpath = Lightpath(
    lightpath_id=1,
    remaining_bandwidth_gbps=70,
    request_allocations={41: 50, 42: 30},
)
```

### Processing Steps

```
1. Orchestrator.handle_release(request, network_state)
   |
   v
2. For each lightpath_id in request.lightpath_ids:
   |
   v
3. Get lightpath from network_state
   |
   v
4. Remove request from lightpath.request_allocations
   - MUTATION: del lightpath.request_allocations[request_id]
   |
   v
5. Restore capacity
   - MUTATION: lightpath.remaining_bandwidth_gbps += bandwidth
   |
   v
6. If lightpath has no more requests:
   - NetworkState.release_lightpath()
   - MUTATION: Remove from _lightpaths
   - MUTATION: Free spectrum
   |
   v
7. Update request.status = RELEASED
```

### State After

```python
# Request 42 released, request 41 still using lightpath
lightpath = Lightpath(
    lightpath_id=1,
    remaining_bandwidth_gbps=100,  # 70 + 30 restored
    request_allocations={41: 50},   # 42 removed
)

request_42 = Request(
    request_id=42,
    status=RequestStatus.RELEASED,
)

# If request 41 also releases, lightpath is deleted:
# network_state._lightpaths = {}
# spectrum freed
```

---

## Slicing Flow

When bandwidth is too high for a single lightpath, request is split.

```
1. SpectrumPipeline.find_spectrum() fails (no single allocation fits)
   |
   v
2. SlicingPipeline.try_slice()
   - Splits request into N slices
   - For each slice:
     - find_spectrum()
     - create_lightpath()
     - validate SNR
   - MUTATION: Creates multiple lightpaths
   - Returns SlicingResult(success=True, num_slices=4)
   |
   v
3. All lightpaths linked to request
   - request.lightpath_ids = [1, 2, 3, 4]
   - request.is_sliced = True
```

### State After Slicing

```python
request = Request(
    request_id=42,
    bandwidth_gbps=400,
    status=RequestStatus.ROUTED,
    lightpath_ids=[1, 2, 3, 4],
    is_sliced=True,
)

# 4 lightpaths, each 100 Gbps
lightpaths = {
    1: Lightpath(total_bandwidth_gbps=100, request_allocations={42: 100}),
    2: Lightpath(total_bandwidth_gbps=100, request_allocations={42: 100}),
    3: Lightpath(total_bandwidth_gbps=100, request_allocations={42: 100}),
    4: Lightpath(total_bandwidth_gbps=100, request_allocations={42: 100}),
}
```

---

## Protection Flow

1+1 protection allocates spectrum on two disjoint paths.

### Allocation

```
1. ProtectedRoutingPipeline.find_routes()
   - Returns disjoint primary + backup pairs
   |
   v
2. SpectrumPipeline.find_protected_spectrum()
   - Finds common slots available on BOTH paths
   |
   v
3. NetworkState.create_lightpath(backup_path=...)
   - MUTATION: Allocates spectrum on primary path
   - MUTATION: Allocates spectrum on backup path
   - MUTATION: Creates single Lightpath with backup_path set
```

### State After

```python
lightpath = Lightpath(
    lightpath_id=1,
    path=["A", "C", "B"],           # Primary
    backup_path=["A", "D", "B"],    # Backup
    start_slot=10,
    end_slot=18,
    is_protected=True,
    active_path="primary",
)

# Spectrum allocated on BOTH paths
# network_state._spectrum[("A","C")] slots 10-18 = 1
# network_state._spectrum[("C","B")] slots 10-18 = 1
# network_state._spectrum[("A","D")] slots 10-18 = 1
# network_state._spectrum[("D","B")] slots 10-18 = 1
```

### Failure Switchover

```
1. FailureManager.handle_link_failure(("C", "B"))
   |
   v
2. Find lightpaths on failed link
   - lightpath 1 uses link ("C", "B") on primary path
   |
   v
3. Switch to backup
   - MUTATION: lightpath.active_path = "backup"
   |
   v
4. Traffic now flows on backup path
   - No spectrum changes (already allocated)
```

---

## Rollback Scenarios

### SNR Failure Rollback

```python
# Lightpath created
lp = network_state.create_lightpath(...)

# SNR check fails
if not snr_pipeline.validate(lp, network_state):
    # Rollback: release spectrum and delete lightpath
    network_state.release_lightpath(lp.lightpath_id)
    # Try next path...
```

### Partial Slicing Rollback

```python
# Creating slices
created = []
for i in range(num_slices):
    result = spectrum_pipeline.find_spectrum(...)
    if not result.is_free:
        # Rollback all created slices
        for lp_id in created:
            network_state.release_lightpath(lp_id)
        # Try different slice configuration...
        break

    lp = network_state.create_lightpath(...)

    if snr_pipeline and not snr_pipeline.validate(lp, network_state):
        # Rollback this slice and all previous
        network_state.release_lightpath(lp.lightpath_id)
        for lp_id in created:
            network_state.release_lightpath(lp_id)
        break

    created.append(lp.lightpath_id)
```

### Grooming Rollback

```python
# If new lightpath allocation fails after partial grooming,
# must rollback the grooming allocations

groomed_allocations = []
for lp in groomed_lightpaths:
    lp.request_allocations[request.request_id] = allocated_bw
    lp.remaining_bandwidth_gbps -= allocated_bw
    groomed_allocations.append((lp, allocated_bw))

# New lightpath allocation fails...

# Rollback grooming
for lp, bw in groomed_allocations:
    del lp.request_allocations[request.request_id]
    lp.remaining_bandwidth_gbps += bw
```

---

## State Invariants

The following must always hold:

### NetworkState Invariants

1. **Spectrum consistency**: Allocated slots match lightpath assignments
2. **Lightpath registry**: All active lightpaths are in `_lightpaths`
3. **Bidirectional allocation**: Both directions of each link are allocated

### Lightpath Invariants

1. **Capacity**: `remaining_bandwidth_gbps >= 0`
2. **Allocation sum**: `sum(request_allocations.values()) <= total_bandwidth_gbps`
3. **Protection**: `is_protected == (backup_path is not None)`

### Request Invariants

1. **Routed has lightpaths**: `status == ROUTED` implies `lightpath_ids` non-empty
2. **Blocked has reason**: `status == BLOCKED` implies `block_reason` is set
3. **Terminal states**: `BLOCKED` and `RELEASED` are terminal
