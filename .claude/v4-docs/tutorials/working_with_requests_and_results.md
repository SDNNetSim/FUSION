# Working with Requests and Results

This tutorial provides step-by-step examples of creating a Request, running it through the core logic, and inspecting the various result objects (`RouteResult`, `SpectrumResult`, `AllocationResult`, etc.).

## Prerequisites

- [Getting Started with Domain Model](./getting_started_with_domain_model.md)
- Understanding of the pipeline architecture

## Related Documentation

- [Architecture: Result Objects](../architecture/result_objects.md)
- [Architecture: Routing Strategies](../architecture/routing_strategies.md)
- [ADR-0003: Result Object Design](../decisions/0003-result-object-design.md)
- [Migration: Before/After Examples](../migration/before_after_examples.md)

---

## Overview: The Request Flow

A request flows through multiple pipeline stages, each producing a typed result:

```
Request
    │
    ▼
RoutingPipeline.find_routes()
    │
    └──> RouteResult (candidate paths)
            │
            ▼
        SpectrumPipeline.find_spectrum()
            │
            └──> SpectrumResult (slot assignment)
                    │
                    ▼ (optional)
                SNRPipeline.validate()
                    │
                    └──> SNRResult (quality check)
                            │
                            ▼
                        AllocationResult (final outcome)
```

---

## Step 1: Create a Request

### From Scratch

```python
from fusion.domain.request import Request, RequestStatus

request = Request(
    request_id=42,
    source="Chicago",
    destination="NewYork",
    bandwidth_gbps=100,
    arrive_time=0.5,
    depart_time=1.5,
    status=RequestStatus.PENDING,
)

# Verify initial state
assert request.is_arrival  # True - status is PENDING
assert request.holding_time == 1.0  # depart - arrive
assert request.endpoint_key == ("Chicago", "NewYork")  # Sorted tuple
assert request.lightpath_ids == []  # No allocations yet
```

### From Legacy Dict

```python
# Legacy format
time_key = (42, 0.5)  # (request_id, arrive_time)
request_dict = {
    "source": "Chicago",
    "destination": "NewYork",
    "bandwidth": 100,
    "arrive": 0.5,
    "depart": 1.5,
}

# Convert to V4 Request
request = Request.from_legacy_dict(time_key, request_dict)

# Roundtrip verification
legacy_dict = request.to_legacy_dict()
assert legacy_dict["source"] == "Chicago"
```

---

## Step 2: Get Routes (RouteResult)

The routing pipeline returns a `RouteResult` containing candidate paths.

### Calling the Routing Pipeline

```python
from fusion.domain.results import RouteResult

# Using the routing pipeline
route_result = routing_pipeline.find_routes(
    source=request.source,
    destination=request.destination,
    bandwidth_gbps=request.bandwidth_gbps,
    network_state=network_state,
)
```

### Inspecting RouteResult

```python
# Check if routes were found
if route_result.is_empty:
    print("No routes available")
    # Handle blocking: BlockReason.NO_ROUTE
else:
    print(f"Found {route_result.num_paths} candidate paths")

# Iterate through paths
for i in range(route_result.num_paths):
    path = route_result.get_path(i)
    weight = route_result.get_weight(i)
    modulations = route_result.get_modulations(i)

    print(f"Path {i}: {' -> '.join(path)}")
    print(f"  Length: {weight:.1f} km")
    print(f"  Valid modulations: {modulations}")
```

### Example RouteResult

```python
route_result = RouteResult(
    paths=[
        ["Chicago", "Cleveland", "NewYork"],
        ["Chicago", "Detroit", "NewYork"],
    ],
    weights_km=[800.0, 950.0],
    modulations=[
        ["QPSK", "16-QAM"],  # Path 0: two valid modulations
        ["BPSK", "QPSK"],    # Path 1: two valid modulations
    ],
    strategy_name="k_shortest_path",
)

# Access patterns
assert route_result.paths[0] == ["Chicago", "Cleveland", "NewYork"]
assert route_result.weights_km[0] == 800.0
assert "QPSK" in route_result.modulations[0]
```

### RouteResult with Protection

```python
# 1+1 protection includes backup paths
protected_result = RouteResult(
    paths=[["Chicago", "Cleveland", "NewYork"]],
    weights_km=[800.0],
    modulations=[["QPSK"]],
    # Backup paths for protection
    backup_paths=[["Chicago", "Detroit", "Buffalo", "NewYork"]],
    backup_weights_km=[1100.0],
    backup_modulations=[["BPSK"]],
    strategy_name="1plus1_protection",
)

# Check for protection
if protected_result.has_protection:
    backup = protected_result.backup_paths[0]
    print(f"Backup path: {' -> '.join(backup)}")
```

---

## Step 3: Find Spectrum (SpectrumResult)

The spectrum pipeline returns a `SpectrumResult` indicating whether spectrum was found.

### Calling the Spectrum Pipeline

```python
from fusion.domain.results import SpectrumResult

# Try each path until spectrum is found
for i in range(route_result.num_paths):
    path = route_result.get_path(i)
    modulations = route_result.get_modulations(i)

    spectrum_result = spectrum_pipeline.find_spectrum(
        path=path,
        modulations=modulations,
        bandwidth_gbps=request.bandwidth_gbps,
        network_state=network_state,
    )

    if spectrum_result.is_free:
        print(f"Found spectrum on path {i}")
        break
```

### Inspecting SpectrumResult

```python
# Check if spectrum was found
if spectrum_result.is_free:
    print(f"Start slot: {spectrum_result.start_slot}")
    print(f"End slot: {spectrum_result.end_slot}")
    print(f"Core: {spectrum_result.core}")
    print(f"Band: {spectrum_result.band}")
    print(f"Modulation: {spectrum_result.modulation}")
    print(f"Slots needed: {spectrum_result.slots_needed}")
    print(f"Num slots: {spectrum_result.num_slots}")  # Computed property
else:
    print("No spectrum available")
    # Handle blocking: BlockReason.NO_SPECTRUM
```

### Example SpectrumResult

```python
# Successful spectrum assignment
spectrum_result = SpectrumResult(
    is_free=True,
    start_slot=10,
    end_slot=18,
    core=0,
    band="c",
    modulation="QPSK",
    slots_needed=8,
)

# Failed spectrum search
blocked_result = SpectrumResult(is_free=False)
assert blocked_result.num_slots == 0  # No allocation
```

---

## Step 4: Validate SNR (SNRResult)

If SNR checking is enabled, the SNR pipeline validates signal quality.

### Calling the SNR Pipeline

```python
from fusion.domain.results import SNRResult

# After creating lightpath, validate SNR
snr_result = snr_pipeline.validate(
    lightpath=lightpath,
    network_state=network_state,
)
```

### Inspecting SNRResult

```python
if snr_result.passed:
    print(f"SNR: {snr_result.snr_db:.2f} dB")
    print(f"Required: {snr_result.required_snr_db:.2f} dB")
    print(f"Margin: {snr_result.margin_db:.2f} dB")
else:
    print(f"SNR failed: {snr_result.failure_reason}")
    # Handle blocking: BlockReason.SNR_FAILURE
```

### Example SNRResult

```python
# Passed validation
passed = SNRResult(
    passed=True,
    snr_db=18.5,
    required_snr_db=15.0,
    margin_db=3.5,
)

# Failed validation
failed = SNRResult(
    passed=False,
    snr_db=12.3,
    required_snr_db=15.0,
    margin_db=-2.7,
    failure_reason="SNR below threshold for 16-QAM",
)
```

---

## Step 5: Handle Grooming (GroomingResult)

If grooming is enabled, check for existing lightpath capacity first.

### Calling the Grooming Pipeline

```python
from fusion.domain.results import GroomingResult

# Check for grooming opportunities before routing
groom_result = grooming_pipeline.try_groom(
    request=request,
    network_state=network_state,
)
```

### Inspecting GroomingResult

```python
if groom_result.fully_groomed:
    # Entire request served by existing lightpaths
    print(f"Fully groomed using lightpaths: {groom_result.lightpaths_used}")
    print(f"Bandwidth groomed: {groom_result.bandwidth_groomed_gbps} Gbps")
    # Skip routing and spectrum - we're done!

elif groom_result.partially_groomed:
    # Some bandwidth groomed, more needed
    print(f"Partially groomed: {groom_result.bandwidth_groomed_gbps} Gbps")
    print(f"Remaining: {groom_result.remaining_bandwidth_gbps} Gbps")
    print(f"Forced path: {groom_result.forced_path}")
    # Continue with routing for remaining bandwidth

else:
    # No grooming possible
    print("No grooming candidates found")
    # Proceed with normal routing
```

### Example GroomingResult

```python
# Fully groomed - no new lightpath needed
full_groom = GroomingResult(
    fully_groomed=True,
    partially_groomed=False,
    bandwidth_groomed_gbps=100,
    remaining_bandwidth_gbps=0,
    lightpaths_used=[5, 7],
)

# Partially groomed - need new lightpath for remainder
partial_groom = GroomingResult(
    fully_groomed=False,
    partially_groomed=True,
    bandwidth_groomed_gbps=50,
    remaining_bandwidth_gbps=50,
    lightpaths_used=[5],
    forced_path=["Chicago", "Cleveland", "NewYork"],  # Must use this path
)

# Helper property
assert partial_groom.needs_new_lightpath  # True
```

---

## Step 6: Handle Slicing (SlicingResult)

If the request is too large for a single lightpath, slicing splits it.

### Inspecting SlicingResult

```python
from fusion.domain.results import SlicingResult

# Slicing attempted when spectrum search fails
slicing_result = slicing_pipeline.try_slice(
    request=request,
    path=path,
    modulations=modulations,
    bandwidth_gbps=remaining_bandwidth,
    network_state=network_state,
    spectrum_pipeline=spectrum_pipeline,
    snr_pipeline=snr_pipeline,
)

if slicing_result.success:
    print(f"Sliced into {slicing_result.num_slices} lightpaths")
    print(f"Bandwidth per slice: {slicing_result.slice_bandwidth_gbps} Gbps")
    print(f"Created lightpaths: {slicing_result.lightpaths_created}")
else:
    print("Slicing failed")
    # Handle blocking
```

### Example SlicingResult

```python
# Successful 4-way slice
sliced = SlicingResult(
    success=True,
    num_slices=4,
    slice_bandwidth_gbps=100,
    lightpaths_created=[10, 11, 12, 13],
    total_bandwidth_gbps=400,
)

# Failed slicing
failed = SlicingResult(success=False)
```

---

## Step 7: Final Result (AllocationResult)

The orchestrator combines all pipeline results into a final `AllocationResult`.

### Inspecting AllocationResult

```python
from fusion.domain.results import AllocationResult
from fusion.domain.request import BlockReason

# Orchestrator returns final result
result = orchestrator.handle_arrival(request, network_state)

if result.success:
    print("Request allocated successfully!")
    print(f"Created lightpaths: {result.lightpaths_created}")
    print(f"Groomed lightpaths: {result.lightpaths_groomed}")
    print(f"Total bandwidth: {result.total_bandwidth_allocated_gbps} Gbps")

    # Check feature flags
    if result.is_groomed:
        print("  - Used grooming")
    if result.is_partially_groomed:
        print("  - Partial grooming (groom + new lightpath)")
    if result.is_sliced:
        print("  - Request was sliced")
    if result.is_protected:
        print("  - 1+1 protection allocated")
else:
    print(f"Request blocked: {result.block_reason.value}")
    # block_reason is a BlockReason enum
```

### Example AllocationResult

```python
# Simple success
simple = AllocationResult(
    success=True,
    lightpaths_created=[15],
    total_bandwidth_allocated_gbps=100,
)

# Grooming + new lightpath
partial_groom = AllocationResult(
    success=True,
    lightpaths_created=[16],
    lightpaths_groomed=[5],
    total_bandwidth_allocated_gbps=150,
    is_groomed=True,
    is_partially_groomed=True,
)

# Sliced request
sliced = AllocationResult(
    success=True,
    lightpaths_created=[10, 11, 12, 13],
    total_bandwidth_allocated_gbps=400,
    is_sliced=True,
)

# Protected request
protected = AllocationResult(
    success=True,
    lightpaths_created=[20],
    total_bandwidth_allocated_gbps=100,
    is_protected=True,
)

# Blocked request
blocked = AllocationResult(
    success=False,
    block_reason=BlockReason.NO_SPECTRUM,
)
```

---

## Complete Example: Request Processing Flow

Here's a complete example showing the full flow:

```python
from fusion.domain.config import SimulationConfig
from fusion.domain.request import Request, RequestStatus
from fusion.domain.network_state import NetworkState
from fusion.domain.results import RouteResult, SpectrumResult, AllocationResult
from fusion.domain.request import BlockReason

def process_request(
    request: Request,
    config: SimulationConfig,
    network_state: NetworkState,
    routing_pipeline,
    spectrum_pipeline,
    grooming_pipeline,
    snr_pipeline,
) -> AllocationResult:
    """Process a request through all pipeline stages."""

    groomed_lightpaths = []
    remaining_bw = request.bandwidth_gbps

    # Step 1: Try grooming first
    if config.grooming_enabled and grooming_pipeline:
        groom_result = grooming_pipeline.try_groom(request, network_state)

        if groom_result.fully_groomed:
            # Done - fully served by existing lightpaths
            request.status = RequestStatus.ROUTED
            request.is_groomed = True
            request.lightpath_ids.extend(groom_result.lightpaths_used)
            return AllocationResult(
                success=True,
                lightpaths_groomed=groom_result.lightpaths_used,
                total_bandwidth_allocated_gbps=request.bandwidth_gbps,
                is_groomed=True,
            )

        if groom_result.partially_groomed:
            groomed_lightpaths = groom_result.lightpaths_used
            remaining_bw = groom_result.remaining_bandwidth_gbps
            forced_path = groom_result.forced_path

    # Step 2: Find routes
    route_result = routing_pipeline.find_routes(
        request.source,
        request.destination,
        remaining_bw,
        network_state,
    )

    if route_result.is_empty:
        # Rollback grooming if we can't complete allocation
        if groomed_lightpaths and not config.can_partially_serve:
            grooming_pipeline.rollback(request, groomed_lightpaths, network_state)
        request.status = RequestStatus.BLOCKED
        request.block_reason = BlockReason.NO_ROUTE.value
        return AllocationResult(success=False, block_reason=BlockReason.NO_ROUTE)

    # Step 3: Try each path
    for i in range(route_result.num_paths):
        path = route_result.get_path(i)
        modulations = route_result.get_modulations(i)
        weight = route_result.get_weight(i)

        # Step 3a: Find spectrum
        spectrum_result = spectrum_pipeline.find_spectrum(
            path, modulations, remaining_bw, network_state
        )

        if not spectrum_result.is_free:
            continue  # Try next path

        # Step 3b: Create lightpath
        lightpath = network_state.create_lightpath(
            path=path,
            start_slot=spectrum_result.start_slot,
            end_slot=spectrum_result.end_slot,
            core=spectrum_result.core,
            band=spectrum_result.band,
            modulation=spectrum_result.modulation,
            bandwidth_gbps=remaining_bw,
            path_weight_km=weight,
        )

        # Step 3c: Validate SNR (if enabled)
        if config.snr_enabled and snr_pipeline:
            snr_result = snr_pipeline.validate(lightpath, network_state)
            if not snr_result.passed:
                # Rollback and try next path
                network_state.release_lightpath(lightpath.lightpath_id)
                continue

        # Success!
        lightpath.request_allocations[request.request_id] = remaining_bw
        request.status = RequestStatus.ROUTED
        request.lightpath_ids.append(lightpath.lightpath_id)
        request.lightpath_ids.extend(groomed_lightpaths)

        return AllocationResult(
            success=True,
            lightpaths_created=[lightpath.lightpath_id],
            lightpaths_groomed=groomed_lightpaths,
            total_bandwidth_allocated_gbps=request.bandwidth_gbps,
            is_groomed=len(groomed_lightpaths) > 0,
            is_partially_groomed=len(groomed_lightpaths) > 0,
        )

    # All paths failed
    if groomed_lightpaths and not config.can_partially_serve:
        grooming_pipeline.rollback(request, groomed_lightpaths, network_state)

    request.status = RequestStatus.BLOCKED
    request.block_reason = BlockReason.NO_SPECTRUM.value
    return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)
```

---

## Before/After: Legacy vs V4

### Before: Dictionary-Based Results

```python
# OLD: Results stored by mutating props objects
def handle_arrival(sdn_props, route_props, engine_props):
    routing.get_route()  # Mutates route_props

    # Check success by inspecting mutated state
    if not route_props.paths_matrix:
        sdn_props.block_reason = "no_route"
        return False

    for path_idx, path in enumerate(route_props.paths_matrix):
        # Spectrum search mutates sdn_props
        if spectrum.find_and_allocate():
            sdn_props.routed = True
            return True

    sdn_props.block_reason = "no_spectrum"
    return False
```

### After: Typed Result Objects

```python
# NEW: Results returned as typed objects
def handle_arrival(request, network_state):
    route_result = routing.find_routes(...)  # Returns RouteResult

    if route_result.is_empty:
        return AllocationResult(success=False, block_reason=BlockReason.NO_ROUTE)

    for i in range(route_result.num_paths):
        spectrum_result = spectrum.find_spectrum(...)  # Returns SpectrumResult

        if spectrum_result.is_free:
            lightpath = network_state.create_lightpath(...)
            return AllocationResult(success=True, lightpaths_created=[...])

    return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)
```

---

## Checklist: Working with Results

When processing requests and results:

- [ ] Check `route_result.is_empty` before iterating paths
- [ ] Check `spectrum_result.is_free` before using slot values
- [ ] Check `snr_result.passed` before proceeding
- [ ] Check `groom_result.fully_groomed` to skip routing
- [ ] Check `groom_result.needs_new_lightpath` for partial grooming
- [ ] Always set `request.status` on success or failure
- [ ] Always set `request.block_reason` on failure
- [ ] Rollback partial state on failure (release lightpaths)
- [ ] Return `AllocationResult` with appropriate flags set

---

## Next Steps

- [Adding a New Routing Strategy](./adding_a_new_routing_strategy.md) - Extend routing
- [Migrating Legacy Code to V4](./migrating_legacy_code_to_v4_domain_model.md) - Migration guide
