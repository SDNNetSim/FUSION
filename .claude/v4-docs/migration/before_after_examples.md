# Before/After Examples

This document shows code examples comparing legacy patterns with V4 domain model patterns.

## Configuration Access

### Before: engine_props dict

```python
# Reading configuration
def process_request(engine_props: dict, request_dict: dict):
    k_paths = engine_props.get("k_paths", 3)
    grooming_enabled = engine_props.get("is_grooming_enabled", False)
    snr_type = engine_props.get("snr_type")

    # Type uncertainty - is it None or "None"?
    if snr_type and snr_type != "None":
        check_snr = True

    # No IDE autocomplete for keys
    # Typos not caught: engine_props.get("k_path")  # Missing 's'
```

### After: SimulationConfig

```python
# Reading configuration
def process_request(config: SimulationConfig, request: Request):
    k_paths = config.k_paths            # IDE autocomplete
    grooming_enabled = config.grooming_enabled  # Clear naming

    # Explicit boolean flag
    if config.snr_enabled:
        check_snr = True

    # Typos caught by mypy: config.k_path  # AttributeError
```

---

## Request Handling

### Before: request_dict

```python
# Processing a request
def handle_arrival(reqs_dict: dict, time_key: tuple[int, float], sdn_props):
    request_dict = reqs_dict[time_key]

    source = request_dict["source"]
    destination = request_dict["destination"]
    bandwidth = request_dict["bandwidth"]

    # Implicit state in sdn_props
    sdn_props.source = source
    sdn_props.destination = destination
    sdn_props.bandwidth = bandwidth

    # Process...

    # Status tracked via separate variables
    if blocked:
        sdn_props.block_reason = "no_spectrum"
        stats.blocked_requests += 1
    else:
        sdn_props.routed = True
        stats.successful_requests += 1
```

### After: Request object

```python
# Processing a request
def handle_arrival(request: Request, network_state: NetworkState) -> AllocationResult:
    # All data in request object
    source = request.source
    destination = request.destination
    bandwidth = request.bandwidth_gbps

    # Process...

    if blocked:
        request.status = RequestStatus.BLOCKED
        request.block_reason = BlockReason.NO_SPECTRUM.value
        return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)
    else:
        request.status = RequestStatus.ROUTED
        request.lightpath_ids.append(lp.lightpath_id)
        return AllocationResult(success=True, lightpaths_created=[lp.lightpath_id])
```

---

## Lightpath Access

### Before: lightpath_status_dict

```python
# Finding lightpath with capacity
def find_grooming_candidate(sdn_props, source: str, dest: str, needed_bw: int):
    # Construct key manually
    key = tuple(sorted([source, dest]))

    if key not in sdn_props.lightpath_status_dict:
        return None

    # Nested dict access
    lp_group = sdn_props.lightpath_status_dict[key]

    best_lp_id = None
    max_remaining = 0

    for lp_id, lp_info in lp_group.items():
        remaining = lp_info["remaining_bandwidth"]
        if remaining >= needed_bw and remaining > max_remaining:
            max_remaining = remaining
            best_lp_id = lp_id

    if best_lp_id is None:
        return None

    return lp_group[best_lp_id]
```

### After: Lightpath objects

```python
# Finding lightpath with capacity
def find_grooming_candidate(
    network_state: NetworkState,
    source: str,
    dest: str,
    needed_bw: int
) -> Lightpath | None:
    # Direct method call - no manual key construction
    lightpaths = network_state.get_lightpaths_with_capacity(source, dest, needed_bw)

    if not lightpaths:
        return None

    # Use typed objects with properties
    return max(lightpaths, key=lambda lp: lp.remaining_bandwidth_gbps)
```

---

## Spectrum Checking

### Before: cores_matrix access

```python
# Check if spectrum is free
def is_spectrum_free(sdn_props, path: list[str], start: int, end: int, core: int, band: str):
    for i in range(len(path) - 1):
        link = (path[i], path[i + 1])

        # Deep dict access
        if link not in sdn_props.network_spectrum_dict:
            return False

        link_dict = sdn_props.network_spectrum_dict[link]
        cores_matrix = link_dict["cores_matrix"]

        # NumPy array access
        band_matrix = cores_matrix[band]
        if not np.all(band_matrix[core][start:end] == 0):
            return False

    return True
```

### After: NetworkState method

```python
# Check if spectrum is free
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
```

---

## Routing Results

### Before: Mutating sdn_props

```python
# Getting routes
def get_routes(engine_props, sdn_props, route_props):
    routing = Routing(engine_props, sdn_props, route_props)
    routing.get_route()

    # Results stored in route_props - side effect
    paths = route_props.paths_matrix
    weights = route_props.weights_list
    mods = route_props.modulation_formats_matrix

    # Check if any routes found
    if not paths:
        sdn_props.block_reason = "no_route"
        return False

    return True
```

### After: Returning RouteResult

```python
# Getting routes
def get_routes(
    routing: RoutingPipeline,
    request: Request,
    network_state: NetworkState
) -> RouteResult:
    # Pure function - returns result, no side effects
    result = routing.find_routes(
        request.source,
        request.destination,
        request.bandwidth_gbps,
        network_state
    )

    # Typed access to results
    paths = result.paths
    weights = result.weights_km
    mods = result.modulations

    # Clear check
    if result.is_empty:
        # Caller handles blocking
        pass

    return result
```

---

## Statistics Collection

### Before: Scattered updates

```python
# Recording statistics
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
# Recording statistics
def record_result(
    stats: StatsCollector,
    request: Request,
    result: AllocationResult
) -> None:
    # Single method call with typed objects
    stats.record_arrival(request, result)

    # Inside StatsCollector.record_arrival:
    # - Automatically tracks total
    # - Automatically tracks success/blocked
    # - Automatically tracks features (groomed, sliced, protected)
    # - Automatically tracks block reasons
```

---

## Feature Flag Checking

### Before: String comparisons

```python
# Checking features
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

### After: Boolean flags

```python
# Checking features
def process_request(config: SimulationConfig):
    if config.snr_enabled:  # Clear boolean
        check_snr()

    if config.grooming_enabled:  # Clear boolean
        try_grooming()

    if config.slicing_enabled:  # Clear boolean
        try_slicing()
```

---

## Endpoint Key Construction

### Before: Manual tuple creation

```python
# Creating endpoint key
def get_lightpaths_between(sdn_props, source: str, dest: str):
    # Must remember to sort
    key = tuple(sorted([source, dest]))

    # Easy to forget sorting:
    # key = (source, dest)  # BUG: order matters

    return sdn_props.lightpath_status_dict.get(key, {})
```

### After: Property on domain object

```python
# Using endpoint key
request = Request(source="Z", destination="A", ...)
key = request.endpoint_key  # Automatically sorted: ("A", "Z")

lightpath = Lightpath(path=["Z", "B", "A"], ...)
key = lightpath.endpoint_key  # Automatically sorted: ("A", "Z")

# Consistent everywhere, no manual construction needed
```

---

## Error Handling

### Before: String-based errors

```python
# Handling errors
def allocate(sdn_props, engine_props):
    if not can_route:
        sdn_props.block_reason = "no_route"
        return False

    if not can_allocate_spectrum:
        sdn_props.block_reason = "no_spectrum"
        return False

    if snr_failed:
        sdn_props.block_reason = "snr_failure"
        return False

    return True
```

### After: Enum-based errors

```python
# Handling errors
def allocate(
    request: Request,
    network_state: NetworkState
) -> AllocationResult:
    if not can_route:
        return AllocationResult(success=False, block_reason=BlockReason.NO_ROUTE)

    if not can_allocate_spectrum:
        return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)

    if snr_failed:
        return AllocationResult(success=False, block_reason=BlockReason.SNR_FAILURE)

    return AllocationResult(success=True, lightpaths_created=[lp.lightpath_id])
```

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Configuration | Dict with string keys | Frozen dataclass |
| Request state | Spread across dicts | Single Request object |
| Lightpath access | Nested dicts | Typed Lightpath objects |
| Spectrum check | Direct array access | NetworkState methods |
| Routing output | Side effects on Props | Return RouteResult |
| Statistics | Scattered updates | StatsCollector |
| Feature flags | String comparisons | Boolean properties |
| Error handling | String constants | BlockReason enum |
| Type safety | None | Full mypy checking |
