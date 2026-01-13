# Getting Started with the V4 Domain Model

This tutorial provides a high-level walkthrough of the core domain objects in the V4 architecture: `SimulationConfig`, `Request`, `Lightpath`, and `NetworkState`.

## Prerequisites

- Familiarity with FUSION's optical network simulation concepts
- Understanding of Python dataclasses

## Related Documentation

- [Architecture: Domain Model](../architecture/domain_model.md)
- [Architecture: Result Objects](../architecture/result_objects.md)
- [ADR-0001: Frozen Dataclasses](../decisions/0001-frozen-dataclasses.md)
- [ADR-0005: Legacy Compatibility](../decisions/0005-legacy-compatibility.md)

---

## Overview: From Dicts to Domain Objects

The V4 architecture replaces dictionary-based data structures with typed, validated dataclasses:

| Legacy Pattern | V4 Domain Object | Purpose |
|----------------|------------------|---------|
| `engine_props` dict | `SimulationConfig` | Immutable simulation configuration |
| Request dict in `reqs_dict` | `Request` | Network service request with lifecycle |
| Nested dict in `lightpath_status_dict` | `Lightpath` | Allocated optical path with capacity |
| Spread across Props objects | `NetworkState` | Single source of truth for network state |

---

## SimulationConfig: Immutable Configuration

`SimulationConfig` is a **frozen dataclass** created once at simulation start. It replaces the mutable `engine_props` dictionary.

### Why Frozen?

- **Prevents accidental mutation**: Configuration cannot change mid-simulation
- **Thread-safe**: Safe to share across components without locks
- **Hashable**: Can be used as dict keys or in sets
- **Clear contract**: What you see at start is what you get throughout

### Key Fields

```python
from fusion.domain.config import SimulationConfig

# SimulationConfig captures all simulation settings
config = SimulationConfig(
    # Network topology
    network_name="USbackbone60",
    cores_per_link=7,
    band_list=("c", "l"),           # Tuple, not list (immutable)
    band_slots={"c": 320, "l": 320},
    guard_slots=1,

    # Traffic generation
    num_requests=1000,
    erlang=100.0,
    holding_time=1.0,

    # Routing
    route_method="k_shortest_path",
    k_paths=3,
    allocation_method="first_fit",

    # Feature flags (explicit booleans)
    grooming_enabled=True,
    slicing_enabled=False,
    max_slices=1,
    snr_enabled=True,
    snr_type="snr_e2e",
    snr_recheck=False,
    can_partially_serve=False,

    # Modulation
    modulation_formats={...},
    mod_per_bw={...},
    snr_thresholds={"BPSK": 6.0, "QPSK": 9.0, ...},
)
```

### Converting From Legacy

```python
# Create from existing engine_props dict
config = SimulationConfig.from_engine_props(engine_props)

# Convert back for legacy code compatibility
legacy_dict = config.to_engine_props()
```

### Mental Model

Think of `SimulationConfig` as the "rules of the game":
- It defines what features are enabled
- It specifies network parameters
- It never changes during a simulation run
- Multiple components read it, none modify it

---

## Request: Tracking Service Lifecycle

`Request` represents a network service request through its lifecycle, from arrival to departure.

### Request Lifecycle

```
PENDING ──(allocation)──> ROUTED ──(release)──> RELEASED
    │
    └──(blocked)──> BLOCKED
```

### Key Fields

```python
from fusion.domain.request import Request, RequestStatus

request = Request(
    # Identity (immutable after creation)
    request_id=42,
    source="Chicago",
    destination="NewYork",
    bandwidth_gbps=100,
    arrive_time=0.5,
    depart_time=1.5,

    # Allocation state (mutable during processing)
    status=RequestStatus.PENDING,
    lightpath_ids=[],        # Will be populated on success
    block_reason=None,       # Will be set on failure

    # Feature flags (set during allocation)
    is_sliced=False,
    is_groomed=False,
    is_partially_groomed=False,
    is_protected=False,
)
```

### Computed Properties

```python
# Check if this is an arrival event
if request.is_arrival:
    process_arrival(request)

# Get canonical endpoint key (always sorted)
key = request.endpoint_key  # ("Chicago", "NewYork") - sorted alphabetically

# Calculate holding time
duration = request.holding_time  # depart_time - arrive_time
```

### Lifecycle Transitions

```python
# On successful allocation
request.status = RequestStatus.ROUTED
request.lightpath_ids.append(lightpath.lightpath_id)

# On failure
request.status = RequestStatus.BLOCKED
request.block_reason = "no_spectrum"

# On release (departure event)
request.status = RequestStatus.RELEASED
```

### Mental Model

Think of `Request` as a "work order":
- It describes what the customer wants (source, dest, bandwidth)
- It tracks the current status (pending, routed, blocked, released)
- It records what resources were assigned (lightpath_ids)
- Multiple lightpaths can serve one request (slicing, grooming)

---

## Lightpath: Allocated Optical Path

`Lightpath` represents an allocated optical path with spectrum assignment and capacity tracking.

### Key Fields

```python
from fusion.domain.lightpath import Lightpath

lightpath = Lightpath(
    # Identity
    lightpath_id=1,
    path=["Chicago", "Cleveland", "NewYork"],

    # Spectrum assignment
    start_slot=10,
    end_slot=18,          # 8 slots allocated
    core=0,
    band="c",
    modulation="QPSK",

    # Capacity
    total_bandwidth_gbps=100,
    remaining_bandwidth_gbps=50,   # 50 Gbps still available for grooming
    path_weight_km=800.0,

    # Quality metrics
    snr_db=18.5,
    xt_cost=0.02,
    is_degraded=False,

    # Protection (1+1)
    backup_path=["Chicago", "Detroit", "Buffalo", "NewYork"],
    is_protected=True,
    active_path="primary",

    # Request tracking
    request_allocations={42: 50, 43: 50},  # request_id -> bandwidth_gbps
)
```

### Computed Properties

```python
# Get canonical endpoint key
key = lightpath.endpoint_key  # ("Chicago", "NewYork")

# Get number of slots used
slots = lightpath.num_slots  # end_slot - start_slot = 8

# Get number of hops
hops = lightpath.num_hops  # len(path) - 1 = 2

# Get utilization percentage
util = lightpath.utilization  # (100-50)/100 = 0.5
```

### Grooming: Sharing Capacity

```python
# Allocate bandwidth from lightpath to a new request
new_request_id = 44
bw_to_allocate = 25

lightpath.request_allocations[new_request_id] = bw_to_allocate
lightpath.remaining_bandwidth_gbps -= bw_to_allocate

# On release: restore capacity
del lightpath.request_allocations[new_request_id]
lightpath.remaining_bandwidth_gbps += bw_to_allocate
```

### Mental Model

Think of `Lightpath` as a "pipe" in the network:
- It has a physical route (path) and spectral location (slots, core, band)
- It has total capacity and remaining capacity
- Multiple requests can share it (grooming)
- It may have a backup route (protection)

---

## NetworkState: Single Source of Truth

`NetworkState` is the **single source of truth** for all network state. There is exactly one instance per simulation.

### Ownership Model

```
SimulationEngine (OWNS)
    │
    └── NetworkState (SINGLE INSTANCE)
            │
            ├── Spectrum state (per link)
            └── Lightpath registry
```

### Key Rules

1. **One instance**: SimulationEngine creates and owns the single NetworkState
2. **Pass by reference**: Orchestrator and pipelines receive it per call
3. **No caching**: Pipelines never store or cache NetworkState
4. **No copying**: Never copy internal dicts (stale data bug)

### Read Operations

```python
from fusion.domain.network_state import NetworkState

# Get network topology
topology = network_state.topology

# Find lightpath by ID
lp = network_state.get_lightpath(lightpath_id=1)

# Find lightpaths between endpoints
lightpaths = network_state.get_lightpaths_between("Chicago", "NewYork")

# Find lightpaths with available capacity
candidates = network_state.get_lightpaths_with_capacity(
    source="Chicago",
    destination="NewYork",
    min_bandwidth=50,
)

# Check if spectrum is available on a path
is_free = network_state.is_spectrum_available(
    path=["Chicago", "Cleveland", "NewYork"],
    start=10,
    end=18,
    core=0,
    band="c",
)
```

### Write Operations

```python
# Create a new lightpath (allocates spectrum automatically)
lightpath = network_state.create_lightpath(
    path=["Chicago", "Cleveland", "NewYork"],
    start_slot=10,
    end_slot=18,
    core=0,
    band="c",
    modulation="QPSK",
    bandwidth_gbps=100,
    path_weight_km=800.0,
    backup_path=None,  # Set for 1+1 protection
)

# Release a lightpath (frees spectrum automatically)
network_state.release_lightpath(lightpath_id=1)
```

### Anti-Pattern: Local Caching

```python
# WRONG: Caching spectrum state
class BadPipeline:
    def __init__(self, network_state):
        # This will become stale!
        self._cached_spectrum = network_state.network_spectrum_dict.copy()

# CORRECT: Always query fresh state
class GoodPipeline:
    def find_spectrum(self, path, network_state):
        # Fresh query every time
        if network_state.is_spectrum_available(path, start, end, core, band):
            ...
```

### Mental Model

Think of `NetworkState` as the "ground truth" database:
- All reads go through it for fresh data
- All writes go through it for consistency
- Pipelines are stateless - they query NetworkState each time
- No local caches means no stale data bugs

---

## Putting It Together

Here's how the domain objects interact in a typical request flow:

```python
# 1. Configuration is created once at simulation start
config = SimulationConfig.from_engine_props(engine_props)

# 2. NetworkState is owned by SimulationEngine
network_state = NetworkState(topology, config)

# 3. Request arrives
request = Request(
    request_id=42,
    source="Chicago",
    destination="NewYork",
    bandwidth_gbps=100,
    arrive_time=0.5,
    depart_time=1.5,
)

# 4. Find spectrum (query NetworkState)
if network_state.is_spectrum_available(path, start, end, core, band):
    # 5. Create lightpath (mutates NetworkState)
    lightpath = network_state.create_lightpath(...)

    # 6. Update request
    request.status = RequestStatus.ROUTED
    request.lightpath_ids.append(lightpath.lightpath_id)

    # 7. Link lightpath to request
    lightpath.request_allocations[request.request_id] = request.bandwidth_gbps

# 8. On departure: release
network_state.release_lightpath(lightpath.lightpath_id)
request.status = RequestStatus.RELEASED
```

---

## Checklist: Developer Quick Reference

When working with V4 domain objects:

- [ ] Use `SimulationConfig` instead of `engine_props` dict
- [ ] Check `config.grooming_enabled` not `engine_props.get("is_grooming_enabled")`
- [ ] Create `Request` objects from legacy dicts using `Request.from_legacy_dict()`
- [ ] Use `request.endpoint_key` instead of manually sorting source/dest
- [ ] Query `NetworkState` for lightpaths, don't access `lightpath_status_dict`
- [ ] Never cache `NetworkState` data in pipelines
- [ ] Pass `network_state` by reference to pipelines, don't store it
- [ ] Use `network_state.create_lightpath()` not direct dict manipulation
- [ ] Use `network_state.release_lightpath()` for cleanup

---

## Next Steps

- [Working with Requests and Results](./working_with_requests_and_results.md) - Step-by-step examples
- [Adding a New Routing Strategy](./adding_a_new_routing_strategy.md) - Extending the system
- [Migrating Legacy Code to V4](./migrating_legacy_code_to_v4_domain_model.md) - Migration guide
