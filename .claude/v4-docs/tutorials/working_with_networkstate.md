# Working with NetworkState

This tutorial covers how to use `NetworkState`, the authoritative source for all mutable network data in the V4 architecture.

## Prerequisites

- [Getting Started with Domain Model](./getting_started_with_domain_model.md)
- Basic understanding of optical network concepts (spectrum, lightpaths)

## Related Documentation

- [Architecture: NetworkState](../architecture/network_state.md)
- [ADR-0006: NetworkState Authority](../decisions/0006-networkstate-authority.md)

---

## Overview

`NetworkState` is the single source of truth for:

- Spectrum allocation per link
- Active lightpaths
- Lightpath-to-request mappings

### Core Principle

```
SimulationEngine (OWNER)
    |
    +-- _network_state: NetworkState  <-- Single instance
            |
            +-- All pipelines receive this by reference
            +-- Only NetworkState methods mutate state
```

---

## Creating NetworkState

### Basic Initialization

```python
import networkx as nx
from fusion.domain.network_state import NetworkState
from fusion.domain.config import SimulationConfig

# Create topology
topology = nx.Graph()
topology.add_edge("A", "B", weight=100.0)
topology.add_edge("B", "C", weight=100.0)
topology.add_edge("A", "D", weight=150.0)
topology.add_edge("D", "C", weight=150.0)

# Create configuration
config = SimulationConfig(
    network_name="example",
    cores_per_link=7,
    band_list=("c",),
    band_slots={"c": 320},
    guard_slots=1,
    # ... other required fields
)

# Initialize NetworkState
network_state = NetworkState(topology, config)
```

### What Happens on Initialization

1. Creates `LinkSpectrum` for each edge (both directions)
2. Initializes empty spectrum matrices (all zeros)
3. Sets up empty lightpath dictionary
4. Initializes lightpath ID counter to 1

```python
# After initialization:
# - network_state._spectrum[("A", "B")] exists
# - network_state._spectrum[("B", "A")] exists (both directions)
# - All spectrum slots are 0 (free)
# - network_state._lightpaths == {}
# - network_state._next_lightpath_id == 1
```

---

## Reading Spectrum Data

### Check Spectrum Availability

```python
# Check if slots 0-10 on core 0, band "c" are free along path
is_free = network_state.is_spectrum_available(
    path=["A", "B", "C"],
    start_slot=0,
    end_slot=10,
    core=0,
    band="c",
)

if is_free:
    print("Spectrum is available")
else:
    print("Spectrum is occupied")
```

### Get Link Spectrum

```python
# Get spectrum for a specific link
link_spectrum = network_state.get_link_spectrum(("A", "B"))

# Access the cores matrix for band "c"
matrix = link_spectrum.cores_matrix["c"]

# matrix shape: (num_cores, num_slots)
# matrix[core_idx, slot_idx] == 0 means free
# matrix[core_idx, slot_idx] == lightpath_id means occupied

# Example: Check specific slot
if matrix[0, 5] == 0:
    print("Slot 5 on core 0 is free")
else:
    print(f"Slot 5 occupied by lightpath {matrix[0, 5]}")
```

### Find Free Spectrum Range

```python
# Custom function to find first free range
def find_first_free_range(
    network_state: NetworkState,
    path: list[str],
    slots_needed: int,
    core: int,
    band: str,
) -> tuple[int, int] | None:
    """Find first available slot range on path."""
    # Get slot count from first link
    first_link = (path[0], path[1])
    link_spectrum = network_state.get_link_spectrum(first_link)
    total_slots = link_spectrum.cores_matrix[band].shape[1]

    for start in range(total_slots - slots_needed + 1):
        end = start + slots_needed
        if network_state.is_spectrum_available(path, start, end, core, band):
            return (start, end)

    return None

# Usage
result = find_first_free_range(network_state, ["A", "B", "C"], 8, 0, "c")
if result:
    start, end = result
    print(f"Found free range: slots {start}-{end}")
```

---

## Creating Lightpaths

### Basic Lightpath Creation

```python
lightpath = network_state.create_lightpath(
    path=["A", "B", "C"],
    start_slot=0,
    end_slot=8,
    core=0,
    band="c",
    modulation="QPSK",
    bandwidth_gbps=100,
    path_weight_km=200.0,
)

print(f"Created lightpath {lightpath.lightpath_id}")
print(f"Path: {lightpath.path}")
print(f"Slots: {lightpath.start_slot}-{lightpath.end_slot}")
```

### What Happens on Creation

1. Validates spectrum is available (raises `ValueError` if not)
2. Allocates spectrum on all path links (both directions)
3. Creates `Lightpath` object with unique ID
4. Stores lightpath in internal dictionary
5. Increments ID counter

```python
# After creation:
# - Spectrum slots 0-8 on links A-B, B-A, B-C, C-B are marked with lightpath ID
# - network_state.get_lightpath(1) returns the lightpath
# - is_spectrum_available(["A", "B", "C"], 0, 8, 0, "c") returns False
```

### Protected Lightpath Creation

For 1+1 protection, allocate on both primary and backup paths:

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

print(f"Protected lightpath: {lightpath.lightpath_id}")
print(f"Primary: {lightpath.path}")
print(f"Backup: {lightpath.backup_path}")
```

---

## Releasing Lightpaths

### Basic Release

```python
# Release by lightpath ID
network_state.release_lightpath(lightpath_id=1)

# After release:
# - Spectrum is freed on all path links
# - Lightpath removed from internal dictionary
# - is_spectrum_available() returns True for those slots
```

### Safe Release (Non-existent ID)

```python
# Releasing non-existent lightpath is safe (no-op)
network_state.release_lightpath(999)  # Does not raise error
```

### Partial Release for Grooming

When a groomed request departs, don't release the lightpath - just free the bandwidth:

```python
# Get the lightpath
lightpath = network_state.get_lightpath(lightpath_id)

# Remove request's bandwidth allocation
allocated_bw = lightpath.request_allocations.pop(request_id)

# Restore available bandwidth
lightpath.remaining_bandwidth_gbps += allocated_bw

# Only release lightpath if no requests remain
if not lightpath.request_allocations:
    network_state.release_lightpath(lightpath_id)
```

---

## Querying Lightpaths

### Get by ID

```python
lightpath = network_state.get_lightpath(lightpath_id=1)

if lightpath is None:
    print("Lightpath not found")
else:
    print(f"Found: {lightpath.path}")
```

### Get by Endpoints

```python
# Find all lightpaths between A and C (any path)
lightpaths = network_state.get_lightpaths_between("A", "C")

for lp in lightpaths:
    print(f"LP {lp.lightpath_id}: {lp.path}")

# Note: Order of endpoints doesn't matter
# get_lightpaths_between("A", "C") == get_lightpaths_between("C", "A")
```

### Get with Available Capacity

```python
# Find lightpaths with at least 50 Gbps available (for grooming)
lightpaths = network_state.get_lightpaths_with_capacity(
    source="A",
    destination="C",
    min_bw=50,
)

for lp in lightpaths:
    print(f"LP {lp.lightpath_id}: {lp.remaining_bandwidth_gbps} Gbps available")
```

### Iterate All Lightpaths

```python
# Access all lightpaths (read-only)
for lp_id, lightpath in network_state._lightpaths.items():
    print(f"{lp_id}: {lightpath.source} -> {lightpath.destination}")
```

---

## Common Patterns

### Pattern 1: Allocate Request to Lightpath

```python
def allocate_request_to_lightpath(
    network_state: NetworkState,
    request: Request,
    lightpath_id: int,
    bandwidth: int,
) -> bool:
    """Allocate request bandwidth to existing lightpath."""
    lightpath = network_state.get_lightpath(lightpath_id)
    if lightpath is None:
        return False

    if lightpath.remaining_bandwidth_gbps < bandwidth:
        return False

    # Allocate
    lightpath.remaining_bandwidth_gbps -= bandwidth
    lightpath.request_allocations[request.request_id] = bandwidth

    return True
```

### Pattern 2: Find Best Lightpath for Grooming

```python
def find_best_grooming_candidate(
    network_state: NetworkState,
    source: str,
    destination: str,
    required_bw: int,
) -> Lightpath | None:
    """Find lightpath with most available capacity."""
    candidates = network_state.get_lightpaths_with_capacity(
        source, destination, min_bw=required_bw
    )

    if not candidates:
        return None

    # Sort by available capacity (descending)
    candidates.sort(key=lambda lp: lp.remaining_bandwidth_gbps, reverse=True)

    return candidates[0]
```

### Pattern 3: Compute Link Utilization

```python
def compute_link_utilization(
    network_state: NetworkState,
    link: tuple[str, str],
) -> float:
    """Compute utilization ratio for a link (0.0 to 1.0)."""
    link_spectrum = network_state.get_link_spectrum(link)

    total_slots = 0
    used_slots = 0

    for band, matrix in link_spectrum.cores_matrix.items():
        total_slots += matrix.size
        used_slots += (matrix != 0).sum()

    if total_slots == 0:
        return 0.0

    return used_slots / total_slots
```

### Pattern 4: Snapshot for Comparison

```python
def capture_spectrum_snapshot(network_state: NetworkState) -> dict:
    """Capture deep copy of spectrum state."""
    import numpy as np

    return {
        link: {
            band: matrix.copy()
            for band, matrix in ls.cores_matrix.items()
        }
        for link, ls in network_state._spectrum.items()
    }

def compare_snapshots(snap1: dict, snap2: dict) -> bool:
    """Check if two snapshots are equal."""
    import numpy as np

    if snap1.keys() != snap2.keys():
        return False

    for link in snap1:
        for band in snap1[link]:
            if not np.array_equal(snap1[link][band], snap2[link][band]):
                return False

    return True
```

---

## Anti-Patterns to Avoid

### 1. Storing NetworkState in Pipeline

```python
# BAD: Storing state reference
class BadPipeline:
    def __init__(self, network_state: NetworkState):
        self._state = network_state  # NEVER DO THIS

# GOOD: Receive per call
class GoodPipeline:
    def find_routes(self, ..., network_state: NetworkState):
        # Use network_state only within this call
        ...
```

### 2. Direct Matrix Modification

```python
# BAD: Direct modification
matrix = network_state.get_link_spectrum(("A", "B")).cores_matrix["c"]
matrix[0, 0:8] = 1  # WRONG - bypasses NetworkState

# GOOD: Use NetworkState methods
network_state.create_lightpath(
    path=["A", "B"], start_slot=0, end_slot=8, ...
)
```

### 3. Caching Query Results

```python
# BAD: Caching (stale data risk)
class BadPipeline:
    def __init__(self):
        self._cached_lightpaths = {}

    def process(self, network_state):
        if "A-C" not in self._cached_lightpaths:
            self._cached_lightpaths["A-C"] = network_state.get_lightpaths_between("A", "C")
        return self._cached_lightpaths["A-C"]  # May be stale!

# GOOD: Fresh query each time
class GoodPipeline:
    def process(self, network_state):
        return network_state.get_lightpaths_between("A", "C")
```

### 4. Modifying Returned Lightpath Without Method

```python
# BAD: Direct modification
lp = network_state.get_lightpath(1)
lp.remaining_bandwidth_gbps = 50  # Works but bypasses validation

# BETTER: Use structured update (if available)
network_state.update_lightpath_bandwidth(1, remaining=50)
```

---

## Legacy Compatibility (Migration Period)

During migration, `NetworkState` provides legacy properties:

### network_spectrum_dict

```python
# Legacy format access (deprecated)
legacy_dict = network_state.network_spectrum_dict

# Structure:
# {
#     ("A", "B"): {
#         "cores_matrix": {"c": np.array(...), "l": np.array(...)},
#     },
#     ...
# }

# Use for:
# - Legacy code that expects dict format
# - Gradual migration

# Returns COPY, not reference (safe to read)
```

### lightpath_status_dict

```python
# Legacy format access (deprecated)
legacy_dict = network_state.lightpath_status_dict

# Structure:
# {
#     ("A", "C"): {  # Sorted endpoint tuple
#         1: {
#             "path": ["A", "B", "C"],
#             "core": 0,
#             "band": "c",
#             "start_slot": 0,
#             "end_slot": 8,
#             "mod_format": "QPSK",
#             "lightpath_bandwidth": 100,
#             "remaining_bandwidth": 100,
#             "requests_dict": {},
#         },
#     },
# }
```

### Migration Path

```python
# BEFORE (legacy)
key = tuple(sorted([src, dst]))
if key in sdn_props.lightpath_status_dict:
    for lp_id, lp_info in sdn_props.lightpath_status_dict[key].items():
        if lp_info["remaining_bandwidth"] >= bw:
            ...

# AFTER (new)
for lp in network_state.get_lightpaths_with_capacity(src, dst, min_bw=bw):
    # Use typed Lightpath object
    ...
```

---

## Testing NetworkState

### Unit Test Example

```python
import pytest
import networkx as nx
from fusion.domain.network_state import NetworkState
from fusion.domain.config import SimulationConfig


@pytest.fixture
def network_state():
    topology = nx.Graph()
    topology.add_edge("A", "B", weight=100)
    topology.add_edge("B", "C", weight=100)

    config = SimulationConfig(
        network_name="test",
        cores_per_link=1,
        band_list=("c",),
        band_slots={"c": 320},
        # ...
    )

    return NetworkState(topology, config)


class TestNetworkState:
    def test_initial_spectrum_is_free(self, network_state):
        assert network_state.is_spectrum_available(
            ["A", "B", "C"], 0, 10, 0, "c"
        )

    def test_create_lightpath_allocates_spectrum(self, network_state):
        network_state.create_lightpath(
            path=["A", "B"],
            start_slot=0,
            end_slot=8,
            core=0,
            band="c",
            modulation="QPSK",
            bandwidth_gbps=100,
            path_weight_km=100.0,
        )

        assert not network_state.is_spectrum_available(
            ["A", "B"], 0, 8, 0, "c"
        )

    def test_release_restores_spectrum(self, network_state):
        lp = network_state.create_lightpath(...)
        network_state.release_lightpath(lp.lightpath_id)

        assert network_state.is_spectrum_available(
            ["A", "B"], 0, 8, 0, "c"
        )
```

---

## Next Steps

- [Writing a New Pipeline](./writing_a_new_pipeline.md) - Use NetworkState in pipelines
- [Adding a New Routing Strategy](./adding_a_new_routing_strategy.md) - Query NetworkState in routing
- [Architecture: NetworkState](../architecture/network_state.md) - Full specification
