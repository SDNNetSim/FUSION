# FUSION Domain Module

## Purpose

Typed domain objects for the FUSION simulation framework. This module provides immutable, validated data structures that represent the core concepts in optical network simulation: configuration, requests, lightpaths, network state, and pipeline results.

All domain objects are designed for:
- **Type Safety**: Full type annotations with mypy compliance
- **Immutability**: Frozen dataclasses where appropriate to prevent accidental mutation
- **Validation**: Comprehensive `__post_init__` validation for data integrity
- **Legacy Interop**: Bidirectional conversion with legacy dictionary formats via `from_legacy_dict`/`to_legacy_dict`

**Example**: *This module defines what a request looks like, how lightpaths track capacity, and what results pipelines return - the shared vocabulary for all FUSION components.*

## Quick Start

### Creating a Simulation Configuration
```python
from fusion.domain import SimulationConfig

# From engine_props dictionary (legacy interop)
config = SimulationConfig.from_engine_props(engine_props)

# Access configuration values
print(f"Network: {config.network_name}")
print(f"Cores per link: {config.cores_per_link}")
print(f"Band slots: {config.band_slots}")
```

### Working with Requests
```python
from fusion.domain import Request, RequestType, RequestStatus

# Create a new request
request = Request(
    request_id=1,
    source="0",
    destination="5",
    bandwidth_gbps=100,
    arrive=0.5,
    depart=1.5,
    request_type=RequestType.ARRIVAL,
)

# Check request state
if request.status == RequestStatus.PENDING:
    print(f"Request {request.request_id}: {request.source} -> {request.destination}")
```

### Managing Lightpaths
```python
from fusion.domain import Lightpath

# Create a lightpath
lp = Lightpath(
    lightpath_id=1,
    path=["0", "2", "5"],
    start_slot=10,
    end_slot=18,
    core=0,
    band="c",
    modulation="QPSK",
    total_bandwidth_gbps=100,
    remaining_bandwidth_gbps=100,
)

# Allocate bandwidth to a request
if lp.can_accommodate(50):
    lp.allocate_bandwidth(request_id=42, bandwidth_gbps=50)
    print(f"Utilization: {lp.utilization:.1%}")
```

### Network State Management
```python
from fusion.domain import NetworkState, SimulationConfig

# Initialize network state from configuration
config = SimulationConfig.from_engine_props(engine_props)
state = NetworkState.from_config(config, topology)

# Check spectrum availability
link_id = ("0", "2")
is_free = state.is_spectrum_free(link_id, core=0, band="c", start=10, end=18)

# Allocate spectrum
if is_free:
    state.allocate_spectrum(link_id, core=0, band="c", start=10, end=18, lightpath_id=1)
```

## Architecture

```
domain/
├── __init__.py           # Public API exports
├── config.py             # SimulationConfig - immutable configuration
├── request.py            # Request, RequestType, RequestStatus, BlockReason
├── lightpath.py          # Lightpath - optical path with capacity tracking
├── network_state.py      # NetworkState, LinkSpectrum - mutable network state
├── results.py            # Pipeline result objects (RouteResult, SpectrumResult, etc.)
├── README.md             # This file
├── TODO.md               # Development roadmap
└── tests/                # Unit tests
    └── test_*.py
```

## Key Components

### SimulationConfig
Immutable configuration container that:
- Captures all simulation parameters from engine_props
- Provides type-safe access to configuration values
- Supports bidirectional legacy dictionary conversion
- Validates configuration consistency at creation time

### Request
Network service request that:
- Tracks request lifecycle (PENDING -> ALLOCATED/BLOCKED)
- Supports multiple request types (ARRIVAL, DEPARTURE, FAILURE, RECOVERY)
- Records block reasons for analytics
- Maintains protection status for survivability experiments

### Lightpath
Allocated optical path that:
- Manages spectrum assignment (path, slots, core, band)
- Tracks capacity and bandwidth allocations per request
- Supports traffic grooming with utilization tracking
- Handles 1+1 protection with backup paths (beta)

### NetworkState
Mutable network state container that:
- Owns all link spectrum via LinkSpectrum objects
- Manages lightpath registry and request associations
- Provides spectrum query and allocation methods
- Maintains consistency between spectrum and lightpath state

### LinkSpectrum
Per-link spectrum array that:
- Stores spectrum occupancy as NumPy arrays per core/band
- Tracks which lightpath owns each slot
- Supports efficient spectrum queries
- Handles guard band management

### Result Objects
Immutable pipeline stage outputs including:
- **RouteResult**: Candidate paths from routing stage
- **SpectrumResult**: Spectrum assignment outcome
- **GroomingResult**: Traffic grooming decision
- **SlicingResult**: Bandwidth slicing allocation
- **SNRResult**: Signal quality validation
- **ProtectionResult**: Disjoint backup path assignment
- **AllocationResult**: Final allocation outcome

## Usage Examples

### Example 1: Pipeline Integration
```python
from fusion.domain import Request, NetworkState, AllocationResult

def handle_arrival(request: Request, state: NetworkState) -> AllocationResult:
    """Process request through allocation pipeline."""
    # Route request
    route_result = routing_pipeline.route(request, state)
    if not route_result.paths:
        return AllocationResult(success=False, block_reason=BlockReason.NO_PATH)

    # Assign spectrum
    spectrum_result = spectrum_pipeline.assign(request, state, route_result)
    if not spectrum_result.success:
        return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)

    # Validate SNR
    snr_result = snr_pipeline.validate(request, state, spectrum_result)
    if not snr_result.passed:
        return AllocationResult(success=False, block_reason=BlockReason.SNR_VIOLATION)

    return AllocationResult(success=True, lightpath_id=spectrum_result.lightpath_id)
```

### Example 2: Traffic Grooming
```python
from fusion.domain import Lightpath, Request

def groom_request(request: Request, lightpaths: list[Lightpath]) -> Lightpath | None:
    """Find existing lightpath with capacity for request."""
    for lp in lightpaths:
        if lp.source == request.source and lp.destination == request.destination:
            if lp.can_accommodate(request.bandwidth_gbps):
                lp.allocate_bandwidth(request.request_id, request.bandwidth_gbps)
                return lp
    return None
```

### Example 3: Network State Queries
```python
from fusion.domain import NetworkState

def find_available_spectrum(
    state: NetworkState,
    path: list[str],
    slots_needed: int,
) -> tuple[int, int] | None:
    """Find contiguous free spectrum on all links in path."""
    for link_id in state.path_to_links(path):
        # Use first-fit to find available spectrum
        slot = state.find_first_fit(link_id, core=0, band="c", slots_needed=slots_needed)
        if slot is not None:
            return (slot, slot + slots_needed)
    return None
```

## Data Structures

### Request Status Flow
```
PENDING -> ALLOCATED (success)
        -> BLOCKED (no resources)
        -> RELEASED (departed)
        -> FAILED (link failure)
        -> RECOVERED (restored)
```

### Lightpath Capacity Model
```
total_bandwidth_gbps: 100 Gbps (fixed at creation)
remaining_bandwidth_gbps: 60 Gbps (decreases with allocations)
request_allocations: {req_42: 30, req_57: 10} (tracks per-request usage)
utilization: 0.4 (40% used)
```

### Spectrum Representation
```
LinkSpectrum.spectrum[core][band] = np.array([0, 0, 1, 1, 1, 0, 0, ...])
  - 0: Free slot
  - lightpath_id (>0): Slot allocated to that lightpath
```

## Testing

```bash
# Run domain module tests
pytest fusion/domain/tests/

# Run with coverage
pytest --cov=fusion.domain fusion/domain/tests/

# Run specific test file
pytest fusion/domain/tests/test_config.py
pytest fusion/domain/tests/test_request.py
pytest fusion/domain/tests/test_lightpath.py
pytest fusion/domain/tests/test_network_state.py
pytest fusion/domain/tests/test_results.py
```

## Error Handling

### Validation in __post_init__
All domain objects validate their invariants at creation:

```python
@dataclass
class Lightpath:
    def __post_init__(self) -> None:
        if len(self.path) < 2:
            raise ValueError("path must have at least 2 nodes")
        if self.start_slot >= self.end_slot:
            raise ValueError("start_slot must be < end_slot")
```

### Method-Level Validation
```python
def allocate_bandwidth(self, request_id: int, bandwidth_gbps: int) -> bool:
    if request_id in self.request_allocations:
        raise ValueError(f"Request {request_id} already allocated")
    if bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be > 0")
```

## Dependencies

### Internal Dependencies
- `fusion.utils`: Logging configuration
- **Standard Library**: `dataclasses`, `enum`, `typing`

### External Dependencies
- `numpy`: Spectrum array storage and operations
- `networkx`: Topology queries in NetworkState

## Migration Notes

This module was introduced in v6.0 as part of the pipeline architecture. Legacy code can interoperate using:

```python
# Legacy dict -> Domain object
config = SimulationConfig.from_engine_props(engine_props)
lp = Lightpath.from_legacy_dict(lp_id, lp_info)

# Domain object -> Legacy dict
engine_props = config.to_engine_props()
lp_info = lp.to_legacy_dict()
```

See `TODO.md` for planned deprecation of legacy adapters in v6.1.
