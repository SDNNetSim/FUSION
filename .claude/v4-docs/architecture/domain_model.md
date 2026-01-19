# Domain Model

This document describes the core domain objects that form the foundation of the V4 architecture.

## Overview

The domain model replaces dictionary-based data structures with typed, validated dataclasses:

| Legacy | V4 Domain Object | Purpose |
|--------|------------------|---------|
| `engine_props` dict | `SimulationConfig` | Immutable simulation configuration |
| Request dict in `reqs_dict` | `Request` | Network service request with lifecycle |
| Nested dict in `lightpath_status_dict` | `Lightpath` | Allocated optical path with capacity |
| Various dicts | `RouteResult`, `SpectrumResult`, etc. | Pipeline stage outputs |

---

## SimulationConfig

**File**: `fusion/domain/config.py`

Immutable configuration object created once at simulation start. Replaces the mutable `engine_props` dictionary.

### Fields

```python
@dataclass(frozen=True)
class SimulationConfig:
    """Immutable simulation configuration."""

    # Network topology
    network_name: str
    cores_per_link: int
    band_list: tuple[str, ...]          # ("c",) or ("c", "l") or ("c", "l", "s")
    band_slots: dict[str, int]          # {"c": 320, "l": 320, "s": 320}
    guard_slots: int

    # Traffic generation
    num_requests: int
    erlang: float
    holding_time: float

    # Routing
    route_method: str                   # "k_shortest_path", "1plus1_protection", etc.
    k_paths: int
    allocation_method: str              # "first_fit", "best_fit", etc.

    # Feature flags
    grooming_enabled: bool
    slicing_enabled: bool
    max_slices: int
    snr_enabled: bool
    snr_type: str | None                # "snr_e2e", "snr_segment", None
    snr_recheck: bool
    can_partially_serve: bool

    # Modulation
    modulation_formats: dict            # Format specs
    mod_per_bw: dict                    # Bandwidth -> modulation mapping
    snr_thresholds: dict[str, float]    # Modulation -> required SNR
```

### Conversion Methods

```python
@classmethod
def from_engine_props(cls, engine_props: dict) -> "SimulationConfig":
    """Create from legacy engine_props dictionary."""

def to_engine_props(self) -> dict:
    """Convert back to legacy format for compatibility."""
```

### Design Rationale

1. **Frozen dataclass**: Prevents accidental mutation during simulation
2. **Tuple for band_list**: Immutable sequence, hashable
3. **Explicit feature flags**: Clear on/off for each feature vs. checking None values
4. **Type-safe**: Full type annotations enable mypy checking

---

## Request

**File**: `fusion/domain/request.py`

Represents a network service request through its lifecycle.

### RequestStatus Enum

```python
class RequestStatus(Enum):
    """Lifecycle states for a network request."""
    PENDING = auto()      # Created, not yet processed
    ROUTED = auto()       # Successfully allocated
    BLOCKED = auto()      # Failed allocation
    RELEASED = auto()     # Departed, resources freed
```

### Request Dataclass

```python
@dataclass
class Request:
    """Network service request with lifecycle tracking."""

    # Identity (immutable after creation)
    request_id: int
    source: str
    destination: str
    bandwidth_gbps: int
    arrive_time: float
    depart_time: float

    # Allocation state (mutable during processing)
    status: RequestStatus = RequestStatus.PENDING
    lightpath_ids: list[int] = field(default_factory=list)
    block_reason: str | None = None

    # Feature flags (set during allocation)
    is_sliced: bool = False
    is_groomed: bool = False
    is_partially_groomed: bool = False
    is_protected: bool = False
```

### Computed Properties

```python
@property
def is_arrival(self) -> bool:
    """True if this is an arrival event."""
    return self.status == RequestStatus.PENDING

@property
def endpoint_key(self) -> tuple[str, str]:
    """Canonical (sorted) endpoint tuple for lookups."""
    return tuple(sorted([self.source, self.destination]))

@property
def holding_time(self) -> float:
    """Duration request holds resources."""
    return self.depart_time - self.arrive_time
```

### BlockReason Enum

```python
class BlockReason(Enum):
    """Reasons a request may be blocked."""
    NO_ROUTE = "no_route"
    NO_SPECTRUM = "no_spectrum"
    SNR_FAILURE = "snr_failure"
    XT_FAILURE = "xt_failure"
    CONGESTION = "congestion"
    NO_MODULATION = "no_modulation"
    PROTECTION_UNAVAILABLE = "protection_unavailable"
    GROOMING_FAILED = "grooming_failed"
    SLICING_FAILED = "slicing_failed"
```

---

## Lightpath

**File**: `fusion/domain/lightpath.py`

Represents an allocated optical path with spectrum assignment and capacity tracking.

### Lightpath Dataclass

```python
@dataclass
class Lightpath:
    """Allocated optical lightpath with capacity tracking."""

    # Identity
    lightpath_id: int
    path: list[str]                     # [node1, node2, node3, ...]

    # Spectrum assignment
    start_slot: int
    end_slot: int
    core: int
    band: str                           # "c", "l", or "s"
    modulation: str                     # "QPSK", "16-QAM", etc.

    # Capacity
    total_bandwidth_gbps: int
    remaining_bandwidth_gbps: int
    path_weight_km: float

    # Quality metrics
    snr_db: float | None = None
    xt_cost: float | None = None
    is_degraded: bool = False

    # Protection (1+1)
    backup_path: list[str] | None = None
    is_protected: bool = False
    active_path: str = "primary"        # "primary" or "backup"

    # Request tracking
    request_allocations: dict[int, int] = field(default_factory=dict)
    # Maps request_id -> bandwidth_gbps allocated from this lightpath
```

### Computed Properties

```python
@property
def endpoint_key(self) -> tuple[str, str]:
    """Canonical (sorted) endpoint tuple."""
    return tuple(sorted([self.path[0], self.path[-1]]))

@property
def num_slots(self) -> int:
    """Number of spectrum slots used."""
    return self.end_slot - self.start_slot

@property
def num_hops(self) -> int:
    """Number of links in path."""
    return len(self.path) - 1

@property
def utilization(self) -> float:
    """Fraction of bandwidth currently in use."""
    used = self.total_bandwidth_gbps - self.remaining_bandwidth_gbps
    return used / self.total_bandwidth_gbps if self.total_bandwidth_gbps > 0 else 0.0
```

### Legacy Conversion

```python
@classmethod
def from_legacy_dict(cls, lightpath_id: int, lp_info: dict) -> "Lightpath":
    """Create from legacy lightpath_status_dict entry."""
    return cls(
        lightpath_id=lightpath_id,
        path=lp_info["path"],
        start_slot=lp_info["start_slot"],
        end_slot=lp_info["end_slot"],
        core=lp_info["core"],
        band=lp_info["band"],
        modulation=lp_info["mod_format"],
        total_bandwidth_gbps=lp_info["lightpath_bandwidth"],
        remaining_bandwidth_gbps=lp_info["remaining_bandwidth"],
        path_weight_km=lp_info["path_weight"],
        snr_db=lp_info.get("snr_cost"),
        xt_cost=lp_info.get("xt_cost"),
        is_degraded=lp_info.get("is_degraded", False),
        request_allocations=lp_info.get("requests_dict", {}).copy(),
    )

def to_legacy_dict(self) -> dict:
    """Convert to legacy format for compatibility."""
    return {
        "path": self.path,
        "core": self.core,
        "band": self.band,
        "start_slot": self.start_slot,
        "end_slot": self.end_slot,
        "mod_format": self.modulation,
        "lightpath_bandwidth": self.total_bandwidth_gbps,
        "remaining_bandwidth": self.remaining_bandwidth_gbps,
        "snr_cost": self.snr_db,
        "xt_cost": self.xt_cost,
        "path_weight": self.path_weight_km,
        "is_degraded": self.is_degraded,
        "requests_dict": self.request_allocations.copy(),
    }
```

---

## Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                      SimulationConfig                            │
│  (frozen, created once at start)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ configures
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Request                                  │
│  request_id: 42                                                  │
│  source: "A", destination: "B"                                   │
│  bandwidth_gbps: 100                                             │
│  status: ROUTED                                                  │
│  lightpath_ids: [1, 2]  ─────────────────┐                      │
└─────────────────────────────────────────────────────────────────┘
                                            │
                                            │ references
                                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Lightpath                                 │
│  lightpath_id: 1                                                 │
│  path: ["A", "C", "B"]                                          │
│  start_slot: 10, end_slot: 18                                   │
│  request_allocations: {42: 50}  ◄───── back-reference           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage Patterns

### Creating Domain Objects

```python
# Configuration from legacy
config = SimulationConfig.from_engine_props(engine_props)

# Request from legacy arrival event
time_key = (42, 0.5)
request = Request.from_legacy_dict(time_key, reqs_dict[time_key])

# Lightpath created by NetworkState (Phase 2)
lightpath = network_state.create_lightpath(
    path=["A", "C", "B"],
    start_slot=10,
    end_slot=18,
    core=0,
    band="c",
    modulation="QPSK",
    bandwidth_gbps=100,
    path_weight_km=500.0,
)
```

### Updating State

```python
# Request lifecycle
request.status = RequestStatus.ROUTED
request.lightpath_ids.append(lightpath.lightpath_id)
request.is_groomed = True

# Lightpath capacity
lightpath.request_allocations[request.request_id] = 50
lightpath.remaining_bandwidth_gbps -= 50

# On release
del lightpath.request_allocations[request.request_id]
lightpath.remaining_bandwidth_gbps += 50
request.status = RequestStatus.RELEASED
```

---

## Testing Strategy

Each domain object has corresponding tests:

| File | Tests |
|------|-------|
| `test_config.py` | Creation, from_engine_props, to_engine_props, immutability |
| `test_request.py` | Creation, status transitions, computed properties, legacy conversion |
| `test_lightpath.py` | Creation, capacity tracking, computed properties, legacy conversion |

### Key Test Cases

1. **Roundtrip conversion**: `from_legacy_dict` -> `to_legacy_dict` preserves all data
2. **Immutability**: `SimulationConfig` fields cannot be modified
3. **Computed properties**: Correct calculation of derived values
4. **Edge cases**: Empty paths, zero bandwidth, boundary values
