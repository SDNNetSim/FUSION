# Validation Rules

This document defines naming conventions, type requirements, and state shape validation for V4 domain objects.

## Naming Conventions

### Field Names

| Category | Convention | Examples |
|----------|------------|----------|
| Identity | Singular noun | `request_id`, `lightpath_id`, `source` |
| Collections | Plural noun | `paths`, `modulations`, `lightpath_ids` |
| Booleans | `is_` or `has_` prefix | `is_groomed`, `is_protected`, `has_backup` |
| Counts | `num_` prefix or `_count` suffix | `num_slots`, `usage_count` |
| Measurements | Unit suffix | `bandwidth_gbps`, `path_weight_km`, `snr_db` |
| Time | `_time` suffix | `arrive_time`, `depart_time` |

### Type Annotations

All fields must have explicit type annotations:

```python
# Good
request_id: int
source: str
paths: list[list[str]]
snr_db: float | None

# Bad
request_id  # Missing annotation
source = ""  # Inferred, not explicit
```

### Enum Values

Enum string values match legacy constants:

```python
class BlockReason(Enum):
    NO_ROUTE = "no_route"        # Matches legacy "no_route" string
    NO_SPECTRUM = "no_spectrum"  # Matches legacy "no_spectrum" string
```

---

## Type Requirements

### Immutable vs Mutable

| Object | Mutability | Reason |
|--------|------------|--------|
| `SimulationConfig` | Frozen | Configuration never changes during simulation |
| `RouteResult` | Frozen | Pipeline outputs are immutable |
| `SpectrumResult` | Frozen | Pipeline outputs are immutable |
| `GroomingResult` | Frozen | Pipeline outputs are immutable |
| `AllocationResult` | Frozen | Pipeline outputs are immutable |
| `Request` | Mutable | Status changes during lifecycle |
| `Lightpath` | Mutable | Capacity changes during grooming |
| `NetworkState` | Mutable | State changes during simulation |

### Container Types

```python
# Tuples for immutable sequences
band_list: tuple[str, ...]  # Not list

# Lists for mutable/ordered sequences
paths: list[list[str]]
lightpath_ids: list[int]

# Dicts for mappings
band_slots: dict[str, int]
request_allocations: dict[int, int]

# Use defaultdict for counters
block_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
```

### Optional Types

Use `| None` syntax (Python 3.10+):

```python
# Good
snr_db: float | None = None
backup_path: list[str] | None = None

# Avoid
snr_db: Optional[float] = None  # Less readable
```

---

## State Shape Validation

### SimulationConfig Validation

```python
def validate_config(config: SimulationConfig) -> list[str]:
    """Validate configuration, return list of errors."""
    errors = []

    # Network validation
    if config.cores_per_link < 1:
        errors.append("cores_per_link must be >= 1")

    if not config.band_list:
        errors.append("band_list cannot be empty")

    for band in config.band_list:
        if band not in ("c", "l", "s"):
            errors.append(f"Invalid band: {band}")
        if config.band_slots.get(band, 0) <= 0:
            errors.append(f"band_slots[{band}] must be > 0")

    if config.guard_slots < 0:
        errors.append("guard_slots must be >= 0")

    # Traffic validation
    if config.num_requests < 1:
        errors.append("num_requests must be >= 1")

    if config.erlang <= 0:
        errors.append("erlang must be > 0")

    if config.holding_time <= 0:
        errors.append("holding_time must be > 0")

    # Routing validation
    if config.k_paths < 1:
        errors.append("k_paths must be >= 1")

    # Slicing validation
    if config.slicing_enabled and config.max_slices < 2:
        errors.append("max_slices must be >= 2 when slicing enabled")

    return errors
```

### Request Validation

```python
def validate_request(request: Request) -> list[str]:
    """Validate request, return list of errors."""
    errors = []

    if request.request_id < 0:
        errors.append("request_id must be >= 0")

    if not request.source:
        errors.append("source cannot be empty")

    if not request.destination:
        errors.append("destination cannot be empty")

    if request.source == request.destination:
        errors.append("source and destination must differ")

    if request.bandwidth_gbps <= 0:
        errors.append("bandwidth_gbps must be > 0")

    if request.arrive_time < 0:
        errors.append("arrive_time must be >= 0")

    if request.depart_time <= request.arrive_time:
        errors.append("depart_time must be > arrive_time")

    # State consistency
    if request.status == RequestStatus.ROUTED and not request.lightpath_ids:
        errors.append("ROUTED request must have lightpath_ids")

    if request.status == RequestStatus.BLOCKED and not request.block_reason:
        errors.append("BLOCKED request must have block_reason")

    return errors
```

### Lightpath Validation

```python
def validate_lightpath(lightpath: Lightpath, config: SimulationConfig) -> list[str]:
    """Validate lightpath, return list of errors."""
    errors = []

    if lightpath.lightpath_id < 0:
        errors.append("lightpath_id must be >= 0")

    if len(lightpath.path) < 2:
        errors.append("path must have at least 2 nodes")

    # Spectrum bounds
    if lightpath.start_slot < 0:
        errors.append("start_slot must be >= 0")

    if lightpath.end_slot <= lightpath.start_slot:
        errors.append("end_slot must be > start_slot")

    band_slots = config.band_slots.get(lightpath.band, 0)
    if lightpath.end_slot > band_slots:
        errors.append(f"end_slot exceeds band capacity ({band_slots})")

    if lightpath.core < 0 or lightpath.core >= config.cores_per_link:
        errors.append(f"core must be in [0, {config.cores_per_link})")

    if lightpath.band not in config.band_list:
        errors.append(f"band {lightpath.band} not in band_list")

    # Capacity
    if lightpath.total_bandwidth_gbps <= 0:
        errors.append("total_bandwidth_gbps must be > 0")

    if lightpath.remaining_bandwidth_gbps < 0:
        errors.append("remaining_bandwidth_gbps must be >= 0")

    if lightpath.remaining_bandwidth_gbps > lightpath.total_bandwidth_gbps:
        errors.append("remaining_bandwidth_gbps exceeds total")

    # Protection consistency
    if lightpath.is_protected and not lightpath.backup_path:
        errors.append("protected lightpath must have backup_path")

    if lightpath.backup_path and not lightpath.is_protected:
        errors.append("backup_path set but is_protected is False")

    if lightpath.active_path not in ("primary", "backup"):
        errors.append("active_path must be 'primary' or 'backup'")

    return errors
```

### RouteResult Validation

```python
def validate_route_result(result: RouteResult) -> list[str]:
    """Validate route result, return list of errors."""
    errors = []

    # Length consistency
    if len(result.paths) != len(result.weights_km):
        errors.append("paths and weights_km length mismatch")

    if len(result.paths) != len(result.modulations):
        errors.append("paths and modulations length mismatch")

    # Path validity
    for i, path in enumerate(result.paths):
        if len(path) < 2:
            errors.append(f"path[{i}] must have at least 2 nodes")

    # Modulation validity
    for i, mods in enumerate(result.modulations):
        if not mods:
            errors.append(f"modulations[{i}] cannot be empty")

    # Backup consistency
    if result.backup_paths is not None:
        if len(result.backup_paths) != len(result.paths):
            errors.append("backup_paths length must match paths")

        if result.backup_weights_km is None:
            errors.append("backup_weights_km required when backup_paths set")

        if result.backup_modulations is None:
            errors.append("backup_modulations required when backup_paths set")

    return errors
```

---

## State Transition Rules

### Request State Machine

```
                    +---------+
                    | PENDING |
                    +---------+
                         |
            +------------+------------+
            |                         |
            v                         v
       +---------+              +---------+
       | ROUTED  |              | BLOCKED |
       +---------+              +---------+
            |
            v
       +----------+
       | RELEASED |
       +----------+
```

**Valid Transitions**:
- PENDING -> ROUTED (allocation succeeded)
- PENDING -> BLOCKED (allocation failed)
- ROUTED -> RELEASED (departure event)

**Invalid Transitions**:
- BLOCKED -> ROUTED (cannot recover)
- RELEASED -> * (terminal state)
- ROUTED -> BLOCKED (cannot fail after success)

### Lightpath Capacity Invariant

```python
# Invariant: sum of allocations <= total bandwidth
assert sum(lightpath.request_allocations.values()) <= lightpath.total_bandwidth_gbps

# Invariant: remaining = total - sum of allocations
allocated = sum(lightpath.request_allocations.values())
assert lightpath.remaining_bandwidth_gbps == lightpath.total_bandwidth_gbps - allocated
```

---

## Validation Enforcement

### Creation-Time Validation

```python
@dataclass
class Request:
    # ... fields ...

    def __post_init__(self):
        """Validate on creation."""
        if self.source == self.destination:
            raise ValueError("source and destination must differ")
        if self.bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be positive")
        if self.depart_time <= self.arrive_time:
            raise ValueError("depart_time must be after arrive_time")
```

### Runtime Assertions

```python
def allocate_bandwidth(lightpath: Lightpath, request_id: int, bandwidth: int) -> None:
    """Allocate bandwidth from lightpath to request."""
    assert bandwidth > 0, "bandwidth must be positive"
    assert bandwidth <= lightpath.remaining_bandwidth_gbps, "insufficient capacity"
    assert request_id not in lightpath.request_allocations, "request already allocated"

    lightpath.request_allocations[request_id] = bandwidth
    lightpath.remaining_bandwidth_gbps -= bandwidth

    # Verify invariant
    assert lightpath.remaining_bandwidth_gbps >= 0
```

### Test-Time Validation

```python
def test_lightpath_capacity_invariant():
    """Lightpath capacity tracking maintains invariant."""
    lp = Lightpath(
        lightpath_id=1,
        total_bandwidth_gbps=100,
        remaining_bandwidth_gbps=100,
        # ... other fields
    )

    # Allocate 30
    lp.request_allocations[1] = 30
    lp.remaining_bandwidth_gbps = 70

    # Verify invariant
    allocated = sum(lp.request_allocations.values())
    assert lp.remaining_bandwidth_gbps == lp.total_bandwidth_gbps - allocated

    # Allocate another 50
    lp.request_allocations[2] = 50
    lp.remaining_bandwidth_gbps = 20

    # Still holds
    allocated = sum(lp.request_allocations.values())
    assert lp.remaining_bandwidth_gbps == lp.total_bandwidth_gbps - allocated
```

---

## Error Messages

Error messages should be:
1. **Specific**: Identify the exact field and problem
2. **Actionable**: Suggest what the correct value should be
3. **Contextual**: Include actual values when helpful

```python
# Good
f"end_slot ({lightpath.end_slot}) exceeds band capacity ({band_slots})"
f"core must be in [0, {config.cores_per_link}), got {lightpath.core}"

# Bad
"invalid end_slot"
"core out of range"
```
