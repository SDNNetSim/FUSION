# Result Objects

This document specifies all result types used for communication between pipeline stages.

## Overview

Result objects are immutable dataclasses returned by each pipeline stage. They provide:

1. **Type safety**: Typed fields instead of dictionary access
2. **Validation**: Invariants checked at construction
3. **Clear contracts**: Each pipeline's output is well-defined

## Result Type Summary

| Type | Created By | Consumed By | Purpose |
|------|-----------|-------------|---------|
| `RouteResult` | RoutingPipeline | Orchestrator, SpectrumPipeline | Candidate paths with modulations |
| `SpectrumResult` | SpectrumPipeline | Orchestrator, NetworkState | Slot assignment details |
| `GroomingResult` | GroomingPipeline | Orchestrator | Grooming outcome and residual |
| `SlicingResult` | SlicingPipeline | Orchestrator | Slice configuration |
| `SNRResult` | SNRPipeline | Orchestrator | SNR validation outcome |
| `AllocationResult` | Orchestrator | SimulationEngine, StatsCollector | Final allocation outcome |

---

## RouteResult

**Created by**: `RoutingPipeline.find_routes()`

**Purpose**: Contains candidate paths between source and destination, with valid modulation formats for each path.

```python
@dataclass(frozen=True)
class RouteResult:
    """Result of routing computation."""

    # Primary paths
    paths: list[list[str]]
    """Ordered list of paths, each path is list of node IDs."""

    weights_km: list[float]
    """Path lengths in kilometers, parallel to paths."""

    modulations: list[list[str | None]]
    """Valid modulations per path. None if no valid modulation for that path."""

    # Backup paths (for 1+1 protection)
    backup_paths: list[list[str]] | None = None
    """Backup paths, parallel to primary paths. None if not using protection."""

    backup_weights_km: list[float] | None = None
    """Backup path lengths, parallel to backup_paths."""

    backup_modulations: list[list[str | None]] | None = None
    """Valid modulations for backup paths."""

    # Metadata
    strategy_name: str = ""
    """Name of routing strategy used (e.g., "ksp", "disjoint")."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional strategy-specific data."""

    # Computed properties
    @property
    def is_empty(self) -> bool:
        """True if no routes found."""
        return len(self.paths) == 0

    @property
    def num_paths(self) -> int:
        """Number of candidate paths."""
        return len(self.paths)

    @property
    def has_protection(self) -> bool:
        """True if backup paths are available."""
        return self.backup_paths is not None and len(self.backup_paths) > 0
```

### Invariants

- `len(paths) == len(weights_km) == len(modulations)`
- If `backup_paths` is not None: `len(backup_paths) == len(paths)`
- Each path has at least 2 nodes (source and destination)
- Weights are non-negative

### Example

```python
route_result = RouteResult(
    paths=[["A", "B", "C"], ["A", "D", "C"]],
    weights_km=[200.0, 350.0],
    modulations=[["QPSK", "16-QAM"], ["QPSK"]],
    strategy_name="ksp",
)

# With protection
protected_result = RouteResult(
    paths=[["A", "B", "C"]],
    weights_km=[200.0],
    modulations=[["QPSK"]],
    backup_paths=[["A", "D", "E", "C"]],
    backup_weights_km=[400.0],
    backup_modulations=[["BPSK"]],
    strategy_name="disjoint",
)
```

---

## SpectrumResult

**Created by**: `SpectrumPipeline.find_spectrum()`

**Purpose**: Contains spectrum slot assignment for a path.

```python
@dataclass(frozen=True)
class SpectrumResult:
    """Result of spectrum assignment."""

    is_free: bool
    """True if spectrum was found."""

    # Slot assignment (only valid if is_free=True)
    start_slot: int = 0
    """First slot index (inclusive)."""

    end_slot: int = 0
    """Last slot index (exclusive)."""

    core: int = 0
    """Core index for multi-core fiber."""

    band: str = "c"
    """Spectrum band: "c", "l", or "s"."""

    modulation: str = ""
    """Selected modulation format."""

    slots_needed: int = 0
    """Number of slots required (including guard band)."""

    # Protection spectrum (if applicable)
    backup_start_slot: int | None = None
    backup_end_slot: int | None = None
    backup_core: int | None = None
    backup_band: str | None = None

    @property
    def num_slots(self) -> int:
        """Number of slots allocated."""
        return self.end_slot - self.start_slot if self.is_free else 0
```

### Invariants

- If `is_free=False`, slot fields are ignored
- `end_slot > start_slot` when `is_free=True`
- `slots_needed == end_slot - start_slot` (or includes guard)

### Example

```python
# Found spectrum
spectrum_result = SpectrumResult(
    is_free=True,
    start_slot=10,
    end_slot=18,
    core=0,
    band="c",
    modulation="QPSK",
    slots_needed=8,
)

# No spectrum available
blocked_result = SpectrumResult(is_free=False)
```

---

## GroomingResult

**Created by**: `GroomingPipeline.try_groom()`

**Purpose**: Indicates whether request can be served (fully or partially) by existing lightpaths.

```python
@dataclass(frozen=True)
class GroomingResult:
    """Result of grooming attempt."""

    fully_groomed: bool
    """True if entire request fits in existing lightpaths."""

    partially_groomed: bool
    """True if some bandwidth groomed but more needed."""

    bandwidth_groomed_gbps: int
    """Bandwidth successfully groomed."""

    remaining_bandwidth_gbps: int
    """Bandwidth still needing new lightpath."""

    lightpaths_used: list[int]
    """IDs of lightpaths used for grooming."""

    forced_path: list[str] | None = None
    """If partially groomed, new lightpath must use this path."""

    @property
    def needs_new_lightpath(self) -> bool:
        """True if additional spectrum allocation needed."""
        return not self.fully_groomed and self.remaining_bandwidth_gbps > 0
```

### Example

```python
# Fully groomed
full_result = GroomingResult(
    fully_groomed=True,
    partially_groomed=False,
    bandwidth_groomed_gbps=100,
    remaining_bandwidth_gbps=0,
    lightpaths_used=[5, 7],
)

# Partially groomed
partial_result = GroomingResult(
    fully_groomed=False,
    partially_groomed=True,
    bandwidth_groomed_gbps=50,
    remaining_bandwidth_gbps=50,
    lightpaths_used=[5],
    forced_path=["A", "B", "C"],
)

# No grooming possible
no_groom = GroomingResult(
    fully_groomed=False,
    partially_groomed=False,
    bandwidth_groomed_gbps=0,
    remaining_bandwidth_gbps=100,
    lightpaths_used=[],
)
```

---

## SlicingResult

**Created by**: `SlicingPipeline.try_slice()`

**Purpose**: Contains slicing configuration when request is split across multiple lightpaths.

```python
@dataclass(frozen=True)
class SlicingResult:
    """Result of slicing attempt."""

    success: bool
    """True if slicing succeeded."""

    num_slices: int = 0
    """Number of slices created."""

    slice_bandwidth_gbps: int = 0
    """Bandwidth per slice."""

    lightpaths_created: list[int] = field(default_factory=list)
    """IDs of lightpaths created for slices."""

    total_bandwidth_gbps: int = 0
    """Total bandwidth allocated across slices."""
```

### Example

```python
# Successful slicing
sliced = SlicingResult(
    success=True,
    num_slices=4,
    slice_bandwidth_gbps=100,
    lightpaths_created=[10, 11, 12, 13],
    total_bandwidth_gbps=400,
)

# Slicing failed
failed = SlicingResult(success=False)
```

---

## SNRResult

**Created by**: `SNRPipeline.validate()`

**Purpose**: Contains SNR validation outcome for a lightpath.

```python
@dataclass(frozen=True)
class SNRResult:
    """Result of SNR validation."""

    passed: bool
    """True if SNR meets threshold."""

    snr_db: float = 0.0
    """Calculated SNR in dB."""

    required_snr_db: float = 0.0
    """Required SNR threshold for modulation."""

    margin_db: float = 0.0
    """SNR margin (snr_db - required_snr_db)."""

    failure_reason: str | None = None
    """Explanation if validation failed."""

    # Per-link breakdown (optional)
    link_snr_values: dict[tuple[str, str], float] = field(default_factory=dict)
    """SNR contribution per link."""
```

### Example

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

## AllocationResult

**Created by**: `SDNOrchestrator.handle_arrival()`

**Purpose**: Final outcome of request processing, consumed by engine and stats.

```python
@dataclass(frozen=True)
class AllocationResult:
    """Final result of allocation attempt."""

    success: bool
    """True if request was served (fully or partially)."""

    # Lightpath tracking
    lightpaths_created: list[int] = field(default_factory=list)
    """IDs of newly created lightpaths."""

    lightpaths_groomed: list[int] = field(default_factory=list)
    """IDs of existing lightpaths used for grooming."""

    # Bandwidth tracking
    total_bandwidth_allocated_gbps: int = 0
    """Total bandwidth allocated to request."""

    # Feature flags
    is_groomed: bool = False
    """True if any grooming was used."""

    is_partially_groomed: bool = False
    """True if grooming + new lightpath."""

    is_sliced: bool = False
    """True if request was sliced."""

    is_protected: bool = False
    """True if 1+1 protection allocated."""

    # Failure info
    block_reason: BlockReason | None = None
    """Reason for blocking (if success=False)."""

    # Detailed results (for debugging/analysis)
    route_result: RouteResult | None = None
    spectrum_result: SpectrumResult | None = None
    snr_result: SNRResult | None = None
```

### Invariants

- If `success=True`: at least one lightpath created or groomed
- If `success=False`: `block_reason` is set
- `total_bandwidth_allocated_gbps > 0` when `success=True`

### Example

```python
# Successful allocation
success_result = AllocationResult(
    success=True,
    lightpaths_created=[15],
    total_bandwidth_allocated_gbps=100,
)

# Groomed + new lightpath
partial_result = AllocationResult(
    success=True,
    lightpaths_created=[16],
    lightpaths_groomed=[5],
    total_bandwidth_allocated_gbps=150,
    is_groomed=True,
    is_partially_groomed=True,
)

# Blocked
blocked_result = AllocationResult(
    success=False,
    block_reason=BlockReason.NO_SPECTRUM,
)
```

---

## Feasibility Authority

Each result type has a primary feasibility indicator:

| Result Type | Feasibility Field | Meaning |
|-------------|-------------------|---------|
| `RouteResult` | `is_empty` | No paths found |
| `SpectrumResult` | `is_free` | Spectrum available |
| `GroomingResult` | `fully_groomed` or `partially_groomed` | Grooming possible |
| `SlicingResult` | `success` | Slicing worked |
| `SNRResult` | `passed` | SNR acceptable |
| `AllocationResult` | `success` | **Final authority** |

The orchestrator combines these to produce the final `AllocationResult.success`.

---

## File Organization

All result types are defined in `fusion/domain/results.py`:

```python
# fusion/domain/results.py

from dataclasses import dataclass, field
from typing import Any
from fusion.domain.request import BlockReason

__all__ = [
    "RouteResult",
    "SpectrumResult",
    "GroomingResult",
    "SlicingResult",
    "SNRResult",
    "AllocationResult",
]
```

The `__init__.py` re-exports all result types:

```python
# fusion/domain/__init__.py

from fusion.domain.results import (
    RouteResult,
    SpectrumResult,
    GroomingResult,
    SlicingResult,
    SNRResult,
    AllocationResult,
)
```
