# Phase 1: Core Domain Model

This phase introduces the foundational domain objects without modifying existing code.

## Objectives

- Create `SimulationConfig`, `Request`, `Lightpath` dataclasses
- Create all result types (`RouteResult`, `SpectrumResult`, etc.)
- Create `StatsCollector` skeleton
- All new code is additive only
- No changes to existing function signatures

## Constraints

- `run_comparison.py` must pass unchanged
- No modifications to existing function signatures
- All new code is in new files
- Existing tests must continue to pass

---

## Micro-Phases

### P1.1: Domain Scaffolding

**Files Created**:
- `fusion/domain/__init__.py`
- `fusion/domain/config.py`

**Content**:

```python
# fusion/domain/__init__.py
from fusion.domain.config import SimulationConfig

__all__ = ["SimulationConfig"]
```

```python
# fusion/domain/config.py
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SimulationConfig:
    """Immutable simulation configuration."""

    # Network
    network_name: str
    cores_per_link: int
    band_list: tuple[str, ...]
    band_slots: dict[str, int]
    guard_slots: int

    # Traffic
    num_requests: int
    erlang: float
    holding_time: float

    # Routing
    route_method: str
    k_paths: int
    allocation_method: str

    # Features
    grooming_enabled: bool
    slicing_enabled: bool
    max_slices: int
    snr_enabled: bool
    snr_type: str | None
    snr_recheck: bool
    can_partially_serve: bool

    # Modulation
    modulation_formats: dict[str, Any]
    mod_per_bw: dict[str, Any]
    snr_thresholds: dict[str, float]

    @classmethod
    def from_engine_props(cls, engine_props: dict) -> "SimulationConfig":
        """Create from legacy engine_props dictionary."""
        return cls(
            network_name=engine_props.get("network", ""),
            cores_per_link=engine_props.get("cores_per_link", 1),
            band_list=tuple(engine_props.get("band_list", ["c"])),
            band_slots={
                "c": engine_props.get("c_band", 320),
                "l": engine_props.get("l_band", 0),
                "s": engine_props.get("s_band", 0),
            },
            guard_slots=engine_props.get("guard_slots", 1),
            num_requests=engine_props.get("num_requests", 1000),
            erlang=engine_props.get("erlang", 100.0),
            holding_time=engine_props.get("holding_time", 1.0),
            route_method=engine_props.get("route_method", "k_shortest_path"),
            k_paths=engine_props.get("k_paths", 3),
            allocation_method=engine_props.get("allocation_method", "first_fit"),
            grooming_enabled=engine_props.get("is_grooming_enabled", False),
            slicing_enabled=engine_props.get("max_segments", 1) > 1,
            max_slices=engine_props.get("max_segments", 1),
            snr_enabled=engine_props.get("snr_type") not in (None, "None", ""),
            snr_type=engine_props.get("snr_type"),
            snr_recheck=engine_props.get("snr_recheck", False),
            can_partially_serve=engine_props.get("can_partially_serve", False),
            modulation_formats=engine_props.get("modulation_formats_dict", {}),
            mod_per_bw=engine_props.get("mod_per_bw", {}),
            snr_thresholds=engine_props.get("req_snr", {}),
        )

    def to_engine_props(self) -> dict:
        """Convert back to legacy format."""
        return {
            "network": self.network_name,
            "cores_per_link": self.cores_per_link,
            "band_list": list(self.band_list),
            "c_band": self.band_slots.get("c", 0),
            "l_band": self.band_slots.get("l", 0),
            "s_band": self.band_slots.get("s", 0),
            "guard_slots": self.guard_slots,
            "num_requests": self.num_requests,
            "erlang": self.erlang,
            "holding_time": self.holding_time,
            "route_method": self.route_method,
            "k_paths": self.k_paths,
            "allocation_method": self.allocation_method,
            "is_grooming_enabled": self.grooming_enabled,
            "max_segments": self.max_slices,
            "snr_type": self.snr_type,
            "snr_recheck": self.snr_recheck,
            "can_partially_serve": self.can_partially_serve,
            "modulation_formats_dict": self.modulation_formats,
            "mod_per_bw": self.mod_per_bw,
            "req_snr": self.snr_thresholds,
        }
```

**Verification**:
```bash
pytest fusion/tests/domain/test_config.py -v
ruff check fusion/domain/
mypy fusion/domain/
```

---

### P1.2: Request Wrapper

**Files Created**:
- `fusion/domain/request.py`

**Files Modified**:
- `fusion/domain/__init__.py` (add exports)

**Content**:

```python
# fusion/domain/request.py
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

class RequestStatus(Enum):
    """Lifecycle states for a network request."""
    PENDING = auto()
    ROUTED = auto()
    BLOCKED = auto()
    RELEASED = auto()

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

    # Allocation state (mutable)
    status: RequestStatus = RequestStatus.PENDING
    lightpath_ids: list[int] = field(default_factory=list)
    block_reason: Optional[str] = None

    # Feature flags
    is_sliced: bool = False
    is_groomed: bool = False
    is_partially_groomed: bool = False
    is_protected: bool = False

    @property
    def is_arrival(self) -> bool:
        return self.status == RequestStatus.PENDING

    @property
    def is_successful(self) -> bool:
        return self.status == RequestStatus.ROUTED

    @property
    def is_blocked(self) -> bool:
        return self.status == RequestStatus.BLOCKED

    @property
    def endpoint_key(self) -> tuple[str, str]:
        return tuple(sorted([self.source, self.destination]))

    @property
    def holding_time(self) -> float:
        return self.depart_time - self.arrive_time

    @classmethod
    def from_legacy_dict(
        cls,
        time_key: tuple[int, float],
        request_dict: dict,
        request_id: Optional[int] = None,
    ) -> "Request":
        req_num, arrive = time_key
        return cls(
            request_id=request_id if request_id is not None else req_num,
            source=request_dict["source"],
            destination=request_dict["destination"],
            bandwidth_gbps=request_dict["bandwidth"],
            arrive_time=arrive,
            depart_time=request_dict["depart"],
        )

    def to_legacy_dict(self) -> dict:
        return {
            "source": self.source,
            "destination": self.destination,
            "bandwidth": self.bandwidth_gbps,
            "arrive": self.arrive_time,
            "depart": self.depart_time,
            "mod_format": None,
            "path": None,
        }
```

**Verification**:
```bash
pytest fusion/tests/domain/test_request.py -v
ruff check fusion/domain/request.py
mypy fusion/domain/request.py
```

---

### P1.3: Lightpath Wrapper

**Files Created**:
- `fusion/domain/lightpath.py`

**Files Modified**:
- `fusion/domain/__init__.py` (add exports)

**Content**:

```python
# fusion/domain/lightpath.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Lightpath:
    """Allocated optical lightpath with capacity tracking."""

    # Identity
    lightpath_id: int
    path: list[str]

    # Spectrum assignment
    start_slot: int
    end_slot: int
    core: int
    band: str
    modulation: str

    # Capacity
    total_bandwidth_gbps: int
    remaining_bandwidth_gbps: int
    path_weight_km: float

    # Quality metrics
    snr_db: Optional[float] = None
    xt_cost: Optional[float] = None
    is_degraded: bool = False

    # Protection
    backup_path: Optional[list[str]] = None
    is_protected: bool = False
    active_path: str = "primary"

    # Request tracking
    request_allocations: dict[int, int] = field(default_factory=dict)

    @property
    def endpoint_key(self) -> tuple[str, str]:
        return tuple(sorted([self.path[0], self.path[-1]]))

    @property
    def num_slots(self) -> int:
        return self.end_slot - self.start_slot

    @property
    def num_hops(self) -> int:
        return len(self.path) - 1

    @property
    def utilization(self) -> float:
        used = self.total_bandwidth_gbps - self.remaining_bandwidth_gbps
        return used / self.total_bandwidth_gbps if self.total_bandwidth_gbps > 0 else 0.0

    @classmethod
    def from_legacy_dict(cls, lightpath_id: int, lp_info: dict) -> "Lightpath":
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

**Verification**:
```bash
pytest fusion/tests/domain/test_lightpath.py -v
ruff check fusion/domain/lightpath.py
mypy fusion/domain/lightpath.py
```

---

### P1.4: Result Objects

**Files Created**:
- `fusion/domain/results.py`

**Files Modified**:
- `fusion/domain/__init__.py` (add exports)

**Content**: See [Result Objects](../architecture/result_objects.md) for full specification.

The file contains:
- `RouteResult` - Routing pipeline output
- `SpectrumResult` - Spectrum pipeline output
- `GroomingResult` - Grooming pipeline output
- `SlicingResult` - Slicing pipeline output
- `SNRResult` - SNR validation output
- `AllocationResult` - Final orchestrator output

**Verification**:
```bash
pytest fusion/tests/domain/test_results.py -v
ruff check fusion/domain/results.py
mypy fusion/domain/results.py
```

---

### P1.5: StatsCollector Skeleton

**Files Created**:
- `fusion/stats/__init__.py`
- `fusion/stats/collector.py`

**Content**:

```python
# fusion/stats/__init__.py
from fusion.stats.collector import StatsCollector

__all__ = ["StatsCollector"]
```

```python
# fusion/stats/collector.py
from dataclasses import dataclass, field
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fusion.domain.config import SimulationConfig
    from fusion.domain.request import Request
    from fusion.domain.results import AllocationResult

@dataclass
class StatsCollector:
    """Collects simulation statistics."""

    config: "SimulationConfig"

    # Request counters
    total_requests: int = 0
    successful_requests: int = 0
    blocked_requests: int = 0

    # Blocking breakdown
    block_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Feature tracking
    groomed_requests: int = 0
    sliced_requests: int = 0
    protected_requests: int = 0

    # SNR tracking
    snr_values: list[float] = field(default_factory=list)

    # Modulation tracking
    modulations_used: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_arrival(
        self,
        request: "Request",
        result: "AllocationResult",
    ) -> None:
        """Record arrival event outcome."""
        self.total_requests += 1

        if result.success:
            self.successful_requests += 1
            if result.is_groomed:
                self.groomed_requests += 1
            if result.is_sliced:
                self.sliced_requests += 1
            if result.is_protected:
                self.protected_requests += 1
        else:
            self.blocked_requests += 1
            if result.block_reason:
                self.block_reasons[result.block_reason.value] += 1

    def record_snr(self, snr_db: float) -> None:
        """Record SNR measurement."""
        self.snr_values.append(snr_db)

    @property
    def blocking_probability(self) -> float:
        """Calculate current blocking probability."""
        if self.total_requests == 0:
            return 0.0
        return self.blocked_requests / self.total_requests

    def to_comparison_format(self) -> dict:
        """Export in format expected by run_comparison.py."""
        return {
            "blocking_probability": self.blocking_probability,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "blocked_requests": self.blocked_requests,
            "block_reasons": dict(self.block_reasons),
            "groomed_requests": self.groomed_requests,
            "sliced_requests": self.sliced_requests,
            "protected_requests": self.protected_requests,
        }
```

**Verification**:
```bash
pytest fusion/tests/stats/test_collector.py -v
ruff check fusion/stats/
mypy fusion/stats/
```

---

## Phase 1 Exit Criteria

Before proceeding to Phase 2:

- [ ] All domain classes created with full type annotations
- [ ] `from_legacy_dict()` and `to_legacy_dict()` work correctly
- [ ] All computed properties implemented and tested
- [ ] All result types defined with frozen dataclasses
- [ ] StatsCollector skeleton created
- [ ] All unit tests pass
- [ ] `ruff check fusion/domain/ fusion/stats/` passes
- [ ] `mypy fusion/domain/ fusion/stats/` passes
- [ ] Existing tests still pass (no regressions)
- [ ] `run_comparison.py` passes unchanged

---

## Test File Structure

```
fusion/tests/
├── domain/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_request.py
│   ├── test_lightpath.py
│   └── test_results.py
└── stats/
    ├── __init__.py
    └── test_collector.py
```

---

## Documentation Updated

- [Domain Model](../architecture/domain_model.md) - SimulationConfig, Request, Lightpath
- [Result Objects](../architecture/result_objects.md) - All result types
