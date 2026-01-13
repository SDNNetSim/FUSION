# FUSION Architecture Refactoring Plan - V2

## Overview

This document provides a grounded, concrete architecture plan based on inventory of the actual current data structures and flows.

**The Four Goals:**
1. Centralized state and breaking up god objects
2. Standardized return shapes and naming
3. Interaction of functionalities (grooming, slicing, SNR, etc.) without breaking everything
4. SDN controller as a high-level router to wrappers/pipelines (and avoiding the RL mini-simulator problem)

---

# STEP 0 - Inventory of Current Data Structures

## Restating the Four Goals

**Goal 1: Centralized State / Break Up God Objects**
The problem is that `engine_props` and `SDNProps` are bloated containers mixing unrelated concerns - config, topology, current request state, global tracking dicts, per-request lists. When you need to understand what state exists or where it's mutated, you have to chase through thousands of lines. The fix needs to separate: (a) immutable config, (b) network topology/spectrum state, (c) per-request transient state, (d) statistics collection.

**Goal 2: Standardized Return Shapes and Naming**
Functions return inconsistent shapes - sometimes dicts, sometimes by mutating props objects, sometimes both. The same concept has multiple names (`lightpath_id` vs `lp_id`). This makes it impossible to know what a function actually returns or requires without reading its implementation. The fix needs standard result types and a single naming convention.

**Goal 3: Feature Composition Without Breaking Everything**
Adding grooming broke core assumptions because it introduced multi-lightpath requests. Adding slicing made it worse. Adding SNR added more conditionals. Each new feature requires touching the SDN controller plus multiple algorithms. The RL module gave up and cloned the simulator. The fix needs features to be composable units that don't require modifying other features.

**Goal 4: SDN Controller as Thin Router**
Currently the SDN controller contains ~1400 lines with business logic for grooming, slicing, SNR, protection all interleaved. Adding a feature means editing this monolith. The fix needs the controller to only orchestrate - delegating actual logic to dedicated pipeline/wrapper classes. RL should use the same pipelines, not clone them.

---

## Current Data Structure Inventory

### 1. `engine_props` (dict)

**What it contains:**
```python
{
    # Network topology
    "network": "USbackbone60",
    "topology_info": {...},           # Physical topology from create_pt()
    "topology": nx.Graph(),           # NetworkX graph object

    # Spectrum configuration
    "band_list": ["c", "l"],
    "c_band": 200,                    # C-band slots
    "l_band": 200,                    # L-band slots
    "cores_per_link": 7,
    "guard_slots": 1,

    # Routing configuration
    "route_method": "k_shortest_path",
    "k_paths": 3,
    "allocation_method": "first_fit",

    # Traffic configuration
    "erlang": 50,
    "num_requests": 1000,
    "holding_time": 1,

    # Feature flags
    "is_grooming_enabled": False,
    "dynamic_lps": False,
    "can_partially_serve": False,
    "snr_type": None,
    "snr_recheck": False,
    "max_segments": 1,

    # Modulation data
    "modulation_formats_dict": {...},
    "mod_per_bw": {...},
    "mod_assumption": "Flexible",

    # Runtime
    "seed": 42,
    # ... many more
}
```

**How it's used:**
- Created in `network_simulator.py` (line 108)
- Passed to `SimulationEngine.__init__(engine_props)`
- From there, passed to SDNController, Routing, SpectrumAssignment, Grooming, etc.
- Accessed everywhere via `self.engine_props`

**How it's mutated:**
- Mostly read-only after creation
- But `topology` object is mutable and used throughout
- Some runtime state gets added during simulation

**Pain points:**
- Mixes immutable config with mutable runtime state
- No clear documentation of what keys exist
- Passed everywhere - every class takes it
- Type is just `dict` - no IDE support, no validation

---

### 2. `SDNProps` (class in `fusion/core/properties.py`, lines 321-546)

**What it contains:**
```python
class SDNProps:
    # Network references (should be in NetworkState)
    topology: Any | None = None
    network_spectrum_dict: dict[tuple, dict] | None = None

    # CURRENT REQUEST state (transient, reset per request)
    request_id: int | None = None
    source: str | None = None
    destination: str | None = None
    bandwidth: float | None = None
    arrive: float | None = None
    depart: float | None = None
    request_type: str | None = None
    slots_needed: int | None = None

    # Path state for current request
    path_list: list[int] | None = None
    path_index: int | None = None
    was_routed: bool | None = None

    # Per-lightpath results for current request (lists that grow)
    bandwidth_list: list[float] = []
    modulation_list: list[str] = []
    core_list: list[int] = []
    band_list: list[str] = []
    crosstalk_list: list[float] = []
    snr_list: list[float] = []
    xt_list: list[float] = []
    start_slot_list: list[int] = []
    end_slot_list: list[int] = []
    lightpath_id_list: list[int] = []
    lightpath_bandwidth_list: list[float] = []

    # GLOBAL tracking dicts (should NOT be here)
    lightpath_status_dict: dict | None = None
    transponder_usage_dict: dict | None = None
    lp_bw_utilization_dict: dict | None = None
    allocated_requests: dict[int, dict[str, Any]] = {}

    # Grooming flags for current request
    was_groomed: bool | None = None
    was_partially_groomed: bool = False
    remaining_bw: float | None = None  # After partial grooming

    # 1+1 Protection attributes
    primary_path: list[int] | None = None
    backup_path: list[int] | None = None
    is_protected: bool = False
    active_path: str = "primary"

    # Counter
    _lightpath_id_counter: int = 0

    def get_lightpath_id(self) -> int:
        self._lightpath_id_counter += 1
        return self._lightpath_id_counter
```

**How it's used:**
- Created once per simulation in SDNController
- Reset/populated at start of each request in `handle_event()`
- Mutated throughout request processing by routing, spectrum, grooming
- Read at end to extract results

**How it's mutated:**
- `handle_event()` sets request fields
- `Routing.get_route()` reads source/dest, routing writes path results elsewhere (route_props)
- `SpectrumAssignment.get_spectrum()` reads path, writes start/end slots to lists
- `Grooming.handle_grooming()` reads/writes lightpath_status_dict, sets was_groomed flags
- `allocate()` reads all the lists to write to network_spectrum_dict

**Pain points:**
- Mixes 3 different concerns: (1) current request params, (2) current request results, (3) global tracking dicts
- 60+ attributes - impossible to understand without reading all code
- No separation between input and output
- Global dicts (lightpath_status_dict) shouldn't live here
- Lists grow per-lightpath but naming suggests per-request

---

### 3. `SpectrumProps` (class, lines 74-123)

**What it contains:**
```python
class SpectrumProps:
    # INPUT: what we're looking for
    path_list: list[int] | None = None
    backup_path: list[int] | None = None
    slots_needed: int | None = None
    modulation: str | None = None

    # INPUT: constraints/forcing
    forced_core: int | None = None
    forced_index: int | None = None
    forced_band: str | None = None

    # OUTPUT: what we found
    is_free: bool = False
    start_slot: int | None = None
    end_slot: int | None = None
    core_number: int | None = None
    current_band: str | None = None
    crosstalk_cost: float | None = None

    # REFERENCE: for reading network state
    cores_matrix: dict[str, Any] | None = None
    reverse_cores_matrix: dict[str, Any] | None = None

    # For grooming tracking (shouldn't be here)
    lightpath_id: int | None = None
    lightpath_bandwidth: float | None = None
```

**Pain points:**
- Mixes input parameters with output results
- Contains reference to cores_matrix (state leak)
- lightpath_id shouldn't be here (that's SDN's job)

---

### 4. `RoutingProps` (class, lines 30-71)

**What it contains:**
```python
class RoutingProps:
    # OUTPUT: computed paths
    paths_matrix: list[Any] = []            # List of k paths
    modulation_formats_matrix: list[list[str]] = []
    weights_list: list[float] = []
    path_index_list: list[int] = []

    # OUTPUT: for 1+1 protection
    backup_paths_matrix: list[list[int] | None] = []
    backup_modulation_formats_matrix: list[list[str]] = []

    # CONFIG: physical layer parameters (shouldn't be here)
    input_power: float = DEFAULT_INPUT_POWER
    frequency_spacing: float = DEFAULT_FREQUENCY_SPACING
    mci_worst: float = WORST_CASE_MCI
    span_length: float = DEFAULT_SPAN_LENGTH
    max_link_length: float | None = None
    max_span: int | None = None
```

**Pain points:**
- Mixes routing output with physical layer config
- Physical params should come from config, not routing props

---

### 5. `SNRProps` (class, lines 126-295)

**What it contains:**
```python
class SNRProps:
    # Physical constants
    light_frequency: float = LIGHT_FREQUENCY_CENTER
    planck_constant: float = PLANCK_CONSTANT

    # System parameters (should be config)
    request_bit_rate: float = 12.5
    request_snr: float = 8.5
    nsp: dict[str, float] = {"c": 1.77, "l": 1.99, ...}
    req_snr: dict[str, float] = {"BPSK": 3.71, "QPSK": 6.72, ...}

    # Current calculation params
    center_frequency: float | None = None
    bandwidth: float | None = None
    length: float | None = None
    number_of_spans: int | None = None

    # Pre-calculated file mappings
    file_mapping_dict: dict = {...}
```

**Pain points:**
- Physical constants should be module-level or config
- Mixes constants with per-calculation state

---

### 6. `network_spectrum_dict` (dict)

**Structure:**
```python
network_spectrum_dict[(src_node, dst_node)] = {
    "cores_matrix": {
        "c": np.ndarray((num_cores, c_band_slots), dtype=float),
        "l": np.ndarray((num_cores, l_band_slots), dtype=float),
        # etc for other bands
    },
    "link_num": int,
    "usage_count": int,
    "throughput": float,
}
# Values in cores_matrix:
# 0 = free
# positive int = lightpath_id occupying that slot
# negative int = guard band for that lightpath (-lightpath_id)
```

**How it's used:**
- Created in `SimulationEngine.create_topology()` (lines 571-589)
- Referenced from `sdn_props.network_spectrum_dict`
- Read by SpectrumAssignment to check availability
- Written by `SDNController.allocate()` to mark occupied slots
- Written by `SDNController.release()` to zero out slots

**How it's mutated:**
- `allocate()`: `cores_matrix[band][core][start:end] = lightpath_id`
- Guard bands: `cores_matrix[band][core][end] = -lightpath_id`
- `release()`: `cores_matrix[band][core][slots] = 0`

**Pain points:**
- Direct numpy mutations from multiple places
- No abstraction - everyone knows the internal structure
- Both (a,b) and (b,a) stored (bidirectional links)

---

### 7. `lightpath_status_dict` (nested dict, inside SDNProps)

**Structure:**
```python
lightpath_status_dict = {
    (src, dst): {  # Key is SORTED tuple of endpoints
        lightpath_id_1: {
            "path": [1, 2, 3, 4],
            "core": 2,
            "band": "c",
            "start_slot": 50,
            "end_slot": 60,
            "mod_format": "16-QAM",
            "lightpath_bandwidth": 100,      # Total capacity Gbps
            "remaining_bandwidth": 25,       # Available for grooming
            "snr_cost": 18.5,
            "xt_cost": 0.5,
            "path_weight": 5.2,
            "requests_dict": {
                req_id_1: 50,                # request_id -> allocated_bandwidth
                req_id_2: 25,
            },
            "time_bw_usage": {
                arrival_time: 50.0,          # utilization % at each time
            },
            "is_degraded": False,
        },
        lightpath_id_2: {...},
    },
    (other_src, other_dst): {...},
}
```

**How it's used:**
- Created/managed in `fusion/core/grooming.py`
- Read by grooming to find existing lightpaths with remaining bandwidth
- Written when new lightpaths established or requests groomed
- Updated when requests release

**Pain points:**
- Lives inside SDNProps (wrong owner)
- Deeply nested - 3 levels of dicts
- Keyed by sorted endpoint tuple - awkward to query
- Contains both spectrum info (should be in network state) and grooming info

---

### 8. Request Dict

**Structure:**
```python
# In generate_simulation_requests()
requests_dict = {
    (request_id, arrival_time): {
        "req_id": 1,
        "source": "node_a",
        "destination": "node_z",
        "arrive": 1.5,
        "depart": 11.8,
        "request_type": "arrival",      # or "release"
        "bandwidth": "50GHz",           # STRING key into mod_per_bw
        "mod_formats": {...},           # Available modulations for this bw
    },
    (request_id, depart_time): {
        "req_id": 1,
        # ... same but request_type="release"
    },
}
```

**Pain points:**
- Tuple key `(id, time)` is awkward
- Two entries per request (arrival + release)
- `bandwidth` is a string key, not numeric
- Contains `mod_formats` which is derived data

---

### 9. `StatsProps` (class, lines 549-612)

**What it contains:**
```python
class StatsProps:
    snapshots_dict: dict[int, dict] = {}
    cores_dict: dict[int, int] = {}
    modulations_used_dict: dict[str, Any] = {}
    bandwidth_blocking_dict: dict = {}
    link_usage_dict: dict[str, dict] = {}
    block_reasons_dict: dict[str, int] = {
        "distance": None,
        "congestion": None,
        "xt_threshold": None,
        "failure": None,
    }
    simulation_blocking_list: list[float] = []
    protection_switchovers: int = 0
    protection_failures: int = 0
    # ... many more tracking lists
```

**Pain points:**
- Reasonable as a stats container
- But some "stats" are really outcomes that should be on Request

---

# STEP 1 - Centralized State / Breaking Up God Objects

## Mapping Table: Current to Proposed

| Current Structure | Current Location | New Owner | Mutability | Compatibility View |
|-------------------|------------------|-----------|------------|-------------------|
| **engine_props** config keys (network, erlang, route_method, etc.) | dict passed everywhere | `SimulationConfig` | Immutable (frozen dataclass) | `SimulationConfig.to_engine_props()` |
| **engine_props** topology keys (topology, topology_info) | dict | `NetworkState` | Mutable (topology is fixed, spectrum changes) | `NetworkState.topology` property |
| **engine_props** modulation data (mod_per_bw, modulation_formats_dict) | dict | `SimulationConfig` | Immutable | Via config |
| **network_spectrum_dict** | `sdn_props.network_spectrum_dict` | `NetworkState` | Mutable (controlled methods) | `NetworkState.network_spectrum_dict` read-only property |
| **lightpath_status_dict** | `sdn_props.lightpath_status_dict` | `NetworkState` | Mutable (controlled methods) | `NetworkState.lightpath_status_dict` read-only property |
| **SDNProps** current request fields (request_id, source, dest, etc.) | SDNProps instance | `Request` object (passed as param) | Mutable | None needed |
| **SDNProps** result lists (core_list, band_list, etc.) | SDNProps instance | `AllocationResult` or list of `Lightpath` | Immutable result | None needed |
| **SDNProps** global tracking (allocated_requests, transponder_usage) | SDNProps instance | `NetworkState` | Mutable (controlled methods) | None needed |
| **SpectrumProps** input fields | SpectrumProps instance | Function parameters | N/A (not stored) | None needed |
| **SpectrumProps** output fields | SpectrumProps instance | `SpectrumResult` dataclass | Immutable result | None needed |
| **RoutingProps** output fields | RoutingProps instance | `RouteResult` dataclass | Immutable result | None needed |
| **RoutingProps** physical params | RoutingProps instance | `SimulationConfig` | Immutable | Via config |
| **Request dict** | `reqs_dict[(id, time)]` | `Request` dataclass | Mutable (status changes) | `Request.to_legacy_dict()` |
| **StatsProps** | StatsProps instance | `StatsCollector` class | Mutable | Keep as-is initially |

---

## Concrete Before/After Examples

### Example 1: A Single Request

**BEFORE (current dict structure):**
```python
# In reqs_dict with tuple key
reqs_dict[(42, 1.5)] = {
    "req_id": 42,
    "source": "node_0",
    "destination": "node_5",
    "arrive": 1.5,
    "depart": 11.8,
    "request_type": "arrival",
    "bandwidth": "100GHz",           # String! Key into mod_per_bw
    "mod_formats": {"16-QAM": {...}, "QPSK": {...}},
}

# After processing, outcome is scattered:
# - sdn_props.was_routed = True
# - sdn_props.lightpath_id_list = [7]
# - sdn_props.path_list = [0, 2, 5]
# - Stats updated separately
# - If blocked, sdn_props.was_routed = False and reason... somewhere
```

**AFTER (Request class):**
```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class RequestStatus(Enum):
    PENDING = "pending"
    ROUTED = "routed"
    BLOCKED = "blocked"
    RELEASED = "released"

@dataclass
class Request:
    # Identity (immutable after creation)
    request_id: int
    source: str
    destination: str
    bandwidth_gbps: int              # Numeric! e.g., 100
    arrive_time: float
    depart_time: float

    # Outcome (set during processing)
    status: RequestStatus = RequestStatus.PENDING
    block_reason: Optional[str] = None
    lightpath_ids: list[int] = field(default_factory=list)
    allocated_bandwidth_gbps: int = 0

    # Flags
    is_groomed: bool = False
    is_partially_groomed: bool = False
    is_sliced: bool = False

    @classmethod
    def from_legacy_dict(cls, key: tuple, data: dict, config: 'SimulationConfig') -> 'Request':
        """Create from current reqs_dict entry."""
        return cls(
            request_id=data["req_id"],
            source=data["source"],
            destination=data["destination"],
            bandwidth_gbps=config.bandwidth_key_to_gbps(data["bandwidth"]),
            arrive_time=data["arrive"],
            depart_time=data["depart"],
        )

    def to_legacy_dict(self) -> dict:
        """For backward compatibility during migration."""
        return {
            "req_id": self.request_id,
            "source": self.source,
            "destination": self.destination,
            # ...
        }

# Usage:
request = Request(
    request_id=42,
    source="node_0",
    destination="node_5",
    bandwidth_gbps=100,
    arrive_time=1.5,
    depart_time=11.8,
)

# After processing:
request.status = RequestStatus.ROUTED
request.lightpath_ids = [7]
# Everything about this request is ON the request object
```

---

### Example 2: Lightpath + Spectrum

**BEFORE (scattered across dicts):**
```python
# Spectrum in network_spectrum_dict
network_spectrum_dict[("node_0", "node_2")]["cores_matrix"]["c"][2][50:60] = 7  # lp_id=7
network_spectrum_dict[("node_0", "node_2")]["cores_matrix"]["c"][2][60] = -7    # guard

# Lightpath info in lightpath_status_dict (keyed by sorted endpoints!)
lightpath_status_dict[("node_0", "node_5")][7] = {
    "path": [0, 2, 5],
    "core": 2,
    "band": "c",
    "start_slot": 50,
    "end_slot": 60,
    "mod_format": "16-QAM",
    "lightpath_bandwidth": 100,
    "remaining_bandwidth": 100,
    "requests_dict": {42: 100},
    "time_bw_usage": {1.5: 100.0},
    "snr_cost": 18.5,
    "xt_cost": 0.3,
    "path_weight": 450.0,
    "is_degraded": False,
}

# To find "lightpaths from node_0 to node_5", you must:
# 1. Create sorted tuple: tuple(sorted(["node_0", "node_5"]))
# 2. Look up that key
# 3. Filter by actual path direction if needed
```

**AFTER (Lightpath class + NetworkState):**
```python
@dataclass
class Lightpath:
    lightpath_id: int
    path: list[str]                    # ["node_0", "node_2", "node_5"]

    # Spectrum allocation
    start_slot: int
    end_slot: int
    core: int
    band: str
    modulation: str

    # Capacity
    total_bandwidth_gbps: int
    remaining_bandwidth_gbps: int

    # Physical metrics
    path_weight_km: float
    snr_db: Optional[float] = None
    xt_cost: Optional[float] = None
    is_degraded: bool = False

    # Grooming: which requests use this lightpath
    request_allocations: dict[int, int] = field(default_factory=dict)  # req_id -> gbps

    @property
    def source(self) -> str:
        return self.path[0]

    @property
    def destination(self) -> str:
        return self.path[-1]

    @property
    def endpoint_key(self) -> tuple[str, str]:
        """Sorted endpoints for grouping (backward compat)."""
        return tuple(sorted([self.source, self.destination]))


class NetworkState:
    def __init__(self, topology: nx.Graph, config: 'SimulationConfig'):
        self._topology = topology
        self._config = config
        self._spectrum: dict[tuple[str, str], LinkSpectrum] = {}
        self._lightpaths: dict[int, Lightpath] = {}
        self._next_lightpath_id = 1
        self._initialize_spectrum()

    # --- Lightpath management ---

    def create_lightpath(
        self,
        path: list[str],
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
        modulation: str,
        bandwidth_gbps: int,
        path_weight_km: float,
    ) -> Lightpath:
        """Create and register a new lightpath."""
        lp = Lightpath(
            lightpath_id=self._next_lightpath_id,
            path=path,
            start_slot=start_slot,
            end_slot=end_slot,
            core=core,
            band=band,
            modulation=modulation,
            total_bandwidth_gbps=bandwidth_gbps,
            remaining_bandwidth_gbps=bandwidth_gbps,
            path_weight_km=path_weight_km,
        )
        self._next_lightpath_id += 1
        self._lightpaths[lp.lightpath_id] = lp
        self._allocate_spectrum_for_lightpath(lp)
        return lp

    def get_lightpaths_between(self, source: str, dest: str) -> list[Lightpath]:
        """Find all lightpaths between two endpoints (either direction)."""
        key = tuple(sorted([source, dest]))
        return [lp for lp in self._lightpaths.values() if lp.endpoint_key == key]

    def get_lightpaths_with_capacity(self, source: str, dest: str, min_bandwidth: int) -> list[Lightpath]:
        """Find lightpaths that can groom additional bandwidth."""
        return [
            lp for lp in self.get_lightpaths_between(source, dest)
            if lp.remaining_bandwidth_gbps >= min_bandwidth
        ]

    # --- Spectrum management ---

    def is_spectrum_available(
        self, path: list[str], start_slot: int, end_slot: int, core: int, band: str
    ) -> bool:
        """Check if spectrum range is free on all links of path."""
        for i in range(len(path) - 1):
            link_spectrum = self._spectrum[(path[i], path[i+1])]
            if not link_spectrum.is_range_free(band, core, start_slot, end_slot):
                return False
        return True

    def _allocate_spectrum_for_lightpath(self, lp: Lightpath) -> None:
        """Mark spectrum as occupied (internal)."""
        for i in range(len(lp.path) - 1):
            link = (lp.path[i], lp.path[i+1])
            self._spectrum[link].allocate(
                lp.band, lp.core, lp.start_slot, lp.end_slot, lp.lightpath_id
            )

    # --- Legacy compatibility ---

    @property
    def network_spectrum_dict(self) -> dict:
        """Read-only view in legacy format."""
        return {
            link: {
                'cores_matrix': ls.cores_matrix,
                'usage_count': ls.usage_count,
                'throughput': ls.throughput,
                'link_num': ls.link_num,
            }
            for link, ls in self._spectrum.items()
        }

    @property
    def lightpath_status_dict(self) -> dict:
        """Read-only view in legacy format."""
        result: dict = {}
        for lp in self._lightpaths.values():
            key = lp.endpoint_key
            if key not in result:
                result[key] = {}
            result[key][lp.lightpath_id] = {
                'path': lp.path,
                'core': lp.core,
                'band': lp.band,
                'start_slot': lp.start_slot,
                'end_slot': lp.end_slot,
                'mod_format': lp.modulation,
                'lightpath_bandwidth': lp.total_bandwidth_gbps,
                'remaining_bandwidth': lp.remaining_bandwidth_gbps,
                'requests_dict': lp.request_allocations.copy(),
                'snr_cost': lp.snr_db,
                'xt_cost': lp.xt_cost,
                'path_weight': lp.path_weight_km,
                'is_degraded': lp.is_degraded,
            }
        return result
```

---

### Example 3: Avoiding engine_props Becoming Another God Object

**The key**: Separate concerns into distinct objects that don't all get passed everywhere.

```python
# BEFORE: Everything in engine_props, passed to everything
class SDNController:
    def __init__(self, engine_props: dict):
        self.engine_props = engine_props  # Has topology, config, spectrum, everything

class Routing:
    def __init__(self, engine_props: dict):
        self.engine_props = engine_props  # Same huge dict

# AFTER: Each component gets only what it needs

@dataclass(frozen=True)
class SimulationConfig:
    """Immutable config - created once, never changes."""
    network_name: str
    route_method: str
    k_paths: int
    allocation_method: str
    # ... all the config values
    # NO topology, NO spectrum state, NO runtime

class NetworkState:
    """Mutable network state - owned by SimulationEngine."""
    # topology, spectrum, lightpaths, transponders
    # Provides controlled mutation methods

class SDNOrchestrator:
    def __init__(self, config: SimulationConfig, pipelines: PipelineSet):
        self.config = config        # Immutable, safe to store
        self.pipelines = pipelines  # References to pipeline objects
        # Does NOT own NetworkState - receives it per call

    def handle_arrival(self, request: Request, network_state: NetworkState) -> AllocationResult:
        # network_state passed in, not stored
        # This makes dependencies explicit

class RoutingPipeline:
    def __init__(self, config: SimulationConfig):
        self.config = config  # Only config, not network state

    def find_routes(self, source: str, dest: str, network_state: NetworkState) -> RouteResult:
        # Reads topology from network_state
        # Does not mutate network_state
        pass
```

**Why this works:**
- `SimulationConfig` is immutable - safe to store anywhere, no mutation bugs
- `NetworkState` is mutable but passed explicitly - you can see what functions need it
- Components store config (read-only) but receive state (read-write) as parameters
- No god object that everything holds a reference to

---

## How State Gets Centralized

### Single Source of Truth Table

| State Type | Single Source of Truth | How Others Access | How Others Mutate |
|------------|------------------------|-------------------|-------------------|
| **Configuration** | `SimulationConfig` (immutable) | Stored reference | Cannot mutate |
| **Topology** | `NetworkState._topology` | `network_state.topology` property | Cannot mutate (fixed at creation) |
| **Spectrum** | `NetworkState._spectrum` | `network_state.is_spectrum_available()`, `network_state.get_link_spectrum()` | `network_state.create_lightpath()`, `network_state.release_lightpath()` |
| **Lightpaths** | `NetworkState._lightpaths` | `network_state.get_lightpath()`, `network_state.get_lightpaths_between()` | Via `create_lightpath()`, `release_lightpath()` methods |
| **Per-request state** | `Request` object | Passed as function parameter | Mutate the passed object |
| **Statistics** | `StatsCollector` (owned by SimulationEngine) | `stats.record_allocation()` etc. | Via recording methods |

### Access Pattern

```python
class SimulationEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.network_state = NetworkState(topology, config)
        self.stats = StatsCollector()
        self.orchestrator = PipelineFactory.create_orchestrator(config)

    def handle_request(self, request: Request):
        if request.is_arrival:
            # Pass network_state - orchestrator can read/write via methods
            result = self.orchestrator.handle_arrival(request, self.network_state)
            self.stats.record_result(request, result)
        else:
            self.orchestrator.handle_release(request, self.network_state)
```

---

# STEP 2 - Standardized Return Shapes and Naming

## Naming Standard Table

| Concept | Old Names (inconsistent) | New Standard Name | Type | Lives In |
|---------|--------------------------|-------------------|------|----------|
| Lightpath identifier | `lightpath_id`, `lp_id`, `light_id` | `lightpath_id` | `int` | `Lightpath.lightpath_id` |
| Request identifier | `request_id`, `req_id` | `request_id` | `int` | `Request.request_id` |
| List of k paths | `paths_matrix`, `route_matrix` | `paths` | `list[list[str]]` | `RouteResult.paths` |
| Single path | `path_list`, `path`, `chosen_path_list` | `path` | `list[str]` | `Lightpath.path`, function params |
| Bandwidth | `bandwidth` (str), `bw`, `lightpath_bandwidth` | `bandwidth_gbps` | `int` | `Request.bandwidth_gbps`, `Lightpath.total_bandwidth_gbps` |
| Remaining capacity | `remaining_bandwidth`, `remaining_bw` | `remaining_bandwidth_gbps` | `int` | `Lightpath.remaining_bandwidth_gbps` |
| Spectrum start | `start_slot`, `start_index` | `start_slot` | `int` | `SpectrumResult.start_slot`, `Lightpath.start_slot` |
| Spectrum end | `end_slot`, `end_index` | `end_slot` | `int` | `SpectrumResult.end_slot`, `Lightpath.end_slot` |
| Core number | `core`, `core_num`, `core_number` | `core` | `int` | `SpectrumResult.core`, `Lightpath.core` |
| Optical band | `band`, `current_band` | `band` | `str` | `SpectrumResult.band`, `Lightpath.band` |
| Modulation format | `modulation`, `mod_format`, `modulation_format` | `modulation` | `str` | `SpectrumResult.modulation`, `Lightpath.modulation` |
| Path length/weight | `path_weight`, `weight`, `path_length` | `path_weight_km` | `float` | `RouteResult.weights_km`, `Lightpath.path_weight_km` |
| Was request groomed | `was_groomed` | `is_groomed` | `bool` | `Request.is_groomed` |
| Partial grooming | `was_partially_groomed` | `is_partially_groomed` | `bool` | `Request.is_partially_groomed` |
| Allocation success | `was_routed`, `is_free`, `success` | `success` | `bool` | `AllocationResult.success` |

---

## Standard Return Types

### Layer Responsibilities

| Layer | Input | Output | Responsibility |
|-------|-------|--------|----------------|
| **Routing** | source, dest, request | `RouteResult` | Find candidate paths. NO feasibility check. |
| **Spectrum** | path, modulations | `SpectrumResult` | Find available slots. Feasibility = `is_free`. |
| **Allocation** | request, path, spectrum result | `AllocationResult` | Actually allocate. Final success/failure. |
| **Grooming** | request, network_state | `GroomingResult` | Check existing lightpaths. Bandwidth check. |
| **Slicing** | request, path | `AllocationResult` | Multiple smaller allocations. |

### Dataclass Definitions

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class BlockReason(Enum):
    NONE = "none"
    NO_ROUTE = "no_route"           # No path exists
    NO_SPECTRUM = "no_spectrum"     # Path exists but no slots
    SNR_FAILURE = "snr_failure"     # SNR below threshold
    XT_THRESHOLD = "xt_threshold"   # Crosstalk too high
    DISTANCE = "distance"           # Path too long for any modulation


@dataclass(frozen=True)
class RouteResult:
    """Output of routing - candidate paths only, no feasibility."""
    paths: list[list[str]]                      # k candidate paths
    modulations: list[list[str]]                # Valid modulations per path
    weights_km: list[float]                     # Path lengths in km
    backup_paths: Optional[list[list[str]]] = None  # For 1+1 protection
    backup_modulations: Optional[list[list[str]]] = None


@dataclass(frozen=True)
class SpectrumResult:
    """Output of spectrum search - feasibility at spectrum level."""
    is_free: bool                               # Did we find available spectrum?
    start_slot: Optional[int] = None
    end_slot: Optional[int] = None
    core: Optional[int] = None
    band: Optional[str] = None
    modulation: Optional[str] = None
    slots_needed: Optional[int] = None

    # For backup path (1+1)
    backup_start_slot: Optional[int] = None
    backup_end_slot: Optional[int] = None
    backup_core: Optional[int] = None
    backup_band: Optional[str] = None


@dataclass(frozen=True)
class GroomingResult:
    """Output of grooming attempt."""
    fully_groomed: bool                         # Request fully satisfied
    partially_groomed: bool                     # Some bandwidth groomed
    bandwidth_groomed_gbps: int                 # How much we groomed
    remaining_bandwidth_gbps: int               # Still needed
    lightpaths_used: list[int]                  # Lightpath IDs used
    forced_path: Optional[list[str]] = None    # For partial: continue on this path


@dataclass
class AllocationResult:
    """Final outcome of allocation attempt."""
    success: bool
    block_reason: BlockReason = BlockReason.NONE
    lightpaths_created: list[int] = field(default_factory=list)
    lightpaths_groomed: list[int] = field(default_factory=list)
    total_bandwidth_allocated_gbps: int = 0
    is_sliced: bool = False
    is_groomed: bool = False
    is_partially_groomed: bool = False
```

---

## Before/After: Simple KSP Flow (No Grooming/Slicing/SNR)

### BEFORE (Current Code)

```python
# In SDNController.handle_event() - simplified
def handle_event(self, request_dict: dict, request_type: str):
    if request_type == "arrival":
        # Populate sdn_props with request data
        self.sdn_props.request_id = request_dict["req_id"]
        self.sdn_props.source = request_dict["source"]
        self.sdn_props.destination = request_dict["destination"]
        self.sdn_props.bandwidth = request_dict["bandwidth"]
        # ... more fields

        # Call routing - mutates route_props
        self.routing_obj.get_route()
        # Results now in self.routing_obj.route_props.paths_matrix

        # Try each path
        for path_index in range(len(self.route_props.paths_matrix)):
            self.sdn_props.path_list = self.route_props.paths_matrix[path_index]
            self.sdn_props.path_index = path_index

            # Call spectrum - mutates spectrum_props
            mod_format_list = self.route_props.modulation_formats_matrix[path_index]
            self.spectrum_obj.get_spectrum(mod_format_list)

            if self.spectrum_obj.spectrum_props.is_free:
                # Generate lightpath ID
                lp_id = self.sdn_props.get_lightpath_id()

                # Allocate - mutates network_spectrum_dict
                self.allocate()

                # Update result lists
                self.sdn_props.lightpath_id_list.append(lp_id)
                self.sdn_props.was_routed = True
                return

        # All paths failed
        self.sdn_props.was_routed = False

# Return type: None (results scattered in sdn_props)
# Caller must read: sdn_props.was_routed, sdn_props.lightpath_id_list, etc.
```

### AFTER (New Design)

```python
# In RoutingPipeline
class KSPRoutingPipeline:
    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState
    ) -> RouteResult:
        """Find k-shortest paths. Returns structure only."""
        paths = self._compute_k_shortest(source, destination, network_state.topology)
        modulations = [self._get_valid_modulations(p, bandwidth_gbps) for p in paths]
        weights = [self._compute_path_weight(p, network_state.topology) for p in paths]

        return RouteResult(
            paths=paths,
            modulations=modulations,
            weights_km=weights,
        )


# In SpectrumPipeline
class FirstFitSpectrumPipeline:
    def find_spectrum(
        self,
        path: list[str],
        modulations: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """Find available spectrum on path. Returns feasibility."""
        for modulation in modulations:
            if modulation is None:
                continue
            slots_needed = self._calculate_slots(bandwidth_gbps, modulation)

            for band in self.config.band_list:
                for core in range(self.config.cores_per_link):
                    start = self._find_first_fit(path, band, core, slots_needed, network_state)
                    if start is not None:
                        return SpectrumResult(
                            is_free=True,
                            start_slot=start,
                            end_slot=start + slots_needed,
                            core=core,
                            band=band,
                            modulation=modulation,
                            slots_needed=slots_needed,
                        )

        return SpectrumResult(is_free=False)


# In SDNOrchestrator - THIN, no business logic
class SDNOrchestrator:
    def handle_arrival(
        self,
        request: Request,
        network_state: NetworkState
    ) -> AllocationResult:
        """Orchestrate allocation - delegates to pipelines."""

        # 1. Route
        route_result = self.routing.find_routes(
            request.source, request.destination, request.bandwidth_gbps, network_state
        )

        if not route_result.paths:
            return AllocationResult(success=False, block_reason=BlockReason.NO_ROUTE)

        # 2. Try each path
        for i, path in enumerate(route_result.paths):
            spectrum_result = self.spectrum.find_spectrum(
                path, route_result.modulations[i], request.bandwidth_gbps, network_state
            )

            if spectrum_result.is_free:
                # 3. Allocate
                lightpath = network_state.create_lightpath(
                    path=path,
                    start_slot=spectrum_result.start_slot,
                    end_slot=spectrum_result.end_slot,
                    core=spectrum_result.core,
                    band=spectrum_result.band,
                    modulation=spectrum_result.modulation,
                    bandwidth_gbps=request.bandwidth_gbps,
                    path_weight_km=route_result.weights_km[i],
                )

                # 4. Link request to lightpath
                lightpath.request_allocations[request.request_id] = request.bandwidth_gbps
                request.lightpath_ids.append(lightpath.lightpath_id)
                request.status = RequestStatus.ROUTED

                return AllocationResult(
                    success=True,
                    lightpaths_created=[lightpath.lightpath_id],
                    total_bandwidth_allocated_gbps=request.bandwidth_gbps,
                )

        return AllocationResult(success=False, block_reason=BlockReason.NO_SPECTRUM)


# CALLER USAGE:
request = Request(request_id=42, source="A", destination="Z", bandwidth_gbps=100, ...)
result = orchestrator.handle_arrival(request, network_state)

if result.success:
    print(f"Allocated on lightpaths: {result.lightpaths_created}")
else:
    print(f"Blocked: {result.block_reason}")
# Everything is in the result object - no scattered state to read
```

---

# STEP 3 - Interaction of Functionalities

## Restating the Problem

**The original nightmare:**

1. **Grooming** introduced the concept that one request can use multiple existing lightpaths (partial grooming) or establish new ones. This broke code that assumed `request_id` maps 1:1 to `lightpath_id`.

2. **Slicing** splits a single request into multiple smaller lightpaths (different from grooming - these are NEW lightpaths, not reusing existing ones). This added another dimension of "one request -> many lightpaths."

3. **SNR** adds validation that can fail AFTER spectrum is found but BEFORE allocation is finalized. This requires rollback capability.

4. **Combinations** like "grooming + slicing + SNR" create a matrix of behaviors. The current SDN controller handles this with nested conditionals, making it unmaintainable.

5. **RL gave up** and created its own mini-simulator because it couldn't cleanly hook into this mess. It duplicates feasibility checking in `mock_handle_arrival()`.

---

## Composition Model: Ordered Pipeline with Optional Stages

**Model chosen: Linear pipeline with optional stages.**

```
Request
   |
   v
+-----------------------------------------------------+
| STAGE 1: GROOMING (optional)                        |
| - Check existing lightpaths                         |
| - If fully groomed -> DONE                          |
| - If partial -> reduce remaining_bw, set forced_path|
+-----------------------------------------------------+
   |
   v
+-----------------------------------------------------+
| STAGE 2: ROUTING                                    |
| - Find k candidate paths                            |
| - (May be constrained by forced_path from grooming) |
+-----------------------------------------------------+
   |
   v
+-----------------------------------------------------+
| STAGE 3: SPECTRUM + ALLOCATION                      |
| - For each path, find spectrum                      |
| - If found, allocate                                |
| - If SNR enabled: validate SNR                      |
|   - If SNR fails: rollback, try next                |
| - If slicing enabled and standard fails:            |
|   - Try slicing (multiple smaller lightpaths)       |
+-----------------------------------------------------+
   |
   v
AllocationResult
```

**Key design decisions:**

1. **Grooming runs first** because it can short-circuit everything (fully groomed = done).
2. **Routing is always required** (even if path is forced, we need modulation formats).
3. **Spectrum + Allocation + SNR are tightly coupled** - SNR validation happens immediately after allocation with rollback on failure.
4. **Slicing is a fallback** when standard allocation fails.

**Configuration determines which stages are active:**

```python
@dataclass
class PipelineConfig:
    grooming_enabled: bool = False
    slicing_enabled: bool = False
    max_slices: int = 1
    snr_enabled: bool = False
    snr_recheck: bool = False
```

---

## Scenario Walkthroughs

### Scenario A: Plain KSP (no grooming, no slicing, no SNR)

**Config:** `grooming_enabled=False, slicing_enabled=False, snr_enabled=False`

**Flow:**
```
Request(id=1, src=A, dst=Z, bw=100)
   |
   | [GROOMING: skipped - not enabled]
   |
   v
ROUTING.find_routes(A, Z)
   -> RouteResult(paths=[[A,B,Z], [A,C,Z]], modulations=[[QPSK,16QAM], [QPSK]], weights=[200, 250])
   |
   v
For path [A,B,Z]:
   SPECTRUM.find_spectrum(path=[A,B,Z], mods=[QPSK,16QAM], bw=100)
   -> SpectrumResult(is_free=True, start=10, end=18, core=0, band=c, mod=QPSK)
   |
   v
   [SNR: skipped - not enabled]
   |
   v
NETWORK_STATE.create_lightpath(path=[A,B,Z], slots=10-18, core=0, band=c, mod=QPSK, bw=100)
   -> Lightpath(id=1, ...)
   |
   v
AllocationResult(success=True, lightpaths_created=[1], bw_allocated=100)
```

**Objects passed:**
- `Request` -> Routing (for bandwidth)
- `RouteResult` -> Spectrum (for paths, modulations)
- `SpectrumResult` -> NetworkState.create_lightpath
- `AllocationResult` <- returned to caller

---

### Scenario B: Grooming + SNR, no slicing

**Config:** `grooming_enabled=True, slicing_enabled=False, snr_enabled=True`

**Flow:**
```
Request(id=2, src=A, dst=Z, bw=50)
   |
   v
GROOMING.try_groom(request, network_state)
   - Finds Lightpath(id=1) with remaining_bw=50
   - Allocates 50 Gbps from Lightpath 1 to Request 2
   -> GroomingResult(fully_groomed=True, bw_groomed=50, lightpaths_used=[1])
   |
   v
   [Fully groomed - DONE, skip routing/spectrum]
   |
   v
AllocationResult(success=True, lightpaths_groomed=[1], is_groomed=True, bw_allocated=50)
```

**Partial grooming case:**
```
Request(id=3, src=A, dst=Z, bw=150)
   |
   v
GROOMING.try_groom(request, network_state)
   - Finds Lightpath(id=1) with remaining_bw=0 (used up)
   - Finds Lightpath(id=7) with remaining_bw=80, same path [A,B,Z]
   - Allocates 80 Gbps from Lightpath 7
   -> GroomingResult(fully_groomed=False, partially_groomed=True,
                    bw_groomed=80, remaining_bw=70,
                    lightpaths_used=[7], forced_path=[A,B,Z])
   |
   v
ROUTING.find_routes(A, Z, forced_path=[A,B,Z])
   -> RouteResult(paths=[[A,B,Z]], ...)  # Only the forced path
   |
   v
For path [A,B,Z]:
   SPECTRUM.find_spectrum(path=[A,B,Z], bw=70)  # Remaining bandwidth
   -> SpectrumResult(is_free=True, start=50, end=57, ...)
   |
   v
   NETWORK_STATE.create_lightpath(..., bw=70)
   -> Lightpath(id=8)
   |
   v
   SNR.validate(lightpath=8, network_state)
   -> True  # SNR OK
   |
   v
AllocationResult(success=True, lightpaths_created=[8], lightpaths_groomed=[7],
                 is_partially_groomed=True, bw_allocated=150)
```

---

### Scenario C: Slicing only, no grooming, with SNR

**Config:** `grooming_enabled=False, slicing_enabled=True, max_slices=4, snr_enabled=True`

**Flow:**
```
Request(id=4, src=A, dst=Z, bw=400)
   |
   | [GROOMING: skipped]
   |
   v
ROUTING.find_routes(A, Z)
   -> RouteResult(paths=[[A,B,Z], [A,C,Z]], mods=[[None], [QPSK]], weights=[200, 350])
   # Path [A,B,Z] is too short for 400Gbps - no valid modulation (None)
   # Path [A,C,Z] only supports QPSK for 400Gbps
   |
   v
For path [A,B,Z]:
   SPECTRUM.find_spectrum(path=[A,B,Z], mods=[None], bw=400)
   -> SpectrumResult(is_free=False)  # No valid modulation
   |
   v
   [Slicing enabled - try to slice]
   SLICING.try_slice(request, path=[A,B,Z], network_state)
      - 400 Gbps can be split into 4 x 100 Gbps (each can use 16-QAM)
      - For each slice:
         - find_spectrum(bw=100) -> success
         - create_lightpath -> Lightpath
         - SNR.validate -> pass/fail
            - If fail: rollback this slice, try next config
      -> AllocationResult(success=True, lightpaths_created=[9,10,11,12], is_sliced=True)
   |
   v
AllocationResult(success=True, lightpaths_created=[9,10,11,12], is_sliced=True, bw=400)
```

---

### Scenario D: Grooming + Slicing + SNR (The Hard Case)

**Config:** `grooming_enabled=True, slicing_enabled=True, max_slices=4, snr_enabled=True`

**Flow:**
```
Request(id=5, src=A, dst=Z, bw=500)
   |
   v
GROOMING.try_groom(request, network_state)
   - Finds Lightpath(id=7) with remaining_bw=100
   - Allocates 100 Gbps
   -> GroomingResult(partially_groomed=True, bw_groomed=100, remaining_bw=400,
                    forced_path=[A,B,Z], lightpaths_used=[7])
   |
   v
ROUTING.find_routes(A, Z, forced_path=[A,B,Z])
   -> RouteResult(paths=[[A,B,Z]], mods=[[None]], ...)  # No mod for 400Gbps on this path
   |
   v
For path [A,B,Z]:
   SPECTRUM.find_spectrum(bw=400) -> SpectrumResult(is_free=False)
   |
   v
   [Slicing fallback]
   SLICING.try_slice(bw=400, path=[A,B,Z])
      - Split into 4 x 100 Gbps
      - For each slice:
         - find_spectrum -> success
         - create_lightpath -> Lightpath(id=13..16)
         - SNR.validate ->
            - Slices 13,14,15 pass
            - Slice 16 FAILS SNR
            - Rollback slice 16
            - Try alternate slicing: 2 x 200 Gbps? -> also fails
            - All slicing attempts exhausted
      -> AllocationResult(success=False)
   |
   v
   [Path [A,B,Z] fully failed, but we already groomed!]
   [Decision point: do we accept partial? Check config.can_partially_serve]
   |
   v
   If can_partially_serve=True:
      # Accept the groomed 100 + successfully sliced lightpaths
      -> AllocationResult(success=True, partial=True,
                         lightpaths_groomed=[7], lightpaths_created=[13,14,15],
                         bw_allocated=400, bw_remaining=100)
   |
   If can_partially_serve=False:
      # Rollback grooming allocation too
      GROOMING.rollback(request, lightpath=7)
      -> AllocationResult(success=False, block_reason=SNR_FAILURE)
```

**Key decisions made by configuration:**
- `grooming_enabled` -> whether to check existing lightpaths first
- `slicing_enabled` + `max_slices` -> whether to try slicing as fallback
- `snr_enabled` -> whether to validate SNR after allocation
- `can_partially_serve` -> whether to accept partial allocation

**No wrapper-of-wrappers:** Each stage is a flat call. The orchestrator manages the flow, but each pipeline is independent.

---

## RL Integration

### What Disappears from RL

**BEFORE (current RL):**
```python
# In CoreUtilHelpers
def mock_handle_arrival(self, path_list, mod_format):
    """DUPLICATE feasibility check - creates temporary SpectrumAssignment."""
    spectrum_obj = SpectrumAssignment(self.engine_props, self.sdn_props, ...)
    spectrum_obj.spectrum_props.forced_index = None
    spectrum_obj.get_spectrum(mod_format_list=mod_format_list)
    return spectrum_obj.spectrum_props.is_free is True

# In SimEnvObs
def _get_paths_slots(self, bandwidth):
    """Check each path feasibility - ANOTHER duplication."""
    for path in paths:
        is_feasible = self._mock_check_path(path, bandwidth)
        # ...
```

**AFTER (unified):**
```python
# mock_handle_arrival is DELETED
# _get_paths_slots uses the real pipelines
```

### RL Adapter

```python
class RLSimulationAdapter:
    """
    RL-friendly interface that uses the SAME pipelines as simulation.
    No duplicated logic.
    """

    def __init__(
        self,
        config: SimulationConfig,
        routing: RoutingPipeline,
        spectrum: SpectrumPipeline,
    ):
        self.config = config
        self.routing = routing
        self.spectrum = spectrum

    def get_path_options(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> list[PathOption]:
        """
        Get candidate paths with feasibility info.
        Uses SAME routing and spectrum pipelines - no duplication.
        """
        # Use real routing pipeline
        route_result = self.routing.find_routes(source, destination, bandwidth_gbps, network_state)

        options = []
        for i, path in enumerate(route_result.paths):
            # Use real spectrum pipeline for feasibility
            spectrum_result = self.spectrum.find_spectrum(
                path, route_result.modulations[i], bandwidth_gbps, network_state
            )

            options.append(PathOption(
                path_index=i,
                path=path,
                weight_km=route_result.weights_km[i],
                is_feasible=spectrum_result.is_free,
                modulation=spectrum_result.modulation if spectrum_result.is_free else None,
            ))

        return options

    def get_action_mask(self, options: list[PathOption]) -> list[bool]:
        """Which actions (path indices) are valid."""
        return [opt.is_feasible for opt in options]

    def apply_action(
        self,
        action: int,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
        orchestrator: SDNOrchestrator,
    ) -> tuple[bool, float]:
        """Apply RL action using real orchestrator. Returns (success, reward)."""
        if action >= len(options) or not options[action].is_feasible:
            return False, -1.0

        # Force the chosen path
        forced_path = options[action].path

        # Use REAL orchestrator - same code path as non-RL
        result = orchestrator.handle_arrival(
            request,
            network_state,
            forced_path=forced_path,
        )

        reward = 1.0 if result.success else -1.0
        return result.success, reward


@dataclass
class PathOption:
    path_index: int
    path: list[str]
    weight_km: float
    is_feasible: bool
    modulation: Optional[str]
```

### Updated RL Environment

```python
class UnifiedSimEnv(gym.Env):
    """Gymnasium env using unified adapter."""

    def __init__(self, config: SimulationConfig, topology: nx.Graph):
        self.config = config
        self.network_state = NetworkState(topology, config)
        self.orchestrator = PipelineFactory.create_orchestrator(config)
        self.adapter = RLSimulationAdapter(
            config=config,
            routing=self.orchestrator.routing,      # SAME routing
            spectrum=self.orchestrator.spectrum,    # SAME spectrum
        )

    def reset(self, seed=None, options=None):
        self.network_state = NetworkState(self.topology, self.config)
        self.requests = self._generate_requests()
        self.current_idx = 0
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        request = self.requests[self.current_idx]
        self.path_options = self.adapter.get_path_options(
            request.source, request.destination, request.bandwidth_gbps, self.network_state
        )
        # Build observation from path_options
        return {...}

    def _get_info(self):
        return {'action_mask': self.adapter.get_action_mask(self.path_options)}

    def step(self, action: int):
        request = self.requests[self.current_idx]

        # Use adapter which uses REAL orchestrator
        success, reward = self.adapter.apply_action(
            action, request, self.path_options, self.network_state, self.orchestrator
        )

        self.current_idx += 1
        terminated = self.current_idx >= len(self.requests)

        return self._get_obs(), reward, terminated, False, self._get_info()
```

**What this achieves:**
- RL uses `routing.find_routes()` - same as simulation
- RL uses `spectrum.find_spectrum()` - same as simulation
- RL uses `orchestrator.handle_arrival()` - same as simulation
- No `mock_handle_arrival()`, no `_get_paths_slots()` duplication
- Adding SNR/grooming/slicing to RL = just enable it in config

---

# STEP 4 - SDN Controller as Thin Router

## Intended Role of SDN Controller

The SDN controller (now `SDNOrchestrator`):

1. **Receives** an incoming `Request` object
2. **Knows** which pipelines are enabled via `PipelineConfig`
3. **Delegates** to pipeline classes in the correct order
4. **Returns** an `AllocationResult`
5. **Does NOT** contain grooming/slicing/SNR algorithm code
6. **Does NOT** have if/else branches based on feature internals

---

## Corrected Design

### Module Diagram

```
+-------------------------------------------------------------------------+
|                          SimulationEngine                               |
|  - owns NetworkState                                                    |
|  - owns StatsCollector                                                  |
|  - owns SDNOrchestrator                                                 |
|  - calls orchestrator.handle_arrival(request, network_state)            |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                          SDNOrchestrator                                |
|  - THIN: ~100 lines, no algorithm code                                  |
|  - holds references to pipelines                                        |
|  - orchestrates: grooming? -> routing -> spectrum+snr -> slicing        |
+-------------------------------------------------------------------------+
           |              |              |              |
           v              v              v              v
    +----------+   +----------+   +----------+   +----------+
    | Grooming |   | Routing  |   | Spectrum |   | Slicing  |
    | Pipeline |   | Pipeline |   | Pipeline |   | Pipeline |
    |          |   |          |   |          |   |          |
    | try_groom|   |find_routes|  |find_spec |   |try_slice |
    | release  |   |          |   | allocate |   |          |
    +----------+   +----------+   +----------+   +----------+
                                        |
                                        v
                                 +----------+
                                 |   SNR    |
                                 | Pipeline |
                                 |          |
                                 | validate |
                                 | recheck  |
                                 +----------+
                                        |
                                        v
+-------------------------------------------------------------------------+
|                           NetworkState                                  |
|  - _topology: nx.Graph                                                  |
|  - _spectrum: dict[link, LinkSpectrum]                                  |
|  - _lightpaths: dict[id, Lightpath]                                     |
|  - create_lightpath(), release_lightpath()                              |
|  - is_spectrum_available()                                              |
|  - get_lightpaths_between()                                             |
+-------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------+
|                    Domain Objects                                       |
|  Request          Lightpath          SimulationConfig                   |
|  - request_id     - lightpath_id     - route_method                     |
|  - source         - path             - k_paths                          |
|  - destination    - start_slot       - grooming_enabled                 |
|  - bandwidth_gbps - end_slot         - slicing_enabled                  |
|  - status         - core, band       - snr_enabled                      |
|  - lightpath_ids  - remaining_bw     - ...                              |
+-------------------------------------------------------------------------+
```

---

### Pseudocode for `handle_arrival(request)`

```python
class SDNOrchestrator:
    """
    THIN orchestrator - no algorithm code.
    Just coordinates pipeline calls in order.
    """

    def __init__(self, config: SimulationConfig, pipelines: PipelineSet):
        self.config = config
        self.grooming = pipelines.grooming      # Optional[GroomingPipeline]
        self.routing = pipelines.routing        # RoutingPipeline
        self.spectrum = pipelines.spectrum      # SpectrumPipeline
        self.snr = pipelines.snr                # Optional[SNRPipeline]
        self.slicing = pipelines.slicing        # Optional[SlicingPipeline]

    def handle_arrival(
        self,
        request: Request,
        network_state: NetworkState,
        forced_path: Optional[list[str]] = None,
    ) -> AllocationResult:
        """
        Orchestrate allocation attempt.

        This method contains NO algorithm logic - only pipeline coordination.
        """
        groomed_lightpaths = []
        remaining_bw = request.bandwidth_gbps

        # --- STAGE 1: Grooming (if enabled) ---
        if self.grooming is not None and self.config.grooming_enabled:
            groom_result = self.grooming.try_groom(request, network_state)

            if groom_result.fully_groomed:
                request.status = RequestStatus.ROUTED
                request.is_groomed = True
                return AllocationResult(
                    success=True,
                    lightpaths_groomed=groom_result.lightpaths_used,
                    is_groomed=True,
                    total_bandwidth_allocated_gbps=groom_result.bandwidth_groomed_gbps,
                )

            if groom_result.partially_groomed:
                groomed_lightpaths = groom_result.lightpaths_used
                remaining_bw = groom_result.remaining_bandwidth_gbps
                forced_path = groom_result.forced_path  # Must use same path

        # --- STAGE 2: Routing ---
        route_result = self.routing.find_routes(
            request.source, request.destination, remaining_bw, network_state, forced_path
        )

        if not route_result.paths:
            return self._handle_failure(request, groomed_lightpaths, BlockReason.NO_ROUTE, network_state)

        # --- STAGE 3: Try each path ---
        for path_idx, path in enumerate(route_result.paths):
            result = self._try_allocate_on_path(
                request, path, route_result.modulations[path_idx],
                route_result.weights_km[path_idx], remaining_bw, network_state
            )

            if result is not None:
                # Success - combine with groomed lightpaths
                return AllocationResult(
                    success=True,
                    lightpaths_created=result.lightpaths_created,
                    lightpaths_groomed=groomed_lightpaths,
                    is_groomed=len(groomed_lightpaths) > 0,
                    is_partially_groomed=len(groomed_lightpaths) > 0 and len(result.lightpaths_created) > 0,
                    is_sliced=result.is_sliced,
                    total_bandwidth_allocated_gbps=request.bandwidth_gbps,
                )

        # All paths failed
        return self._handle_failure(request, groomed_lightpaths, BlockReason.NO_SPECTRUM, network_state)

    def _try_allocate_on_path(
        self,
        request: Request,
        path: list[str],
        modulations: list[str],
        weight_km: float,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> Optional[AllocationResult]:
        """Try standard allocation, then slicing fallback."""

        # Try standard allocation
        spectrum_result = self.spectrum.find_spectrum(path, modulations, bandwidth_gbps, network_state)

        if spectrum_result.is_free:
            alloc_result = self._allocate_and_validate(
                request, path, spectrum_result, weight_km, bandwidth_gbps, network_state
            )
            if alloc_result is not None:
                return alloc_result

        # Slicing fallback (if enabled)
        if self.slicing is not None and self.config.slicing_enabled:
            return self.slicing.try_slice(
                request, path, modulations, bandwidth_gbps, network_state,
                self.spectrum, self.snr
            )

        return None

    def _allocate_and_validate(
        self,
        request: Request,
        path: list[str],
        spectrum_result: SpectrumResult,
        weight_km: float,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> Optional[AllocationResult]:
        """Allocate spectrum, validate SNR, rollback on failure."""

        lightpath = network_state.create_lightpath(
            path=path,
            start_slot=spectrum_result.start_slot,
            end_slot=spectrum_result.end_slot,
            core=spectrum_result.core,
            band=spectrum_result.band,
            modulation=spectrum_result.modulation,
            bandwidth_gbps=bandwidth_gbps,
            path_weight_km=weight_km,
        )

        # SNR validation (if enabled)
        if self.snr is not None and self.config.snr_enabled:
            snr_ok = self.snr.validate(lightpath, network_state)
            if not snr_ok:
                network_state.release_lightpath(lightpath.lightpath_id)
                return None

        # Success
        lightpath.request_allocations[request.request_id] = bandwidth_gbps
        request.lightpath_ids.append(lightpath.lightpath_id)

        return AllocationResult(
            success=True,
            lightpaths_created=[lightpath.lightpath_id],
            total_bandwidth_allocated_gbps=bandwidth_gbps,
        )

    def _handle_failure(
        self,
        request: Request,
        groomed_lightpaths: list[int],
        reason: BlockReason,
        network_state: NetworkState,
    ) -> AllocationResult:
        """Handle allocation failure - possibly accept partial or rollback."""

        if groomed_lightpaths and self.config.can_partially_serve:
            # Accept partial grooming
            request.status = RequestStatus.ROUTED
            request.is_partially_groomed = True
            return AllocationResult(
                success=True,
                lightpaths_groomed=groomed_lightpaths,
                is_partially_groomed=True,
                block_reason=reason,  # Record why we couldn't fully serve
            )

        if groomed_lightpaths:
            # Rollback grooming
            self.grooming.rollback(request, groomed_lightpaths, network_state)

        request.status = RequestStatus.BLOCKED
        request.block_reason = reason.value
        return AllocationResult(success=False, block_reason=reason)
```

**Note what's NOT in SDNOrchestrator:**
- No K-shortest path algorithm
- No first-fit/best-fit spectrum search
- No SNR calculation formulas
- No slicing segmentation logic
- No grooming bandwidth matching
- Just: call grooming -> call routing -> call spectrum -> call snr -> call slicing

---

### PipelineFactory and PipelineSet

```python
@dataclass
class PipelineSet:
    """Container for all pipeline instances."""
    routing: RoutingPipeline
    spectrum: SpectrumPipeline
    grooming: Optional[GroomingPipeline] = None
    snr: Optional[SNRPipeline] = None
    slicing: Optional[SlicingPipeline] = None


class PipelineFactory:
    """Creates pipelines based on configuration."""

    @staticmethod
    def create_routing(config: SimulationConfig) -> RoutingPipeline:
        """Create routing pipeline based on config.route_method."""
        if config.route_method == "k_shortest_path":
            return KSPRoutingPipeline(config)
        elif config.route_method == "congestion_aware":
            return CongestionAwareRoutingPipeline(config)
        # ... etc
        raise ValueError(f"Unknown route method: {config.route_method}")

    @staticmethod
    def create_spectrum(config: SimulationConfig) -> SpectrumPipeline:
        if config.allocation_method == "first_fit":
            return FirstFitSpectrumPipeline(config)
        elif config.allocation_method == "best_fit":
            return BestFitSpectrumPipeline(config)
        # ...

    @staticmethod
    def create_grooming(config: SimulationConfig) -> Optional[GroomingPipeline]:
        if not config.grooming_enabled:
            return None
        return GroomingPipelineImpl(config)

    @staticmethod
    def create_snr(config: SimulationConfig) -> Optional[SNRPipeline]:
        if not config.snr_enabled:
            return None
        if config.snr_type == "snr_e2e":
            return E2ESNRPipeline(config)
        # ...

    @staticmethod
    def create_slicing(config: SimulationConfig) -> Optional[SlicingPipeline]:
        if not config.slicing_enabled:
            return None
        return SlicingPipelineImpl(config)

    @classmethod
    def create_orchestrator(cls, config: SimulationConfig) -> SDNOrchestrator:
        pipelines = PipelineSet(
            routing=cls.create_routing(config),
            spectrum=cls.create_spectrum(config),
            grooming=cls.create_grooming(config),
            snr=cls.create_snr(config),
            slicing=cls.create_slicing(config),
        )
        return SDNOrchestrator(config, pipelines)
```

---

### Adding/Removing a Feature is Local

**Example: Adding a new "QoS Priority" feature**

1. **Create** `QoSPipeline` protocol:
   ```python
   class QoSPipeline(Protocol):
       def prioritize_paths(self, paths: list[list[str]], request: Request) -> list[list[str]]:
           """Reorder paths based on QoS requirements."""
           ...
   ```

2. **Implement** `QoSPriorityPipeline`:
   ```python
   class QoSPriorityPipeline:
       def prioritize_paths(self, paths, request):
           if request.qos_class == "premium":
               return sorted(paths, key=lambda p: len(p))  # Shortest first
           return paths
   ```

3. **Add to** `PipelineSet`:
   ```python
   @dataclass
   class PipelineSet:
       # ...existing...
       qos: Optional[QoSPipeline] = None
   ```

4. **Add to** `PipelineFactory`:
   ```python
   @staticmethod
   def create_qos(config) -> Optional[QoSPipeline]:
       if not config.qos_enabled:
           return None
       return QoSPriorityPipeline(config)
   ```

5. **Update** `SDNOrchestrator.handle_arrival()`:
   ```python
   # After routing
   if self.qos is not None:
       route_result = RouteResult(
           paths=self.qos.prioritize_paths(route_result.paths, request),
           # ...
       )
   ```

**Files changed:**
- `fusion/interfaces/pipelines.py` - add protocol
- `fusion/pipelines/qos_pipeline.py` - new file
- `fusion/core/pipeline_factory.py` - add factory method
- `fusion/core/orchestrator.py` - add 3-line integration

**Files NOT changed:**
- Routing pipeline
- Spectrum pipeline
- Grooming pipeline
- SNR pipeline
- Slicing pipeline
- RL adapter
- NetworkState
- Request/Lightpath

---

### How This Prevents RL Mini-Simulator

**Problem**: RL needed its own feasibility checking because hooking into SDNController was too complex.

**Solution**: RL uses the same pipeline objects:

```python
class RLSimulationAdapter:
    def __init__(self, orchestrator: SDNOrchestrator):
        # REUSE the same pipeline instances
        self.routing = orchestrator.routing
        self.spectrum = orchestrator.spectrum
        self.orchestrator = orchestrator

    def get_path_options(self, request, network_state):
        # Uses SAME routing pipeline
        route_result = self.routing.find_routes(...)

        options = []
        for path in route_result.paths:
            # Uses SAME spectrum pipeline
            spectrum_result = self.spectrum.find_spectrum(...)
            options.append(PathOption(is_feasible=spectrum_result.is_free, ...))
        return options

    def apply_action(self, action, request, network_state):
        # Uses SAME orchestrator
        return self.orchestrator.handle_arrival(request, network_state, forced_path=...)
```

**What disappears from RL:**
- `mock_handle_arrival()` - deleted
- `_get_paths_slots()` - replaced with `adapter.get_path_options()`
- Manual spectrum feasibility checks - use `spectrum.find_spectrum()`
- Separate arrival/release handling - use `orchestrator.handle_arrival()`/`handle_release()`

**When grooming/slicing/SNR are added:**
- RL config enables them: `config.grooming_enabled = True`
- No RL code changes needed
- The orchestrator handles it automatically

---

# Summary: How V2 Addresses the Four Goals

| Goal | V1 Problem | V2 Solution |
|------|------------|-------------|
| **1. Centralized State** | V1 proposed classes but didn't show migration from current structures | V2 provides concrete mapping table, before/after examples, and shows exactly where each piece of current state goes |
| **2. Standardized Returns** | V1 defined result types but mixed feasibility across layers | V2 defines clear layer responsibilities (routing=structure, spectrum=feasibility, allocation=final), naming table, before/after for real code |
| **3. Feature Composition** | V1 showed one happy path | V2 walks through 4 scenarios including the hard case (grooming+slicing+SNR), shows exactly where decisions are made |
| **4. SDN as Thin Router** | V1 had business logic in orchestrator (slicing decisions, force_route hacks) | V2 orchestrator is truly thin (~100 lines), all logic in pipelines, RL reuses same pipelines |

---

# Migration Path (High-Level)

## Phase 1: Domain Objects
- Create `fusion/domain/` with `Request`, `Lightpath`, `SimulationConfig`
- Add `from_legacy_dict()` and `to_legacy_dict()` methods
- No changes to existing code yet

## Phase 2: NetworkState
- Create `NetworkState` class with controlled methods
- Add legacy compatibility properties (`network_spectrum_dict`, `lightpath_status_dict`)
- Migrate spectrum and lightpath management to NetworkState

## Phase 3: Result Types
- Create `RouteResult`, `SpectrumResult`, `GroomingResult`, `AllocationResult`
- Create pipeline protocols (interfaces)
- Create adapter classes that wrap existing implementations

## Phase 4: Orchestrator
- Create `SDNOrchestrator` with feature flag
- Run both old SDNController and new orchestrator, compare outputs
- Verify `run_comparison.py` passes

## Phase 5: RL Integration
- Create `RLSimulationAdapter`
- Update gymnasium env to use adapter
- Remove `mock_handle_arrival()` and duplicated code

## Phase 6: Cleanup
- Remove legacy compatibility properties
- Remove old SDNController
- Remove unused Props classes
