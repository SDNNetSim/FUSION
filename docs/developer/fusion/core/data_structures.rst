.. _core-data-structures:

===============
Data Structures
===============

This document covers the key data structures used in the core module:
Domain objects (shared simulation concepts), Result objects (orchestrator outputs),
and Properties classes (legacy state containers).

Overview
========

FUSION uses three families of data structures:

**Domain Objects** (Shared)
   Core simulation concepts in ``fusion/domain/``.
   Used by both legacy and orchestrator architectures: NetworkState, Request,
   Lightpath, SimulationConfig.

**Result Objects** (v6.0+ Orchestrator)
   Immutable frozen dataclasses in ``fusion/domain/results.py``.
   Capture pipeline outputs with validation and factory methods.

**Properties Classes** (Legacy)
   Mutable classes in ``fusion/core/properties.py``.
   Carry state between legacy components.

.. code-block:: text

   +------------------------+     +------------------------+
   |    RESULT OBJECTS      |     |   PROPERTIES CLASSES   |
   |    (Orchestrator)      |     |   (Legacy)             |
   +------------------------+     +------------------------+
   | - RouteResult          |     | - RoutingProps         |
   | - SpectrumResult       |     | - SpectrumProps        |
   | - GroomingResult       |     | - SDNProps             |
   | - SlicingResult        |     | - SNRProps             |
   | - SNRResult            |     | - StatsProps           |
   | - AllocationResult     |     | - GroomingProps        |
   | - ProtectionResult     |     |                        |
   +------------------------+     +------------------------+

   +------------------------+
   |    DOMAIN OBJECTS      |
   |    (Shared)            |
   +------------------------+
   | - NetworkState         |
   | - LinkSpectrum         |
   | - Request              |
   | - Lightpath            |
   | - SimulationConfig     |
   +------------------------+

Domain Objects
==============

Domain objects in ``fusion/domain/`` are the shared data structures used by both
the legacy and orchestrator architectures. They represent the core simulation
concepts.

NetworkState
------------

**The single source of truth for network state during simulation.**

:Location: ``fusion/domain/network_state.py``
:Key Classes: ``NetworkState``, ``LinkSpectrum``

The ``NetworkState`` class provides immutable access to the current network state,
including topology, spectrum allocation, and active lightpaths.

**NetworkState fields:**

.. code-block:: python

   @dataclass
   class NetworkState:
       topology: nx.Graph                          # Network graph
       link_spectrum: dict[tuple[str, str], LinkSpectrum]  # Per-link spectrum
       lightpaths: dict[int, Lightpath] = field(default_factory=dict)
       active_failures: set[tuple[str, str]] = field(default_factory=set)

**Key methods:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - ``get_link_spectrum(src, dst)``
     - Get LinkSpectrum for a link
   * - ``is_range_free(path, core, start, end)``
     - Check if spectrum range is free on all path links
   * - ``create_lightpath(lightpath)``
     - Create new lightpath (returns new NetworkState)
   * - ``release_lightpath(lightpath_id)``
     - Release lightpath (returns new NetworkState)
   * - ``get_lightpaths_with_capacity(src, dst, bandwidth)``
     - Find lightpaths that can groom this request
   * - ``get_links_on_path(path)``
     - Get all links for a path
   * - ``is_link_failed(src, dst)``
     - Check if a link is currently failed

**Usage:**

.. code-block:: python

   from fusion.domain import NetworkState

   # Create from simulation state
   network_state = NetworkState(
       topology=topology,
       link_spectrum=link_spectrum_dict,
       lightpaths=active_lightpaths,
   )

   # Check spectrum availability
   if network_state.is_range_free(path, core=0, start=100, end=108):
       # Allocate...

   # Find grooming opportunities
   candidates = network_state.get_lightpaths_with_capacity(
       source="0",
       destination="5",
       bandwidth_gbps=100,
   )

LinkSpectrum
------------

**Per-link spectrum management with slot allocation tracking.**

:Location: ``fusion/domain/network_state.py``

.. code-block:: python

   @dataclass
   class LinkSpectrum:
       num_cores: int
       slots_per_core: int
       spectrum: np.ndarray  # Shape: (num_cores, slots_per_core)

**Key methods:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - ``is_range_free(core, start, end)``
     - Check if slot range is free on a core
   * - ``allocate_range(core, start, end, request_id)``
     - Mark slots as allocated (returns new LinkSpectrum)
   * - ``release_range(core, start, end)``
     - Mark slots as free (returns new LinkSpectrum)
   * - ``get_utilization(core)``
     - Get utilization ratio for a core
   * - ``find_first_free_block(core, num_slots)``
     - Find first free contiguous block

**Usage:**

.. code-block:: python

   # Check if slots 100-108 are free on core 0
   if link_spectrum.is_range_free(core=0, start=100, end=108):
       new_spectrum = link_spectrum.allocate_range(
           core=0, start=100, end=108, request_id=42
       )

Request
-------

**Represents a network connection request with lifecycle tracking.**

:Location: ``fusion/domain/request.py``
:Key Classes: ``Request``, ``RequestType``, ``RequestStatus``, ``BlockReason``, ``ProtectionStatus``

**Request dataclass:**

.. code-block:: python

   @dataclass
   class Request:
       request_id: int
       source: str
       destination: str
       bandwidth_gbps: int
       arrive_time: float
       depart_time: float

       # Lifecycle tracking
       request_type: RequestType = RequestType.ARRIVAL
       status: RequestStatus = RequestStatus.PENDING

       # Allocation results (set after processing)
       allocated_path: tuple[str, ...] | None = None
       allocated_core: int | None = None
       allocated_start_slot: int | None = None
       allocated_end_slot: int | None = None
       allocated_modulation: str | None = None
       lightpath_id: int | None = None

       # Protection (1+1)
       protection_status: ProtectionStatus = ProtectionStatus.UNPROTECTED
       backup_path: tuple[str, ...] | None = None
       backup_core: int | None = None
       backup_start_slot: int | None = None
       backup_end_slot: int | None = None

       # Blocking info
       block_reason: BlockReason | None = None

**RequestType enum:**

.. code-block:: python

   class RequestType(Enum):
       ARRIVAL = "arrival"    # New connection request
       RELEASE = "release"    # Connection teardown

**RequestStatus enum (state machine):**

.. code-block:: python

   class RequestStatus(Enum):
       PENDING = "pending"          # Not yet processed
       ROUTED = "routed"            # Path found
       ALLOCATED = "allocated"      # Spectrum assigned
       ACTIVE = "active"            # Currently using resources
       BLOCKED = "blocked"          # Could not be served
       RELEASED = "released"        # Resources freed
       FAILED = "failed"            # Failed during active state

**BlockReason enum:**

.. code-block:: python

   class BlockReason(Enum):
       CONGESTION = "congestion"       # No spectrum available
       DISTANCE = "distance"           # Path too long for modulation
       XT_THRESHOLD = "xt_threshold"   # Cross-talk too high
       SNR_THRESHOLD = "snr_threshold" # SNR too low
       NO_PATH = "no_path"             # No route exists
       FAILURE = "failure"             # Network failure blocked request

**ProtectionStatus enum:**

.. code-block:: python

   class ProtectionStatus(Enum):
       UNPROTECTED = "unprotected"     # No backup path
       PROTECTED = "protected"          # Backup path established
       SWITCHED = "switched"            # Using backup path
       FAILED = "failed"                # Both paths failed

**Key methods:**

.. code-block:: python

   # Duration calculation
   request.holding_time  # depart_time - arrive_time

   # State transitions
   request.mark_routed(path)
   request.mark_allocated(core, start, end, modulation)
   request.mark_blocked(reason)
   request.mark_released()

Lightpath
---------

**Represents an established optical connection with capacity management.**

:Location: ``fusion/domain/lightpath.py``

.. code-block:: python

   @dataclass
   class Lightpath:
       lightpath_id: int
       path: tuple[str, ...]
       core: int
       start_slot: int
       end_slot: int               # Exclusive
       modulation: str
       capacity_gbps: int          # Total capacity
       allocated_gbps: int = 0     # Currently used

       # Protection
       is_protected: bool = False
       backup_path: tuple[str, ...] | None = None
       backup_core: int | None = None
       backup_start_slot: int | None = None
       backup_end_slot: int | None = None

       # Request tracking
       request_ids: set[int] = field(default_factory=set)

**Key methods:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - ``available_capacity``
     - Property: ``capacity_gbps - allocated_gbps``
   * - ``num_slots``
     - Property: ``end_slot - start_slot``
   * - ``can_accommodate(bandwidth)``
     - Check if bandwidth fits in available capacity
   * - ``allocate_bandwidth(bandwidth, request_id)``
     - Add bandwidth allocation (returns new Lightpath)
   * - ``release_bandwidth(bandwidth, request_id)``
     - Remove bandwidth allocation (returns new Lightpath)
   * - ``uses_link(src, dst)``
     - Check if lightpath traverses a link

**Usage:**

.. code-block:: python

   from fusion.domain import Lightpath

   # Create lightpath
   lp = Lightpath(
       lightpath_id=1,
       path=("0", "2", "5"),
       core=0,
       start_slot=100,
       end_slot=108,
       modulation="QPSK",
       capacity_gbps=100,
   )

   # Check grooming possibility
   if lp.can_accommodate(50):
       lp = lp.allocate_bandwidth(50, request_id=42)

   # Check available capacity
   print(f"Available: {lp.available_capacity} Gbps")

SimulationConfig
----------------

**Immutable configuration replacing mutable engine_props dict.**

:Location: ``fusion/domain/config.py``

The ``SimulationConfig`` dataclass provides type-safe, validated configuration
for simulations. It replaces the legacy ``engine_props`` dictionary approach.

.. code-block:: python

   @dataclass(frozen=True)
   class SimulationConfig:
       # Simulation control
       max_iters: int = 10
       num_requests: int = 1000
       erlang: float = 300.0
       holding_time: float = 3600.0
       seed: int | None = None

       # Network configuration
       network: str = "NSFNet"
       cores_per_link: int = 7
       c_band: int = 320
       l_band: int = 0
       s_band: int = 0

       # Algorithm selection
       route_method: str = "k_shortest_path"
       allocation_method: str = "first_fit"
       k_paths: int = 3

       # Feature flags
       use_orchestrator: bool = False
       enable_grooming: bool = True
       enable_slicing: bool = False
       enable_protection: bool = False

       # Survivability
       failure_enabled: bool = False
       failure_type: str | None = None

       # Output control
       output_train_data: bool = False
       save_snapshots: bool = False
       print_step: int = 0

**Key methods:**

.. code-block:: python

   # Create from legacy dict
   config = SimulationConfig.from_engine_props(engine_props)

   # Convert back to dict (for legacy components)
   engine_props = config.to_engine_props()

   # Access with type safety
   if config.enable_protection:
       # Protection is enabled
       ...

**Advantages over engine_props dict:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Aspect
     - Benefit
   * - Type safety
     - IDE autocompletion, mypy validation
   * - Immutability
     - Cannot be accidentally modified
   * - Defaults
     - Clear default values in one place
   * - Validation
     - ``__post_init__`` checks constraints
   * - Documentation
     - Field types and docstrings

Result Objects
==============

Result objects are immutable (``frozen=True``) dataclasses that capture the output
of each pipeline stage. They provide:

- Validation via ``__post_init__``
- Factory methods (``blocked()``, ``success()``, etc.)
- Conversion from legacy properties (``from_routing_props()``)

AllocationResult
----------------

**The SINGLE SOURCE OF TRUTH for request allocation.**

:Location: ``fusion/domain/results.py``
:Immutability: Frozen dataclass

The ``success`` field is the **final authority** on whether a request was served.

**Fields:**

.. code-block:: python

   @dataclass(frozen=True)
   class AllocationResult:
       # Final authority
       success: bool

       # Lightpath tracking
       lightpaths_created: tuple[int, ...] = ()
       lightpaths_groomed: tuple[int, ...] = ()
       total_bandwidth_allocated_gbps: int = 0

       # Feature flags
       is_groomed: bool = False
       is_partially_groomed: bool = False
       is_sliced: bool = False
       is_protected: bool = False

       # Failure info
       block_reason: BlockReason | None = None

       # Per-segment tracking (for sliced allocations)
       bandwidth_allocations: tuple[int, ...] = ()
       modulations: tuple[str, ...] = ()
       cores: tuple[int, ...] = ()
       bands: tuple[str, ...] = ()
       start_slots: tuple[int, ...] = ()
       end_slots: tuple[int, ...] = ()
       snr_values: tuple[float, ...] = ()

       # Nested results (for debugging)
       route_result: RouteResult | None = None
       spectrum_result: SpectrumResult | None = None
       grooming_result: GroomingResult | None = None
       slicing_result: SlicingResult | None = None
       snr_result: SNRResult | None = None
       protection_result: ProtectionResult | None = None

**Invariants:**

- If ``success=True``: at least one lightpath (created or groomed)
- If ``success=True``: ``total_bandwidth_allocated_gbps > 0``
- If ``success=False``: ``block_reason`` is set

**Factory methods:**

.. code-block:: python

   # Blocked request
   result = AllocationResult.blocked(BlockReason.CONGESTION)

   # New lightpath
   result = AllocationResult.success_new_lightpath(
       lightpath_id=42,
       bandwidth_gbps=100,
   )

   # Groomed request
   result = AllocationResult.success_groomed(
       lightpath_ids=[1, 2],
       bandwidth_gbps=100,
   )

   # Sliced request
   result = AllocationResult.success_sliced(
       lightpath_ids=[1, 2, 3],
       bandwidth_gbps=300,
   )

RouteResult
-----------

**Output of the routing pipeline - candidate paths with modulation options.**

.. code-block:: python

   @dataclass(frozen=True)
   class RouteResult:
       # Primary paths
       paths: tuple[tuple[str, ...], ...] = ()
       weights_km: tuple[float, ...] = ()
       modulations: tuple[tuple[str, ...], ...] = ()

       # Backup paths (for 1+1 protection)
       backup_paths: tuple[tuple[str, ...], ...] | None = None
       backup_weights_km: tuple[float, ...] | None = None
       backup_modulations: tuple[tuple[str, ...], ...] | None = None

       # Metadata
       strategy_name: str = ""
       connection_index: int | None = None  # For external SNR lookup

**Key properties:**

.. code-block:: python

   result.is_empty          # True if no paths found
   result.num_paths         # Number of candidate paths
   result.has_protection    # True if backup paths available
   result.best_path         # First (best) path or None
   result.best_weight       # Weight of best path

**Usage:**

.. code-block:: python

   result = RouteResult(
       paths=(("0", "2", "5"),),
       weights_km=(100.0,),
       modulations=(("QPSK", "16-QAM"),),
       strategy_name="k_shortest_path",
   )

   for i in range(result.num_paths):
       path = result.get_path(i)
       mods = result.get_modulations_for_path(i)

SpectrumResult
--------------

**Output of spectrum assignment - allocated slot range.**

.. code-block:: python

   @dataclass(frozen=True)
   class SpectrumResult:
       is_free: bool                 # Whether allocation succeeded

       # Allocation details (valid only if is_free=True)
       start_slot: int = 0
       end_slot: int = 0             # Exclusive
       core: int = 0
       band: str = "c"
       modulation: str = ""
       slots_needed: int = 0

       # Dynamic LP slicing
       achieved_bandwidth_gbps: int | None = None  # May be less than requested

       # Backup spectrum (for 1+1 protection)
       backup_start_slot: int | None = None
       backup_end_slot: int | None = None
       backup_core: int | None = None
       backup_band: str | None = None

**Key properties:**

.. code-block:: python

   result.num_slots    # Slots allocated (0 if not free)
   result.has_backup   # True if backup spectrum allocated

**Factory methods:**

.. code-block:: python

   # No spectrum found
   result = SpectrumResult.not_found(slots_needed=8)

   # Successful allocation
   result = SpectrumResult(
       is_free=True,
       start_slot=100,
       end_slot=108,
       core=0,
       band="c",
       modulation="QPSK",
       slots_needed=8,
   )

GroomingResult
--------------

**Output of grooming pipeline - using existing lightpath capacity.**

.. code-block:: python

   @dataclass(frozen=True)
   class GroomingResult:
       fully_groomed: bool = False
       partially_groomed: bool = False
       bandwidth_groomed_gbps: int = 0
       remaining_bandwidth_gbps: int = 0
       lightpaths_used: tuple[int, ...] = ()
       forced_path: tuple[str, ...] | None = None

**Invariants:**

- ``fully_groomed`` and ``partially_groomed`` are mutually exclusive
- If ``fully_groomed``: ``remaining_bandwidth_gbps == 0``
- If groomed: ``len(lightpaths_used) > 0``

**Factory methods:**

.. code-block:: python

   # No grooming possible
   result = GroomingResult.no_grooming(bandwidth_gbps=100)

   # Fully groomed
   result = GroomingResult.full(
       bandwidth_gbps=100,
       lightpath_ids=[1, 2],
   )

   # Partially groomed
   result = GroomingResult.partial(
       bandwidth_groomed=50,
       remaining=50,
       lightpath_ids=[1],
       forced_path=["0", "2", "5"],
   )

SlicingResult
-------------

**Output of slicing pipeline - request split across lightpaths.**

.. code-block:: python

   @dataclass(frozen=True)
   class SlicingResult:
       success: bool = False
       num_slices: int = 0
       slice_bandwidth_gbps: int = 0
       lightpaths_created: tuple[int, ...] = ()
       total_bandwidth_gbps: int = 0

**Key property:**

.. code-block:: python

   result.is_sliced    # True if num_slices > 1

**Factory methods:**

.. code-block:: python

   # Slicing failed
   result = SlicingResult.failed()

   # Single lightpath (no slicing needed)
   result = SlicingResult.single_lightpath(
       bandwidth_gbps=100,
       lightpath_id=1,
   )

   # Sliced into multiple lightpaths
   result = SlicingResult.sliced(
       num_slices=4,
       slice_bandwidth=25,
       lightpath_ids=[1, 2, 3, 4],
   )

SNRResult
---------

**Output of SNR validation - signal quality check.**

.. code-block:: python

   @dataclass(frozen=True)
   class SNRResult:
       passed: bool
       snr_db: float = 0.0
       required_snr_db: float = 0.0
       margin_db: float = 0.0
       failure_reason: str | None = None
       link_snr_values: dict[tuple[str, str], float] = field(default_factory=dict)

**Key properties:**

.. code-block:: python

   result.is_degraded        # Passed but margin < 1 dB
   result.has_link_breakdown # Per-link values available

**Factory methods:**

.. code-block:: python

   # SNR passed
   result = SNRResult.success(snr_db=18.5, required_snr_db=15.0)

   # SNR failed
   result = SNRResult.failure(
       snr_db=12.0,
       required_snr_db=15.0,
       reason="SNR below threshold",
   )

   # SNR validation skipped
   result = SNRResult.skipped()

ProtectionResult
----------------

**Output of protection pipeline - 1+1 path establishment.**

.. code-block:: python

   @dataclass(frozen=True)
   class ProtectionResult:
       primary_established: bool = False
       backup_established: bool = False
       primary_spectrum: SpectrumResult | None = None
       backup_spectrum: SpectrumResult | None = None
       switchover_triggered: bool = False
       switchover_success: bool = False
       switchover_time_ms: float | None = None
       failure_type: str | None = None
       recovery_type: str | None = None

**Key properties:**

.. code-block:: python

   result.is_fully_protected  # Both paths established
   result.recovery_duration_ms  # Time to recover (if completed)

Properties Classes (Legacy)
===========================

Properties classes are mutable state containers used by legacy components.

RoutingProps
------------

**State for path computation.**

.. code-block:: python

   class RoutingProps:
       def __init__(self):
           # Path results
           self.paths_matrix: list[Any] = []
           self.modulation_formats_matrix: list[list[str]] = []
           self.weights_list: list[float] = []
           self.path_index_list: list[int] = []

           # Backup paths (1+1 protection)
           self.backup_paths_matrix: list[list[int] | None] = []
           self.backup_modulation_formats_matrix: list[list[str]] = []

           # Physical parameters
           self.input_power: float = DEFAULT_INPUT_POWER
           self.frequency_spacing: float = DEFAULT_FREQUENCY_SPACING
           self.span_length: float = DEFAULT_SPAN_LENGTH

SpectrumProps
-------------

**State for spectrum assignment.**

.. code-block:: python

   class SpectrumProps:
       def __init__(self):
           # Path and requirements
           self.path_list: list[int] | None = None
           self.backup_path: list[int] | None = None
           self.slots_needed: int | None = None
           self.modulation: str | None = None

           # Core assignment
           self.forced_core: int | None = None
           self.core_number: int | None = None

           # Allocation results
           self.is_free: bool = False
           self.start_slot: int | None = None
           self.end_slot: int | None = None
           self.current_band: str | None = None

           # Grooming
           self.lightpath_id: int | None = None
           self.lightpath_bandwidth: float | None = None

SDNProps
--------

**State for SDN controller operations.**

.. code-block:: python

   class SDNProps:
       def __init__(self):
           # Network state
           self.topology: Any | None = None
           self.network_spectrum_dict: dict | None = None

           # Current request
           self.request_id: int | None = None
           self.source: str | None = None
           self.destination: str | None = None
           self.bandwidth: float | None = None

           # Allocation results
           self.was_routed: bool | None = None
           self.block_reason: str | None = None

           # Per-segment tracking (slicing)
           self.modulation_list: list[str] = []
           self.core_list: list[int] = []
           self.start_slot_list: list[int] = []
           self.end_slot_list: list[int] = []
           # ... many more lists

           # Protection (1+1)
           self.primary_path: list[int] | None = None
           self.backup_path: list[int] | None = None
           self.is_protected: bool = False

SNRProps - Multi-Fiber Mappings
-------------------------------

**State for SNR calculations, including file mappings for pre-calculated data.**

.. code-block:: python

   class SNRProps:
       def __init__(self):
           # Physical constants
           self.light_frequency: float = LIGHT_FREQUENCY_CENTER
           self.planck_constant: float = PLANCK_CONSTANT

           # SNR thresholds per modulation
           self.req_snr: dict[str, float] = {
               "BPSK": 3.71,
               "QPSK": 6.72,
               "8-QAM": 10.84,
               "16-QAM": 13.24,
               "32-QAM": 16.16,
               "64-QAM": 19.01,
           }

           # Pre-calculated SNR file mappings
           self.file_mapping_dict = {...}  # See below

**Multi-Fiber File Mappings:**

The ``file_mapping_dict`` maps topology and core configurations to pre-calculated
SNR data files:

.. code-block:: python

   self.file_mapping_dict = {
       "USbackbone60": {
           # Multi-fiber (single core per fiber)
           "multi_fiber": {
               "mf": "MF-USB6014-MF.npy",
               "gsnr": "GSNR-USB6014-MF.npy",
           },
           # Multi-core fiber: (center_core, total_cores)
           (2, 4): {  # 4-core fiber, center core 2
               "mf": "MF-USB6014-MCF4-C2.npy",
               "gsnr": "GSNR-USB6014-MCF4-C2.npy",
           },
           (6, 7): {  # 7-core fiber, center core 6
               "mf": "MF-USB6014-MCF7-C6.npy",
               "gsnr": "GSNR-USB6014-MCF7-C6.npy",
           },
           (3, 7): {  # 7-core fiber, edge core 3
               "mf": "MF-USB6014-MCF7-C3.npy",
               "gsnr": "GSNR-USB6014-MCF7-C3.npy",
           },
           # ... 13-core, 19-core configurations
       },
       "Spainbackbone30": {
           # Similar structure for Spain topology
       },
   }

**Understanding the mappings:**

- ``"multi_fiber"``: Standard single-core fibers (one fiber per core)
- ``(center_core, total_cores)``: Multi-core fiber configuration
- ``"mf"``: Modulation format selection file
- ``"gsnr"``: Generalized SNR values file

**Example usage:**

.. code-block:: python

   # Get files for 7-core MCF on USbackbone60
   topology = "USbackbone60"
   center_core = 6
   total_cores = 7

   files = snr_props.file_mapping_dict[topology][(center_core, total_cores)]
   mf_file = files["mf"]    # "MF-USB6014-MCF7-C6.npy"
   gsnr_file = files["gsnr"]  # "GSNR-USB6014-MCF7-C6.npy"

Converting Between Legacy and Result Objects
============================================

Adapters convert between legacy properties and result objects:

.. code-block:: python

   # Legacy to Result
   route_result = RouteResult.from_routing_props(routing_props)
   spectrum_result = SpectrumResult.from_spectrum_props(spectrum_props)

   # Result to Legacy (for backward compatibility)
   allocation_dict = spectrum_result.to_allocation_dict()

Design Principles
=================

1. **Result objects are immutable**
   - Use ``frozen=True`` for all dataclasses
   - Prevents accidental mutation
   - Enables safe caching and sharing

2. **Validation on creation**
   - ``__post_init__`` validates invariants
   - Fail fast if data is inconsistent

3. **Factory methods for common cases**
   - ``blocked()``, ``success()``, etc.
   - Clearer code than constructor calls

4. **Conversion methods for migration**
   - ``from_*_props()`` for legacy compatibility
   - ``to_*_dict()`` for backward compatibility

See Also
========

- :doc:`orchestrator` - How pipelines use result objects
- :doc:`adapters` - Converting between legacy and result objects
- :doc:`metrics` - StatsProps documentation
- ``fusion/domain/results.py`` - Result object source code
- ``fusion/domain/network_state.py`` - NetworkState and LinkSpectrum source code
- ``fusion/domain/request.py`` - Request and enums source code
- ``fusion/domain/lightpath.py`` - Lightpath source code
- ``fusion/domain/config.py`` - SimulationConfig source code
- ``fusion/core/properties.py`` - Properties class source code
