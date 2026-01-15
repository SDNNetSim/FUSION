.. _domain-objects:

==============
Domain Objects
==============

This document provides detailed documentation for each domain object, including
field descriptions, validation rules, and usage patterns.

Overview
========

Domain objects are organized into four categories:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Category
     - Objects
     - Purpose
   * - Configuration
     - ``SimulationConfig``
     - Immutable simulation parameters
   * - Request Lifecycle
     - ``Request``, enums
     - Service request tracking
   * - Network Resources
     - ``Lightpath``, ``NetworkState``, ``LinkSpectrum``
     - Optical network state
   * - Pipeline Results
     - ``*Result`` classes
     - Immutable pipeline outputs

SimulationConfig
================

**Location:** ``fusion/domain/config.py``

Immutable configuration container that replaces the mutable ``engine_props`` dictionary.

Fields
------

.. list-table::
   :header-rows: 1
   :widths: 25 18 57

   * - Field
     - Type
     - Description
   * - ``network_name``
     - **str**
     - Network topology identifier (e.g., "USbackbone60")
   * - ``cores_per_link``
     - **int**
     - Number of cores per fiber link (MCF support)
   * - ``band_list``
     - **tuple[str, ...]**
     - Available frequency bands as immutable tuple
   * - ``band_slots``
     - **dict[str, int]**
     - Slot count per band (e.g., {"c": 320, "l": 320})
   * - ``guard_slots``
     - **int**
     - Guard band slots between allocations
   * - ``span_length``
     - **float**
     - Default span length in km
   * - ``max_link_length``
     - **float | None**
     - Maximum link length constraint
   * - ``modulation_formats``
     - **tuple[str, ...]**
     - Available modulation formats
   * - ``bw_per_slot``
     - **float**
     - Bandwidth per slot in GHz
   * - ``route_method``
     - **str**
     - Routing algorithm name
   * - ``allocation_method``
     - **str**
     - Spectrum allocation strategy
   * - ``k_paths``
     - **int**
     - Number of candidate paths for k-shortest
   * - ``snr_type``
     - **str**
     - SNR calculation method
   * - ``grooming_enabled``
     - **bool**
     - Enable traffic grooming
   * - ``slicing_enabled``
     - **bool**
     - Enable request slicing
   * - ``protection_enabled``
     - **bool**
     - Enable 1+1 protection

Usage
-----

.. code-block:: python

   from fusion.domain import SimulationConfig

   # Create from engine_props (legacy interop)
   config = SimulationConfig.from_engine_props(engine_props)

   # Access fields with type safety
   if config.grooming_enabled:
       print(f"Grooming enabled with {config.cores_per_link} cores")

   # Convert back to dict (for legacy code)
   engine_props = config.to_engine_props()

Request
=======

**Location:** ``fusion/domain/request.py``

Represents a network service request with full lifecycle tracking.

Fields
------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Type
     - Description
   * - ``request_id``
     - **int**
     - Unique identifier
   * - ``source``
     - **str**
     - Source node ID
   * - ``destination``
     - **str**
     - Destination node ID
   * - ``bandwidth_gbps``
     - **int**
     - Requested bandwidth in Gbps
   * - ``arrive``
     - **float**
     - Arrival time (simulation time)
   * - ``depart``
     - **float**
     - Departure time (simulation time)
   * - ``request_type``
     - **RequestType**
     - ARRIVAL, DEPARTURE, FAILURE, RECOVERY
   * - ``status``
     - **RequestStatus**
     - Current lifecycle state
   * - ``block_reason``
     - **BlockReason | None**
     - Why request was blocked (if applicable)
   * - ``protection_status``
     - **ProtectionStatus**
     - Protection state for survivability
   * - ``lightpath_id``
     - **int | None**
     - Assigned lightpath (if allocated)
   * - ``is_protected``
     - **bool**
     - Whether request requires 1+1 protection
   * - ``sla_availability``
     - **float**
     - Required availability (e.g., 0.9999)

Enums
-----

**RequestType** - Type of request event:

.. code-block:: python

   class RequestType(Enum):
       ARRIVAL = "arrival"      # New connection request
       DEPARTURE = "departure"  # Connection teardown
       FAILURE = "failure"      # Link/node failure event
       RECOVERY = "recovery"    # Recovery from failure

**RequestStatus** - Lifecycle state:

.. code-block:: python

   class RequestStatus(Enum):
       PENDING = "pending"        # Not yet processed
       ALLOCATED = "allocated"    # Successfully allocated
       BLOCKED = "blocked"        # Could not be served
       RELEASED = "released"      # Connection ended normally
       FAILED = "failed"          # Failed during active state
       RECOVERED = "recovered"    # Restored after failure

**BlockReason** - Why a request was blocked:

.. code-block:: python

   class BlockReason(Enum):
       NO_PATH = "no_path"               # No route exists
       NO_SPECTRUM = "no_spectrum"       # Spectrum unavailable
       SNR_VIOLATION = "snr_violation"   # SNR below threshold
       XT_VIOLATION = "xt_violation"     # Cross-talk too high
       NO_PROTECTION = "no_protection"   # No disjoint backup path
       DISTANCE = "distance"             # Path too long for modulation

**ProtectionStatus** - Protection state:

.. code-block:: python

   class ProtectionStatus(Enum):
       UNPROTECTED = "unprotected"    # No backup path
       PROTECTED = "protected"        # Backup path established
       SWITCHED = "switched"          # Using backup path
       FAILED = "failed"              # Both paths failed

Usage
-----

.. code-block:: python

   from fusion.domain import Request, RequestType, RequestStatus, BlockReason

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

   # Check status
   if request.status == RequestStatus.PENDING:
       # Process request...
       pass

   # Computed properties
   print(f"Holding time: {request.holding_time}")
   print(f"Endpoint pair: {request.endpoint_key}")

Lightpath
=========

**Location:** ``fusion/domain/lightpath.py``

Represents an allocated optical path with capacity management for traffic grooming.

Fields
------

.. list-table::
   :header-rows: 1
   :widths: 28 17 55

   * - Field
     - Type
     - Description
   * - ``lightpath_id``
     - **int**
     - Unique identifier
   * - ``path``
     - **list[str]**
     - Ordered node IDs from source to destination
   * - ``start_slot``
     - **int**
     - First allocated slot index (inclusive)
   * - ``end_slot``
     - **int**
     - Last allocated slot index (exclusive)
   * - ``core``
     - **int**
     - Core number for MCF (0-indexed)
   * - ``band``
     - **str**
     - Frequency band ("c", "l", "s")
   * - ``modulation``
     - **str**
     - Modulation format ("QPSK", "16-QAM", etc.)
   * - ``total_bandwidth_gbps``
     - **int**
     - Maximum capacity
   * - ``remaining_bandwidth_gbps``
     - **int**
     - Available for new requests
   * - ``path_weight_km``
     - **float**
     - Physical path length in km
   * - ``request_allocations``
     - **dict[int, int]**
     - Maps request_id to allocated bandwidth
   * - ``snr_db``
     - **float | None**
     - Signal-to-noise ratio
   * - ``is_degraded``
     - **bool**
     - True if quality has degraded

**Protection fields (beta):**

.. list-table::
   :header-rows: 1
   :widths: 28 17 55

   * - Field
     - Type
     - Description
   * - ``backup_path``
     - **list[str] | None**
     - Disjoint backup path
   * - ``backup_start_slot``
     - **int | None**
     - Backup spectrum start slot
   * - ``backup_end_slot``
     - **int | None**
     - Backup spectrum end slot
   * - ``backup_core``
     - **int | None**
     - Backup core number
   * - ``is_protected``
     - **bool**
     - Has backup path
   * - ``active_path``
     - **str**
     - Currently active ("primary" or "backup")

Computed Properties
-------------------

.. code-block:: python

   lp.source              # First node in path
   lp.destination         # Last node in path
   lp.endpoint_key        # Sorted (source, dest) tuple
   lp.num_slots           # end_slot - start_slot
   lp.num_hops            # len(path) - 1
   lp.utilization         # Fraction of capacity used (0.0 to 1.0)
   lp.has_capacity        # remaining_bandwidth_gbps > 0
   lp.num_requests        # Number of requests using this lightpath
   lp.is_empty            # No requests allocated
   lp.current_path        # Active path (primary or backup)

Methods
-------

.. code-block:: python

   # Capacity checking
   lp.can_accommodate(bandwidth_gbps: int) -> bool

   # Bandwidth allocation (modifies lightpath)
   lp.allocate_bandwidth(request_id: int, bandwidth_gbps: int, timestamp: float | None) -> bool

   # Bandwidth release (modifies lightpath)
   lp.release_bandwidth(request_id: int, timestamp: float | None) -> int

   # Get allocation for specific request
   lp.get_allocation(request_id: int) -> int | None

   # Protection switching (beta)
   lp.switch_to_backup() -> bool
   lp.switch_to_primary() -> bool

Usage
-----

.. code-block:: python

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

   # Traffic grooming - allocate bandwidth
   if lp.can_accommodate(50):
       success = lp.allocate_bandwidth(request_id=42, bandwidth_gbps=50)
       print(f"Utilization: {lp.utilization:.1%}")  # 50%

   # Release bandwidth when request departs
   released = lp.release_bandwidth(request_id=42)
   print(f"Released {released} Gbps")

NetworkState
============

**Location:** ``fusion/domain/network_state.py``

The single source of truth for network state during simulation.

.. important::

   ``NetworkState`` is **mutable** - it changes as requests are processed.
   This is different from result objects which are immutable.

Fields
------

NetworkState uses private fields with property accessors:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Description
   * - ``topology``
     - NetworkX graph (read-only access)
   * - ``lightpath_count``
     - Number of active lightpaths
   * - ``link_count``
     - Number of links in topology
   * - ``node_count``
     - Number of nodes in topology
   * - ``next_lightpath_id``
     - Next ID that will be assigned

Key Methods
-----------

**Lightpath Management:**

.. code-block:: python

   # Get lightpath by ID
   state.get_lightpath(lightpath_id: int) -> Lightpath | None

   # Get all lightpaths
   state.get_all_lightpaths() -> list[Lightpath]

   # Get lightpaths for endpoint pair
   state.get_lightpaths_for_endpoints(src: str, dst: str) -> list[Lightpath]

   # Create new lightpath (allocates spectrum)
   state.create_lightpath(
       path: list[str],
       start_slot: int,
       end_slot: int,
       core: int,
       band: str,
       modulation: str,
       total_bandwidth_gbps: int,
       ...
   ) -> int  # Returns lightpath_id

   # Remove lightpath (releases spectrum)
   state.remove_lightpath(lightpath_id: int) -> None

**Spectrum Queries:**

.. code-block:: python

   # Check if spectrum range is free on all links in path
   state.is_spectrum_available(
       path: list[str],
       start_slot: int,
       end_slot: int,
       core: int,
       band: str,
   ) -> bool

   # Get spectrum array for a link
   state.get_link_spectrum(link: tuple[str, str]) -> LinkSpectrum

   # Check if specific slot is free
   state.is_spectrum_free(
       link: tuple[str, str],
       core: int,
       band: str,
       start: int,
       end: int,
   ) -> bool

**Topology Queries:**

.. code-block:: python

   # Get neighboring nodes
   state.get_neighbors(node: str) -> list[str]

   # Check if link exists
   state.has_link(link: tuple[str, str]) -> bool

   # Get link length
   state.get_link_length(link: tuple[str, str]) -> float

   # Convert path to list of links
   state.path_to_links(path: list[str]) -> list[tuple[str, str]]

**Utilization:**

.. code-block:: python

   # Get overall spectrum utilization
   state.get_spectrum_utilization(band: str | None = None) -> float

Usage
-----

.. code-block:: python

   from fusion.domain import NetworkState, SimulationConfig

   # Create from configuration
   config = SimulationConfig.from_engine_props(engine_props)
   state = NetworkState.from_config(config, topology)

   # Check spectrum and create lightpath
   path = ["0", "2", "5"]
   if state.is_spectrum_available(path, start=10, end=18, core=0, band="c"):
       lp_id = state.create_lightpath(
           path=path,
           start_slot=10,
           end_slot=18,
           core=0,
           band="c",
           modulation="QPSK",
           total_bandwidth_gbps=100,
       )
       print(f"Created lightpath {lp_id}")

LinkSpectrum
============

**Location:** ``fusion/domain/network_state.py``

Per-link spectrum management with slot allocation tracking.

Fields
------

.. list-table::
   :header-rows: 1
   :widths: 28 17 55

   * - Field
     - Type
     - Description
   * - ``link_id``
     - **tuple[str, str]**
     - (source, destination) node IDs
   * - ``length_km``
     - **float**
     - Physical link length
   * - ``num_cores``
     - **int**
     - Number of cores (MCF)

Spectrum Array Values
---------------------

The spectrum is stored as NumPy arrays per core/band:

.. code-block:: text

   spectrum[core][band] = np.array([0, 0, 1, 1, 1, 0, 0, ...])
     - 0: Free slot
     - lightpath_id (>0): Slot allocated to that lightpath

Methods
-------

.. code-block:: python

   # Get spectrum array for a band
   get_spectrum_array(band: str) -> np.ndarray

   # Get all configured bands
   get_bands() -> list[str]

   # Check if range is free
   is_range_free(core: int, band: str, start: int, end: int) -> bool

   # Allocate spectrum (marks slots with lightpath_id)
   allocate_range(core: int, band: str, start: int, end: int, lightpath_id: int) -> None

   # Release spectrum (sets slots to 0)
   release_range(core: int, band: str, start: int, end: int) -> None

Result Objects
==============

All result objects are in ``fusion/domain/results.py`` and share these characteristics:

- Immutable (``@dataclass(frozen=True)``)
- Validated in ``__post_init__``
- Factory methods for common cases
- Conversion methods for legacy interop

AllocationResult
----------------

The **final authority** on whether a request was served.

.. code-block:: python

   @dataclass(frozen=True)
   class AllocationResult:
       success: bool                              # THE authority
       lightpaths_created: tuple[int, ...] = ()   # New lightpath IDs
       lightpaths_groomed: tuple[int, ...] = ()   # Existing lightpaths used
       total_bandwidth_allocated_gbps: int = 0

       # Feature flags
       is_groomed: bool = False
       is_partially_groomed: bool = False
       is_sliced: bool = False
       is_protected: bool = False

       # Failure info
       block_reason: BlockReason | None = None

       # Nested results (for debugging)
       route_result: RouteResult | None = None
       spectrum_result: SpectrumResult | None = None
       grooming_result: GroomingResult | None = None
       snr_result: SNRResult | None = None
       slicing_result: SlicingResult | None = None
       protection_result: ProtectionResult | None = None

RouteResult
-----------

Output of routing pipeline - candidate paths.

.. code-block:: python

   @dataclass(frozen=True)
   class RouteResult:
       paths: tuple[tuple[str, ...], ...] = ()         # Candidate paths
       weights_km: tuple[float, ...] = ()              # Path lengths
       modulations: tuple[tuple[str, ...], ...] = ()   # Valid modulations per path

       # For 1+1 protection
       backup_paths: tuple[tuple[str, ...], ...] | None = None

       # Metadata
       strategy_name: str = ""

   # Properties
   result.is_empty       # True if no paths found
   result.num_paths      # Number of candidate paths
   result.has_protection # True if backup paths available

SpectrumResult
--------------

Output of spectrum assignment - allocated slot range.

.. code-block:: python

   @dataclass(frozen=True)
   class SpectrumResult:
       is_free: bool                   # Whether allocation succeeded
       start_slot: int = 0             # First slot (inclusive)
       end_slot: int = 0               # Last slot (exclusive)
       core: int = 0                   # Core number
       band: str = "c"                 # Frequency band
       modulation: str = ""            # Selected modulation
       slots_needed: int = 0           # Slots including guard band

       # For dynamic slicing
       achieved_bandwidth_gbps: int | None = None

   # Properties
   result.num_slots      # end_slot - start_slot

GroomingResult
--------------

Output of grooming pipeline - using existing capacity.

.. code-block:: python

   @dataclass(frozen=True)
   class GroomingResult:
       fully_groomed: bool = False        # Entire request served
       partially_groomed: bool = False    # Some bandwidth groomed
       bandwidth_groomed_gbps: int = 0    # Amount successfully groomed
       remaining_bandwidth_gbps: int = 0  # Needs new lightpath
       lightpaths_used: tuple[int, ...] = ()
       forced_path: tuple[str, ...] | None = None

SNRResult
---------

Output of SNR validation.

.. code-block:: python

   @dataclass(frozen=True)
   class SNRResult:
       passed: bool                       # Whether SNR check passed
       snr_db: float = 0.0                # Calculated SNR
       required_snr_db: float = 0.0       # Threshold for modulation
       margin_db: float = 0.0             # SNR - required

   # Properties
   result.is_degraded    # Passed but margin < 1 dB

SlicingResult
-------------

Output of slicing pipeline - request split across lightpaths.

.. code-block:: python

   @dataclass(frozen=True)
   class SlicingResult:
       success: bool = False
       num_slices: int = 0
       slice_bandwidth_gbps: int = 0
       lightpath_ids: tuple[int, ...] = ()

   # Properties
   result.is_sliced      # True if num_slices > 1

ProtectionResult
----------------

Output of protection pipeline - 1+1 path establishment.

.. code-block:: python

   @dataclass(frozen=True)
   class ProtectionResult:
       primary_established: bool = False
       backup_established: bool = False
       primary_spectrum: SpectrumResult | None = None
       backup_spectrum: SpectrumResult | None = None

   # Properties
   result.is_fully_protected  # Both paths established

See Also
========

- :doc:`lifecycle` - How these objects flow through the pipeline
- :ref:`core-orchestrator` - Pipeline stages that produce result objects
- :ref:`core-data-structures` - Additional detail on data structures
