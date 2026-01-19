.. _core-orchestrator:

==================
Orchestrator Guide
==================

This document provides a comprehensive guide to the ``SDNOrchestrator`` and its
pipeline-based request processing. Understanding these flows is essential for
working with survivability features, RL integration, and adding new capabilities.

Overview
========

The ``SDNOrchestrator`` is a **thin coordination layer** that routes requests
through configurable pipelines. It implements a 6-stage allocation flow:

.. code-block:: text

   Request Arrival
        |
        v
   +--------------------+
   | Stage 1: Grooming  |  Try to use existing lightpath capacity
   +--------------------+
        |
        v
   +--------------------+
   | Stage 2: Routing   |  Find candidate paths (k-shortest, etc.)
   +--------------------+
        |
        v
   +--------------------+
   | Stage 3: Standard  |  Try all paths without slicing
   | Allocation         |
   +--------------------+
        |
        v
   +--------------------+
   | Stage 4: Dynamic   |  Dynamic LP slicing (fixed-grid only)
   | LP Slicing         |
   +--------------------+
        |
        v
   +--------------------+
   | Stage 5: Segment   |  Segment slicing (flex-grid)
   | Slicing            |
   +--------------------+
        |
        v
   +--------------------+
   | Stage 6: Blocking  |  No allocation possible
   +--------------------+

Pipeline Architecture
=====================

PipelineSet
-----------

The orchestrator receives a ``PipelineSet`` containing all pipeline instances:

.. code-block:: python

   from dataclasses import dataclass
   from fusion.interfaces.pipelines import (
       RoutingPipeline, SpectrumPipeline,
       GroomingPipeline, SNRPipeline, SlicingPipeline
   )

   @dataclass(frozen=True)
   class PipelineSet:
       """Container for all pipeline implementations."""

       # Required pipelines
       routing: RoutingPipeline
       spectrum: SpectrumPipeline

       # Optional pipelines
       grooming: GroomingPipeline | None = None
       snr: SNRPipeline | None = None
       slicing: SlicingPipeline | None = None

       @property
       def has_grooming(self) -> bool:
           return self.grooming is not None

       @property
       def has_snr(self) -> bool:
           return self.snr is not None

       @property
       def has_slicing(self) -> bool:
           return self.slicing is not None

PipelineFactory
---------------

The ``PipelineFactory`` creates pipelines based on configuration:

.. code-block:: python

   from fusion.core.pipeline_factory import PipelineFactory
   from fusion.domain.config import SimulationConfig

   config = SimulationConfig.from_engine_props(engine_props)

   # Create individual pipelines
   routing = PipelineFactory.create_routing(config)
   spectrum = PipelineFactory.create_spectrum(config)
   grooming = PipelineFactory.create_grooming(config)  # Returns None if disabled
   snr = PipelineFactory.create_snr(config)            # Returns None if disabled

   # Or create the complete set
   pipelines = PipelineFactory.create_pipeline_set(config)

   # Create orchestrator directly
   orchestrator = PipelineFactory.create_orchestrator(config)

Stage-by-Stage Flow
===================

Stage 1: Grooming
-----------------

**Purpose:** Use existing lightpath capacity before creating new lightpaths.

**When enabled:** ``config.grooming_enabled = True``

**Flow:**

.. code-block:: text

   Request (source, dest, bandwidth)
        |
        v
   +----------------------------------+
   | GroomingPipeline.try_groom()     |
   |                                  |
   | 1. Find lightpaths on path       |
   |    from source to dest           |
   | 2. Check available capacity      |
   | 3. Return GroomingResult         |
   +----------------------------------+
        |
        +---> fully_groomed=True
        |         |
        |         v
        |    Return AllocationResult.success_groomed()
        |
        +---> partially_groomed=True
        |         |
        |         v
        |    remaining_bw = result.remaining_bandwidth_gbps
        |    forced_path = result.forced_path
        |    Continue to Stage 2 (routing)
        |
        +---> no grooming possible
              |
              v
         Continue to Stage 2 (routing)

**GroomingResult fields:**

.. code-block:: python

   @dataclass(frozen=True)
   class GroomingResult:
       fully_groomed: bool = False        # Entire request served
       partially_groomed: bool = False    # Some bandwidth groomed
       bandwidth_groomed_gbps: int = 0    # Amount successfully groomed
       remaining_bandwidth_gbps: int = 0  # Needs new lightpath
       lightpaths_used: tuple[int, ...] = ()  # IDs used
       forced_path: tuple[str, ...] | None = None  # Required path for remainder

**How grooming works:**

1. Find existing lightpaths between source and destination
2. Group lightpaths by physical path
3. Check available capacity on each lightpath
4. If total capacity >= request bandwidth: fully groomed
5. If some capacity available: partially groomed, track remaining
6. If partial, the new lightpath must use same physical path (forced_path)

Stage 2: Routing
----------------

**Purpose:** Find candidate paths from source to destination.

**Flow:**

.. code-block:: text

   Request (source, dest, bandwidth, [forced_path])
        |
        v
   +----------------------------------+
   | RoutingPipeline.find_routes()    |
   |                                  |
   | 1. If forced_path: use it        |
   | 2. Else: run routing algorithm   |
   |    (k-shortest, NLI-aware, etc.) |
   | 3. Filter by modulation reach    |
   | 4. Return RouteResult            |
   +----------------------------------+
        |
        +---> is_empty=True (no paths found)
        |         |
        |         v
        |    Return AllocationResult.blocked(BlockReason.NO_PATH)
        |
        +---> paths found
              |
              v
         Continue to Stage 3 (standard allocation)

**RouteResult fields:**

.. code-block:: python

   @dataclass(frozen=True)
   class RouteResult:
       paths: tuple[tuple[str, ...], ...] = ()     # Candidate paths
       weights_km: tuple[float, ...] = ()          # Path lengths
       modulations: tuple[tuple[str, ...], ...] = ()  # Valid modulations per path
       backup_paths: tuple[tuple[str, ...], ...] | None = None  # For 1+1 protection
       strategy_name: str = ""                      # Algorithm used

**Routing algorithms available:**

- ``k_shortest_path`` - K shortest paths by weight
- ``least_congested`` - Paths with most available spectrum
- ``nli_aware`` - Non-linear interference aware routing
- ``xt_aware`` - Cross-talk aware routing
- ``frag_aware`` - Fragmentation aware routing

Stage 3: Standard Allocation
----------------------------

**Purpose:** Try to allocate spectrum on each candidate path.

**Flow:**

.. code-block:: text

   For each path in route_result.paths:
        |
        v
   +----------------------------------+
   | SpectrumPipeline.find_spectrum() |
   |                                  |
   | 1. Calculate slots needed        |
   | 2. Search for contiguous block   |
   | 3. Check all links on path       |
   | 4. Return SpectrumResult         |
   +----------------------------------+
        |
        +---> is_free=False (no spectrum)
        |         |
        |         v
        |    Try next path
        |
        +---> is_free=True
              |
              v
   +----------------------------------+
   | SNRPipeline.validate_snr()       |  (if SNR enabled)
   |                                  |
   | 1. Calculate path SNR            |
   | 2. Check against threshold       |
   | 3. Return SNRResult              |
   +----------------------------------+
        |
        +---> passed=False
        |         |
        |         v
        |    Try next path (SNR too low)
        |
        +---> passed=True
              |
              v
         Return AllocationResult.success_new_lightpath()

**SpectrumResult fields:**

.. code-block:: python

   @dataclass(frozen=True)
   class SpectrumResult:
       is_free: bool                    # Whether allocation succeeded
       start_slot: int = 0              # First slot index
       end_slot: int = 0                # Last slot (exclusive)
       core: int = 0                    # Core number (MCF)
       band: str = "c"                  # Frequency band
       modulation: str = ""             # Selected modulation
       slots_needed: int = 0            # Slots including guard band
       snr_db: float | None = None      # SNR value (if calculated)

**Spectrum allocation strategies:**

- ``first_fit`` - First available contiguous block
- ``best_fit`` - Smallest sufficient block
- ``last_fit`` - Last available block
- ``priority_based`` - Based on configurable priority

Stage 4: Dynamic LP Slicing (Fixed-Grid)
----------------------------------------

**Purpose:** Split request using dynamic lightpath slicing when standard allocation fails.

**When enabled:** ``config.dynamic_lps = True`` and ``config.fixed_grid = True``

**Flow:**

.. code-block:: text

   Standard allocation failed for all paths
        |
        v
   +----------------------------------+
   | Try allocation with              |
   | allow_slicing=True               |
   | use_dynamic_slicing=True         |
   +----------------------------------+
        |
        v
   Spectrum assignment uses dynamic
   lightpath capacity calculation
        |
        +---> Success
        |         |
        |         v
        |    Return AllocationResult with achieved_bandwidth
        |
        +---> Failure
              |
              v
         Continue to Stage 5 (segment slicing)

**Dynamic LP slicing behavior:**

In fixed-grid mode, the spectrum assignment can allocate less bandwidth than
requested if the full request cannot fit. The ``achieved_bandwidth_gbps`` field
in ``SpectrumResult`` indicates how much was actually allocated.

Stage 5: Segment Slicing (Flex-Grid)
------------------------------------

**Purpose:** Split a large request into multiple smaller lightpaths (segments).

**When enabled:** ``config.slicing_enabled = True``

**Flow:**

.. code-block:: text

   All previous stages failed
        |
        v
   +----------------------------------+
   | SlicingPipeline.slice_request()  |
   |                                  |
   | 1. Determine segment size        |
   | 2. For each segment:             |
   |    a. Find spectrum              |
   |    b. Validate SNR               |
   |    c. Allocate                   |
   | 3. Return SlicingResult          |
   +----------------------------------+
        |
        +---> success=True
        |         |
        |         v
        |    Return AllocationResult.success_sliced(
        |        lightpath_ids=[...],
        |        num_slices=N
        |    )
        |
        +---> success=False
              |
              v
         Continue to Stage 6 (blocking)

**SlicingResult fields:**

.. code-block:: python

   @dataclass(frozen=True)
   class SlicingResult:
       success: bool = False              # Whether slicing succeeded
       num_slices: int = 0                # Number of segments
       slice_bandwidth_gbps: int = 0      # Bandwidth per segment
       lightpaths_created: tuple[int, ...] = ()  # Segment IDs
       total_bandwidth_gbps: int = 0      # Total allocated

**Slicing strategy:**

The slicing pipeline divides the request bandwidth into segments that can each
be served by a single lightpath. This is useful when:

- The full bandwidth requires a modulation format that cannot reach
- Spectrum fragmentation prevents contiguous allocation
- Multiple paths can serve different segments

Stage 6: Blocking
-----------------

**Purpose:** Handle the case when no allocation is possible.

**Flow:**

.. code-block:: text

   All allocation attempts failed
        |
        v
   +----------------------------------+
   | Determine block reason:          |
   |                                  |
   | - NO_PATH: No route found        |
   | - CONGESTION: No spectrum        |
   | - SNR_THRESHOLD: SNR too low     |
   | - PROTECTION_FAIL: No backup     |
   +----------------------------------+
        |
        v
   Return AllocationResult.blocked(reason)

**BlockReason enum:**

.. code-block:: python

   class BlockReason(Enum):
       NO_PATH = "no_path"           # Routing found no paths
       CONGESTION = "congestion"     # No spectrum available
       SNR_THRESHOLD = "snr_threshold"  # SNR below modulation threshold
       PROTECTION_FAIL = "protection_fail"  # No disjoint backup path
       FAILURE = "failure"           # Failed due to network failure

Protection Pipeline (1+1)
=========================

**Purpose:** Establish both primary and backup paths for protected requests.

**When enabled:** ``config.protection_enabled = True`` and ``request.protection_required = True``

**Flow:**

.. code-block:: text

   Protected request arrives
        |
        v
   +----------------------------------+
   | SDNOrchestrator._handle_         |
   | protected_arrival()              |
   +----------------------------------+
        |
        v
   +----------------------------------+
   | ProtectionPipeline.establish_    |
   | protected_connection()           |
   |                                  |
   | 1. Find disjoint path pair       |
   | 2. Allocate primary spectrum     |
   | 3. Allocate backup spectrum      |
   | 4. Return ProtectionResult       |
   +----------------------------------+
        |
        +---> is_fully_protected=True
        |         |
        |         v
        |    Return AllocationResult with
        |    is_protected=True,
        |    protection_result=ProtectionResult
        |
        +---> backup_established=False
        |         |
        |         v
        |    Return AllocationResult with
        |    is_protected=False (primary only)
        |
        +---> primary_established=False
              |
              v
         Return AllocationResult.blocked(BlockReason.PROTECTION_FAIL)

**ProtectionResult fields:**

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

RL and Policy Integration
=========================

The orchestrator integrates with RL/ML policies via the ``ControlPolicy`` interface.

ControlPolicy Interface
-----------------------

.. code-block:: python

   from fusion.interfaces.control_policy import ControlPolicy

   class ControlPolicy(Protocol):
       def select_action(
           self,
           request: Request,
           network_state: NetworkState,
           route_result: RouteResult,
       ) -> tuple[int, str | None]:
           """
           Select which path and modulation to use.

           Returns:
               tuple of (path_index, modulation_format or None)
           """
           ...

RLSimulationAdapter
-------------------

For RL policies, the orchestrator uses an ``RLSimulationAdapter``:

.. code-block:: python

   from fusion.modules.rl.adapter import RLSimulationAdapter

   # Create adapter with trained model
   rl_adapter = RLSimulationAdapter(
       model_path="path/to/model.zip",
       config=config,
   )

   # Create orchestrator with RL adapter
   orchestrator = SDNOrchestrator(
       config=config,
       pipelines=pipeline_set,
       rl_adapter=rl_adapter,
   )

**RL integration flow:**

.. code-block:: text

   Request arrives
        |
        v
   Routing finds k candidate paths
        |
        v
   +----------------------------------+
   | RL Adapter builds observation    |
   | from network_state + paths       |
   +----------------------------------+
        |
        v
   +----------------------------------+
   | RL Model predicts action         |
   | (path_index, modulation)         |
   +----------------------------------+
        |
        v
   Orchestrator uses selected path
   for spectrum allocation

Configuration Options
=====================

Orchestrator-related configuration options:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``use_orchestrator``
     - ``False``
     - Enable orchestrator path (vs. legacy)
   * - ``grooming_enabled``
     - ``False``
     - Enable grooming pipeline
   * - ``snr_enabled``
     - ``True``
     - Enable SNR validation pipeline
   * - ``slicing_enabled``
     - ``False``
     - Enable segment slicing pipeline
   * - ``protection_enabled``
     - ``False``
     - Enable 1+1 protection pipeline
   * - ``dynamic_lps``
     - ``False``
     - Enable dynamic LP slicing
   * - ``fixed_grid``
     - ``True``
     - Use fixed-grid spectrum allocation

Example: Complete Flow
======================

Here's a complete example showing all stages:

.. code-block:: python

   from fusion.core import SDNOrchestrator, PipelineFactory
   from fusion.domain.config import SimulationConfig
   from fusion.domain.request import Request
   from fusion.domain.network_state import NetworkState

   # Setup
   config = SimulationConfig.from_engine_props({
       'use_orchestrator': True,
       'grooming_enabled': True,
       'snr_enabled': True,
       'k_paths': 3,
       # ... other config
   })

   orchestrator = PipelineFactory.create_orchestrator(config)

   # Create request
   request = Request(
       request_id=1,
       source="0",
       destination="5",
       bandwidth_gbps=100,
       arrive_time=0.0,
       depart_time=3600.0,
   )

   # Get network state snapshot
   network_state = NetworkState(
       topology=topology,
       network_spectrum_dict=spectrum_dict,
       lightpath_status_dict=lp_dict,
   )

   # Process request
   result = orchestrator.handle_arrival(request, network_state)

   # Check result
   if result.success:
       print(f"Allocated: {result.total_bandwidth_allocated_gbps} Gbps")
       print(f"Lightpaths: {result.all_lightpath_ids}")
       if result.is_groomed:
           print("Used existing lightpath capacity")
       if result.is_sliced:
           print(f"Split into {len(result.lightpaths_created)} segments")
   else:
       print(f"Blocked: {result.block_reason}")

See Also
========

- :doc:`architecture` - Legacy vs. orchestrator architecture
- :doc:`adapters` - How adapters wrap legacy code
- :doc:`data_structures` - Result object documentation
- ``fusion/interfaces/pipelines.py`` - Pipeline protocol definitions
- ``fusion/pipelines/`` - Pipeline implementations
