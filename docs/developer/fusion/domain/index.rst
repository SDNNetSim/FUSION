.. _domain-module:

=============
Domain Module
=============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Typed domain objects representing core simulation concepts
   :Location: ``fusion/domain/``
   :Key Files: ``config.py``, ``request.py``, ``lightpath.py``, ``network_state.py``, ``results.py``
   :Depends On: ``numpy``, ``networkx``
   :Used By: ``fusion.core``, ``fusion.pipelines``, ``fusion.interfaces``, ``fusion.rl``

The ``domain`` module defines the **shared vocabulary** for the entire FUSION simulator.
It contains typed, validated dataclasses that represent fundamental simulation concepts:
what a request looks like, how lightpaths track capacity, what network state contains,
and what results pipelines return.

**Why this module matters:**

- Every other module imports from ``domain`` to work with these core concepts
- Type safety catches bugs at development time, not runtime
- Immutable result objects prevent accidental state corruption
- Legacy conversion methods enable gradual migration from dict-based code

.. rubric:: In This Section

* :doc:`objects` - Detailed documentation of each domain object
* :doc:`lifecycle` - How objects flow through the request pipeline

**When you work here:**

- Adding new fields to existing domain objects
- Creating new result types for pipeline stages
- Understanding how data flows through the simulator
- Debugging request processing or allocation issues

How Domain Differs from Configs and CLI
=======================================

New developers often confuse these three modules. Here's how they relate:

.. code-block:: text

   +------------------+     +------------------+     +------------------+
   |      CLI         |     |     Configs      |     |     Domain       |
   |   (Entry Point)  |     |  (File Parsing)  |     | (Runtime Objects)|
   +------------------+     +------------------+     +------------------+
          |                        |                        ^
          | User runs              | INI files              | Used throughout
          | fusion-sim             | parsed into            | simulation
          v                        v                        |
   +------------------+     +------------------+            |
   | Parse arguments  |---->| Merge with INI   |            |
   | --network NSFNet |     | configuration    |            |
   +------------------+     +------------------+            |
                                   |                        |
                                   v                        |
                            +------------------+            |
                            | Create engine_   |----------->|
                            | props dict       |            |
                            +------------------+            |
                                   |                        |
                                   v                        |
                            +------------------+     +------+-------+
                            | SimulationConfig |---->| Orchestrator |
                            | (domain object)  |     | uses config  |
                            +------------------+     +--------------+

.. list-table:: Module Comparison
   :header-rows: 1
   :widths: 20 27 27 26

   * - Aspect
     - **CLI** (``fusion/cli/``)
     - **Configs** (``fusion/configs/``)
     - **Domain** (``fusion/domain/``)
   * - Purpose
     - Entry point, argument parsing
     - INI file parsing, validation
     - Runtime data structures
   * - When Used
     - At startup only
     - At startup only
     - Throughout simulation
   * - Key Classes
     - ``run_sim.py`` CLI
     - ``cli_to_config.py``
     - ``SimulationConfig``, ``Request``, etc.
   * - Data Flow
     - Args -> engine_props
     - INI -> engine_props
     - engine_props -> typed objects
   * - Mutability
     - N/A (procedural)
     - Mutable dicts
     - Immutable dataclasses

**In short:**

- **CLI**: How users invoke the simulator (command-line arguments)
- **Configs**: How configuration files are parsed (INI templates, validation)
- **Domain**: The typed objects used during simulation (requests, lightpaths, results)

Key Concepts
============

Understanding these concepts is essential before working with domain objects:

Immutability
   Most domain objects use ``@dataclass(frozen=True)`` making them immutable after
   creation. This prevents bugs where objects are accidentally modified and enables
   safe sharing between components.

Validation on Creation
   All domain objects validate their invariants in ``__post_init__``. Invalid data
   fails fast with clear error messages rather than causing subtle bugs later.

Legacy Conversion
   Domain objects provide ``from_legacy_dict()`` and ``to_legacy_dict()`` methods
   for interoperability with older code that uses plain dictionaries. These will
   be removed in v6.1.0 once migration is complete.

Result Objects
   Pipeline stages return immutable result objects (``RouteResult``, ``SpectrumResult``,
   etc.) that capture the outcome. The ``success`` field is the single source of truth
   for whether an operation succeeded.

State vs. Configuration
   ``SimulationConfig`` is immutable configuration set once at startup.
   ``NetworkState`` is mutable state that changes as requests are processed.
   ``Request`` and ``Lightpath`` track individual connection lifecycle.

Architecture
============

Module Structure
----------------

.. code-block:: text

   fusion/domain/
   |-- __init__.py          # Public API exports
   |-- config.py            # SimulationConfig - immutable configuration
   |-- request.py           # Request, RequestType, RequestStatus, BlockReason
   |-- lightpath.py         # Lightpath - optical path with capacity tracking
   |-- network_state.py     # NetworkState, LinkSpectrum - mutable network state
   |-- results.py           # Pipeline result objects (frozen dataclasses)
   |-- README.md            # Module documentation
   |-- TODO.md              # Development roadmap
   `-- tests/               # Unit tests
       |-- test_config.py
       |-- test_request.py
       |-- test_network_state.py
       `-- test_results.py

Object Relationship Diagram
---------------------------

The following diagram shows how domain objects relate to each other:

.. code-block:: text

   +-------------------+
   | SimulationConfig  |  (immutable, created once at startup)
   +-------------------+
            |
            | provides parameters for
            v
   +-------------------+         +-------------------+
   |   NetworkState    |<------->|     Lightpath     |
   | (mutable state)   |  owns   | (capacity mgmt)   |
   +-------------------+         +-------------------+
            ^                            ^
            |                            |
            | queries/updates            | creates/uses
            |                            |
   +-------------------+         +-------------------+
   |      Request      |-------->|  AllocationResult |
   | (service request) | yields  | (pipeline output) |
   +-------------------+         +-------------------+
                                         |
                                         | contains
                                         v
                                 +-------------------+
                                 | RouteResult       |
                                 | SpectrumResult    |
                                 | GroomingResult    |
                                 | SNRResult         |
                                 | SlicingResult     |
                                 | ProtectionResult  |
                                 +-------------------+

Data Flow Summary
-----------------

1. **Configuration**: ``SimulationConfig`` is created from ``engine_props`` at startup
2. **State Initialization**: ``NetworkState`` is created with topology and empty spectrum
3. **Request Arrival**: ``Request`` objects are generated with source, destination, bandwidth
4. **Pipeline Processing**: Orchestrator processes request through pipeline stages
5. **Result Objects**: Each stage returns an immutable result (``RouteResult``, etc.)
6. **State Updates**: ``NetworkState`` is updated with new lightpaths and spectrum allocations
7. **Final Result**: ``AllocationResult`` captures whether request was served or blocked

Components
==========

config.py
---------

:Purpose: Immutable simulation configuration
:Key Classes: ``SimulationConfig``

Replaces the mutable ``engine_props`` dictionary with a typed, validated, frozen
dataclass. Provides type safety, IDE autocompletion, and clear defaults.

.. code-block:: python

   from fusion.domain import SimulationConfig

   # Create from legacy dict
   config = SimulationConfig.from_engine_props(engine_props)

   # Access with type safety
   print(config.network_name)      # IDE knows this is str
   print(config.cores_per_link)    # IDE knows this is int

request.py
----------

:Purpose: Network service request with lifecycle tracking
:Key Classes: ``Request``, ``RequestType``, ``RequestStatus``, ``BlockReason``, ``ProtectionStatus``

Represents a connection request from source to destination with bandwidth requirement.
Tracks lifecycle state from PENDING through ALLOCATED or BLOCKED.

.. code-block:: python

   from fusion.domain import Request, RequestType, RequestStatus

   request = Request(
       request_id=1,
       source="0",
       destination="5",
       bandwidth_gbps=100,
       arrive=0.5,
       depart=1.5,
   )

   # Lifecycle tracking
   print(request.status)  # RequestStatus.PENDING

lightpath.py
------------

:Purpose: Allocated optical path with capacity management
:Key Classes: ``Lightpath``

Represents an established optical connection with spectrum assignment. Supports
traffic grooming by tracking capacity and per-request bandwidth allocations.

.. code-block:: python

   from fusion.domain import Lightpath

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

   # Capacity management
   if lp.can_accommodate(50):
       lp.allocate_bandwidth(request_id=42, bandwidth_gbps=50)

network_state.py
----------------

:Purpose: Mutable network state container
:Key Classes: ``NetworkState``, ``LinkSpectrum``

The single source of truth for network state during simulation. Owns all
``LinkSpectrum`` objects and the lightpath registry.

.. code-block:: python

   from fusion.domain import NetworkState

   # Check spectrum availability
   if state.is_spectrum_available(path, start, end, core, band):
       lightpath_id = state.create_lightpath(...)

results.py
----------

:Purpose: Immutable pipeline stage outputs
:Key Classes: ``RouteResult``, ``SpectrumResult``, ``GroomingResult``, ``SlicingResult``, ``SNRResult``, ``ProtectionResult``, ``AllocationResult``

Each pipeline stage returns a frozen result object. ``AllocationResult`` is the
final output indicating success or failure with detailed tracking.

.. code-block:: python

   from fusion.domain import AllocationResult, BlockReason

   # Successful allocation
   result = AllocationResult(
       success=True,
       lightpaths_created=(1,),
       total_bandwidth_allocated_gbps=100,
   )

   # Blocked request
   result = AllocationResult(
       success=False,
       block_reason=BlockReason.CONGESTION,
   )

Dependencies
============

This Module Depends On
----------------------

- ``numpy`` - Spectrum array storage in LinkSpectrum
- ``networkx`` - Topology queries in NetworkState
- **Standard Library**: ``dataclasses``, ``enum``, ``typing``

Modules That Depend On This
---------------------------

- ``fusion.core`` - SimulationEngine, SDNOrchestrator, SDNController
- ``fusion.pipelines`` - All pipeline implementations
- ``fusion.interfaces`` - Pipeline protocols reference domain types
- ``fusion.rl`` - RL adapters work with Request and NetworkState
- ``fusion.modules`` - Routing, spectrum, SNR modules

Testing
=======

:Test Location: ``fusion/domain/tests/``
:Run Tests: ``pytest fusion/domain/tests/ -v``
:Coverage Target: 80%+

**Test files:**

- ``test_config.py`` - SimulationConfig creation and conversion
- ``test_request.py`` - Request lifecycle and enums
- ``test_network_state.py`` - NetworkState and LinkSpectrum operations
- ``test_results.py`` - All result dataclasses

**Adding new tests:**

.. code-block:: python

   def test_lightpath_when_capacity_exceeded_then_returns_false():
       """Test that allocate_bandwidth returns False when insufficient capacity."""
       # Arrange
       lp = Lightpath(
           lightpath_id=1,
           path=["0", "5"],
           start_slot=0,
           end_slot=8,
           core=0,
           band="c",
           modulation="QPSK",
           total_bandwidth_gbps=100,
           remaining_bandwidth_gbps=50,
       )

       # Act
       result = lp.allocate_bandwidth(request_id=1, bandwidth_gbps=100)

       # Assert
       assert result is False
       assert lp.remaining_bandwidth_gbps == 50  # Unchanged

Related Documentation
=====================

- :ref:`core-module` - How core uses domain objects
- :ref:`core-orchestrator` - Pipeline flows using result objects
- :ref:`core-data-structures` - Detailed result object documentation
- :ref:`configs-module` - Configuration parsing that produces engine_props

.. toctree::
   :maxdepth: 2
   :caption: Contents

   objects
   lifecycle
