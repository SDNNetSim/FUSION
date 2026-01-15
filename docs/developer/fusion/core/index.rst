.. _core-module:

===========
Core Module
===========

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Central simulation engine and network coordination components
   :Location: ``fusion/core/``
   :Key Files: ``simulation.py``, ``orchestrator.py``, ``sdn_controller.py``, ``metrics.py``
   :Depends On: ``fusion.domain``, ``fusion.interfaces``, ``fusion.modules.*``, ``fusion.configs``
   :Used By: ``fusion.cli``, ``fusion.rl``, ``fusion.pipelines``

The ``core`` module is the heart of FUSION, implementing the discrete-event simulation
engine and Software-Defined Networking (SDN) coordination logic. It handles request
processing, routing, spectrum assignment, signal quality validation, and comprehensive
metrics collection.

**This module supports two simulation architectures:**

Legacy Engine (Pre-v6.0)
   The original ``SimulationEngine`` and ``SDNController`` for backward compatibility.
   Uses mutable ``engine_props`` dict and direct component orchestration.

Orchestrator (v6.0+)
   The new ``SDNOrchestrator`` with pipeline-based architecture for survivability
   experiments, policy-based routing, and RL integration. Uses immutable
   ``SimulationConfig`` dataclass and adapter pattern for gradual migration.

.. rubric:: In This Section

* :doc:`architecture` - Legacy vs. Orchestrator architecture comparison
* :doc:`orchestrator` - Pipeline flows: grooming, slicing, SNR, protection
* :doc:`simulation` - How the simulation engine works
* :doc:`adapters` - Adapter pattern and migration status
* :doc:`metrics` - Statistics collection, persistence, and ML training data
* :doc:`data_structures` - Result objects, properties classes, multi-fiber mappings
* :doc:`tutorial` - Step-by-step guide for new contributors
* :doc:`contributing` - How to contribute to the core module

**When you work here:**

- Modifying simulation behavior or request processing logic
- Adding new pipeline stages to the orchestrator
- Implementing new routing or spectrum assignment strategies
- Extending metrics collection or adding new statistics
- Integrating RL/ML policies with the simulation
- Working on survivability features (protection, failure handling)

.. important::

   This project was developed by 2 PhD students while conducting full-time research
   in other areas. Some files in this module are large and complex due to time
   constraints. We plan to improve the organization in future versions and welcome
   contributions from the open-source community.

Key Concepts
============

Understanding these concepts is essential before working with the core module:

Discrete Event Simulation
   FUSION uses event-driven simulation where requests arrive and depart at specific
   times. The simulation processes events in chronological order, tracking network
   state changes after each event.

RSA (Routing and Spectrum Assignment)
   The fundamental resource allocation problem in optical networks. Given a request
   (source, destination, bandwidth), RSA finds a path and assigns contiguous spectrum
   slots on all links of that path.

Lightpath
   An end-to-end optical connection with allocated spectrum. A single request may
   use multiple lightpaths (slicing) or share an existing lightpath (grooming).

Pipeline
   A modular processing stage that handles one aspect of request allocation (routing,
   spectrum assignment, SNR validation, etc.). Pipelines can be composed and configured
   via the orchestrator.

Adapter
   A wrapper that makes legacy code compatible with the new pipeline protocol.
   Adapters are temporary migration layers that will be replaced with clean
   implementations in v6.1.0.

NetworkState
   An immutable snapshot of the network at a point in time, including topology,
   spectrum allocation, and lightpath status. Passed to the orchestrator per-call
   (stateless design).

.. tip::

   New contributors should start with the :doc:`architecture` document to understand
   the difference between legacy and orchestrator paths, then read the :doc:`tutorial`
   for hands-on guidance.

Architecture
============

Module Structure
----------------

.. code-block:: text

   fusion/core/
   |-- Legacy Engine
   |   |-- simulation.py        # Main simulation orchestrator (2,209 lines)
   |   |-- sdn_controller.py    # SDN network controller (1,290 lines)
   |   `-- request.py           # Request generation and management
   |-- Orchestrator (v6.0+)
   |   |-- orchestrator.py      # SDNOrchestrator - thin coordination layer (1,315 lines)
   |   |-- pipeline_factory.py  # PipelineFactory and PipelineSet
   |   `-- adapters/            # Temporary migration layers
   |       |-- routing_adapter.py
   |       |-- spectrum_adapter.py
   |       |-- snr_adapter.py
   |       `-- grooming_adapter.py
   |-- Network Components
   |   |-- routing.py           # Path computation dispatcher
   |   |-- spectrum_assignment.py  # Spectrum allocation (992 lines)
   |   |-- snr_measurements.py  # Signal quality calculations (1,802 lines)
   |   `-- grooming.py          # Traffic grooming
   |-- Data & Metrics
   |   |-- properties.py        # Core data structures (612 lines)
   |   |-- metrics.py           # Statistics collection (2,027 lines)
   |   |-- ml_metrics.py        # ML training data collection (163 lines)
   |   `-- persistence.py       # Data persistence and storage
   `-- tests/
       `-- test_*.py            # 19 test files

Visual: Request Processing Flow
-------------------------------

The following diagram shows how a request flows through both architectures:

.. code-block:: text

   +---------------------------+
   |    Request Arrives        |
   |  (source, dest, bandwidth)|
   +-------------+-------------+
                 |
                 v
   +---------------------------+
   | use_orchestrator flag?    |
   +-------------+-------------+
                 |
       +---------+---------+
       |                   |
       v                   v
   +-------+           +-------+
   | False |           | True  |
   +---+---+           +---+---+
       |                   |
       v                   v
   +-------------------+   +---------------------+
   | LEGACY PATH       |   | ORCHESTRATOR PATH   |
   |                   |   |                     |
   | SimulationEngine  |   | SDNOrchestrator     |
   |       |           |   |       |             |
   |       v           |   |       v             |
   | SDNController     |   | GroomingPipeline    |
   |   .allocate()     |   |       |             |
   |       |           |   |       v             |
   |       v           |   | RoutingPipeline     |
   | Routing           |   |   (via adapter)     |
   | .get_route()      |   |       |             |
   |       |           |   |       v             |
   |       v           |   | SpectrumPipeline    |
   | SpectrumAssignment|   |   (via adapter)     |
   | .get_spectrum()   |   |       |             |
   |       |           |   |       v             |
   |       v           |   | SNRPipeline         |
   | SnrMeasurements   |   |   (via adapter)     |
   | .check_snr()      |   |       |             |
   |       |           |   |       v             |
   |       v           |   | SlicingPipeline     |
   | Grooming          |   |       |             |
   | .handle_grooming()|   |       v             |
   +--------+----------+   | ProtectionPipeline  |
            |              +----------+----------+
            |                         |
            v                         v
   +------------------------------------------+
   |              SimStats                    |
   |          .iter_update()                  |
   |    (metrics collection for both paths)   |
   +------------------------------------------+
            |
            v
   +------------------------------------------+
   |           StatsPersistence               |
   |         (save to JSON/CSV)               |
   +------------------------------------------+

Choosing Between Architectures
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Legacy (SDNController)
     - Orchestrator (v6.0+)
   * - Configuration
     - ``engine_props`` dict
     - ``SimulationConfig`` dataclass
   * - State Management
     - Mutable, stored in controller
     - Immutable, passed per-call
   * - Extensibility
     - Modify source code
     - Add new pipelines
   * - RL Integration
     - Direct model calls
     - ControlPolicy interface
   * - Protection (1+1)
     - Limited support
     - Full pipeline support
   * - Grooming
     - Built-in
     - Via GroomingPipeline
   * - Slicing
     - Via LightPathSlicingManager
     - Via SlicingPipeline
   * - Testing
     - Integration tests
     - Unit testable pipelines
   * - Recommended For
     - Legacy experiments
     - New development

Components
==========

simulation.py
-------------

:Purpose: Main simulation orchestrator coordinating all components
:Size: 2,209 lines (needs refactoring - tracked in TODO.md)
:Key Classes: ``SimulationEngine``

The ``SimulationEngine`` is the entry point for running simulations:

.. code-block:: python

   from fusion.core import SimulationEngine

   engine_props = {
       'max_iters': 10,
       'num_requests': 1000,
       'network': 'NSFNet',
       'cores_per_link': 7,
       # ... other configuration
   }

   engine = SimulationEngine(engine_props)
   engine.create_topology()
   completed = engine.run()

**Key responsibilities:**

- Network topology initialization and validation
- Request generation using Poisson arrival process
- Feature flag resolution (legacy vs. orchestrator path)
- Coordination between controller/orchestrator and metrics collection
- Failure manager integration for survivability experiments
- Multiprocessing and batch execution support

See :doc:`simulation` for detailed documentation.

orchestrator.py
---------------

:Purpose: Thin coordination layer for pipeline-based request handling
:Size: 1,315 lines
:Key Classes: ``SDNOrchestrator``

The ``SDNOrchestrator`` routes requests through pipelines without implementing
algorithm logic (that's delegated to pipelines):

.. code-block:: python

   from fusion.core import SDNOrchestrator, PipelineFactory
   from fusion.domain.config import SimulationConfig

   config = SimulationConfig.from_engine_props(engine_props)
   orchestrator = PipelineFactory.create_orchestrator(config)

   # Handle request
   result = orchestrator.handle_arrival(request, network_state)

**Design rules (enforced by code review):**

- No algorithm logic (K-shortest-path, first-fit, SNR calculation)
- No direct numpy access
- No hardcoded slicing/grooming logic
- Receives NetworkState per call, never stores it

See :doc:`orchestrator` for detailed pipeline flow documentation.

metrics.py
----------

:Purpose: Comprehensive statistics collection and analysis
:Size: 2,027 lines (needs splitting - tracked in TODO.md)
:Key Classes: ``SimStats``, ``StatsProps``

.. note::

   This file's organization will be improved in a future version. The file should
   be split into focused modules (blocking_stats.py, resource_stats.py,
   survivability_stats.py, etc.). Contributions welcome!

**Metrics tracked:**

- Blocking probability (requests and bit-rate)
- Path lengths and hop counts
- SNR and crosstalk values
- Transponder counts
- Fragmentation scores
- Recovery times and decision times
- Confidence intervals for statistical validity

See :doc:`metrics` for detailed documentation.

snr_measurements.py
-------------------

:Purpose: Signal quality assessment and calculations
:Size: 1,802 lines
:Key Classes: ``SnrMeasurements``
:Status: **New file, not heavily tested yet**

**Capabilities:**

- SNR, cross-talk, and non-linear interference calculations
- Support for multiple core configurations (4, 7, 13, 19 cores)
- External SNR data source integration
- Comprehensive noise modeling
- Signal quality threshold validation

ml_metrics.py
-------------

:Purpose: Machine learning training data collection
:Size: 163 lines
:Key Classes: ``MLMetricsCollector``
:Status: **New file, not heavily tested yet**

.. important::

   Despite the "ML" naming, this module is specifically for supervised learning (SL)
   and unsupervised learning (USL) data collection. Reinforcement learning (RL)
   metrics are handled separately in ``fusion.rl``.

Dependencies
============

This Module Depends On
----------------------

- ``fusion.domain`` - Request, NetworkState, SimulationConfig, Result objects
- ``fusion.interfaces`` - Pipeline protocols, ControlPolicy interface
- ``fusion.modules.routing`` - Routing algorithms via registry
- ``fusion.modules.spectrum`` - Spectrum assignment helpers
- ``fusion.modules.snr`` - SNR utilities
- ``fusion.modules.failures`` - FailureManager for survivability
- ``fusion.configs`` - Configuration management
- ``fusion.utils`` - Logging, random, network utilities
- ``numpy`` - Spectrum matrices and numerical operations
- ``networkx`` - Network topology and graph operations

Modules That Depend On This
---------------------------

- ``fusion.cli`` - Uses SimulationEngine for running simulations
- ``fusion.rl`` - Uses orchestrator with RL policies
- ``fusion.pipelines`` - Extends orchestrator with additional pipelines
- ``fusion.analysis`` - Consumes simulation results

Testing
=======

:Test Location: ``fusion/core/tests/``
:Run Tests: ``pytest fusion/core/tests/ -v``
:Coverage Target: 90%+ for critical components

**19 test files covering:**

- **Simulation**: ``test_simulation.py``, ``test_feature_flag.py``
- **Orchestrator**: ``test_orchestrator.py``, ``test_orchestrator_policy.py``
- **Core algorithms**: ``test_routing.py``, ``test_spectrum_assignment.py``, ``test_snr_measurements.py``
- **Metrics**: ``test_metrics.py``, ``test_ml_metrics.py``, ``test_recovery_metrics.py``
- **Supporting**: ``test_grooming.py``, ``test_properties.py``, ``test_request.py``
- **Quality**: ``test_determinism.py``

Related Documentation
=====================

- :doc:`architecture` - Deep dive into legacy vs. orchestrator
- :doc:`orchestrator` - Pipeline configuration and flow
- :doc:`adapters` - Migration status and adapter implementation
- :doc:`metrics` - Statistics collection details
- :doc:`data_structures` - Result objects and properties
- :doc:`tutorial` - Hands-on guide for new contributors
- :ref:`configs-module` - Configuration system documentation

.. toctree::
   :maxdepth: 2
   :caption: Contents

   architecture
   orchestrator
   simulation
   adapters
   metrics
   data_structures
   tutorial
