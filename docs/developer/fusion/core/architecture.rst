.. _core-architecture:

============
Architecture
============

This document explains the two simulation architectures in FUSION: the **Legacy Engine**
(pre-v6.0) and the **Orchestrator** (v6.0+). Understanding these architectures is
essential for contributing to the core module.

Overview
========

FUSION evolved from a monolithic design to a modular, pipeline-based architecture.
Both architectures are fully supported and share the same underlying algorithms,
but differ in how they organize and coordinate components.

.. code-block:: text

   +-----------------------------------------------------------------+
   |                        FUSION Core                              |
   +-----------------------------------------------------------------+
   |                                                                 |
   |  +---------------------------+  +---------------------------+   |
   |  |     LEGACY ENGINE         |  |      ORCHESTRATOR         |   |
   |  |     (Pre-v6.0)            |  |      (v6.0+)              |   |
   |  +---------------------------+  +---------------------------+   |
   |  |                           |  |                           |   |
   |  | SimulationEngine          |  | SimulationEngine          |   |
   |  |     |                     |  |     |                     |   |
   |  |     v                     |  |     v                     |   |
   |  | SDNController             |  | SDNOrchestrator           |   |
   |  |     |                     |  |     |                     |   |
   |  |     +-> Routing           |  |     +-> RoutingPipeline   |   |
   |  |     +-> SpectrumAssign    |  |     +-> SpectrumPipeline  |   |
   |  |     +-> SnrMeasurements   |  |     +-> SNRPipeline       |   |
   |  |     +-> Grooming          |  |     +-> GroomingPipeline  |   |
   |  |                           |  |     +-> SlicingPipeline   |   |
   |  |                           |  |     +-> ProtectionPipeline|   |
   |  +---------------------------+  +---------------------------+   |
   |                                                                 |
   |  +---------------------------+  +---------------------------+   |
   |  |     SHARED COMPONENTS     |  |      NEW COMPONENTS       |   |
   |  +---------------------------+  +---------------------------+   |
   |  | - SimStats (metrics)      |  | - Adapters (migration)    |   |
   |  | - Request generation      |  | - PipelineFactory         |   |
   |  | - Network topology        |  | - ControlPolicy interface |   |
   |  | - Properties classes      |  | - Result objects          |   |
   |  +---------------------------+  +---------------------------+   |
   |                                                                 |
   +-----------------------------------------------------------------+

Legacy Engine (Pre-v6.0)
========================

The legacy architecture uses direct component orchestration with mutable state.

Key Components
--------------

SimulationEngine
   The main entry point that initializes the network, generates requests,
   and coordinates the simulation loop.

SDNController
   Directly manages request allocation by calling routing, spectrum assignment,
   SNR validation, and grooming methods in sequence.

Properties Classes
   Mutable objects (``RoutingProps``, ``SpectrumProps``, ``SDNProps``, etc.)
   that carry state between components.

Configuration Flow
------------------

.. code-block:: text

   INI File
       |
       v
   config_setup.py
       |
       v
   engine_props (dict)  <-- Mutable, flat key-value pairs
       |
       v
   SimulationEngine
       |
       v
   SDNController
       |
       +-> self.engine_props['k_paths']
       +-> self.engine_props['network']
       +-> ... (direct dict access)

Request Processing Flow
-----------------------

.. code-block:: python

   # Simplified legacy flow in SDNController.allocate()

   def allocate(self, request_data):
       # 1. Initialize request properties
       self.sdn_props.source = request_data['source']
       self.sdn_props.destination = request_data['destination']
       self.sdn_props.bandwidth = request_data['bandwidth']

       # 2. Try grooming first (if enabled)
       if self.engine_props.get('grooming'):
           self.grooming.handle_grooming()
           if self.sdn_props.was_groomed:
               return  # Success via grooming

       # 3. Find routes
       self.routing.get_route()
       if not self.routing.route_props.paths_matrix:
           self.sdn_props.block_reason = 'distance'
           return  # Blocked - no route

       # 4. For each candidate path, try spectrum assignment
       for path_index, path in enumerate(self.routing.route_props.paths_matrix):
           self.spectrum_assignment.get_spectrum(modulation_list)

           if self.spectrum_props.is_free:
               # 5. Validate SNR (if enabled)
               if self.engine_props.get('snr_type'):
                   self.snr_measurements.check_snr()
                   if not snr_valid:
                       continue  # Try next path

               # 6. Success - allocate and return
               self._update_network_state()
               return

       # All paths exhausted
       self.sdn_props.block_reason = 'congestion'

Characteristics
---------------

**Advantages:**

- Simple, straightforward control flow
- Easy to understand for single-path debugging
- All state in one place (SDNController)

**Limitations:**

- Tight coupling between components
- Difficult to test individual components in isolation
- Hard to add new pipeline stages without modifying controller
- Mutable state makes debugging complex scenarios difficult
- No clear interface for external policies (RL/ML)

Orchestrator (v6.0+)
====================

The orchestrator architecture uses a pipeline-based design with immutable state.

Key Components
--------------

SDNOrchestrator
   A thin coordination layer that routes requests through pipelines.
   Contains NO algorithm logic - all computation is delegated to pipelines.

PipelineFactory
   Creates and configures pipelines based on ``SimulationConfig``.
   Uses lazy imports to avoid circular dependencies.

PipelineSet
   A frozen dataclass containing all pipeline instances:

   - ``routing`` (required): Finds candidate paths
   - ``spectrum`` (required): Assigns spectrum slots
   - ``grooming`` (optional): Uses existing lightpath capacity
   - ``snr`` (optional): Validates signal quality
   - ``slicing`` (optional): Splits large requests

Adapters
   Temporary wrappers that make legacy components compatible with
   pipeline protocols. See :doc:`adapters` for details.

SimulationConfig
   An immutable frozen dataclass with typed attributes and computed properties.
   Replaces the mutable ``engine_props`` dict.

Result Objects
   Immutable frozen dataclasses (``RouteResult``, ``SpectrumResult``,
   ``AllocationResult``, etc.) that capture pipeline outputs.
   See :doc:`data_structures` for details.

Configuration Flow
------------------

.. code-block:: text

   INI File
       |
       v
   config_setup.py
       |
       v
   engine_props (dict)
       |
       v
   SimulationConfig.from_engine_props()  <-- Immutable, typed, validated
       |
       v
   PipelineFactory.create_orchestrator()
       |
       v
   SDNOrchestrator
       |
       +-> config.k_paths          # Typed attribute
       +-> config.network_name     # Computed property
       +-> config.total_slots      # Computed property

Request Processing Flow
-----------------------

.. code-block:: python

   # Simplified orchestrator flow in SDNOrchestrator.handle_arrival()

   def handle_arrival(self, request, network_state):
       # network_state is passed per-call (stateless design)

       # Stage 1: Grooming (if enabled)
       if self.grooming and self.config.grooming_enabled:
           groom_result = self.grooming.try_groom(request, network_state)
           if groom_result.fully_groomed:
               return AllocationResult.success_groomed(...)
           if groom_result.partially_groomed:
               remaining_bw = groom_result.remaining_bandwidth_gbps
               forced_path = groom_result.forced_path

       # Stage 2: Routing
       route_result = self.routing.find_routes(
           request.source, request.destination,
           remaining_bw, network_state
       )
       if route_result.is_empty:
           return AllocationResult.blocked(BlockReason.NO_PATH)

       # Stage 3: Try each path
       for path_index in range(route_result.num_paths):
           path = route_result.get_path(path_index)
           modulations = route_result.get_modulations_for_path(path_index)

           # Stage 3a: Spectrum assignment
           spectrum_result = self.spectrum.find_spectrum(
               path, modulations, remaining_bw, network_state
           )
           if not spectrum_result.is_free:
               continue

           # Stage 3b: SNR validation (if enabled)
           if self.snr:
               snr_result = self.snr.validate_snr(
                   path, spectrum_result, network_state
               )
               if not snr_result.passed:
                   continue

           # Success
           return AllocationResult.success_new_lightpath(...)

       # Stage 4: Try slicing (if enabled)
       if self.slicing:
           slicing_result = self.slicing.slice_request(...)
           if slicing_result.success:
               return AllocationResult.success_sliced(...)

       return AllocationResult.blocked(BlockReason.CONGESTION)

Characteristics
---------------

**Advantages:**

- Modular, testable components
- Clear interfaces between stages (pipeline protocols)
- Easy to add new pipeline stages without modifying orchestrator
- Immutable state simplifies debugging
- Clean RL/ML integration via ControlPolicy interface
- Stateless design enables better parallelization

**Limitations:**

- More complex initial setup
- Adapters add indirection during migration period
- Requires understanding of multiple components

How to Choose
=============

Use Legacy (SDNController) When:
--------------------------------

- Running existing experiments that depend on legacy behavior
- Debugging specific v5.x issues
- Working with configurations that explicitly set ``use_orchestrator = False``

Use Orchestrator (SDNOrchestrator) When:
----------------------------------------

- Starting new development or experiments
- Implementing survivability features (protection, failure handling)
- Integrating RL/ML policies
- Adding new pipeline stages
- Writing unit tests for specific components
- Working with configurations that set ``use_orchestrator = True``

Switching Between Architectures
===============================

The architecture is controlled by the ``use_orchestrator`` flag:

Via Configuration File
----------------------

.. code-block:: ini

   [general_settings]
   use_orchestrator = True   ; Use orchestrator path
   ; OR
   use_orchestrator = False  ; Use legacy path (default)

Via Environment Variable
------------------------

.. code-block:: bash

   export FUSION_USE_ORCHESTRATOR=1  # Use orchestrator
   python -m fusion.cli.run_sim ...

Via Code
--------

.. code-block:: python

   engine_props = {
       'use_orchestrator': True,  # or False
       # ... other settings
   }
   engine = SimulationEngine(engine_props)

Resolution Priority
-------------------

1. Environment variable ``FUSION_USE_ORCHESTRATOR`` (highest)
2. Configuration parameter ``use_orchestrator``
3. Default: ``False`` (legacy path)

Migration Status
================

The project is in active migration from legacy to orchestrator:

**Completed:**

- Core orchestrator structure
- Pipeline protocols and interfaces
- All four adapters (routing, spectrum, SNR, grooming)
- SimulationConfig dataclass
- Result objects
- Basic pipeline implementations

**In Progress:**

- Slicing pipeline integration
- Protection pipeline for 1+1

**Planned (v6.1.0):**

- Replace adapters with clean implementations
- Remove legacy adapter code
- Full documentation update

See :doc:`adapters` for detailed migration status of each component.

Design Principles
=================

The orchestrator follows these design principles:

1. **No Algorithm Logic in Orchestrator**

   The orchestrator coordinates pipelines but never implements algorithm logic.
   All computation (K-shortest-path, first-fit, SNR calculation) is delegated
   to pipelines.

2. **Stateless Per-Call Design**

   The orchestrator receives ``NetworkState`` as a parameter for each call
   and never stores it. This enables better testing and parallelization.

3. **Immutable Results**

   All result objects are frozen dataclasses. Once created, they cannot be
   modified. This prevents subtle bugs from state mutation.

4. **Pipeline Protocols**

   Each pipeline implements a protocol (``RoutingPipeline``, ``SpectrumPipeline``,
   etc.) defined in ``fusion.interfaces.pipelines``. This enables substitution
   and testing.

5. **Adapter Pattern for Migration**

   Legacy components are wrapped in adapters to satisfy pipeline protocols.
   Adapters are temporary and will be replaced with clean implementations.

See Also
========

- :doc:`orchestrator` - Detailed pipeline flow documentation
- :doc:`adapters` - Adapter pattern and migration status
- :doc:`simulation` - SimulationEngine documentation
- :ref:`configs-module` - Configuration system
