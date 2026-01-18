.. _pipelines-module:

================
Pipelines Module
================

.. important::

   **Status: BETA**

   The pipelines module is a **new component** introduced as part of the orchestrator
   architecture. It is actively being developed and APIs may evolve. While functional
   and tested, consider it beta-quality for production use.

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: High-level provisioning pipelines for the orchestrator architecture
   :Location: ``fusion/pipelines/``
   :Key Files: ``routing_pipeline.py``, ``routing_strategies.py``, ``slicing_pipeline.py``, ``protection_pipeline.py``, ``disjoint_path_finder.py``
   :Depends On: ``fusion.domain``, ``fusion.interfaces.pipelines``
   :Used By: ``fusion.core.SDNOrchestrator``

The pipelines module provides **orchestrator-specific implementations** for multi-step
network provisioning operations. Unlike the algorithm implementations in ``fusion/modules/``,
pipelines are designed specifically for the new orchestrator architecture and implement
the pipeline protocols defined in ``fusion/interfaces/pipelines.py``.

**What this module does:**

- Provides protected routing with 1+1 disjoint path protection
- Implements request slicing for large bandwidth requests
- Offers pluggable routing strategies (k-shortest, load-balanced, protection-aware)
- Handles disjoint path computation for survivability

**When you would work here:**

- Adding new orchestrator-specific provisioning logic
- Implementing new routing strategies
- Modifying how protection paths are computed
- Extending slicing behavior for large requests

Understanding Pipelines vs. Modules
===================================

.. important::

   This is the most common point of confusion. The codebase has **two** places
   where similar-sounding code lives. Understanding the difference is critical.

.. code-block:: text

   +===========================================================================+
   |                   PIPELINES vs MODULES                                     |
   +===========================================================================+
   |                                                                            |
   |   fusion/modules/routing/              fusion/pipelines/                   |
   |   (ALGORITHM Implementations)          (PIPELINE Implementations)          |
   |                                                                            |
   |   +------------------------+           +------------------------+          |
   |   | KShortestPath          |           | ProtectedRoutingPipeline|         |
   |   | CongestionAware        |           | StandardSlicingPipeline |         |
   |   | LeastCongested         |           | ProtectionPipeline      |         |
   |   | FragmentationAware     |           | DisjointPathFinder      |         |
   |   | NLIAware               |           +------------------------+          |
   |   | XTAware                |                      |                        |
   |   +------------------------+                      |                        |
   |            |                                      |                        |
   |            | Used by BOTH paths                   | Orchestrator ONLY      |
   |            v                                      v                        |
   |   +------------------+                  +------------------+               |
   |   | SDNController    |                  | SDNOrchestrator  |               |
   |   | (Legacy)         |                  | (New)            |               |
   |   +------------------+                  +------------------+               |
   |                                                                            |
   +===========================================================================+

**Key Differences:**

.. list-table::
   :header-rows: 1
   :widths: 15 42 43

   * - Aspect
     - ``fusion/modules/``
     - ``fusion/pipelines/``
   * - **Purpose**
     - Single-responsibility algorithm implementations
     - Multi-step provisioning workflows
   * - **Used By**
     - Both legacy (SDNController) and orchestrator paths
     - Orchestrator only
   * - **Pattern**
     - Registry pattern, algorithm interface
     - Pipeline protocol, strategy pattern
   * - **State**
     - Stateless algorithms
     - Stateless pipelines with pluggable strategies
   * - **Returns**
     - Raw results (dicts, tuples)
     - Domain objects (``RouteResult``, ``SlicingResult``)

What About Routing Strategies?
------------------------------

The ``routing_strategies.py`` file contains **strategy implementations** that are
**separate from** the algorithms in ``fusion/modules/routing/``. Here's the distinction:

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Concept
     - ``fusion/modules/routing/``
     - ``fusion/pipelines/routing_strategies.py``
   * - **Contains**
     - Algorithm implementations (k-shortest path, congestion-aware, etc.)
     - Strategy wrappers for orchestrator pipelines
   * - **Interface**
     - ``AbstractRoutingAlgorithm``
     - ``RoutingStrategy`` protocol
   * - **Use Case**
     - Low-level path computation
     - High-level route selection with constraints
   * - **Example**
     - "Find the 3 shortest paths"
     - "Select routes considering protection, load, and exclusions"

.. admonition:: TODO: Consolidation Planned for v6.X
   :class: warning

   **Why the duplication?** The routing strategies currently contain some logic
   that overlaps with ``fusion/modules/routing/`` algorithms. This was the fastest
   path to getting the orchestrator architecture functional and tested.

   **What's the plan?** In a future v6.X release, we plan to:

   1. Refactor routing strategies to wrap/delegate to the registered algorithms
      in ``fusion/modules/routing/``
   2. Eliminate duplicate path computation logic
   3. Ensure both legacy and orchestrator paths use the same underlying algorithms

   For now, be aware that modifying routing behavior may require changes in
   **both** locations depending on which architecture path you're targeting.

How Pipelines Interact with the Simulator
=========================================

The orchestrator architecture uses pipelines as the primary provisioning mechanism:

.. code-block:: text

   +===========================================================================+
   |                     ORCHESTRATOR DATA FLOW                                 |
   +===========================================================================+
   |                                                                            |
   |   Request Arrives                                                          |
   |         |                                                                  |
   |         v                                                                  |
   |   +------------------+                                                     |
   |   | SDNOrchestrator  |                                                     |
   |   +--------+---------+                                                     |
   |            |                                                               |
   |            |  1. Find routes                                               |
   |            v                                                               |
   |   +------------------+     Uses      +------------------+                  |
   |   | RoutingPipeline  |-------------->| RoutingStrategy  |                  |
   |   | (or Adapter)     |               | (pluggable)      |                  |
   |   +--------+---------+               +------------------+                  |
   |            |                                                               |
   |            |  2. Find spectrum                                             |
   |            v                                                               |
   |   +------------------+                                                     |
   |   | SpectrumPipeline |                                                     |
   |   | (Adapter wraps   |                                                     |
   |   |  legacy module)  |                                                     |
   |   +--------+---------+                                                     |
   |            |                                                               |
   |            |  3. Validate SNR                                              |
   |            v                                                               |
   |   +------------------+                                                     |
   |   | SNRPipeline      |                                                     |
   |   | (Adapter wraps   |                                                     |
   |   |  legacy module)  |                                                     |
   |   +--------+---------+                                                     |
   |            |                                                               |
   |            |  4. Slice if needed                                           |
   |            v                                                               |
   |   +------------------+                                                     |
   |   | SlicingPipeline  |                                                     |
   |   | (NEW: fresh      |                                                     |
   |   |  implementation) |                                                     |
   |   +--------+---------+                                                     |
   |            |                                                               |
   |            |  5. Apply protection                                          |
   |            v                                                               |
   |   +------------------+                                                     |
   |   | ProtectionPipeline|                                                    |
   |   | (NEW: fresh       |                                                    |
   |   |  implementation)  |                                                    |
   |   +------------------+                                                     |
   |            |                                                               |
   |            v                                                               |
   |   Lightpath Created                                                        |
   |                                                                            |
   +===========================================================================+

**Important:** Some pipelines are **adapters** (wrapping legacy modules) while others
are **fresh implementations**:

- **Adapters**: ``RoutingAdapter``, ``SpectrumAdapter``, ``SNRAdapter`` (in ``fusion/core/adapters/``)
- **Fresh**: ``ProtectedRoutingPipeline``, ``StandardSlicingPipeline``, ``ProtectionPipeline``

Pipelines vs. Policies
----------------------

Another common confusion: pipelines vs. control policies.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Concept
     - Pipelines
     - Policies
   * - **Location**
     - ``fusion/pipelines/``
     - ``fusion/modules/rl/policies/``
   * - **Purpose**
     - Multi-step provisioning workflows
     - RL decision-making (action selection)
   * - **Interface**
     - Pipeline protocols (``RoutingPipeline``, etc.)
     - ``ControlPolicy`` protocol
   * - **Example**
     - "Find routes, assign spectrum, validate SNR"
     - "Given observation, select action index"
   * - **When Used**
     - Every request during provisioning
     - Only when RL-based decision making is enabled

**Think of it this way:**

- A **policy** decides *which* action to take (e.g., which path index to select)
- A **pipeline** executes *how* to perform the action (e.g., find spectrum for that path)

Architecture
============

Module Structure
----------------

.. code-block:: text

   fusion/pipelines/
   |-- __init__.py                 # Public API exports
   |-- README.md                   # Module overview
   |-- TODO.md                     # Known issues and planned work
   |
   |-- routing_pipeline.py         # ProtectedRoutingPipeline
   |-- routing_strategies.py       # RoutingStrategy implementations
   |-- slicing_pipeline.py         # StandardSlicingPipeline
   |-- protection_pipeline.py      # ProtectionPipeline, allocation helpers
   |-- disjoint_path_finder.py     # DisjointPathFinder algorithms
   |
   `-- tests/                      # Unit tests
       |-- test_protection_pipeline.py
       `-- test_routing_strategies.py

Data Flow
---------

.. code-block:: text

   1. REQUEST ARRIVES
          |
          v
   2. ROUTING PIPELINE
          |
          | ProtectedRoutingPipeline.find_routes()
          |   -> Uses RoutingStrategy to find paths
          |   -> Finds disjoint protection path
          |   -> Returns RouteResult
          v
   3. SPECTRUM ASSIGNMENT (external pipeline)
          |
          | SpectrumPipeline.find_spectrum()
          v
   4. SLICING (if needed)
          |
          | StandardSlicingPipeline.try_slice()
          |   -> Divides bandwidth across tiers
          |   -> Creates multiple lightpaths
          |   -> Returns SlicingResult
          v
   5. PROTECTION ALLOCATION
          |
          | ProtectionPipeline.allocate()
          |   -> Allocates same spectrum on backup path
          |   -> Returns ProtectedAllocationResult
          v
   6. LIGHTPATH CREATED

Components
==========

routing_pipeline.py
-------------------

:Purpose: Protected routing with 1+1 disjoint path support
:Key Class: ``ProtectedRoutingPipeline``

The ``ProtectedRoutingPipeline`` finds both working and protection paths that are
link-disjoint (or node-disjoint based on configuration).

.. code-block:: python

   from fusion.pipelines import ProtectedRoutingPipeline

   pipeline = ProtectedRoutingPipeline(config)
   result = pipeline.find_routes("A", "Z", 100, network_state)

   if result.has_protection:
       print(f"Working: {result.best_path}")
       print(f"Backup: {result.backup_paths[0]}")

**Key Features:**

- Uses ``ProtectionAwareStrategy`` for route selection
- Supports forced paths (from partial grooming)
- Calculates path weights and selects modulations
- Returns ``RouteResult`` with both working and backup paths

routing_strategies.py
---------------------

:Purpose: Pluggable routing strategy implementations
:Key Classes: ``KShortestPathStrategy``, ``LoadBalancedStrategy``, ``ProtectionAwareStrategy``

Routing strategies implement the Strategy pattern for different route selection algorithms:

.. code-block:: python

   from fusion.pipelines import (
       KShortestPathStrategy,
       LoadBalancedStrategy,
       ProtectionAwareStrategy,
   )

   # Basic k-shortest paths
   ksp = KShortestPathStrategy(k=3)
   result = ksp.select_routes("A", "Z", 100, network_state)

   # Load-balanced (considers link utilization)
   lbs = LoadBalancedStrategy(k=5, utilization_weight=0.5)
   result = lbs.select_routes("A", "Z", 100, network_state)

   # Protection-aware (finds disjoint pairs)
   pas = ProtectionAwareStrategy(node_disjoint=True)
   working, backup = pas.find_disjoint_pair("A", "Z", 100, network_state)

**Available Strategies:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Strategy
     - Description
   * - ``KShortestPathStrategy``
     - Basic k-shortest paths using Yen's algorithm
   * - ``LoadBalancedStrategy``
     - Routes considering link utilization and congestion
   * - ``ProtectionAwareStrategy``
     - Finds link/node-disjoint path pairs for 1+1 protection

slicing_pipeline.py
-------------------

:Purpose: Divides large requests across multiple lightpaths
:Key Class: ``StandardSlicingPipeline``

When a single allocation cannot accommodate the full bandwidth (due to fragmentation
or modulation limitations), slicing divides the request across multiple lightpaths:

.. code-block:: python

   from fusion.pipelines import StandardSlicingPipeline

   pipeline = StandardSlicingPipeline(config)
   result = pipeline.try_slice(
       request=request,
       path=path,
       modulation="QPSK",
       bandwidth_gbps=400,
       network_state=network_state,
       spectrum_pipeline=spectrum_pipeline,
   )

   if result.success:
       print(f"Sliced into {result.num_slices} lightpaths")
       print(f"Lightpath IDs: {result.lightpaths_created}")

**Slicing Modes:**

- **Tier-based (default)**: Iterates through bandwidth tiers from ``mod_per_bw`` config
- **Dynamic (GSNR-based)**: Uses GSNR to determine modulation per slice

.. warning::

   **Known Issue:** When ``spectrum_pipeline`` is None (feasibility check only),
   the method returns hardcoded estimates (``num_slices=2``). See ``TODO.md``
   for details and planned fix.

protection_pipeline.py
----------------------

:Purpose: 1+1 dedicated path protection allocation
:Key Classes: ``ProtectionPipeline``, ``ProtectedAllocationResult``

Allocates the same spectrum on both working and protection paths:

.. code-block:: python

   from fusion.pipelines import ProtectionPipeline

   pipeline = ProtectionPipeline(config)
   result = pipeline.allocate(
       working_path=working,
       backup_path=backup,
       spectrum_info=spectrum_result,
       network_state=network_state,
   )

   if result.success:
       print(f"Protected allocation on slots {result.start_slot}-{result.end_slot}")

disjoint_path_finder.py
-----------------------

:Purpose: Algorithms for finding link-disjoint and node-disjoint path pairs
:Key Classes: ``DisjointPathFinder``, ``DisjointnessType``

Uses NetworkX's ``edge_disjoint_paths`` (Suurballe's algorithm) for path computation:

.. code-block:: python

   from fusion.pipelines import DisjointPathFinder, DisjointnessType

   # Link-disjoint paths
   finder = DisjointPathFinder(DisjointnessType.LINK)
   paths = finder.find_disjoint_pair(topology, "A", "D")

   # Node-disjoint paths
   finder = DisjointPathFinder(DisjointnessType.NODE)
   paths = finder.find_disjoint_pair(topology, "A", "D")

   if paths:
       primary, backup = paths
       print(f"Primary: {primary}")
       print(f"Backup: {backup}")

Configuration
=============

Pipelines are configured via ``SimulationConfig``:

.. list-table:: Slicing Configuration
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``max_slices``
     - ``4``
     - Maximum lightpaths per sliced request
   * - ``dynamic_lps``
     - ``False``
     - Enable GSNR-based dynamic slicing
   * - ``fixed_grid``
     - ``False``
     - Fixed vs. flex grid mode for slicing
   * - ``can_partially_serve``
     - ``False``
     - Accept partial bandwidth allocation
   * - ``mod_per_bw``
     - ``{}``
     - Modulation formats per bandwidth tier

.. list-table:: Protection Configuration
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``node_disjoint_protection``
     - ``False``
     - Require node-disjoint (vs. link-disjoint) paths
   * - ``k_paths``
     - ``3``
     - Number of candidate paths to consider

Development Guide
=================

Adding a New Routing Strategy
-----------------------------

**Step 1: Create the strategy class**

.. code-block:: python

   # In fusion/pipelines/routing_strategies.py
   from fusion.domain.results import RouteResult

   class MyCustomStrategy:
       """My custom routing strategy."""

       def __init__(self, custom_param: float = 1.0) -> None:
           self._custom_param = custom_param

       def select_routes(
           self,
           source: str,
           destination: str,
           bandwidth_gbps: int,
           network_state: NetworkState,
           constraints: RouteConstraints | None = None,
       ) -> RouteResult:
           """Select routes using custom logic."""
           # YOUR CUSTOM LOGIC HERE
           paths = self._find_paths(source, destination, network_state)
           weights = self._calculate_weights(paths, network_state)
           modulations = self._select_modulations(weights)

           return RouteResult(
               paths=paths,
               weights_km=weights,
               modulations=modulations,
               strategy_name="my_custom",
           )

**Step 2: Export in __init__.py**

.. code-block:: python

   from .routing_strategies import MyCustomStrategy

   __all__ = [
       # ... existing exports
       "MyCustomStrategy",
   ]

**Step 3: Add tests**

Create tests in ``fusion/pipelines/tests/test_my_strategy.py``.

Modifying Slicing Behavior
--------------------------

The slicing pipeline supports two extension points:

1. **Tier ordering**: Modify ``_try_allocate_tier_based()`` to change how tiers are selected
2. **Feasibility check**: Modify ``_check_tier_feasibility()`` for custom feasibility logic

.. code-block:: python

   # Example: Custom tier selection
   class CustomSlicingPipeline(StandardSlicingPipeline):
       def _get_tier_order(self, bandwidth_gbps: int) -> list[int]:
           """Custom tier ordering (e.g., prefer medium tiers first)."""
           mod_per_bw = getattr(self._config, "mod_per_bw", {})
           tiers = sorted([int(k) for k in mod_per_bw.keys()])
           # Custom ordering logic
           return sorted(tiers, key=lambda t: abs(t - bandwidth_gbps // 2))

Testing
=======

:Test Location: ``fusion/pipelines/tests/``
:Run Tests: ``pytest fusion/pipelines/tests/ -v``

**Test Files:**

- ``test_protection_pipeline.py``: Protection allocation and disjoint path tests
- ``test_routing_strategies.py``: Strategy implementation tests

.. code-block:: bash

   # Run pipeline tests
   pytest fusion/pipelines/tests/ -v

   # Run with coverage
   pytest --cov=fusion.pipelines fusion/pipelines/tests/

**Test Pattern:**

.. code-block:: python

   def test_slicing_creates_multiple_lightpaths():
       """Test that slicing creates expected number of lightpaths."""
       # Arrange
       config = mock_config(max_slices=4, mod_per_bw={...})
       pipeline = StandardSlicingPipeline(config)

       # Act
       result = pipeline.try_slice(
           request, path, "QPSK", 400, network_state,
           spectrum_pipeline=spectrum_pipeline
       )

       # Assert
       assert result.success
       assert result.num_slices >= 2
       assert len(result.lightpaths_created) == result.num_slices

Troubleshooting
===============

**Issue: No protection path found**

:Symptom: ``find_routes()`` returns result without backup paths
:Cause: No link-disjoint (or node-disjoint) path exists
:Solution: Check topology connectivity; may need to disable protection for this request

**Issue: Slicing returns hardcoded estimates**

:Symptom: ``num_slices=2`` and ``slice_bandwidth_gbps=bandwidth_gbps // 2``
:Cause: Using feasibility check mode (``spectrum_pipeline=None``)
:Solution: Pass actual ``spectrum_pipeline`` for accurate allocation; see TODO.md

**Issue: Strategy not selecting expected paths**

:Symptom: Routes differ from expected k-shortest paths
:Cause: Constraints or load balancing affecting selection
:Solution: Check ``RouteConstraints`` and strategy configuration

Known Issues
============

See ``fusion/pipelines/TODO.md`` for tracked issues:

**High Priority:**

- Hardcoded slicing feasibility estimates (see inline TODO in ``slicing_pipeline.py:175-203``)

**Medium Priority:**

- Legacy bug compatibility in SNR recheck (bandwidth tracking inconsistency)
- ``excluded_modulations`` parameter may not be implemented in all adapters

**Low Priority:**

- Code duplication between tier-based and dynamic slicing methods
- Missing slicing metrics collection

Related Documentation
=====================

- :ref:`modules-directory` - Algorithm implementations (routing, spectrum, SNR)
- :ref:`core-module` - How orchestrator uses pipelines
- :ref:`interfaces-module` - Pipeline protocol definitions
- :ref:`domain-module` - ``RouteResult``, ``SlicingResult`` data classes

.. seealso::

   - ``fusion/interfaces/pipelines.py`` - Pipeline protocol definitions
   - ``fusion/core/adapters/`` - Adapters that wrap legacy modules as pipelines
