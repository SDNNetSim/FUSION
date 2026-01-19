.. _routing-module:

==============
Routing Module
==============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Path computation algorithms for optical network routing
   :Location: ``fusion/modules/routing/``
   :Key Files: ``registry.py``, ``k_shortest_path.py``, ``k_path_cache.py``
   :Depends On: ``fusion.interfaces.router``, ``fusion.core.properties``
   :Used By: ``fusion.core`` (SDNController), ``fusion.core.adapters`` (RoutingAdapter), ``fusion.pipelines``

The routing module provides **algorithm implementations** for computing paths between
source and destination nodes in optical networks. These algorithms are the foundation
of the RSA (Routing and Spectrum Assignment) problem.

**What this module does:**

- Computes K-shortest paths between node pairs
- Implements advanced routing strategies (congestion-aware, fragmentation-aware, etc.)
- Provides 1+1 protection routing for network survivability
- Caches pre-computed paths for performance optimization (K-Path Cache)
- Exposes a registry pattern for algorithm discovery and selection

**When you would work here:**

- Adding a new routing algorithm
- Modifying how paths are selected or weighted
- Implementing custom path constraints
- Optimizing routing performance

Understanding Legacy vs. Orchestrator
=====================================

.. important::

   FUSION supports **two simulation architectures** that coexist. Understanding
   which path your code uses is critical for making modifications.

The routing module algorithms are used by **BOTH** architecture paths:

.. code-block:: text

   +===========================================================================+
   |                     ROUTING MODULE USAGE                                   |
   +===========================================================================+
   |                                                                            |
   |   use_orchestrator = False              use_orchestrator = True            |
   |   (Legacy Path)                         (Orchestrator Path)                |
   |                                                                            |
   |   +------------------+                  +------------------+               |
   |   | SDNController    |                  | SDNOrchestrator  |               |
   |   | (fusion/core/)   |                  | (fusion/core/)   |               |
   |   +--------+---------+                  +--------+---------+               |
   |            |                                     |                         |
   |            | Direct instantiation                | Via RoutingAdapter      |
   |            v                                     v                         |
   |   +------------------+                  +------------------+               |
   |   | fusion/core/     |                  | fusion/core/     |               |
   |   | routing.py       |                  | adapters/        |               |
   |   | (Routing class)  |                  | routing_adapter  |               |
   |   +--------+---------+                  +--------+---------+               |
   |            |                                     |                         |
   |            | Uses                                | Wraps                   |
   |            v                                     v                         |
   |   +-------------------------------------------------------+                |
   |   |              fusion/modules/routing/                   |               |
   |   |                                                        |               |
   |   |   KShortestPath, CongestionAwareRouting,              |                |
   |   |   OnePlusOneProtection, etc.                          |                |
   |   +-------------------------------------------------------+                |
   |                                                                            |
   +===========================================================================+

**Key insight:** The algorithms in ``fusion/modules/routing/`` are the **same code**
used by both paths. The difference is only in how they are invoked.

How Adapters and Pipelines Work with Routing
--------------------------------------------

.. list-table:: Integration Points
   :header-rows: 1
   :widths: 25 35 40

   * - Component
     - Location
     - Role
   * - **RoutingAdapter**
     - ``fusion/core/adapters/routing_adapter.py``
     - Wraps legacy routing for orchestrator. Converts ``NetworkState`` to legacy
       ``SDNProps``, calls legacy routing, converts results to ``RouteResult``.
   * - **RoutingPipeline**
     - ``fusion/pipelines/routing_pipeline.py``
     - Fresh orchestrator implementation for protected routing (1+1). Does NOT
       wrap legacy - uses ``ProtectionAwareStrategy`` directly.
   * - **routing_strategies.py**
     - ``fusion/pipelines/routing_strategies.py``
     - Strategy pattern implementations for orchestrator path selection.

**When to modify which:**

- **Adding a new algorithm** -> ``fusion/modules/routing/`` + register
- **Changing adapter behavior** -> ``fusion/core/adapters/routing_adapter.py``
- **New orchestrator-only features** -> ``fusion/pipelines/``

Key Concepts
============

K-Shortest Paths
   The foundational routing approach. Computes the K shortest paths (by hop count
   or physical length) between two nodes. Most other algorithms build on this.

Path Weight
   The metric used to rank paths. Can be ``hops`` (hop count), ``length`` (physical
   distance in km), or custom weights based on network state.

Modulation Format Selection
   Each path has a maximum reach determined by physical constraints. Longer paths
   require more robust (but less efficient) modulation formats. The routing module
   coordinates with modulation configuration to filter valid formats.

Link Disjointness
   For protection, paths must not share any common links. This ensures a single
   failure cannot affect both working and backup paths.

Action Masking
   In RL contexts, action masking prevents selection of infeasible paths (e.g.,
   paths with insufficient spectrum or that traverse failed links).

.. tip::

   If you're new to optical networking, the key insight is that routing is more
   constrained than IP routing. Paths must satisfy physical layer constraints
   (distance, SNR, available spectrum) in addition to connectivity.

Architecture
============

Module Structure
----------------

.. code-block:: text

   fusion/modules/routing/
   |-- __init__.py              # Public API exports
   |-- README.md                # Module overview
   |-- registry.py              # Algorithm discovery and creation
   |-- k_shortest_path.py       # Base K-SP routing (foundation)
   |-- k_path_cache.py          # Pre-computed path caching
   |-- congestion_aware.py      # Load-balanced routing
   |-- least_congested.py       # Bottleneck-aware routing
   |-- fragmentation_aware.py   # Spectrum fragmentation routing
   |-- nli_aware.py             # Non-linear impairment aware
   |-- xt_aware.py              # Cross-talk aware (multi-core)
   |-- one_plus_one_protection.py  # 1+1 disjoint protection
   |-- utils.py                 # Helper functions
   |
   |-- visualization/           # Visualization plugin (BETA)
   |   |-- __init__.py
   |   `-- routing_plugin.py    # Plugin for routing metrics plots
   |
   `-- tests/                   # Unit tests
       |-- __init__.py
       |-- test_k_path_cache.py
       `-- test_one_plus_one_protection.py

Data Flow
---------

.. code-block:: text

   1. REQUEST ARRIVES
          |
          v
   2. ALGORITHM SELECTION (via registry)
          |
          | create_algorithm("k_shortest_path", engine_props, sdn_props)
          v
   3. PATH COMPUTATION
          |
          | algorithm.route(source, destination, request)
          v
   4. RESULTS STORED IN route_props
          |
          | - paths_matrix: List of candidate paths
          | - modulation_formats_matrix: Valid modulations per path
          | - weights_list: Path weights (distance or hops)
          v
   5. CONSUMER READS route_props
          |
          | (SDNController or RoutingAdapter)
          v
   6. SPECTRUM ASSIGNMENT PROCEEDS

Components
==========

registry.py
-----------

:Purpose: Centralized registry for routing algorithm discovery and instantiation
:Key Classes: ``RoutingRegistry``, ``ROUTING_ALGORITHMS``
:Key Functions: ``create_algorithm()``, ``list_routing_algorithms()``, ``get_routing_algorithm_info()``

The registry pattern enables dynamic algorithm selection based on configuration:

.. code-block:: python

   from fusion.modules.routing import (
       RoutingRegistry,
       create_algorithm,
       list_routing_algorithms,
   )

   # List available algorithms
   algorithms = list_routing_algorithms()
   # ['k_shortest_path', 'congestion_aware', 'least_congested', ...]

   # Create an algorithm instance
   router = create_algorithm("k_shortest_path", engine_props, sdn_props)

   # Route a request
   router.route(source="0", destination="5", request=request_obj)

   # Access results
   paths = router.route_props.paths_matrix

**Registered Algorithms:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Name
     - Description
   * - ``k_shortest_path``
     - Basic K-shortest paths by hop count or length
   * - ``congestion_aware``
     - Routes avoiding congested links based on utilization
   * - ``least_congested``
     - Selects path with least congested bottleneck link
   * - ``fragmentation_aware``
     - Routes considering spectrum fragmentation levels
   * - ``nli_aware``
     - Non-linear impairment aware routing
   * - ``xt_aware``
     - Cross-talk aware routing for multi-core fiber
   * - ``1plus1_protection``
     - Link-disjoint working and backup paths

k_shortest_path.py
------------------

:Purpose: Foundation algorithm - computes K shortest simple paths
:Key Class: ``KShortestPath``

This is the **base routing algorithm** that most others extend or call. It uses
NetworkX's ``shortest_simple_paths`` for efficient path enumeration.

.. code-block:: python

   from fusion.modules.routing import KShortestPath

   router = KShortestPath(engine_props, sdn_props)
   router.route(source=0, destination=5, request=None)

   # Results in route_props
   paths = router.route_props.paths_matrix  # [[0,1,3,5], [0,2,4,5], ...]
   weights = router.route_props.weights_list  # [120.5, 145.2, ...]
   mods = router.route_props.modulation_formats_matrix  # [['QPSK', '16-QAM'], ...]

**Configuration Options:**

- ``k_paths``: Number of paths to compute (default: 3)
- ``routing_weight``: Weight metric - ``"length"`` or ``None`` for hops

k_path_cache.py
---------------

:Purpose: Pre-compute and cache K shortest paths for all node pairs
:Key Class: ``KPathCache``

**What it does:**

The K-Path Cache pre-computes K shortest paths for **all** (source, destination)
pairs at initialization time. This trades memory for latency - path lookups become
O(1) dictionary access instead of on-demand computation.

**When to use it:**

- Large-scale simulations with many requests
- When the same node pairs are queried repeatedly
- When path computation latency is a bottleneck

.. code-block:: python

   from fusion.modules.routing.k_path_cache import KPathCache

   # Pre-compute at initialization
   cache = KPathCache(topology=nx_graph, k=4, ordering="hops")

   # O(1) path lookup during simulation
   paths = cache.get_k_paths(source=0, destination=5)
   # [[0,1,3,5], [0,2,4,5], [0,1,2,4,5], [0,3,4,5]]

   # Get path features for RL
   features = cache.get_path_features(
       path=[0,1,3,5],
       network_spectrum_dict=spectrum_state,
       failure_manager=failure_mgr,
   )
   # {'path_hops': 3, 'min_residual_slots': 15, 'frag_indicator': 0.23, ...}

**Ordering Options:**

- ``"hops"``: Order by hop count (unweighted)
- ``"length"``: Order by physical distance (km)
- ``"latency"``: Order by propagation latency

**Memory Considerations:**

.. code-block:: python

   # Check memory usage
   print(f"Cached pairs: {cache.get_cache_size()}")
   print(f"Memory estimate: {cache.get_memory_estimate_mb():.2f} MB")

one_plus_one_protection.py
--------------------------

:Purpose: 1+1 disjoint protection routing for network survivability
:Key Class: ``OnePlusOneProtection``

.. note::

   **Status: BETA**

   The 1+1 protection implementation is currently in BETA and under active
   development for a future v6.X release. The core algorithm works, but
   integration with failure recovery and spectrum sharing is still being refined.

**How 1+1 Protection Works:**

1. Find ALL link-disjoint paths using max-flow algorithm (Suurballe's)
2. Try pairs until finding one where BOTH paths have feasible modulation
3. Allocate spectrum on BOTH paths simultaneously
4. Traffic transmitted on both; receiver monitors primary
5. On failure, fast switchover to backup (default: 50ms)

.. code-block:: python

   from fusion.modules.routing import OnePlusOneProtection

   router = OnePlusOneProtection(engine_props, sdn_props)
   router.route(source=0, destination=5, request=None)

   # Results include both paths
   primary = router.route_props.paths_matrix[0]
   backup = router.route_props.backup_paths_matrix[0]

   # Paths are guaranteed link-disjoint
   primary_links = set(zip(primary[:-1], primary[1:]))
   backup_links = set(zip(backup[:-1], backup[1:]))
   assert primary_links.isdisjoint(backup_links)

**Configuration Options:**

.. code-block:: python

   engine_props = {
       "protection_settings": {
           "protection_switchover_ms": 50.0,  # Switchover latency
           "revert_to_primary": False,        # Revert after repair
       }
   }

congestion_aware.py / least_congested.py / fragmentation_aware.py
-----------------------------------------------------------------

These algorithms extend K-shortest path with **dynamic weight updates** based on
current network state:

- **CongestionAwareRouting**: Weights links by utilization percentage
- **LeastCongestedRouting**: Selects path with least congested bottleneck
- **FragmentationAwareRouting**: Considers spectrum fragmentation levels

All follow the same pattern:

1. Inherit from ``AbstractRoutingAlgorithm``
2. Override ``update_weights()`` to adjust edge weights
3. Call K-shortest paths with updated weights

nli_aware.py / xt_aware.py
--------------------------

Physical-layer aware routing for advanced optical scenarios:

- **NLIAwareRouting**: Considers non-linear impairment accumulation
- **XTAwareRouting**: Cross-talk aware routing for multi-core fiber (MCF)

These are more specialized and require additional physical parameters in
``engine_props``.

visualization/ (BETA)
---------------------

:Purpose: Visualization plugin for routing algorithm analysis
:Status: **BETA** - API may change in future releases

The visualization submodule provides a plugin that extends FUSION's core
visualization system with routing-specific plots:

- **hop_count_plot**: Hop count distribution analysis
- **path_length_plot**: Physical path length trends
- **computation_time_plot**: Algorithm latency with CDF

See :ref:`routing-visualization` for details.

.. warning::

   The visualization plugin is in BETA. It requires the core visualization
   system at ``fusion/visualization/`` to be properly configured. The API
   and available plot types may change in future releases.

Development Guide
=================

Getting Started
---------------

1. Read the ``AbstractRoutingAlgorithm`` interface in ``fusion/interfaces/router.py``
2. Examine ``k_shortest_path.py`` as the reference implementation
3. Understand how results are stored in ``route_props``
4. Look at existing algorithms for patterns

Adding a New Routing Algorithm
------------------------------

**Step 1: Create the algorithm file**

.. code-block:: python

   # fusion/modules/routing/my_routing.py
   """My custom routing algorithm."""

   from typing import Any
   import networkx as nx
   from fusion.interfaces.router import AbstractRoutingAlgorithm


   class MyRouting(AbstractRoutingAlgorithm):
       """
       My custom routing algorithm.

       Implements [describe what makes it special].
       """

       def __init__(self, engine_props: dict[str, Any], sdn_props: Any) -> None:
           super().__init__(engine_props, sdn_props)
           # Initialize algorithm-specific state

       @property
       def algorithm_name(self) -> str:
           return "my_routing"

       @property
       def supported_topologies(self) -> list[str]:
           return ["NSFNet", "USBackbone60", "Pan-European", "Generic"]

       def validate_environment(self, topology: nx.Graph) -> bool:
           return nx.is_connected(topology)

       def route(self, source: Any, destination: Any, request: Any) -> None:
           # Clear previous results
           self.route_props.paths_matrix = []
           self.route_props.modulation_formats_matrix = []
           self.route_props.weights_list = []

           # YOUR ROUTING LOGIC HERE
           paths = self._compute_paths(source, destination)

           # Populate route_props
           for path in paths:
               self.route_props.paths_matrix.append(path)
               # ... modulation formats and weights

       def get_paths(self, source: Any, destination: Any, k: int = 1) -> list[list[Any]]:
           # Return k paths without populating route_props
           return self._compute_paths(source, destination)[:k]

       def update_weights(self, topology: nx.Graph) -> None:
           # Update edge weights based on network state
           pass

       def get_metrics(self) -> dict[str, Any]:
           return {"algorithm": self.algorithm_name}

**Step 2: Register in registry.py**

Add to the ``_register_default_algorithms`` method in ``RoutingRegistry``:

.. code-block:: python

   from .my_routing import MyRouting

   algorithm_classes = [
       # ... existing algorithms
       MyRouting,
   ]

   algorithm_name_mapping = {
       # ... existing mappings
       MyRouting: "my_routing",
   }

**Step 3: Export in __init__.py**

.. code-block:: python

   from .my_routing import MyRouting

   __all__ = [
       # ... existing exports
       "MyRouting",
   ]

**Step 4: Add tests**

Create ``tests/test_my_routing.py`` following the AAA pattern.

Using K-Path Cache for Performance
----------------------------------

If your algorithm needs frequent path lookups:

.. code-block:: python

   class MyRouting(AbstractRoutingAlgorithm):
       def __init__(self, engine_props, sdn_props):
           super().__init__(engine_props, sdn_props)

           # Initialize cache once
           topology = engine_props.get("topology", sdn_props.topology)
           self._cache = KPathCache(
               topology=topology,
               k=engine_props.get("k_paths", 4),
               ordering="hops"
           )

       def route(self, source, destination, request):
           # O(1) lookup instead of on-demand computation
           paths = self._cache.get_k_paths(source, destination)
           # ... process paths

Configuration
=============

The routing algorithm is selected via configuration:

.. list-table:: Configuration Options
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``route_method``
     - ``k_shortest_path``
     - Algorithm name from registry
   * - ``k_paths``
     - ``3``
     - Number of candidate paths to compute
   * - ``routing_weight``
     - ``length``
     - Path weight metric: ``length``, ``hops``, or custom

**INI Configuration Example:**

.. code-block:: ini

   [simulation_settings]
   route_method = congestion_aware
   k_paths = 5
   routing_weight = length

Testing
=======

:Test Location: ``fusion/modules/routing/tests/``
:Run Tests: ``pytest fusion/modules/routing/tests/ -v``

**Existing Tests:**

- ``test_k_path_cache.py``: Tests pre-computed path caching
- ``test_one_plus_one_protection.py``: Tests disjoint path finding

**Adding New Tests:**

.. code-block:: python

   # tests/test_my_routing.py
   import pytest
   import networkx as nx
   from fusion.modules.routing.my_routing import MyRouting


   @pytest.fixture
   def topology():
       """Create test topology."""
       G = nx.Graph()
       G.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3)])
       return G


   def test_my_routing_finds_path(topology):
       """Test that algorithm finds a valid path."""
       engine_props = {"topology": topology, "k_paths": 3}
       sdn_props = type("SDNProps", (), {"topology": topology})()

       router = MyRouting(engine_props, sdn_props)
       router.route(source=0, destination=3, request=None)

       assert len(router.route_props.paths_matrix) > 0
       assert router.route_props.paths_matrix[0][0] == 0
       assert router.route_props.paths_matrix[0][-1] == 3

Troubleshooting
===============

**Issue: No paths found**

:Symptom: ``route_props.paths_matrix`` is empty after calling ``route()``
:Cause: Source and destination may not be connected, or topology is invalid
:Solution: Verify topology connectivity with ``nx.is_connected(topology)``

**Issue: All modulation formats are False**

:Symptom: ``modulation_formats_matrix`` contains only ``[False, False, ...]``
:Cause: Path length exceeds maximum reach of all modulation formats
:Solution: Check ``modulation_formats_dict`` configuration for ``max_length`` values

**Issue: Protection paths not found**

:Symptom: 1+1 routing returns no backup path
:Cause: Network doesn't have sufficient edge connectivity (< 2)
:Solution: Verify ``nx.edge_connectivity(topology) >= 2`` for 1+1 protection

Related Documentation
=====================

- :ref:`modules-directory` - Overview of all FUSION modules
- :ref:`core-module` - How core uses routing via adapters
- :ref:`interfaces-module` - ``AbstractRoutingAlgorithm`` interface

.. seealso::

   - `NetworkX Shortest Paths <https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html>`_
   - `Yen's K-Shortest Paths Algorithm <https://en.wikipedia.org/wiki/Yen%27s_algorithm>`_

.. _routing-visualization:

Visualization Submodule (BETA)
==============================

.. note::

   **Status: BETA**

   The visualization submodule is in BETA and actively being developed.
   The API may evolve in future releases.

The visualization submodule provides a plugin that extends FUSION's core
visualization system (``fusion/visualization/``) with routing-specific
plot types and metrics.

**What It Provides:**

- ``hop_count_plot``: Hop count analysis with distribution
- ``path_length_plot``: Path length trends vs. traffic volume
- ``computation_time_plot``: Algorithm latency with CDF

**Usage:**

.. code-block:: python

   from fusion.visualization.plugins import get_global_registry

   # Load the plugin
   registry = get_global_registry()
   registry.discover_plugins()
   registry.load_plugin("routing")

   # Generate plots via standard API
   from fusion.visualization.application.use_cases.generate_plot import generate_plot

   result = generate_plot(
       config_path="my_experiment.yml",
       plot_type="hop_count_plot",
       output_path="plots/hop_count.png",
   )

For full details, see the docstrings in ``routing_plugin.py``.
