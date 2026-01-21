.. _modules-directory:

==========================
Modules Directory Overview
==========================

.. admonition:: Why This Page Exists
   :class: important

   The ``fusion/modules/`` directory contains algorithm implementations that can be
   confusing to navigate. This page provides a high-level map to help you understand:

   - What lives where and why
   - Legacy code vs. orchestrator-compatible code
   - How modules connect to the simulation engine
   - Where to go when you want to modify specific functionality

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Pluggable algorithm implementations (routing, spectrum, SNR, RL, etc.)
   :Location: ``fusion/modules/``
   :Contains: routing/, spectrum/, snr/, rl/, ml/, failures/
   :Used By: ``fusion.core`` (via adapters), ``fusion.pipelines``

.. admonition:: Detailed Module Documentation
   :class: seealso

   Looking for specific module docs? Jump directly to:

   - :ref:`failures-module` - Network failure injection (beta)
   - :ref:`ml-module` - Machine learning utilities (beta)
   - :ref:`rl-module` - Reinforcement learning (transitioning to UnifiedSimEnv)
   - :ref:`routing-module` - Path computation algorithms
   - :ref:`snr-module` - Signal quality assessment
   - :ref:`spectrum-module` - Spectrum slot assignment algorithms

The ``modules`` directory contains **algorithm implementations** - the actual logic for
routing paths, assigning spectrum, calculating SNR, and making RL decisions. These are
the building blocks that the simulation engine uses.

**The key confusion point:** There are multiple places where similar-sounding code lives:

- ``fusion/modules/`` - Algorithm implementations (this directory)
- ``fusion/pipelines/`` - Orchestrator pipeline wrappers
- ``fusion/core/adapters/`` - Legacy-to-orchestrator bridges
- ``fusion/interfaces/`` - Protocol definitions (no logic)
- ``fusion/domain/`` - Data structures (no logic)

This page explains how they all connect.

Understanding the Architecture
==============================

Legacy vs. Orchestrator: The Big Picture
----------------------------------------

FUSION supports **two simulation architectures** that coexist:

.. code-block:: text

   +------------------------------------------------------------------+
   |                     SIMULATION ENGINE                             |
   |                                                                   |
   |   use_orchestrator = False          use_orchestrator = True       |
   |   (Legacy Path)                     (Orchestrator Path)           |
   +------------------------------------------------------------------+
            |                                      |
            v                                      v
   +------------------+                  +------------------+
   | SDNController    |                  | SDNOrchestrator  |
   | (fusion/core/)   |                  | (fusion/core/)   |
   +--------+---------+                  +--------+---------+
            |                                      |
            | Direct calls                         | Via Adapters
            |                                      |
            v                                      v
   +------------------+                  +------------------+
   | fusion/modules/  |                  | fusion/core/     |
   | routing/         |<-----------------| adapters/        |
   | spectrum/        |   Adapters wrap  |                  |
   | snr/             |   legacy modules +--------+---------+
   +------------------+                           |
                                                  | Adapters call
                                                  |
                                                  v
                                         +------------------+
                                         | fusion/modules/  |
                                         | (same code!)     |
                                         +------------------+

**Key insight:** The modules in ``fusion/modules/`` are used by BOTH paths. The adapters
in ``fusion/core/adapters/`` simply wrap them to satisfy the orchestrator's pipeline
protocols.

Where Algorithm Logic Lives
---------------------------

.. list-table:: Code Location Guide
   :header-rows: 1
   :widths: 25 40 35

   * - If You Want To...
     - Look In...
     - Notes
   * - Modify routing algorithms
     - ``fusion/modules/routing/``
     - K-shortest-path, congestion-aware, etc.
   * - Modify spectrum assignment
     - ``fusion/modules/spectrum/``
     - First-fit, best-fit, last-fit
   * - Modify SNR calculations
     - ``fusion/modules/snr/`` or ``fusion/core/snr_measurements.py``
     - Module has helpers; core has main logic
   * - Add RL policies
     - ``fusion/modules/rl/policies/``
     - BC, IQL, pointer network policies
   * - Add RL environments
     - ``fusion/modules/rl/gymnasium_envs/``
     - Gym-compatible environments
   * - Modify failure handling
     - ``fusion/modules/failures/``
     - Link, node, SRLG failures
   * - Modify ML preprocessing
     - ``fusion/modules/ml/``
     - Feature engineering, evaluation

What About fusion/pipelines/?
-----------------------------

The ``fusion/pipelines/`` directory contains **orchestrator-specific implementations**
that don't use the legacy module structure:

.. code-block:: text

   fusion/pipelines/
   |-- routing_pipeline.py      # New routing for orchestrator
   |-- routing_strategies.py    # Strategy pattern implementations
   |-- protection_pipeline.py   # 1+1 protection logic
   |-- slicing_pipeline.py      # Request slicing logic
   `-- disjoint_path_finder.py  # Path computation utilities

These are **not** wrappers around ``fusion/modules/``. They are fresh implementations
designed specifically for the orchestrator's pipeline protocol.

**When to use which:**

- Modifying legacy simulation behavior -> ``fusion/modules/``
- Adding new orchestrator features -> ``fusion/pipelines/``
- The adapters bridge the gap when orchestrator needs legacy algorithms

.. seealso::

   For detailed documentation on the pipelines module, see :ref:`pipelines-module`.

Module Directory Structure
==========================

.. code-block:: text

   fusion/modules/
   |
   |-- routing/                 # Path computation algorithms
   |   |-- k_shortest_path.py   # Basic K-SP routing
   |   |-- congestion_aware.py  # Load-balanced routing
   |   |-- least_congested.py   # Bottleneck-aware routing
   |   |-- fragmentation_aware.py
   |   |-- nli_aware.py         # Non-linear impairment aware
   |   |-- xt_aware.py          # Cross-talk aware (multi-core)
   |   |-- one_plus_one_protection.py  # Protection path finding
   |   |-- registry.py          # Algorithm discovery
   |   `-- utils.py
   |
   |-- spectrum/                # Spectrum slot assignment
   |   |-- first_fit.py         # First available slots
   |   |-- best_fit.py          # Smallest fitting gap
   |   |-- last_fit.py          # Last available slots
   |   |-- light_path_slicing.py # Large request segmentation
   |   |-- registry.py
   |   `-- utils.py
   |
   |-- snr/                     # Signal quality utilities
   |   |-- snr.py               # SNR helper functions
   |   |-- registry.py
   |   `-- utils.py
   |
   |-- rl/                      # Reinforcement learning
   |   |-- policies/            # Decision-making policies
   |   |   |-- bc_policy.py     # Behavioral cloning
   |   |   |-- iql_policy.py    # Implicit Q-learning
   |   |   |-- pointer_policy.py # Attention-based
   |   |   |-- ksp_ff_policy.py # K-SP + First-Fit baseline
   |   |   `-- one_plus_one_policy.py
   |   |-- gymnasium_envs/      # RL environments
   |   |-- agents/              # Agent implementations
   |   |-- algorithms/          # RL algorithm configs
   |   |-- feat_extrs/          # Feature extractors
   |   |-- sb3/                 # Stable-Baselines3 integration
   |   `-- utils/               # RL utilities
   |
   |-- ml/                      # Machine learning utilities
   |   |-- preprocessing.py     # Data preprocessing
   |   |-- feature_engineering.py
   |   |-- evaluation.py
   |   `-- model_io.py          # Model save/load
   |
   `-- failures/                # Network failure simulation
       |-- failure_manager.py   # Failure orchestration
       |-- failure_types.py     # F1/F2/F3/F4 failures
       `-- registry.py

How Modules Connect to the Simulator
====================================

The Registry Pattern
--------------------

Each module uses a **registry pattern** for algorithm discovery:

.. code-block:: python

   # In fusion/modules/routing/registry.py
   from fusion.modules.routing import RoutingRegistry, create_algorithm

   # List available algorithms
   algorithms = RoutingRegistry.list_algorithms()
   # ['k_shortest_path', 'congestion_aware', 'least_congested', ...]

   # Create an algorithm instance
   router = create_algorithm("k_shortest_path", engine_props, sdn_props)

   # Use it
   path = router.route(source="0", destination="5", request=request_obj)

**The same pattern applies to:**

- ``fusion.modules.spectrum.SpectrumRegistry``
- ``fusion.modules.snr.SnrRegistry``

Legacy Path: Direct Usage
-------------------------

In the legacy path, ``SDNController`` creates algorithm instances directly:

.. code-block:: python

   # Inside SDNController (simplified)
   from fusion.modules.routing import create_algorithm

   self.router = create_algorithm(
       engine_props["routing_algorithm"],
       engine_props,
       sdn_props
   )

   # Later, during request processing:
   path = self.router.route(source, destination, request)

Orchestrator Path: Via Adapters
-------------------------------

The orchestrator uses adapters that wrap legacy modules:

.. code-block:: python

   # Inside fusion/core/adapters/routing_adapter.py (simplified)
   from fusion.modules.routing import create_algorithm

   class RoutingAdapter:
       """Wraps legacy routing to satisfy RoutingPipeline protocol."""

       def __init__(self, engine_props, sdn_props):
           # Create the legacy algorithm
           self._legacy_router = create_algorithm(
               engine_props["routing_algorithm"],
               engine_props,
               sdn_props
           )

       def find_routes(self, source, dest, bandwidth, network_state, **kwargs):
           # Call legacy router, convert result to RouteResult
           paths = self._legacy_router.route(source, dest, ...)
           return RouteResult(paths=paths, ...)

**The adapter's job:**

1. Accept orchestrator-style parameters (``NetworkState``, etc.)
2. Call the legacy module with its expected parameters
3. Convert the result to a domain object (``RouteResult``, etc.)

Common Confusion Points
=======================

"Where is First-Fit implemented?"
---------------------------------

There are **two** first-fit implementations:

1. **Legacy**: ``fusion/modules/spectrum/first_fit.py``
   - Class: ``FirstFitSpectrum``
   - Used by: ``SDNController`` (legacy path)
   - Also used by: ``SpectrumAdapter`` (orchestrator path, via wrapping)

2. **There is no separate orchestrator first-fit**
   - The orchestrator uses the legacy implementation via ``SpectrumAdapter``

"What's the difference between policies and modules?"
-----------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Concept
     - ``fusion/modules/rl/policies/``
     - ``fusion/interfaces/control_policy.py``
   * - Contains
     - Actual policy implementations
     - Protocol definition (interface)
   * - Purpose
     - Provides BC, IQL, pointer policies
     - Defines what methods a policy must have
   * - Has Logic?
     - Yes - neural networks, decision logic
     - No - just method signatures

**Think of it this way:**

- ``interfaces/control_policy.py`` says "a policy must have ``select_action()``"
- ``modules/rl/policies/iql_policy.py`` provides an actual ``select_action()`` implementation

"Where do I add a new routing algorithm?"
-----------------------------------------

1. Create ``fusion/modules/routing/my_algorithm.py``
2. Inherit from ``AbstractRoutingAlgorithm`` (in ``fusion/interfaces/router.py``)
3. Register in ``fusion/modules/routing/registry.py``
4. **That's it** - both legacy and orchestrator paths will be able to use it

The adapters automatically pick up registered algorithms.

"Why are there adapters if modules already work?"
-------------------------------------------------

The orchestrator uses **typed protocols** (``RoutingPipeline``, ``SpectrumPipeline``)
that expect:

- Immutable ``NetworkState`` parameter
- Return domain objects (``RouteResult``, ``SpectrumResult``)
- Specific method signatures

Legacy modules expect:

- Mutable ``engine_props`` and ``sdn_props`` dicts
- Return plain dictionaries or tuples
- Different method names

**Adapters bridge this gap** so we can reuse existing, tested algorithms without
rewriting them.

Development Guide
=================

Adding a New Algorithm
----------------------

**For routing:**

.. code-block:: python

   # fusion/modules/routing/my_routing.py
   from fusion.interfaces.router import AbstractRoutingAlgorithm

   class MyRouting(AbstractRoutingAlgorithm):
       def __init__(self, engine_props, sdn_props):
           super().__init__(engine_props, sdn_props)

       def route(self, source, destination, request):
           # Your routing logic here
           return {"path": [...], "weight": ...}

   # Register in registry.py
   RoutingRegistry.register("my_routing", MyRouting)

**For spectrum:**

.. code-block:: python

   # fusion/modules/spectrum/my_spectrum.py
   from fusion.interfaces.spectrum import AbstractSpectrumAssigner

   class MySpectrum(AbstractSpectrumAssigner):
       def assign(self, path, request):
           # Your spectrum logic here
           return {"start_slot": ..., "end_slot": ..., "core": ...}

   # Register in registry.py
   SpectrumRegistry.register("my_spectrum", MySpectrum)

Adding a New RL Policy
----------------------

.. code-block:: python

   # fusion/modules/rl/policies/my_policy.py
   from fusion.modules.rl.policies.base import BasePolicy

   class MyPolicy(BasePolicy):
       def select_action(self, observation, action_mask=None):
           # Your decision logic here
           return action_index

       def update(self, *args, **kwargs):
           # Training update (or pass for heuristics)
           pass

Testing
=======

:Test Location: ``fusion/modules/tests/`` (integration) and ``fusion/modules/*/tests/`` (unit)
:Run Tests: ``pytest fusion/modules/ -v``

.. code-block:: bash

   # Run all module tests
   pytest fusion/modules/ -v

   # Run specific submodule tests
   pytest fusion/modules/tests/routing/ -v
   pytest fusion/modules/tests/spectrum/ -v
   pytest fusion/modules/tests/rl/ -v

Related Documentation
=====================

- :ref:`core-module` - How core uses modules via adapters
- :ref:`interfaces-module` - Protocol definitions that modules implement
- :ref:`domain-module` - Data structures returned by adapters

Quick Reference: Where To Go
============================

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - I want to...
     - Go to...
   * - Add a new routing algorithm
     - ``fusion/modules/routing/`` + register
   * - Modify first-fit spectrum
     - ``fusion/modules/spectrum/first_fit.py``
   * - Add an RL policy
     - ``fusion/modules/rl/policies/``
   * - Create an RL environment
     - ``fusion/modules/rl/gymnasium_envs/``
   * - Change how adapters convert data
     - ``fusion/core/adapters/``
   * - Add a new pipeline stage
     - ``fusion/pipelines/``
   * - Define a new protocol
     - ``fusion/interfaces/``
   * - Add a domain object
     - ``fusion/domain/``
   * - Change failure simulation
     - :ref:`failures-module`

.. toctree::
   :maxdepth: 1
   :hidden:

   failures/index
   ml/index
   rl/index
   routing/index
   snr/index
   spectrum/index
