.. _rl-adapter:

==================
RL Adapter Package
==================

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Bridge between RL environments and the v6.0 orchestrator stack
   :Location: ``fusion/modules/rl/adapter/``
   :Key Files: ``rl_adapter.py``, ``path_option.py``
   :Depends On: ``fusion/core/orchestrator``, ``fusion/interfaces/pipelines``
   :Used By: ``UnifiedSimEnv``, offline RL policies (BC, IQL)

The adapter package provides the coordination layer between RL environments and
the v6.0 simulation stack. It ensures RL agents use the **same pipeline instances**
as the SDNOrchestrator, eliminating duplicated simulation logic.

This is the **recommended** approach for RL integration in FUSION v6.0+. The
adapter works with the new orchestrator path (``use_orchestrator=True``), not the
legacy SDNController path.

Why This Module Exists
----------------------

Before the adapter, RL environments duplicated simulation logic:

- RL code had its own path feasibility checks
- Spectrum allocation logic was reimplemented
- State was managed inconsistently between RL and simulation

The adapter solves this by:

1. **Sharing pipeline references** with the orchestrator (identity, not copy)
2. **Providing read-only queries** for path options without allocating
3. **Routing all writes** through the orchestrator's allocation logic

Key Invariants
--------------

These invariants MUST be maintained:

Pipeline Identity
   ``adapter.routing IS orchestrator.routing`` (same object, not equal)

Stateless Adapter
   The adapter never stores ``NetworkState`` - it receives it per-call

Read-Only Queries
   ``get_path_options()`` queries but never allocates spectrum

Write-Through
   ``apply_action()`` routes through orchestrator for actual allocation

Key Concepts
============

PathOption
   Immutable dataclass representing a candidate path with all metadata needed for
   RL decision-making: path geometry, modulation, slots needed, feasibility from
   real spectrum checks, and congestion metrics.

Action Masking
   Boolean array indicating which actions (paths) are valid. Used by MaskablePPO
   and other action-masked algorithms to prevent invalid actions.

RLConfig
   Configuration for RL-specific settings: k_paths, reward values, observation
   space configuration, and GNN settings.

DisasterState
   Immutable dataclass for survivability scenarios. Captures disaster information
   needed by offline RL policies trained on survivability experiments.

Architecture
============

.. code-block:: text

   fusion/modules/rl/adapter/
   |-- __init__.py          # Public API exports
   |-- path_option.py       # PathOption dataclass, ActionMask, helpers
   |-- rl_adapter.py        # RLSimulationAdapter, RLConfig, DisasterState
   `-- tests/               # Unit tests
       |-- test_path_option.py
       `-- test_rl_adapter.py

Data Flow
---------

The adapter sits between RL environments and the orchestrator:

.. code-block:: text

   +------------------+     +------------------+     +------------------+
   | UnifiedSimEnv    |---->| RLSimulation     |---->| SDNOrchestrator  |
   | (Gymnasium env)  |     | Adapter          |     |                  |
   +------------------+     +------------------+     +------------------+
          |                        |                        |
          | 1. Request arrives     |                        |
          |----------------------->|                        |
          |                        |                        |
          | 2. get_path_options()  | 3. Query pipelines     |
          |----------------------->|----------------------->|
          |                        |   (read-only)          |
          |                        |<-----------------------|
          |                        |                        |
          | 4. Return PathOptions  |                        |
          |<-----------------------|                        |
          |                        |                        |
          | 5. RL agent selects    |                        |
          |    action              |                        |
          |                        |                        |
          | 6. apply_action()      | 7. handle_arrival()   |
          |----------------------->|----------------------->|
          |                        |   with forced_path     |
          |                        |<-----------------------|
          | 8. Return result       |                        |
          |<-----------------------|                        |

**Step-by-step flow:**

1. **Input**: Request arrives at UnifiedSimEnv
2. **Query**: Env calls ``adapter.get_path_options(request, network_state)``
3. **Pipeline queries**: Adapter queries routing and spectrum pipelines (read-only)
4. **PathOptions**: Adapter returns list of PathOption with feasibility info
5. **Action selection**: RL agent selects action using observation and action mask
6. **Apply action**: Env calls ``adapter.apply_action(action, request, state, options)``
7. **Orchestrator**: Adapter routes through ``orchestrator.handle_arrival()`` with forced path
8. **Result**: AllocationResult returned indicating success/failure

Components
==========

path_option.py
--------------

:Purpose: Defines PathOption dataclass and action mask utilities
:Key Classes: ``PathOption``
:Key Functions: ``compute_action_mask()``
:Type Aliases: ``PathOptionList``, ``ActionMask``

The ``PathOption`` dataclass is frozen (immutable) and hashable, making it safe
for use in RL pipelines where immutability is important.

**Core fields:**

.. code-block:: python

   @dataclass(frozen=True)
   class PathOption:
       path_index: int              # Index in k-paths list (0 to k-1)
       path: tuple[str, ...]        # Node sequence (immutable)
       weight_km: float             # Path length in kilometers
       num_hops: int                # Number of links
       modulation: str | None       # e.g., "QPSK", "16-QAM"
       slots_needed: int            # Contiguous spectrum slots required
       is_feasible: bool            # From real spectrum check
       congestion: float            # [0, 1], higher = more congested
       available_slots: float       # Ratio on most constrained link

**Optional fields for protection and disasters:**

- ``backup_path``, ``backup_feasible``: For 1+1 protected paths
- ``frag_indicator``, ``failure_mask``, ``dist_to_disaster``: For offline RL

**Factory methods:**

.. code-block:: python

   # From pipeline results
   option = PathOption.from_pipeline_results(
       path_index=0,
       route_result=route_result,
       spectrum_result=spectrum_result,
       congestion=0.3,
       available_slots=0.7,
   )

   # For protected routes (1+1)
   option = PathOption.from_protected_route(
       path_index=0,
       primary_path=["A", "B", "C"],
       backup_path=["A", "D", "E", "C"],
       primary_weight=100.0,
       backup_weight=150.0,
       primary_feasible=True,
       backup_feasible=True,
       primary_modulation="QPSK",
       backup_modulation="8-QAM",
       slots_needed=4,
       congestion=0.3,
   )

**Action masking:**

.. code-block:: python

   from fusion.modules.rl.adapter import compute_action_mask

   # Generate mask from path options
   options = adapter.get_path_options(request, network_state)
   mask = compute_action_mask(options, k_paths=3)
   # mask: [True, False, True] = paths 0 and 2 are feasible

rl_adapter.py
-------------

:Purpose: Main adapter between RL environments and orchestrator
:Key Classes: ``RLSimulationAdapter``, ``RLConfig``, ``DisasterState``, ``OfflinePolicyAdapter``
:Key Functions: ``create_disaster_state_from_engine()``

**RLSimulationAdapter** is the core class:

.. code-block:: python

   from fusion.modules.rl.adapter import RLSimulationAdapter, RLConfig

   # Create adapter (shares pipelines with orchestrator)
   config = RLConfig(k_paths=3, rl_success_reward=1.0)
   adapter = RLSimulationAdapter(orchestrator, config)

   # Verify pipeline identity
   assert adapter.routing is orchestrator.routing
   assert adapter.spectrum is orchestrator.spectrum

   # Query path options (read-only)
   options = adapter.get_path_options(request, network_state)

   # Get action mask for RL
   mask = adapter.get_action_mask(options)

   # Apply selected action (writes through orchestrator)
   result = adapter.apply_action(action, request, network_state, options)

   # Compute reward
   reward = adapter.compute_reward(result, request)

**RLConfig** holds RL-specific configuration:

.. code-block:: python

   @dataclass
   class RLConfig:
       k_paths: int = 3                    # Candidate paths to consider
       rl_success_reward: float = 1.0      # Reward for success
       rl_block_penalty: float = -1.0      # Penalty for blocking
       rl_grooming_bonus: float = 0.1      # Bonus for grooming
       rl_slicing_penalty: float = -0.05   # Penalty for slicing
       rl_bandwidth_weighted: bool = False # Weight by bandwidth
       max_holding_time: float = 100.0     # For normalization
       num_nodes: int = 14                 # Network size
       total_slots: int = 320              # Slots per link
       use_gnn_obs: bool = False           # Include GNN features
       obs_space: str = "obs_8"            # Observation configuration

**DisasterState** for survivability scenarios:

.. code-block:: python

   @dataclass(frozen=True)
   class DisasterState:
       active: bool
       centroid: tuple[float, float] | None = None
       radius: float = 0.0
       failed_links: frozenset[tuple[str, str]] = frozenset()
       network_diameter: float = 1.0

   # Create from legacy engine_props
   disaster_state = create_disaster_state_from_engine(engine_props)

**OfflinePolicyAdapter** bridges offline RL policies with UnifiedSimEnv:

.. code-block:: python

   from fusion.modules.rl.adapter import OfflinePolicyAdapter

   # Load offline policy (BC or IQL)
   bc_policy = BCPolicy("models/bc_model.pt", device="cpu")

   # Create offline adapter
   offline_adapter = OfflinePolicyAdapter(
       policy=bc_policy,
       rl_adapter=env.adapter,
   )

   # Use for action selection
   action = offline_adapter.select_action(
       request=request,
       options=options,
       network_state=network_state,
       action_mask=mask,
       disaster_state=disaster_state,
   )

Integration with RL Module
==========================

The adapter integrates with other RL module components:

.. code-block:: text

   +------------------------------------------------------------------+
   |                    RL MODULE ARCHITECTURE                         |
   +------------------------------------------------------------------+
   |                                                                   |
   |   +---------------------------+                                   |
   |   | UnifiedSimEnv             |                                   |
   |   | (environments/)           |                                   |
   |   +---------------------------+                                   |
   |              |                                                    |
   |              | uses                                               |
   |              v                                                    |
   |   +---------------------------+     +---------------------------+ |
   |   | RLSimulationAdapter       |---->| SDNOrchestrator           | |
   |   | (adapter/)                |     | (core/orchestrator.py)    | |
   |   +---------------------------+     +---------------------------+ |
   |              |                               |                    |
   |              | provides                      | uses               |
   |              v                               v                    |
   |   +---------------------------+     +---------------------------+ |
   |   | PathOption                |     | RoutingPipeline           | |
   |   | ActionMask                |     | SpectrumPipeline          | |
   |   +---------------------------+     +---------------------------+ |
   |                                                                   |
   +------------------------------------------------------------------+

**Used by UnifiedSimEnv:**

The ``UnifiedSimEnv`` gymnasium environment holds an ``RLSimulationAdapter``
instance and uses it for all simulation interactions:

.. code-block:: python

   class UnifiedSimEnv(gymnasium.Env):
       def __init__(self, config):
           self.orchestrator = SDNOrchestrator(config, pipelines)
           self.adapter = RLSimulationAdapter(self.orchestrator)

       def step(self, action):
           options = self.adapter.get_path_options(request, state)
           result = self.adapter.apply_action(action, request, state, options)
           reward = self.adapter.compute_reward(result)
           return obs, reward, done, truncated, info

**Used by Offline RL Policies:**

Offline RL policies (BC, IQL) use the ``OfflinePolicyAdapter`` to convert
gymnasium observations to the state format they were trained on:

.. code-block:: python

   # BC/IQL policies expect different state format than gymnasium obs
   offline_state = adapter.build_offline_state(
       request=request,
       options=options,
       network_state=network_state,
       disaster_state=disaster_state,
   )

Legacy vs Orchestrator Path
===========================

This adapter is designed for the **orchestrator path only**:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Aspect
     - Legacy Path
     - Orchestrator Path (Adapter)
   * - Controller
     - SDNController
     - SDNOrchestrator
   * - RL Environment
     - GeneralSimEnv (deprecated)
     - UnifiedSimEnv
   * - Pipeline Access
     - Direct, often duplicated
     - Shared via adapter
   * - State Management
     - engine_props dict
     - NetworkState domain object
   * - Recommended
     - No (deprecated)
     - **Yes**

If you're working with the legacy path (``use_orchestrator=False``), the adapter
is not applicable. Use ``GeneralSimEnv`` instead (deprecated).

Development Guide
=================

Getting Started
---------------

1. Read the `Key Concepts`_ section to understand PathOption and action masking
2. Examine ``rl_adapter.py`` to understand the adapter's role
3. Look at ``UnifiedSimEnv`` to see how the adapter is used in practice

Common Tasks
------------

**Adding a new observation feature**

1. Add the feature computation to ``RLSimulationAdapter.build_observation()``
2. Update ``RLConfig`` if new configuration is needed
3. Add tests in ``test_rl_adapter.py``

**Supporting a new offline policy format**

1. Add conversion logic to ``RLSimulationAdapter.build_offline_state()``
2. Update ``OfflinePolicyAdapter.select_action()`` if needed
3. Add tests

**Adding a new PathOption field**

1. Add the field to the ``PathOption`` dataclass
2. Update validation in ``__post_init__()`` if needed
3. Update factory methods (``from_pipeline_results``, etc.)
4. Add tests in ``test_path_option.py``

Testing
=======

:Test Location: ``fusion/modules/rl/adapter/tests/``
:Run Tests: ``pytest fusion/modules/rl/adapter/tests/ -v``

**Unit Tests:**

- ``test_path_option.py``: PathOption creation, validation, immutability, action masking
- ``test_rl_adapter.py``: Adapter initialization, pipeline identity, path options, apply action

**Running tests:**

.. code-block:: bash

   # Run all adapter tests
   pytest fusion/modules/rl/adapter/tests/ -v

   # Run with coverage
   pytest fusion/modules/rl/adapter/tests/ -v --cov=fusion.modules.rl.adapter

**Test patterns used:**

.. code-block:: python

   def test_component_when_condition_then_expected():
       """PathOption should be immutable."""
       # Arrange
       opt = create_valid_path_option()

       # Act & Assert
       with pytest.raises(FrozenInstanceError):
           opt.is_feasible = False

Related Documentation
=====================

- :ref:`rl-module` - Parent RL module documentation
- :ref:`core-module` - SDNOrchestrator and simulation engine
- :ref:`interfaces-module` - Pipeline interfaces (RoutingPipeline, SpectrumPipeline)

.. seealso::

   - `Gymnasium Documentation <https://gymnasium.farama.org/>`_ - Environment interface
   - `Stable-Baselines3 Action Masking <https://sb3-contrib.readthedocs.io/en/master/modules/maskable_ppo.html>`_
