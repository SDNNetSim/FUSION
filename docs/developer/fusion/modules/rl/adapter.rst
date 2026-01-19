.. _rl-adapter:

==================
RL Adapter Package
==================

.. admonition:: At a Glance
   :class: tip

   :Purpose: Bridge between RL environments and the v6.0 orchestrator stack
   :Location: ``fusion/modules/rl/adapter/``
   :Key Files: ``rl_adapter.py``, ``path_option.py``
   :Prerequisites: Basic understanding of RL concepts (actions, observations, rewards)

This guide teaches you how the adapter works and how to extend it. If you're new
to FUSION's RL module, start here before diving into environments or algorithms.

What Problem Does This Solve?
=============================

When an RL agent needs to make a routing decision, it needs to:

1. **See** what paths are available (observation)
2. **Know** which paths are actually usable (action masking)
3. **Act** by selecting a path (action)
4. **Learn** from the outcome (reward)

The adapter handles all of this by sitting between your RL environment and the
simulation's orchestrator. It translates between "RL language" (observations,
actions, rewards) and "simulation language" (requests, paths, allocations).

.. code-block:: text

   Without adapter:                    With adapter:
   +--------------+                    +--------------+
   | RL Agent     |                    | RL Agent     |
   +--------------+                    +--------------+
         |                                   |
         | (duplicated logic,                | (clean interface)
         |  inconsistent state)              v
         v                             +--------------+
   +--------------+                    | Adapter      |  <-- You are here
   | Simulation   |                    +--------------+
   +--------------+                          |
                                             v
                                       +--------------+
                                       | Orchestrator |
                                       +--------------+

Quick Start Tutorial
====================

This tutorial walks through the adapter's core workflow. By the end, you'll
understand how a single RL step flows through the system.

Step 1: Create the Adapter
--------------------------

The adapter wraps an orchestrator. They share the same pipeline instances:

.. code-block:: python

   from fusion.modules.rl.adapter import RLSimulationAdapter, RLConfig

   # Assume you have an orchestrator from your environment setup
   # orchestrator = SDNOrchestrator(config, pipelines)

   # Create adapter with custom configuration
   rl_config = RLConfig(
       k_paths=3,              # Consider top 3 shortest paths
       rl_success_reward=1.0,  # Reward when allocation succeeds
       rl_block_penalty=-1.0,  # Penalty when request is blocked
   )
   adapter = RLSimulationAdapter(orchestrator, rl_config)

   # Critical invariant: same pipeline objects, not copies
   assert adapter.routing is orchestrator.routing  # Identity check

Step 2: Get Path Options (Observation Data)
-------------------------------------------

When a request arrives, ask the adapter what paths are available:

.. code-block:: python

   # request: the current network request (source, dest, bandwidth, etc.)
   # network_state: current state of the network (spectrum usage, etc.)

   options = adapter.get_path_options(request, network_state)

   # options is a list of PathOption objects
   for opt in options:
       print(f"Path {opt.path_index}: {opt.path}")
       print(f"  Length: {opt.weight_km} km, Hops: {opt.num_hops}")
       print(f"  Feasible: {opt.is_feasible}, Slots needed: {opt.slots_needed}")

**What happens inside:**

1. Adapter calls ``routing.find_routes()`` to get candidate paths
2. For each path, calls ``spectrum.find_spectrum()`` to check if slots are free
3. Packages everything into immutable ``PathOption`` objects

**Key point:** This is READ-ONLY. No spectrum is allocated yet.

Step 3: Build the Action Mask
-----------------------------

RL agents shouldn't select infeasible paths. The action mask tells them which
actions are valid:

.. code-block:: python

   from fusion.modules.rl.adapter import compute_action_mask

   # Method 1: Use the adapter's convenience method
   mask = adapter.get_action_mask(options)

   # Method 2: Use the standalone function
   mask = compute_action_mask(options, k_paths=3)

   # mask is a numpy boolean array: [True, False, True]
   # meaning: path 0 is feasible, path 1 is blocked, path 2 is feasible

   # Your RL agent uses this to avoid invalid actions
   # MaskablePPO example:
   # action, _ = model.predict(observation, action_masks=mask)

Step 4: Apply the Selected Action
---------------------------------

Once your agent selects a path index, apply it through the adapter:

.. code-block:: python

   action = 0  # Agent selected path 0

   result = adapter.apply_action(
       action=action,
       request=request,
       network_state=network_state,
       options=options,
   )

   # result is an AllocationResult
   if result.success:
       print("Request allocated successfully!")
   else:
       print(f"Request blocked: {result.block_reason}")

**What happens inside:**

1. Adapter finds the PathOption matching the action index
2. Calls ``orchestrator.handle_arrival()`` with the path as ``forced_path``
3. Orchestrator runs full allocation logic (SNR checks, grooming, slicing)
4. Returns the result

Step 5: Compute the Reward
--------------------------

Convert the allocation result to an RL reward signal:

.. code-block:: python

   reward = adapter.compute_reward(result, request)
   # reward = 1.0 if success, -1.0 if blocked (based on RLConfig)

**That's it!** The complete RL step is:

.. code-block:: python

   # Inside your environment's step() method:
   options = adapter.get_path_options(request, network_state)
   mask = adapter.get_action_mask(options)
   # ... agent selects action using mask ...
   result = adapter.apply_action(action, request, network_state, options)
   reward = adapter.compute_reward(result, request)

Understanding PathOption
========================

``PathOption`` is the core data structure. It's an immutable snapshot of
everything known about a candidate path.

Anatomy of a PathOption
-----------------------

.. code-block:: python

   from fusion.modules.rl.adapter import PathOption

   # PathOption is a frozen dataclass - you can't modify it after creation
   opt = PathOption(
       # Required fields
       path_index=0,                    # Which path (0 to k-1)
       path=("A", "B", "C"),            # Node sequence (tuple, not list!)
       weight_km=150.5,                 # Physical length
       num_hops=2,                      # Number of links
       modulation="QPSK",               # Selected modulation format
       slots_needed=4,                  # Contiguous slots required
       is_feasible=True,                # Can this path be allocated?
       congestion=0.3,                  # [0,1] how congested (0=empty)
       available_slots=0.7,             # [0,1] availability ratio

       # Optional fields (default to None or sensible values)
       spectrum_start=100,              # First slot if feasible
       spectrum_end=104,                # Last slot if feasible
       core_index=0,                    # For multi-core fibers
       band="C",                        # Spectrum band
   )

   # Immutable - this raises FrozenInstanceError:
   # opt.is_feasible = False  # Error!

   # Hashable - can use as dict key:
   cache = {opt: "some_value"}

Why Immutable?
--------------

1. **Thread safety**: Can pass between threads without locks
2. **Cache keys**: Can use PathOptions as dictionary keys
3. **No surprises**: Once created, it never changes
4. **RL pipelines**: Many RL frameworks expect immutable observations

Creating PathOptions (Factory Methods)
--------------------------------------

You rarely create PathOptions directly. Use factory methods instead:

.. code-block:: python

   # From pipeline results (used internally by adapter)
   opt = PathOption.from_pipeline_results(
       path_index=0,
       route_result=route_result,    # From routing pipeline
       spectrum_result=spectrum_result,  # From spectrum pipeline
       congestion=0.3,
       available_slots=0.7,
   )

   # For simple unprotected routes
   opt = PathOption.from_unprotected_route(
       path_index=0,
       path=["A", "B", "C"],  # List OK here, converted to tuple
       weight_km=150.0,
       is_feasible=True,
       modulation="QPSK",
       slots_needed=4,
       congestion=0.3,
   )

   # For 1+1 protected routes (primary + backup)
   opt = PathOption.from_protected_route(
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

   # Protected paths have extra properties:
   print(opt.is_protected)        # True
   print(opt.both_paths_feasible) # True only if BOTH paths have spectrum
   print(opt.total_weight_km)     # 250.0 (primary + backup)
   print(opt.backup_hop_count)    # 3

Extending the Adapter
=====================

This section shows how to add new functionality step by step.

Tutorial: Adding a New PathOption Field
---------------------------------------

Let's add a ``latency_ms`` field to PathOption.

**Step 1: Add the field to the dataclass**

Edit ``path_option.py``:

.. code-block:: python

   @dataclass(frozen=True)
   class PathOption:
       # ... existing fields ...

       # Add new optional field with default
       latency_ms: float | None = None

**Step 2: Add validation if needed**

In ``__post_init__``:

.. code-block:: python

   def __post_init__(self) -> None:
       # ... existing validation ...

       if self.latency_ms is not None and self.latency_ms < 0:
           raise ValueError("latency_ms must be non-negative")

**Step 3: Update factory methods**

In ``from_pipeline_results``:

.. code-block:: python

   @classmethod
   def from_pipeline_results(cls, ..., latency_ms: float | None = None) -> PathOption:
       # ... existing code ...
       return cls(
           # ... existing fields ...
           latency_ms=latency_ms,
       )

**Step 4: Populate it in the adapter**

In ``rl_adapter.py``, in ``get_path_options()``:

.. code-block:: python

   # After getting route_result
   latency = self._compute_latency(path, network_state)

   option = PathOption(
       # ... existing fields ...
       latency_ms=latency,
   )

**Step 5: Add tests**

In ``test_path_option.py``:

.. code-block:: python

   def test_latency_field_stored_correctly(self) -> None:
       opt = create_valid_path_option(latency_ms=5.2)
       assert opt.latency_ms == 5.2

   def test_negative_latency_raises(self) -> None:
       with pytest.raises(ValueError, match="latency_ms must be non-negative"):
           create_valid_path_option(latency_ms=-1.0)

Tutorial: Adding a New Observation Feature
------------------------------------------

Let's add network-wide utilization to the observation.

**Step 1: Add to RLConfig if configurable**

In ``rl_adapter.py``:

.. code-block:: python

   @dataclass
   class RLConfig:
       # ... existing fields ...
       include_network_util: bool = True  # New config option

**Step 2: Implement the computation**

Add a private method:

.. code-block:: python

   def _compute_network_utilization(
       self,
       network_state: NetworkState,
   ) -> float:
       """Compute average utilization across all links."""
       if not hasattr(network_state, "get_all_link_utilizations"):
           return 0.0

       utils = network_state.get_all_link_utilizations()
       if not utils:
           return 0.0
       return sum(utils) / len(utils)

**Step 3: Add to build_observation()**

.. code-block:: python

   def build_observation(self, request, options, network_state) -> dict[str, Any]:
       obs = {
           # ... existing observation fields ...
       }

       if self._config.include_network_util:
           obs["network_utilization"] = np.array(
               [self._compute_network_utilization(network_state)],
               dtype=np.float32,
           )

       return obs

**Step 4: Update observation space in UnifiedSimEnv**

The environment needs to know about the new observation dimension.

**Step 5: Add tests**

.. code-block:: python

   def test_observation_includes_network_util_when_enabled(self) -> None:
       config = RLConfig(include_network_util=True)
       adapter = RLSimulationAdapter(mock_orchestrator, config)

       obs = adapter.build_observation(request, options, network_state)

       assert "network_utilization" in obs
       assert obs["network_utilization"].shape == (1,)

Tutorial: Adding a New Reward Component
---------------------------------------

Let's add a fragmentation penalty to the reward.

**Step 1: Add config options**

.. code-block:: python

   @dataclass
   class RLConfig:
       # ... existing fields ...
       rl_fragmentation_penalty: float = -0.1
       use_fragmentation_penalty: bool = False

**Step 2: Modify compute_reward()**

.. code-block:: python

   def compute_reward(
       self,
       result: AllocationResult,
       request: Request | None = None,
       network_state: NetworkState | None = None,  # Add parameter
   ) -> float:
       if not result.success:
           return self._config.rl_block_penalty

       reward = self._config.rl_success_reward

       # New: fragmentation penalty
       if self._config.use_fragmentation_penalty and network_state is not None:
           frag = self._compute_fragmentation_increase(result, network_state)
           reward += frag * self._config.rl_fragmentation_penalty

       return reward

**Step 3: Add helper method**

.. code-block:: python

   def _compute_fragmentation_increase(
       self,
       result: AllocationResult,
       network_state: NetworkState,
   ) -> float:
       """Compute how much fragmentation increased due to this allocation."""
       # Implementation depends on your fragmentation metric
       # Return value in [0, 1] where 1 = maximum fragmentation increase
       pass

Configuration Reference
=======================

RLConfig Fields
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``k_paths``
     - ``3``
     - Number of candidate paths to consider
   * - ``rl_success_reward``
     - ``1.0``
     - Reward when allocation succeeds
   * - ``rl_block_penalty``
     - ``-1.0``
     - Penalty when request is blocked
   * - ``rl_grooming_bonus``
     - ``0.1``
     - Bonus when request is groomed onto existing lightpath
   * - ``rl_slicing_penalty``
     - ``-0.05``
     - Penalty when request requires spectrum slicing
   * - ``max_holding_time``
     - ``100.0``
     - Maximum holding time for normalization
   * - ``num_nodes``
     - ``14``
     - Number of nodes in network (for observation space)
   * - ``total_slots``
     - ``320``
     - Total spectrum slots per link
   * - ``use_gnn_obs``
     - ``False``
     - Include GNN features in observation
   * - ``obs_space``
     - ``"obs_8"``
     - Observation space configuration (obs_1 through obs_8)

Testing Your Changes
====================

The adapter has comprehensive tests. Always run them after changes.

Running Tests
-------------

.. code-block:: bash

   # Run all adapter tests
   pytest fusion/modules/rl/adapter/tests/ -v

   # Run specific test file
   pytest fusion/modules/rl/adapter/tests/test_path_option.py -v

   # Run with coverage
   pytest fusion/modules/rl/adapter/tests/ -v --cov=fusion.modules.rl.adapter

   # Run a specific test
   pytest fusion/modules/rl/adapter/tests/test_path_option.py::TestPathOptionCreation -v

Test Structure
--------------

Tests follow the AAA pattern (Arrange, Act, Assert):

.. code-block:: python

   def test_path_option_validates_congestion_range(self) -> None:
       """Congestion outside [0,1] should raise ValueError."""
       # Arrange - set up test data
       kwargs = {"congestion": 1.5}  # Invalid: > 1.0

       # Act & Assert - verify behavior
       with pytest.raises(ValueError, match="congestion must be in"):
           create_valid_path_option(**kwargs)

Test naming convention: ``test_<what>_<when>_<expected>``

Writing Good Tests
------------------

**Do test:**

- Happy path (normal operation)
- Edge cases (empty lists, zero values, boundaries)
- Error conditions (invalid inputs)
- Invariants (immutability, pipeline identity)

**Don't test:**

- Implementation details that might change
- External libraries (they have their own tests)

Example test for a new field:

.. code-block:: python

   class TestLatencyField:
       """Tests for the new latency_ms field."""

       def test_latency_stored_correctly(self) -> None:
           opt = create_valid_path_option(latency_ms=5.2)
           assert opt.latency_ms == 5.2

       def test_latency_defaults_to_none(self) -> None:
           opt = create_valid_path_option()
           assert opt.latency_ms is None

       def test_negative_latency_raises_value_error(self) -> None:
           with pytest.raises(ValueError):
               create_valid_path_option(latency_ms=-1.0)

       def test_zero_latency_is_valid(self) -> None:
           opt = create_valid_path_option(latency_ms=0.0)
           assert opt.latency_ms == 0.0

Troubleshooting
===============

Common Issues
-------------

**"orchestrator cannot be None"**

.. code-block:: python

   # Wrong:
   adapter = RLSimulationAdapter(None)

   # Right:
   adapter = RLSimulationAdapter(orchestrator)

The adapter requires a valid orchestrator instance.

**Action index out of range**

.. code-block:: python

   # If your agent returns action=5 but k_paths=3
   result = adapter.apply_action(5, request, state, options)
   # result.success will be False, result.block_reason = NO_PATH

The adapter handles this gracefully but your agent should use action masking.

**PathOption is immutable**

.. code-block:: python

   opt = create_valid_path_option()
   opt.is_feasible = False  # FrozenInstanceError!

   # Instead, create a new PathOption with different values
   # (though you rarely need to do this manually)

**Pipeline identity broken**

.. code-block:: python

   # This should ALWAYS be true:
   assert adapter.routing is orchestrator.routing

   # If it's not, something is wrong with initialization

File Reference
==============

.. code-block:: text

   fusion/modules/rl/adapter/
   |-- __init__.py          # Public exports (what you import)
   |-- path_option.py       # PathOption, compute_action_mask
   |-- rl_adapter.py        # RLSimulationAdapter, RLConfig, DisasterState
   `-- tests/
       |-- __init__.py
       |-- test_path_option.py   # PathOption tests
       `-- test_rl_adapter.py    # Adapter tests

**What to import:**

.. code-block:: python

   # Most common imports
   from fusion.modules.rl.adapter import (
       RLSimulationAdapter,
       RLConfig,
       PathOption,
       compute_action_mask,
   )

   # For survivability scenarios
   from fusion.modules.rl.adapter import (
       DisasterState,
       create_disaster_state_from_engine,
   )

   # For offline RL (BC, IQL)
   from fusion.modules.rl.adapter import OfflinePolicyAdapter

Related Documentation
=====================

- :ref:`rl-module` - Parent RL module (environments, algorithms, training)
- `Gymnasium Documentation <https://gymnasium.farama.org/>`_ - Environment interface
- `MaskablePPO <https://sb3-contrib.readthedocs.io/en/master/modules/maskable_ppo.html>`_ - Action masking in SB3
