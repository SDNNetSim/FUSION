.. _rl-environments:

======================
RL Environments Module
======================

.. admonition:: At a Glance
   :class: tip

   :Purpose: Gymnasium-compatible RL environment for optical network path selection
   :Location: ``fusion/modules/rl/environments/``
   :Key Classes: ``UnifiedSimEnv``, ``ActionMaskWrapper``, ``PathEncoder``
   :Prerequisites: Gymnasium, numpy; optionally SB3-contrib for MaskablePPO

This module provides ``UnifiedSimEnv``, a Gymnasium-compatible environment
that enables reinforcement learning agents to learn optimal path selection
for optical network Routing and Spectrum Assignment (RSA).

Architecture Overview
=====================

.. code-block:: text

   +------------------------------------------------------------------+
   |                    ENVIRONMENTS MODULE                           |
   +------------------------------------------------------------------+
   |                                                                  |
   |  UnifiedSimEnv                ActionMaskWrapper    PathEncoder   |
   |  --------------                ----------------    -----------   |
   |  - Gymnasium env               - SB3 adapter       - Path to     |
   |  - Configurable obs            - action_masks()      edge mask   |
   |  - Action masking              - MaskablePPO       - GNN format  |
   |  - Dual mode support             compatible                      |
   |                                                                  |
   |  Operating Modes:                                                |
   |  ----------------                                                |
   |  STANDALONE (testing)    <-->    WIRED (production)              |
   |  - Synthetic requests            - Real SimulationEngine         |
   |  - Random feasibility            - Actual spectrum checks        |
   |  - No dependencies               - Full V4 stack                 |
   |                                                                  |
   +------------------------------------------------------------------+

Key Invariants
--------------

1. **Same Pipelines**: Uses identical pipelines as non-RL simulation (no forked code)
2. **Action Masking**: Invalid actions indicated in ``info["action_mask"]``
3. **Reproducibility**: Deterministic episodes via seeding
4. **Dual Mode**: Works standalone for testing or wired to full simulation

Quick Start: Standalone Mode
============================

The simplest way to use UnifiedSimEnv requires no external dependencies:

.. code-block:: python

   from fusion.modules.rl.environments import UnifiedSimEnv

   # Create environment (standalone mode - no simulation stack needed)
   env = UnifiedSimEnv(num_requests=100)

   # Standard Gymnasium loop
   obs, info = env.reset(seed=42)
   total_reward = 0.0

   while True:
       # Get valid actions from mask
       action_mask = info["action_mask"]
       valid_actions = [i for i, valid in enumerate(action_mask) if valid]

       # Select random valid action
       action = env.np_random.choice(valid_actions)

       # Take step
       obs, reward, terminated, truncated, info = env.step(action)
       total_reward += reward

       if terminated or truncated:
           break

   print(f"Episode complete. Total reward: {total_reward}")

Understanding the Observation Space
===================================

UnifiedSimEnv uses a Dict observation space with configurable features.
The default (obs_8) includes all available features:

.. code-block:: python

   from fusion.modules.rl.environments import UnifiedSimEnv

   env = UnifiedSimEnv()

   # Inspect observation space
   for name, space in env.observation_space.spaces.items():
       print(f"{name}: shape={space.shape}, dtype={space.dtype}")

   # Output:
   # source: shape=(14,), dtype=float32
   # destination: shape=(14,), dtype=float32
   # request_bandwidth: shape=(4,), dtype=float32
   # holding_time: shape=(1,), dtype=float32
   # slots_needed: shape=(3,), dtype=float32
   # path_lengths: shape=(3,), dtype=float32
   # congestion: shape=(3,), dtype=float32
   # available_slots: shape=(3,), dtype=float32
   # is_feasible: shape=(3,), dtype=float32

Feature Descriptions
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Feature
     - Shape
     - Description
   * - ``source``
     - (num_nodes,)
     - One-hot encoded source node
   * - ``destination``
     - (num_nodes,)
     - One-hot encoded destination node
   * - ``request_bandwidth``
     - (num_bw_classes,)
     - One-hot bandwidth class (10/40/100/400 Gbps)
   * - ``holding_time``
     - (1,)
     - Normalized request duration [0, 1]
   * - ``slots_needed``
     - (k_paths,)
     - Spectrum slots required per path
   * - ``path_lengths``
     - (k_paths,)
     - Hop count for each candidate path
   * - ``congestion``
     - (k_paths,)
     - Network utilization per path [0, 1]
   * - ``available_slots``
     - (k_paths,)
     - Free spectrum ratio per path [0, 1]
   * - ``is_feasible``
     - (k_paths,)
     - Binary: can this path accommodate the request?

Observation Space Configurations
--------------------------------

Choose the observation level based on your needs:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Space
     - Features
     - Use Case
   * - ``obs_1``
     - source, destination
     - Minimal routing
   * - ``obs_2``
     - + bandwidth
     - Bandwidth-aware
   * - ``obs_3``
     - + holding_time
     - Time-sensitive
   * - ``obs_4``
     - source, dest, bw, time
     - Standard (good start)
   * - ``obs_5``
     - + slots_needed, path_lengths
     - Resource-aware
   * - ``obs_6``
     - + congestion
     - Congestion-aware
   * - ``obs_7``
     - + available_slots
     - Availability-aware
   * - ``obs_8``
     - + is_feasible
     - Complete (default)

.. code-block:: python

   from fusion.modules.rl.adapter import RLConfig
   from fusion.modules.rl.environments import UnifiedSimEnv

   # Use minimal observation space
   config = RLConfig(obs_space="obs_1")
   env = UnifiedSimEnv(config=config)

   # Only source and destination in observation
   obs, info = env.reset(seed=42)
   print(obs.keys())  # dict_keys(['source', 'destination'])

Action Space and Masking
========================

The action space is ``Discrete(k_paths)`` - the agent selects which
candidate path to use for the current request.

Why Action Masking?
-------------------

Not all paths are always feasible. A path may be infeasible if:

- Insufficient contiguous spectrum slots
- Path doesn't exist for this source-destination pair
- Physical constraints (e.g., SNR threshold)

Selecting an infeasible path results in a blocked request and penalty.
Action masking prevents the agent from learning to select invalid actions.

Using Action Masks
------------------

.. code-block:: python

   from fusion.modules.rl.environments import UnifiedSimEnv

   env = UnifiedSimEnv(num_requests=10)
   obs, info = env.reset(seed=42)

   # Action mask is a boolean array
   action_mask = info["action_mask"]
   print(f"Action mask: {action_mask}")  # e.g., [True, True, False]

   # Only select from valid actions
   valid_actions = [i for i, valid in enumerate(action_mask) if valid]

   if valid_actions:
       action = valid_actions[0]  # Select first valid
   else:
       action = 0  # Fallback (will be penalized)

   obs, reward, terminated, truncated, info = env.step(action)

Training with MaskablePPO
=========================

For production training, use SB3-contrib's MaskablePPO with the
ActionMaskWrapper:

.. code-block:: python

   from sb3_contrib import MaskablePPO
   from fusion.modules.rl.environments import UnifiedSimEnv, ActionMaskWrapper

   # Create and wrap environment
   env = UnifiedSimEnv(num_requests=1000)
   wrapped = ActionMaskWrapper(env)

   # Create MaskablePPO model
   model = MaskablePPO(
       "MultiInputPolicy",  # Required for Dict observation space
       wrapped,
       verbose=1,
       n_steps=2048,
       batch_size=64,
       learning_rate=3e-4,
   )

   # Train
   model.learn(total_timesteps=100_000)

   # Evaluate
   obs, info = wrapped.reset(seed=99)
   total_reward = 0.0

   while True:
       action, _ = model.predict(obs, action_masks=wrapped.action_masks())
       obs, reward, terminated, truncated, info = wrapped.step(int(action))
       total_reward += reward

       if terminated or truncated:
           break

   print(f"Evaluation reward: {total_reward}")

ActionMaskWrapper Reference
---------------------------

The wrapper adapts UnifiedSimEnv for SB3's MaskablePPO:

.. code-block:: python

   from fusion.modules.rl.environments import ActionMaskWrapper

   wrapped = ActionMaskWrapper(env)

   # Key method: returns current action mask
   mask = wrapped.action_masks()  # np.ndarray of bools

   # Used by MaskablePPO during predict()
   action, _ = model.predict(obs, action_masks=wrapped.action_masks())

GNN Observations
================

For graph neural network policies, enable GNN observation mode:

.. code-block:: python

   from fusion.modules.rl.adapter import RLConfig
   from fusion.modules.rl.environments import UnifiedSimEnv

   config = RLConfig(
       use_gnn_obs=True,
       num_nodes=14,
       k_paths=5,
   )
   env = UnifiedSimEnv(config=config)

   obs, info = env.reset(seed=42)

   # Additional GNN features available
   print(obs.keys())
   # Includes: edge_index, edge_attr, path_masks, node_features, adjacency

GNN Feature Reference
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Feature
     - Shape
     - Description
   * - ``edge_index``
     - (2, num_edges)
     - PyG format: [source_indices, target_indices]
   * - ``edge_attr``
     - (num_edges, 2)
     - Edge properties: [utilization, normalized_length]
   * - ``path_masks``
     - (k_paths, num_edges)
     - Binary mask: which edges used by each path
   * - ``node_features``
     - (num_nodes, 4)
     - Node props: [util, degree, centrality, marker]
   * - ``adjacency``
     - (num_nodes, num_nodes)
     - Adjacency matrix (optional)

PathEncoder for Custom GNN Features
-----------------------------------

Convert path node sequences to edge masks:

.. code-block:: python

   from fusion.modules.rl.environments import PathEncoder

   # Create encoder for a network
   encoder = PathEncoder(num_nodes=14, num_edges=42)

   # Encode a path (list of node IDs)
   path = [0, 1, 5, 7]
   edge_mask = encoder.encode_path(path)

   # Result: binary array where 1 = edge used by path
   print(edge_mask.shape)  # (42,)
   print(edge_mask.sum())  # 3 (three edges in path)

Wired Mode: Full Simulation Integration
=======================================

For production use with real network simulation:

.. code-block:: python

   from fusion.core.simulation import SimulationEngine
   from fusion.core.orchestrator import SDNOrchestrator
   from fusion.modules.rl.adapter import RLConfig, RLSimulationAdapter
   from fusion.modules.rl.environments import UnifiedSimEnv

   # Setup simulation components
   engine = SimulationEngine(sim_params)
   orchestrator = SDNOrchestrator(engine_props)
   config = RLConfig(k_paths=3, obs_space="obs_8")
   adapter = RLSimulationAdapter(orchestrator, config)

   # Create wired environment
   env = UnifiedSimEnv(
       config=config,
       engine=engine,
       orchestrator=orchestrator,
       adapter=adapter,
   )

   # Now uses real network state and spectrum checks
   obs, info = env.reset(seed=42)

Wired vs Standalone Comparison
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - Standalone Mode
     - Wired Mode
   * - Dependencies
     - None (self-contained)
     - Full V4 simulation stack
   * - Requests
     - Synthetic (seeded RNG)
     - From SimulationEngine
   * - Feasibility
     - Random (~70% feasible)
     - Real spectrum checks
   * - Network State
     - Synthetic topology
     - Actual NetworkState
   * - Use Case
     - Testing, CI/CD
     - Training, production

Episode Lifecycle
=================

Understanding the Episode Flow
------------------------------

.. code-block:: text

   reset(seed=42)
       |
       v
   Generate N requests (Poisson arrivals)
       |
       v
   request_index = 0
       |
       +---> Build observation for request[0]
       |     Return (obs, info) with action_mask
       |
       v
   step(action)
       |
       +---> Apply action (allocate path)
       |     Compute reward (success/block)
       |     Advance request_index
       |     Check termination
       |
       +---> If request_index < N:
       |         Build next observation
       |         Return (obs, reward, False, False, info)
       |
       +---> If request_index >= N:
                 terminated = True
                 Return (zero_obs, reward, True, False, info)

Episode Properties
------------------

.. code-block:: python

   env = UnifiedSimEnv(num_requests=100)
   obs, info = env.reset(seed=42)

   # Track episode progress
   print(f"Current request: {env.request_index}")
   print(f"Total requests: {env.num_requests}")
   print(f"Episode done: {env.is_episode_done}")

   # Access current request (standalone mode)
   req = env.current_request
   print(f"Source: {req.source}, Dest: {req.destination}")

Seeding and Reproducibility
---------------------------

Same seed produces identical episodes:

.. code-block:: python

   env = UnifiedSimEnv(num_requests=10)

   # First run
   obs1, _ = env.reset(seed=42)
   rewards1 = []
   for _ in range(10):
       obs, reward, term, trunc, info = env.step(0)
       rewards1.append(reward)

   # Second run with same seed
   obs2, _ = env.reset(seed=42)
   rewards2 = []
   for _ in range(10):
       obs, reward, term, trunc, info = env.step(0)
       rewards2.append(reward)

   assert rewards1 == rewards2  # Identical!

Configuration Reference
=======================

RLConfig Parameters
-------------------

.. code-block:: python

   from fusion.modules.rl.adapter import RLConfig

   config = RLConfig(
       # Network parameters
       k_paths=3,                  # Candidate paths per request
       num_nodes=14,               # Network node count
       total_slots=320,            # Spectrum slots per link
       num_bandwidth_classes=4,    # Bandwidth quantization levels

       # Observation settings
       obs_space="obs_8",          # Feature set (obs_1 through obs_8)
       use_gnn_obs=False,          # Enable GNN observations
       num_node_features=4,        # GNN node feature dimension

       # Reward shaping
       rl_success_reward=1.0,      # Reward for successful allocation
       rl_block_penalty=-1.0,      # Penalty for blocked request
       rl_grooming_bonus=0.1,      # Bonus for traffic grooming
       rl_slicing_penalty=-0.1,    # Penalty for spectrum slicing

       # Episode settings
       max_holding_time=1000.0,    # Maximum request duration
   )

   env = UnifiedSimEnv(config=config, num_requests=500)

UnifiedSimEnv Constructor
-------------------------

.. code-block:: python

   env = UnifiedSimEnv(
       config=RLConfig(),          # RL configuration
       num_requests=100,           # Requests per episode
       render_mode=None,           # Gymnasium render mode

       # Wired mode (optional)
       engine=None,                # SimulationEngine instance
       orchestrator=None,          # SDNOrchestrator instance
       adapter=None,               # RLSimulationAdapter instance
   )

Troubleshooting
===============

Common Issues
-------------

**"No valid actions available"**

All paths are infeasible for this request. This can happen with:

- High network load (many active connections)
- Large bandwidth requests
- Limited spectrum availability

.. code-block:: python

   action_mask = info["action_mask"]
   if not any(action_mask):
       # All paths infeasible - must select anyway (will block)
       action = 0
   else:
       valid = [i for i, v in enumerate(action_mask) if v]
       action = valid[0]

**"Step called after episode terminated"**

Call reset() before stepping:

.. code-block:: python

   obs, info = env.reset(seed=42)
   while True:
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           # Must reset before next step
           obs, info = env.reset()
           break

**"MaskablePPO expects ndarray, not dict"**

Use ``MultiInputPolicy`` for Dict observation spaces:

.. code-block:: python

   # Wrong: MlpPolicy expects flat observations
   model = MaskablePPO("MlpPolicy", env)  # Error!

   # Correct: MultiInputPolicy handles Dict spaces
   model = MaskablePPO("MultiInputPolicy", env)

**Observation space mismatch after config change**

Recreate the environment after changing config:

.. code-block:: python

   # Wrong: reusing environment with different config
   env.config.obs_space = "obs_4"  # Doesn't update spaces!

   # Correct: create new environment
   new_config = RLConfig(obs_space="obs_4")
   env = UnifiedSimEnv(config=new_config)

File Reference
==============

.. code-block:: text

   fusion/modules/rl/environments/
   |-- __init__.py              # Public exports
   |-- unified_env.py           # UnifiedSimEnv, SimpleRequest, PathEncoder
   |-- wrappers.py              # ActionMaskWrapper
   `-- tests/
       |-- __init__.py
       `-- test_unified_env.py  # Comprehensive test suite

API Summary
-----------

**From package (``fusion.modules.rl.environments``):**

.. code-block:: python

   from fusion.modules.rl.environments import (
       UnifiedSimEnv,       # Main Gymnasium environment
       ActionMaskWrapper,   # SB3 MaskablePPO adapter
       PathEncoder,         # Path to edge mask converter
   )

**From unified_env module:**

.. code-block:: python

   from fusion.modules.rl.environments.unified_env import (
       UnifiedSimEnv,
       SimpleRequest,  # Lightweight request for standalone mode
       PathEncoder,
   )

Related Documentation
=====================

- :ref:`rl-adapter` - RLSimulationAdapter and RLConfig
- :ref:`rl-args` - Observation space definitions (OBS_DICT)
- :ref:`rl-algorithms` - RL algorithms for training
- :ref:`rl-agents` - PathAgent for non-DRL algorithms
