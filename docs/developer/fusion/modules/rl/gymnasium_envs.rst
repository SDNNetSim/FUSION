.. _rl-gymnasium-envs:

================================
Gymnasium Environments (Legacy)
================================

.. warning::

   **Deprecated Module**

   This module contains the legacy ``SimEnv`` (``GeneralSimEnv``) which is
   **deprecated** and will be removed in v6.X. For new work, use
   :ref:`rl-environments` (``UnifiedSimEnv``) instead.

   **Migration Path:**

   .. code-block:: python

      # Old (deprecated)
      from fusion.modules.rl.gymnasium_envs import SimEnv
      env = SimEnv(sim_dict=config)

      # New (recommended)
      from fusion.modules.rl.environments import UnifiedSimEnv
      env = UnifiedSimEnv(config=rl_config)

      # Or use the factory function
      from fusion.modules.rl.gymnasium_envs import create_sim_env
      env = create_sim_env(config, env_type="unified")

.. admonition:: At a Glance
   :class: tip

   :Purpose: Legacy Gymnasium environment for RL simulation
   :Location: ``fusion/modules/rl/gymnasium_envs/``
   :Key Classes: ``SimEnv``, ``create_sim_env()``
   :Status: Deprecated - use :ref:`rl-environments` instead

.. warning::

   **Spectral Band Limitation:**

   This module currently only supports **C-band** spectrum allocation.
   L-band and multi-band scenarios are not yet supported.

Overview
========

The ``gymnasium_envs`` module provides the original Gymnasium-compatible
environment implementation for reinforcement learning with FUSION network
simulations. It wraps the FUSION simulation engine in a standard RL interface.

**Module Contents:**

- ``SimEnv`` (alias ``GeneralSimEnv``): Legacy environment class
- ``create_sim_env()``: Factory function for environment creation with migration support
- ``EnvType``: Environment type constants for factory function
- Constants for configuration and spectral bands

Factory Function
================

The ``create_sim_env()`` factory function provides a migration path from
the legacy ``SimEnv`` to the new ``UnifiedSimEnv``.

.. code-block:: python

   from fusion.modules.rl.gymnasium_envs import create_sim_env, EnvType

   # Create legacy environment (default)
   env = create_sim_env(config)

   # Create unified environment (recommended)
   env = create_sim_env(config, env_type="unified")

   # Or use EnvType constants
   env = create_sim_env(config, env_type=EnvType.UNIFIED)

Environment Selection
---------------------

The factory function determines which environment to create based on:

1. **Explicit parameter**: ``env_type="legacy"`` or ``env_type="unified"``
2. **Environment variable**: ``RL_ENV_TYPE=unified``
3. **Environment variable**: ``USE_UNIFIED_ENV=1``
4. **Default**: Legacy (for backward compatibility)

.. code-block:: bash

   # Via environment variable
   export USE_UNIFIED_ENV=1
   python train.py  # Will use UnifiedSimEnv

Function Reference
------------------

.. code-block:: python

   def create_sim_env(
       config: dict[str, Any] | SimulationConfig,
       env_type: str | None = None,
       wrap_action_mask: bool = True,
       **kwargs: Any,
   ) -> gym.Env:
       """
       Create RL simulation environment.

       :param config: Simulation configuration dict or SimulationConfig
       :param env_type: "legacy" or "unified" (None checks env vars)
       :param wrap_action_mask: Wrap unified env with ActionMaskWrapper
       :param kwargs: Additional arguments for environment constructor
       :return: Gymnasium environment instance
       """

Legacy SimEnv Usage
===================

.. deprecated:: 4.0
   Use ``UnifiedSimEnv`` instead. ``SimEnv`` will be removed in v6.X.

Basic Setup
-----------

.. code-block:: python

   import os
   # Suppress deprecation warning if needed
   os.environ["SUPPRESS_SIMENV_DEPRECATION"] = "1"

   from fusion.modules.rl.gymnasium_envs import SimEnv

   # Create with configuration
   sim_config = {
       "s1": {
           "path_algorithm": "q_learning",
           "k_paths": 3,
           "cores_per_link": 7,
           "c_band": 320,
           "erlang_start": 100,
           "erlang_stop": 500,
           "erlang_step": 50,
       }
   }
   env = SimEnv(sim_dict=sim_config)

Training Loop
-------------

.. code-block:: python

   # Reset environment
   obs, info = env.reset(seed=42)

   for step in range(max_steps):
       # Select action
       action = env.action_space.sample()

       # Take step
       obs, reward, terminated, truncated, info = env.step(action)

       if terminated or truncated:
           obs, info = env.reset()

Integration with Stable-Baselines3
----------------------------------

.. code-block:: python

   from stable_baselines3 import PPO
   from fusion.modules.rl.gymnasium_envs import SimEnv

   env = SimEnv(sim_dict=config)

   model = PPO("MultiInputPolicy", env, verbose=1)
   model.learn(total_timesteps=10000)

SimEnv Configuration
====================

Required Configuration Keys
---------------------------

The ``sim_dict`` must contain an ``"s1"`` key with simulation parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``path_algorithm``
     - str
     - RL algorithm (q_learning, dqn, ppo, etc.)
   * - ``k_paths``
     - int
     - Number of candidate paths per request
   * - ``cores_per_link``
     - int
     - Fiber cores per network link
   * - ``c_band``
     - int
     - Spectral slots in C-band (only C-band supported)
   * - ``erlang_start``
     - float
     - Starting traffic load (Erlang)
   * - ``erlang_stop``
     - float
     - Ending traffic load
   * - ``erlang_step``
     - float
     - Traffic load increment

Optional Parameters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``is_training``
     - True
     - Training mode (vs inference)
   * - ``optimize``
     - False
     - Enable Optuna optimization
   * - ``reward``
     - 1.0
     - Reward for successful allocation
   * - ``penalty``
     - -10.0
     - Penalty for blocked request

Observation and Action Spaces
=============================

Observation Space
-----------------

SimEnv provides graph-structured observations including:

- Network topology information (node features, edge connectivity)
- Current request details (bandwidth, holding time)
- Available resources and path options
- Congestion and feasibility indicators

The exact observation space depends on the configuration and is constructed
by ``get_obs_space()`` in ``utils/deep_rl.py``.

Action Space
------------

Actions represent path selection decisions:

- ``Discrete(k_paths)``: Select which candidate path to use
- Invalid actions result in blocked requests and penalties

Reward Structure
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Outcome
     - Reward
     - Description
   * - Success
     - ``+reward``
     - Request allocated successfully
   * - Blocked
     - ``penalty``
     - Request could not be allocated (negative value)

Environment Lifecycle
=====================

.. code-block:: text

   __init__(sim_dict)
       |
       +---> Setup RLProps, helpers, agents
       |---> Initial reset() to configure spaces
       |---> Build observation_space, action_space
       |
       v
   reset(seed, options)
       |
       +---> Initialize iteration
       |---> Setup simulation engine
       |---> Generate requests (Poisson arrivals)
       |---> Return initial observation
       |
       v
   step(action) [repeated]
       |
       +---> Process action (path selection)
       |---> Attempt allocation via rl_help_obj
       |---> Calculate reward
       |---> Advance to next request
       |---> Check termination
       |---> Return (obs, reward, terminated, truncated, info)
       |
       v
   [Episode ends when all requests processed]

Internal Components
===================

SimEnv uses several helper classes for its operation:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Purpose
   * - ``RLProps``
     - State container for RL properties (paths, slots, etc.)
   * - ``SetupHelper``
     - Simulation setup and model loading
   * - ``SimEnvObs``
     - Observation construction
   * - ``SimEnvUtils``
     - Step handling and termination checks
   * - ``CoreUtilHelpers``
     - Allocation and resource management
   * - ``PathAgent``
     - Path selection agent (Q-learning, bandits, etc.)

Constants Reference
===================

Defined in ``constants.py``:

.. code-block:: python

   # Configuration keys
   DEFAULT_SIMULATION_KEY = "s1"
   DEFAULT_SAVE_SIMULATION = False

   # Spectral bands (C-band only currently)
   SUPPORTED_SPECTRAL_BANDS = ["c"]

   # Arrival parameter keys
   ARRIVAL_DICT_KEYS = {
       "start": "erlang_start",
       "stop": "erlang_stop",
       "step": "erlang_step",
   }

   # Environment defaults
   DEFAULT_ITERATION = 0
   DEFAULT_ARRIVAL_COUNT = 0

Migration to UnifiedSimEnv
==========================

Why Migrate?
------------

- **Accuracy**: UnifiedSimEnv uses the same code paths as non-RL simulation
- **Maintainability**: Single unified codebase (no forked logic)
- **Features**: Better action masking, GNN observations, configurable obs spaces
- **Testing**: More comprehensive test coverage

Migration Steps
---------------

1. **Update imports:**

   .. code-block:: python

      # Before
      from fusion.modules.rl.gymnasium_envs import SimEnv

      # After
      from fusion.modules.rl.environments import UnifiedSimEnv

2. **Update configuration:**

   .. code-block:: python

      # Before (dict with "s1" key)
      config = {"s1": {"path_algorithm": "dqn", ...}}
      env = SimEnv(sim_dict=config)

      # After (RLConfig object)
      from fusion.modules.rl.adapter import RLConfig
      config = RLConfig(k_paths=3, obs_space="obs_8")
      env = UnifiedSimEnv(config=config)

3. **Update action handling:**

   .. code-block:: python

      # Before (no action masking)
      action = model.predict(obs)

      # After (with action masking)
      action_mask = info["action_mask"]
      action = model.predict(obs, action_masks=action_mask)

4. **Test thoroughly** before removing legacy code

File Reference
==============

.. code-block:: text

   fusion/modules/rl/gymnasium_envs/
   |-- __init__.py              # Factory function, exports
   |-- general_sim_env.py       # SimEnv class (deprecated)
   |-- constants.py             # Configuration constants
   |-- README.md                # Module documentation
   |-- TODO.md                  # Development roadmap
   `-- tests/
       `-- ...                  # Unit tests

**Public API:**

.. code-block:: python

   from fusion.modules.rl.gymnasium_envs import (
       # Factory function (recommended)
       create_sim_env,
       EnvType,

       # Legacy environment (deprecated)
       SimEnv,

       # Constants
       DEFAULT_SIMULATION_KEY,
       DEFAULT_SAVE_SIMULATION,
       SUPPORTED_SPECTRAL_BANDS,
       ARRIVAL_DICT_KEYS,
       DEFAULT_ITERATION,
       DEFAULT_ARRIVAL_COUNT,
   )

Related Documentation
=====================

- :ref:`rl-environments` - UnifiedSimEnv (recommended replacement)
- :ref:`rl-adapter` - RLSimulationAdapter and RLConfig
- :ref:`rl-algorithms` - RL algorithms for training
- :ref:`rl-module` - Parent RL module documentation

.. seealso::

   - `Gymnasium Documentation <https://gymnasium.farama.org/>`_
   - `Stable-Baselines3 Documentation <https://stable-baselines3.readthedocs.io/>`_
