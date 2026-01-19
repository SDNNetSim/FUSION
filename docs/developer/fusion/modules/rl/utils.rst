.. _rl-utils:

======================
RL Utilities (utils)
======================

.. admonition:: At a Glance
   :class: tip

   :Purpose: Infrastructure utilities for RL training, callbacks, and hyperparameter management
   :Location: ``fusion/modules/rl/utils/``
   :Key Classes: ``EpisodicRewardCallback``, ``LearnRateEntCallback``, ``HyperparamConfig``
   :Integration: Used by all RL algorithms, environments, and training workflows

Overview
========

The utils module provides essential infrastructure supporting the entire RL framework.
It handles everything from custom training callbacks to hyperparameter optimization,
model initialization, and simulation data management.

**Key Capabilities:**

- **Custom SB3 Callbacks**: Powerful callbacks for reward tracking and dynamic hyperparameter adjustment
- **Hyperparameter Management**: Flexible configuration with multiple decay strategies
- **Optuna Integration**: Automated hyperparameter search space generation
- **Model Setup**: Factory functions for all supported SB3 algorithms
- **GNN Caching**: Efficient caching of GNN embeddings for faster training

Custom Callbacks
================

FUSION provides custom Stable-Baselines3 callbacks that extend standard training
capabilities. These callbacks integrate seamlessly with any SB3 algorithm and
provide features specifically designed for optical network optimization.

EpisodicRewardCallback
----------------------

Tracks episode rewards across training and saves them periodically for analysis.

**Features:**

- Tracks cumulative rewards per episode
- Saves reward matrices at configurable intervals
- Supports multi-trial reward aggregation
- Compatible with all SB3 algorithms

.. code-block:: python

   from stable_baselines3 import PPO
   from fusion.modules.rl.utils import EpisodicRewardCallback

   # Create callback
   reward_callback = EpisodicRewardCallback(verbose=1)
   reward_callback.sim_dict = sim_dict  # Attach simulation config
   reward_callback.max_iters = 1000

   # Train with callback
   model = PPO("MultiInputPolicy", env, verbose=1)
   model.learn(total_timesteps=100_000, callback=reward_callback)

   # Access rewards after training
   episode_rewards = reward_callback.episode_rewards
   print(f"Average reward: {episode_rewards.mean():.2f}")

**Saved Output Format:**

.. code-block:: text

   logs/<algorithm>/<network>/<date>/<time>/
   `-- rewards_e<erlang>_routes_c<cores>_t<trial>_iter_<iteration>.npy

LearnRateEntCallback
--------------------

Dynamically adjusts learning rate and entropy coefficient during training.
Essential for achieving optimal convergence in policy gradient methods.

**Features:**

- Linear learning rate decay based on training progress
- Exponential entropy coefficient decay (PPO/A2C)
- Episode-based updates (not step-based)
- Configurable start/end values

.. code-block:: python

   from stable_baselines3 import PPO
   from fusion.modules.rl.utils import LearnRateEntCallback

   # Create callback with config
   lr_callback = LearnRateEntCallback(verbose=1)
   lr_callback.sim_dict = {
       "alpha_start": 3e-4,
       "alpha_end": 1e-5,
       "epsilon_start": 0.01,  # entropy coef
       "epsilon_end": 0.001,
       "decay_rate": 0.99,
       "max_iters": 1000,
       "path_algorithm": "ppo",
   }

   # Train with dynamic hyperparameters
   model = PPO("MultiInputPolicy", env)
   model.learn(total_timesteps=100_000, callback=lr_callback)

**Decay Behavior:**

.. code-block:: text

   Learning Rate:  linear decay from alpha_start to alpha_end
   Entropy Coef:   exponential decay with decay_rate (PPO/A2C only)

   Episode 0:   LR=0.000300, Entropy=0.010000
   Episode 100: LR=0.000270, Entropy=0.003660
   Episode 500: LR=0.000150, Entropy=0.000066
   Episode 999: LR=0.000010, Entropy=0.000001

GetModelParams
--------------

Extracts model parameters and value function estimates during training.
Useful for monitoring and debugging policy learning.

.. code-block:: python

   from fusion.modules.rl.utils import GetModelParams

   params_callback = GetModelParams(verbose=0)
   model.learn(total_timesteps=10_000, callback=params_callback)

   # Access extracted data
   model_params = params_callback.model_params
   value_estimate = params_callback.value_estimate

Combining Callbacks
-------------------

Multiple callbacks can be combined for comprehensive training monitoring:

.. code-block:: python

   from stable_baselines3.common.callbacks import CallbackList
   from fusion.modules.rl.utils import (
       EpisodicRewardCallback,
       LearnRateEntCallback,
   )

   # Create callbacks
   reward_cb = EpisodicRewardCallback(verbose=1)
   lr_cb = LearnRateEntCallback(verbose=1)

   # Attach simulation config to both
   for cb in [reward_cb, lr_cb]:
       cb.sim_dict = sim_dict
       cb.max_iters = sim_dict["max_iters"]

   # Combine and train
   callbacks = CallbackList([reward_cb, lr_cb])
   model.learn(total_timesteps=100_000, callback=callbacks)

Hyperparameter Management
=========================

HyperparamConfig
----------------

Manages hyperparameter schedules with multiple decay strategies for tabular
and bandit algorithms.

**Supported Decay Strategies:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Strategy
     - Description
   * - ``linear_decay``
     - Linear interpolation from start to end value
   * - ``exp_decay``
     - Exponential decay using decay_rate
   * - ``softmax``
     - Temperature-based softmax exploration
   * - ``reward_based``
     - Adjusts based on reward improvement
   * - ``state_based``
     - Adjusts based on state visitation counts

.. code-block:: python

   from fusion.modules.rl.utils import HyperparamConfig

   config = HyperparamConfig(
       engine_props=engine_props,
       rl_props=rl_props,
       is_path=True,  # Path agent (vs core agent)
   )

   # Update after each episode
   config.iteration += 1
   config.update_alpha()  # Update learning rate
   config.update_eps()    # Update exploration rate

   # Access current values
   current_lr = config.current_alpha
   current_eps = config.current_epsilon

Optuna Integration
------------------

The ``get_optuna_hyperparams()`` function generates algorithm-specific
search spaces for Optuna optimization:

.. code-block:: python

   import optuna
   from fusion.modules.rl.utils import get_optuna_hyperparams

   def objective(trial):
       # Get hyperparameters for this trial
       params = get_optuna_hyperparams(sim_dict, trial)

       # Create model with suggested params
       model = PPO(
           params["policy"],
           env,
           learning_rate=params["learning_rate"],
           n_steps=params["n_steps"],
           batch_size=params["batch_size"],
           gamma=params["gamma"],
           # ... other params
       )

       # Train and return objective
       model.learn(total_timesteps=params["n_timesteps"])
       return evaluate_model(model)

   study = optuna.create_study(direction="maximize")
   study.optimize(objective, n_trials=100)

**Search Spaces by Algorithm:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Key Parameters
   * - PPO
     - n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, learning_rate, ent_coef
   * - A2C
     - n_steps, gamma, gae_lambda, learning_rate, ent_coef, use_rms_prop
   * - DQN
     - buffer_size, learning_starts, batch_size, tau, gamma, exploration_fraction
   * - QR-DQN
     - DQN params + n_quantiles
   * - Q-Learning
     - alpha, epsilon, discount_factor, decay_rate
   * - Bandits
     - epsilon (greedy) or conf_param (UCB)

Model Setup Utilities
=====================

Factory functions for initializing SB3 models from YAML configuration files:

.. code-block:: python

   from fusion.modules.rl.utils import (
       setup_ppo,
       setup_a2c,
       setup_dqn,
       setup_qr_dqn,
       setup_feature_extractor,
   )

   # Setup PPO with GNN feature extractor
   model = setup_ppo(env, device="cuda")

   # Or setup DQN
   model = setup_dqn(env, device="cpu")

setup_feature_extractor
-----------------------

Configures GNN feature extractors with automatic caching:

.. code-block:: python

   from fusion.modules.rl.utils import setup_feature_extractor

   # Returns (extractor_class, kwargs)
   extr_class, feat_kwargs = setup_feature_extractor(env)

   # Use with SB3 policy
   model = PPO(
       "MultiInputPolicy",
       env,
       policy_kwargs={
           "features_extractor_class": extr_class,
           "features_extractor_kwargs": feat_kwargs,
       },
   )

**Caching Behavior:**

- First run: Computes GNN embeddings and saves to ``gnn_cached/<network>.pt``
- Subsequent runs: Loads cached embeddings for faster initialization

Error Handling
==============

The module provides a custom exception hierarchy for granular error handling:

.. code-block:: python

   from fusion.modules.rl.utils import (
       RLUtilsError,           # Base exception
       ConfigurationError,     # Invalid configuration
       HyperparameterError,    # Hyperparameter issues
       ModelSetupError,        # Model initialization failed
       FeatureExtractorError,  # Feature extractor issues
       DataLoadingError,       # Data loading failed
       SimulationDataError,    # Simulation data issues
       CacheError,             # Caching issues
   )

   try:
       model = setup_ppo(env, device="cuda")
   except ModelSetupError as e:
       logger.error("Failed to setup PPO: %s", e)
   except ConfigurationError as e:
       logger.error("Invalid configuration: %s", e)

File Reference
==============

.. code-block:: text

   fusion/modules/rl/utils/
   |-- __init__.py              # Public API exports
   |-- callbacks.py             # Custom SB3 callbacks
   |-- hyperparams.py           # HyperparamConfig and Optuna
   |-- setup.py                 # Model setup utilities
   |-- deep_rl.py               # Deep RL algorithm utilities
   |-- observation_space.py     # Observation space construction
   |-- topology.py              # Network topology conversion
   |-- sim_env.py               # Simulation environment utilities
   |-- sim_data.py              # Simulation data handling
   |-- sim_filters.py           # Data filtering utilities
   |-- general_utils.py         # General helper functions
   |-- gym_envs.py              # Gymnasium environment utilities
   |-- errors.py                # Custom exception hierarchy
   |-- cache_gnn_once.py        # GNN embedding caching
   |-- unity_hyperparams.py     # Unity cluster utilities
   `-- rl_zoo.py                # RLZoo3 integration utilities

**Public API:**

.. code-block:: python

   from fusion.modules.rl.utils import (
       # Callbacks
       EpisodicRewardCallback,
       GetModelParams,
       LearnRateEntCallback,

       # Deep RL utilities
       get_action_space,
       get_algorithm_instance,
       get_obs_space,

       # Hyperparameters
       HyperparamConfig,
       get_optuna_hyperparams,

       # Setup utilities
       SetupHelper,
       setup_a2c,
       setup_dqn,
       setup_feature_extractor,
       setup_ppo,
       setup_qr_dqn,
       setup_rl_sim,

       # Observation space
       FragmentationTracker,
       get_observation_space,

       # Simulation environment
       SimEnvObs,
       SimEnvUtils,

       # Topology utilities
       convert_networkx_topo,
       load_topology_from_graph,

       # Error classes
       CacheError,
       ConfigurationError,
       DataLoadingError,
       FeatureExtractorError,
       HyperparameterError,
       ModelSetupError,
       RLUtilsError,
       SimulationDataError,
   )

Related Documentation
=====================

- :ref:`rl-module` - Parent RL module documentation
- :ref:`rl-algorithms` - RL algorithm implementations
- :ref:`rl-environments` - UnifiedSimEnv documentation
- :ref:`rl-feat-extrs` - GNN feature extractors

.. seealso::

   - `Stable-Baselines3 Callbacks <https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html>`_
   - `Optuna Documentation <https://optuna.readthedocs.io/>`_
