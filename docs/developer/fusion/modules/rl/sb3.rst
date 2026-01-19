.. _rl-sb3:

=======================================
Stable-Baselines3 Integration (sb3)
=======================================

.. admonition:: At a Glance
   :class: tip

   :Purpose: Environment registration and RLZoo3 integration for SB3 training
   :Location: ``fusion/modules/rl/sb3/``
   :Key Functions: ``copy_yml_file()``, ``main()``
   :External Docs: `RLZoo3 Documentation <https://rl-baselines3-zoo.readthedocs.io/>`_

Overview
========

The sb3 module provides utilities for integrating FUSION environments with the
Stable-Baselines3 (SB3) reinforcement learning framework. It handles:

1. **Environment Registration**: Register custom environments with Gymnasium
2. **RLZoo3 Integration**: Deploy hyperparameter configurations for automated training

**Why This Module?**

Stable-Baselines3 and RLZoo3 provide a powerful ecosystem for:

- Standardized RL algorithm implementations (PPO, DQN, A2C, etc.)
- Hyperparameter optimization with Optuna integration
- Experiment tracking and reproducibility
- Pre-tuned configurations for common environments

This module bridges FUSION's custom environments with this ecosystem.

RLZoo3 Integration
==================

FUSION supports automatic integration with `RLZoo3 <https://github.com/DLR-RM/rl-baselines3-zoo>`_,
a training framework built on Stable-Baselines3 that provides:

- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Experiment Management**: Organized logging and model saving
- **Benchmarking**: Standardized evaluation protocols
- **Pre-tuned Configs**: Curated hyperparameters for various environments

Training Workflow
-----------------

.. code-block:: text

   +------------------+     +------------------+     +------------------+
   | 1. Register Env  |---->| 2. Deploy Config |---->| 3. Train with    |
   |    with Gymnasium|     |    to RLZoo3     |     |    RLZoo3        |
   +------------------+     +------------------+     +------------------+

**Step 1: Register Environment**

.. code-block:: bash

   python -m fusion.modules.rl.sb3.register_env --algo PPO --env-name SimEnv

**Step 2: Train with RLZoo3**

.. code-block:: bash

   # Basic training
   python -m rl_zoo3.train --algo ppo --env SimEnv

   # With hyperparameter optimization
   python -m rl_zoo3.train --algo ppo --env SimEnv -optimize --n-trials 100

   # With custom config
   python -m rl_zoo3.train --algo ppo --env SimEnv --conf-file custom_ppo.yml

**Step 3: Evaluate**

.. code-block:: bash

   python -m rl_zoo3.enjoy --algo ppo --env SimEnv --folder logs/

Environment Registration
========================

The ``main()`` function registers FUSION environments with Gymnasium's registry:

.. code-block:: python

   from gymnasium.envs.registration import register

   # Register custom environment
   register(
       id="SimEnv",
       entry_point="reinforcement_learning.gymnasium_envs.general_sim_env:SimEnv"
   )

Command-Line Interface
----------------------

.. code-block:: bash

   python -m fusion.modules.rl.sb3.register_env --algo ALGO --env-name ENV

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Required
     - Description
   * - ``--algo``
     - Yes
     - Algorithm name (PPO, DQN, A2C, etc.)
   * - ``--env-name``
     - Yes
     - Environment class name to register

**Example:**

.. code-block:: bash

   # Register SimEnv with PPO configuration
   python -m fusion.modules.rl.sb3.register_env --algo PPO --env-name SimEnv

   # Register with DQN
   python -m fusion.modules.rl.sb3.register_env --algo DQN --env-name SimEnv

Configuration Management
========================

copy_yml_file
-------------

Deploys algorithm configuration files to RLZoo3's hyperparameters directory:

.. code-block:: python

   from fusion.modules.rl.sb3 import copy_yml_file

   # Copy PPO configuration to RLZoo3
   copy_yml_file("PPO")

**File Paths:**

.. code-block:: text

   Source:      sb3_scripts/yml/{algorithm}.yml
   Destination: venvs/.../site-packages/rl_zoo3/hyperparams/{algorithm}.yml

Configuration File Format
-------------------------

Algorithm configurations use YAML format compatible with RLZoo3:

**PPO Example:**

.. code-block:: yaml

   SimEnv:
     policy: 'MlpPolicy'
     n_timesteps: !!float 2e6
     learning_rate: lin_3e-4
     n_steps: 2048
     batch_size: 64
     n_epochs: 10
     gamma: 0.99
     gae_lambda: 0.95
     clip_range: 0.2
     ent_coef: 0.0
     vf_coef: 0.5
     max_grad_norm: 0.5

**DQN Example:**

.. code-block:: yaml

   SimEnv:
     policy: 'MlpPolicy'
     n_timesteps: !!float 1e6
     buffer_size: 1000000
     learning_rate: !!float 1e-4
     learning_starts: 50000
     batch_size: 32
     tau: 1.0
     gamma: 0.99
     train_freq: 4
     gradient_steps: 1
     target_update_interval: 10000

**Key Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``policy``
     - Policy architecture (MlpPolicy, MultiInputPolicy, CnnPolicy)
   * - ``n_timesteps``
     - Total training timesteps
   * - ``learning_rate``
     - Learning rate (can use schedules like ``lin_3e-4``)
   * - ``batch_size``
     - Minibatch size for updates
   * - ``gamma``
     - Discount factor

Directory Structure
===================

.. code-block:: text

   project/
   |-- sb3_scripts/yml/           # Source configuration files
   |   |-- PPO.yml
   |   |-- DQN.yml
   |   |-- A2C.yml
   |   `-- ...
   |
   |-- fusion/modules/rl/sb3/     # This module
   |   |-- __init__.py
   |   |-- register_env.py
   |   `-- README.md
   |
   `-- logs/                      # RLZoo3 training outputs
       `-- ppo/
           `-- SimEnv_1/
               |-- model.zip      # Trained model
               |-- config.yml     # Training config
               `-- evaluations/   # Evaluation results

Using with FUSION Environments
==============================

Standard Training Pipeline
--------------------------

.. code-block:: python

   from stable_baselines3 import PPO
   from fusion.modules.rl.gymnasium_envs import create_sim_env

   # Create FUSION environment
   config = {"k_paths": 3, "spectral_slots": 320}
   env = create_sim_env(config, env_type="unified")

   # Train with SB3 directly
   model = PPO("MultiInputPolicy", env, verbose=1)
   model.learn(total_timesteps=100_000)

   # Save model
   model.save("ppo_fusion")

With RLZoo3 Hyperparameter Optimization
---------------------------------------

.. code-block:: bash

   # 1. Register environment
   python -m fusion.modules.rl.sb3.register_env --algo PPO --env-name SimEnv

   # 2. Run hyperparameter optimization
   python -m rl_zoo3.train \
       --algo ppo \
       --env SimEnv \
       -optimize \
       --n-trials 100 \
       --sampler tpe \
       --pruner median

   # 3. Train with best hyperparameters
   python -m rl_zoo3.train \
       --algo ppo \
       --env SimEnv \
       --conf-file logs/ppo/SimEnv_1/best_model/config.yml

With Custom Feature Extractors
------------------------------

.. code-block:: python

   from stable_baselines3 import PPO
   from fusion.modules.rl.feat_extrs import PathGNN
   from fusion.modules.rl.gymnasium_envs import create_sim_env

   env = create_sim_env(config, env_type="unified")

   model = PPO(
       "MultiInputPolicy",
       env,
       policy_kwargs={
           "features_extractor_class": PathGNN,
           "features_extractor_kwargs": {
               "emb_dim": 64,
               "gnn_type": "gat",
               "layers": 2,
           }
       },
       verbose=1,
   )

Error Handling
==============

The module provides clear error messages for common issues:

**FileNotFoundError:**

.. code-block:: text

   Configuration file not found: sb3_scripts/yml/PPO.yml.
   Ensure the algorithm configuration exists in sb3_scripts/yml/

**PermissionError:**

.. code-block:: text

   Cannot write to RLZoo3 directory: .../rl_zoo3/hyperparams/PPO.yml.
   Check file permissions and virtual environment access.

File Reference
==============

.. code-block:: text

   fusion/modules/rl/sb3/
   |-- __init__.py          # Public exports
   |-- register_env.py      # Registration utilities
   `-- README.md            # Module documentation

**Public API:**

.. code-block:: python

   from fusion.modules.rl.sb3 import (
       copy_yml_file,   # Deploy config to RLZoo3
       main,            # CLI entry point
   )

Related Documentation
=====================

- :ref:`rl-algorithms` - RL algorithm wrappers (PPO, DQN, A2C)
- :ref:`rl-feat-extrs` - GNN feature extractors for SB3 policies
- :ref:`rl-environments` - UnifiedSimEnv for SB3 training
- :ref:`rl-module` - Parent RL module documentation

.. seealso::

   - `Stable-Baselines3 Documentation <https://stable-baselines3.readthedocs.io/>`_
   - `RLZoo3 Documentation <https://rl-baselines3-zoo.readthedocs.io/>`_
   - `Gymnasium Documentation <https://gymnasium.farama.org/>`_
