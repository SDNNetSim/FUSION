.. _rl-module:

==============================
Reinforcement Learning Module
==============================

.. note::

   **Status: Transitioning to UnifiedSimEnv**

   This module is transitioning from the legacy ``GeneralSimEnv`` to the new
   ``UnifiedSimEnv``. New work should use ``UnifiedSimEnv``.

   - **Pre v6.0**: ``GeneralSimEnv`` (deprecated, removal planned for v6.1+)
   - **Post v6.0**: ``UnifiedSimEnv`` (recommended for new work)

.. toctree::
   :maxdepth: 2
   :caption: Submodules

   adapter
   agents
   algorithms
   args
   environments
   feat_extrs
   gymnasium_envs
   policies
   sb3
   utils
   visualization

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Reinforcement learning for network resource allocation and optimization
   :Location: ``fusion/modules/rl/``
   :Key Entry Points: ``workflow_runner.py``, ``model_manager.py``
   :CLI Command: ``python -m fusion.cli.run_train --config_path <config.ini> --agent_type rl``
   :External Docs: `Stable-Baselines3 Documentation <https://stable-baselines3.readthedocs.io/>`_

The RL module enables intelligent network optimization through reinforcement learning.
It provides a complete framework for training agents to make routing decisions in
optical networks.

.. important::

   **Current Agent Support:**

   - **Path/Routing Agent**: Fully implemented and supported
   - **Core Agent**: Placeholder - development planned for future versions
   - **Spectrum Agent**: Placeholder - development planned for future versions

   Currently, only the **routing/path selection agent** is functional. Core assignment
   and spectrum allocation use heuristic methods (e.g., ``first_fit``).

.. warning::

   **Spectral Band Limitation:**

   RL environments currently only support **C-band** spectrum allocation.
   L-band and multi-band scenarios are not yet supported. Multi-band support
   is planned for a future v6.X release.

**What This Module Provides:**

- **In-house RL algorithms**: Q-learning, Epsilon-Greedy Bandits, UCB Bandits (actively expanded)
- **Deep RL via Stable-Baselines3**: PPO, A2C, DQN, QR-DQN wrappers
- **Custom SB3 callbacks**: Episodic reward tracking, dynamic learning rate/entropy decay (see :ref:`rl-utils`)
- **RLZoo3 integration**: Automatic hyperparameter optimization and experiment management (see :ref:`rl-sb3`)
- **Offline RL policies** *(beta)*: Behavioral Cloning (BC), Implicit Q-Learning (IQL)
- **GNN-based feature extractors** *(beta)*: GAT, SAGE, GraphConv, Graphormer (see :ref:`rl-feat-extrs`)
- **Hyperparameter optimization**: Optuna integration with configurable pruning
- **Action masking**: Safe RL deployment preventing invalid actions

.. tip::

   **RLZoo3 Integration**: FUSION environments can be registered with
   `RLZoo3 <https://rl-baselines3-zoo.readthedocs.io/>`_ for automated hyperparameter
   tuning, experiment tracking, and benchmarking. See :ref:`rl-sb3` for details.

   We are also developing FUSION-native training infrastructure that will provide
   tighter integration with our simulation stack, custom callbacks, and
   domain-specific optimizations for optical network environments.

.. note::

   **Multi-Processing Limitation:**

   RL training currently runs in **single-process mode**. Multi-environment
   parallelization (e.g., ``SubprocVecEnv``) is not yet supported due to the
   complexity of serializing simulation state across processes. This is planned
   for a future release.

.. tip::

   If you're new to reinforcement learning, we recommend familiarizing yourself with
   `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/>`_ first, as the deep RL
   components build on top of it. However, you can use FUSION's in-house algorithms
   (Q-learning, bandits) without any SB3 knowledge.

Capabilities Overview
=====================

.. code-block:: text

   +===========================================================================+
   |                    REINFORCEMENT LEARNING IN FUSION                       |
   +===========================================================================+
   |                                                                           |
   |   +---------------------------+     +---------------------------+         |
   |   |   IN-HOUSE ALGORITHMS     |     |   DEEP RL (via SB3)       |         |
   |   +---------------------------+     +---------------------------+         |
   |   |                           |     |                           |         |
   |   | Q-Learning (tabular)      |     | PPO (policy gradient)     |         |
   |   | Epsilon-Greedy Bandit     |     | A2C (actor-critic)        |         |
   |   | UCB Bandit                |     | DQN (value-based)         |         |
   |   |                           |     | QR-DQN (distributional)   |         |
   |   | Status: Active expansion  |     |                           |         |
   |   +---------------------------+     +---------------------------+         |
   |              |                                   |                        |
   |              +-----------------------------------+                        |
   |                               |                                           |
   |                               v                                           |
   |   +---------------------------------------------------------------+       |
   |   |                    FUSION RL ENVIRONMENTS                      |      |
   |   +---------------------------------------------------------------+       |
   |   | GeneralSimEnv (legacy) --> UnifiedSimEnv (recommended)         |      |
   |   |                                                                |      |
   |   | Features:                                                      |      |
   |   | - Gymnasium-compatible interface                               |      |
   |   | - Action masking for safe exploration                          |      |
   |   | - GNN-based state representations                              |      |
   |   | - Configurable reward functions                                |      |
   |   +---------------------------------------------------------------+       |
   |                               |                                           |
   |                               v                                           |
   |   +---------------------------------------------------------------+       |
   |   |              OFFLINE RL POLICIES (BETA)                        |      |
   |   +---------------------------------------------------------------+       |
   |   | BCPolicy - Behavioral Cloning from expert demonstrations      |       |
   |   | IQLPolicy - Implicit Q-Learning from offline data             |       |
   |   | KSPFFPolicy - K-Shortest Path First-Fit heuristic baseline    |       |
   |   | OnePlusOnePolicy - 1+1 disjoint path protection               |       |
   |   +---------------------------------------------------------------+       |
   |                                                                           |
   +===========================================================================+

Algorithm Types
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - Algorithm
     - Type
     - Implementation
     - Use Case
   * - ``q_learning``
     - Tabular RL
     - In-house (``algorithms/q_learning.py``)
     - Small state spaces, interpretable policies
   * - ``epsilon_greedy``
     - Multi-armed Bandit
     - In-house (``algorithms/bandits.py``)
     - Path selection with exploration
   * - ``ucb``
     - Multi-armed Bandit
     - In-house (``algorithms/bandits.py``)
     - Optimistic exploration
   * - ``ppo``
     - Deep RL
     - Stable-Baselines3 wrapper
     - Large state spaces, continuous training
   * - ``a2c``
     - Deep RL
     - Stable-Baselines3 wrapper
     - Faster training, simpler architecture
   * - ``dqn``
     - Deep RL
     - Stable-Baselines3 wrapper
     - Discrete actions, replay buffer
   * - ``qr_dqn``
     - Deep RL
     - Stable-Baselines3 wrapper
     - Risk-sensitive decisions

Getting Started
===============

Prerequisites
-------------

1. FUSION installed with RL dependencies: ``pip install -e ".[rl]"``
2. A configuration INI file (see :ref:`rl-configuration`)
3. Basic understanding of RL concepts (recommended)

Quick Start: Training an Agent
------------------------------

**Step 1: Create a configuration file**

Copy the default template and customize the ``[rl_settings]`` section:

.. code-block:: bash

   cp fusion/configs/templates/default.ini my_rl_config.ini

**Step 2: Configure RL parameters**

Edit the ``[rl_settings]`` section in your INI file:

.. code-block:: ini

   [rl_settings]
   # Algorithm selection (only path_algorithm uses RL currently)
   path_algorithm = dqn          # Options: q_learning, dqn, ppo, a2c, etc.
   core_algorithm = first_fit    # Heuristic (RL agent not yet implemented)
   spectrum_algorithm = first_fit # Heuristic (RL agent not yet implemented)

   # Training parameters
   is_training = True
   n_trials = 1                  # Number of training runs
   device = cpu                  # cpu, cuda, or mps

   # Hyperparameters
   gamma = 0.1                   # Discount factor
   epsilon_start = 0.01          # Initial exploration rate
   epsilon_end = 0.01            # Final exploration rate

   # Neural network (for DRL algorithms)
   feature_extractor = path_gnn
   gnn_type = graph_conv
   layers = 2
   emb_dim = 64

**Step 3: Run training**

.. code-block:: bash

   python -m fusion.cli.run_train --config_path my_rl_config.ini --agent_type rl

**Step 4: Check results**

RL-specific outputs (model, rewards, memory) are saved to:

.. code-block:: text

   logs/<algorithm>/<network>/<date>/<time>/

Standard simulation statistics (blocking probability, etc.) are saved to:

.. code-block:: text

   data/output/<network>/<date>/<time>/

.. _rl-configuration:

Configuration Reference
=======================

INI File Settings
-----------------

The ``[rl_settings]`` section in your INI file controls all RL behavior:

.. list-table:: Core Settings
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``path_algorithm``
     - ``dqn``
     - Algorithm for path selection (q_learning, dqn, ppo, a2c, epsilon_greedy, ucb)
   * - ``core_algorithm``
     - ``first_fit``
     - Algorithm for core selection (RL agent not yet implemented)
   * - ``spectrum_algorithm``
     - ``first_fit``
     - Algorithm for spectrum allocation (RL agent not yet implemented)
   * - ``is_training``
     - ``True``
     - Enable training mode (testing mode not yet implemented)
   * - ``device``
     - ``cpu``
     - Compute device (cpu, cuda, mps)

.. list-table:: Training Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``n_trials``
     - ``1``
     - Number of independent training runs
   * - ``gamma``
     - ``0.1``
     - Discount factor for future rewards (0.0-1.0)
   * - ``epsilon_start``
     - ``0.01``
     - Initial exploration rate
   * - ``epsilon_end``
     - ``0.01``
     - Final exploration rate
   * - ``alpha_start``
     - ``0.000215``
     - Initial learning rate
   * - ``reward``
     - ``1``
     - Reward for successful allocation
   * - ``penalty``
     - ``-10``
     - Penalty for blocked requests

.. list-table:: Neural Network Settings (DRL only)
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``feature_extractor``
     - ``path_gnn``
     - Feature extraction method (path_gnn, mlp). See :ref:`rl-feat-extrs`
   * - ``gnn_type``
     - ``gat``
     - GNN architecture: ``gat`` (attention), ``sage`` (sampling), ``graphconv`` (standard)
   * - ``layers``
     - ``2``
     - Number of GNN convolution layers
   * - ``emb_dim``
     - ``64``
     - Embedding dimension size
   * - ``heads``
     - ``4``
     - Attention heads (for GraphTransformer)

Command Line Interface
----------------------

**Basic training:**

.. code-block:: bash

   python -m fusion.cli.run_train --config_path path/to/config.ini --agent_type rl

**Required CLI arguments:**

- ``--config_path``: Path to INI configuration file
- ``--agent_type``: Type of agent (``rl`` for reinforcement learning, ``sl`` for supervised learning)

The CLI reads all parameters from the INI file.

Directory Structure and Data Flow
=================================

Module Structure
----------------

.. code-block:: text

   fusion/modules/rl/
   |-- __init__.py              # Module docstring
   |-- README.md                # Module documentation
   |-- TODO.md                  # Development roadmap
   |-- errors.py                # Custom exception hierarchy
   |-- model_manager.py         # Model creation and persistence
   |-- workflow_runner.py       # Training orchestration (main entry)
   |
   |-- adapter/                 # Simulation integration layer
   |   |-- rl_adapter.py        # RLSimulationAdapter
   |   `-- path_option.py       # PathOption dataclass
   |
   |-- agents/                  # Agent implementations
   |   |-- base_agent.py        # BaseAgent class
   |   |-- path_agent.py        # Path selection agent (IMPLEMENTED)
   |   |-- core_agent.py        # Core assignment (PLACEHOLDER)
   |   `-- spectrum_agent.py    # Spectrum allocation (PLACEHOLDER)
   |
   |-- algorithms/              # Algorithm implementations
   |   |-- q_learning.py        # In-house Q-learning
   |   |-- bandits.py           # In-house bandit algorithms
   |   |-- ppo.py               # SB3 PPO wrapper
   |   |-- a2c.py               # SB3 A2C wrapper
   |   |-- dqn.py               # SB3 DQN wrapper
   |   `-- qr_dqn.py            # SB3 QR-DQN wrapper
   |
   |-- environments/            # Unified environment (recommended)
   |   |-- unified_env.py       # UnifiedSimEnv
   |   `-- wrappers.py          # ActionMaskWrapper
   |
   |-- gymnasium_envs/          # Legacy environment (deprecated)
   |   `-- general_sim_env.py   # GeneralSimEnv (SimEnv)
   |
   |-- feat_extrs/              # GNN feature extractors (BETA)
   |   |-- path_gnn.py          # Standard GNN (GAT, SAGE, GraphConv)
   |   |-- path_gnn_cached.py   # Cached GNN for static graphs
   |   `-- graphormer.py        # Transformer-based GNN (experimental)
   |
   |-- policies/                # Path selection policies (BETA)
   |   |-- bc_policy.py         # Behavioral Cloning
   |   |-- iql_policy.py        # Implicit Q-Learning
   |   |-- ksp_ff_policy.py     # K-Shortest Path First-Fit
   |   `-- one_plus_one_policy.py  # 1+1 protection
   |
   |-- utils/                   # Utility modules (19 files)
   |   |-- setup.py             # Environment setup
   |   |-- hyperparams.py       # Hyperparameter handling and Optuna config
   |   |-- callbacks.py         # Training callbacks
   |   `-- ...
   |
   |-- visualization/           # RL-specific visualization
   |   `-- rl_plugin.py         # Visualization plugin
   |
   `-- sb3/                     # Stable-Baselines3 integration
       `-- register_env.py      # Environment registration

Output Directories
------------------

FUSION separates RL-specific outputs from standard simulation outputs:

.. code-block:: text

   +===========================================================================+
   |                         OUTPUT DIRECTORY STRUCTURE                        |
   +===========================================================================+

   logs/                              <-- RL-SPECIFIC OUTPUTS (SB3 format)
   `-- <algorithm>/                   # e.g., dqn, ppo, q_learning
       `-- <network>/                 # e.g., NSFNet, COST239
           `-- <date>/                # e.g., 2024-01-15
               `-- <time>/            # e.g., 14-30-25
                   |-- model.zip              # Trained model (SB3 format)
                   |-- average_rewards.npy    # Mean rewards per iteration
                   |-- memory_usage.npy       # Memory consumption tracking
                   `-- hyperparams.yml        # Used hyperparameters

   data/output/                       <-- SIMULATION STATISTICS
   `-- <network>/
       `-- <date>/
           `-- <time>/
               |-- blocking_probability.json  # Standard simulation metrics
               |-- fragmentation.json         # Spectrum fragmentation data
               `-- ...                         # Other simulation outputs

   data/input/                        <-- OPTUNA STUDY RESULTS
   `-- <network>/
       `-- <date>/
           `-- <time>/
               `-- hyperparam_study.pkl       # Optuna study object

The ``logs/`` directory structure matches Stable-Baselines3 conventions for compatibility
with SB3 tools and visualization utilities.

Input/Output Flow
-----------------

.. code-block:: text

   INPUT                           PROCESSING                        OUTPUT
   =====                           ==========                        ======

   +------------------+
   | INI Config File  |
   | (rl_settings)    |----+
   +------------------+    |
                           |     +----------------------+
   +------------------+    +---->|                      |     +------------------+
   | Network Topology |--------->| workflow_runner.py   |---->| logs/            |
   | (data/input/)    |--------->|                      |     | (RL outputs)     |
   +------------------+    +---->| - Creates environment|     +------------------+
                           |     | - Trains agent       |              |
   +------------------+    |     | - Saves results      |     +------------------+
   | Hyperparams YAML |----+     +----------------------+     | data/output/     |
   | (configs/)       |                    |                  | (sim stats)      |
   +------------------+                    |                  +------------------+
                                           v
                                  +------------------+
                                  | SimEnv /         |
                                  | UnifiedSimEnv    |
                                  | (Gymnasium env)  |
                                  +------------------+

Hyperparameter Optimization with Optuna
=======================================

FUSION integrates `Optuna <https://optuna.org/>`_ for automated hyperparameter
optimization with configurable pruning strategies.

Enabling Optuna
---------------

In your INI file:

.. code-block:: ini

   [rl_settings]
   optimize_hyperparameters = True
   optuna_trials = 50           # Number of optimization trials

How It Works
------------

1. **Study Creation**: Creates an Optuna study with maximize direction (reward)
2. **Trial Execution**: Each trial samples hyperparameters and trains an agent
3. **Pruning**: Hyperband pruner eliminates poor-performing trials early
4. **Results**: Best hyperparameters saved to ``data/input/<network>/<date>/<time>/``

.. code-block:: text

   +------------------+     +------------------+     +------------------+
   | Optuna Study     |---->| Trial N          |---->| Evaluate         |
   | (maximize reward)|     | Sample params    |     | Report to Optuna |
   +------------------+     | Train agent      |     | Prune if poor    |
          |                 +------------------+     +------------------+
          |                          |                       |
          |                          v                       |
          |                 +------------------+             |
          +---------------->| Best Trial       |<------------+
                            | Save params      |
                            | Save model       |
                            +------------------+

Customizing Optuna Studies
--------------------------

The Optuna study configuration is defined in ``workflow_runner.py``:

.. code-block:: python

   # workflow_runner.py (lines 565-572)
   from optuna.pruners import HyperbandPruner

   pruner = HyperbandPruner(
       min_resource=20,                    # Minimum iterations before pruning
       max_resource=sim_dict["max_iters"], # Maximum iterations (from config)
       reduction_factor=3                  # Pruning aggressiveness
   )
   study = optuna.create_study(
       direction="maximize",               # Maximize reward
       study_name=study_name,
       pruner=pruner
   )
   study.optimize(objective, n_trials=n_trials)

**To customize the pruner**, modify the ``HyperbandPruner`` parameters:

- ``min_resource``: Minimum number of iterations before a trial can be pruned
- ``max_resource``: Maximum number of iterations (typically ``max_iters``)
- ``reduction_factor``: Higher values prune more aggressively

**Alternative pruners** available in Optuna:

.. code-block:: python

   from optuna.pruners import MedianPruner, SuccessiveHalvingPruner

   # Median pruner - prune if below median of completed trials
   pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

   # Successive halving - aggressive early stopping
   pruner = SuccessiveHalvingPruner(min_resource=10, reduction_factor=4)

Hyperparameter Search Spaces
----------------------------

The search spaces are defined in ``utils/hyperparams.py``. Each algorithm has its
own parameter ranges:

**PPO Parameters** (``_ppo_hyperparams``):

.. code-block:: python

   params["n_steps"] = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
   params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
   params["n_epochs"] = trial.suggest_int("n_epochs", 3, 20)
   params["gamma"] = trial.suggest_float("gamma", 0.90, 0.999)
   params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.8, 1.0)
   params["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.4)
   params["alpha_start"] = trial.suggest_float("learning_rate_start", 1e-5, 1e-3, log=True)

**DQN Parameters** (``_dqn_hyperparams``):

.. code-block:: python

   params["buffer_size"] = trial.suggest_int("buffer_size", 50000, 500000, step=50000)
   params["learning_starts"] = trial.suggest_int("learning_starts", 1000, 10000, step=1000)
   params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
   params["tau"] = trial.suggest_float("tau", 0.005, 1.0, log=True)
   params["gamma"] = trial.suggest_float("gamma", 0.90, 0.999)
   params["exploration_fraction"] = trial.suggest_float("exploration_fraction", 0.05, 0.3)

**Q-Learning/Bandit Parameters** (``get_optuna_hyperparams``):

.. code-block:: python

   params["alpha_start"] = trial.suggest_float("alpha_start", 0.01, 0.5, step=0.01)
   params["epsilon_start"] = trial.suggest_float("epsilon_start", 0.01, 0.5, step=0.01)
   params["discount_factor"] = trial.suggest_float("discount_factor", 0.8, 1.0, step=0.01)
   params["decay_rate"] = trial.suggest_float("decay_rate", 0.1, 0.5, step=0.01)

**To customize search spaces**, modify the corresponding functions in ``utils/hyperparams.py``.

Integration with Simulation
===========================

Legacy Path (GeneralSimEnv)
---------------------------

.. code-block:: text

   +------------------+     +------------------+     +------------------+
   | INI Config       |---->| SimEnv           |---->| SDNController    |
   | [rl_settings]    |     | (GeneralSimEnv)  |     | (legacy path)    |
   +------------------+     +------------------+     +------------------+
                                    |
                                    v
                            +------------------+
                            | RL Agent         |
                            | - Observes state |
                            | - Selects action |
                            | - Receives reward|
                            +------------------+

New Path (UnifiedSimEnv)
------------------------

.. code-block:: text

   +------------------+     +------------------+     +------------------+
   | INI Config       |---->| UnifiedSimEnv    |---->| RLSimulation     |
   | [rl_settings]    |     |                  |     | Adapter          |
   +------------------+     +------------------+     +------------------+
                                    |                        |
                                    v                        v
                            +------------------+     +------------------+
                            | RL Agent         |     | SDNOrchestrator  |
                            | - Action masking |     | (new path)       |
                            | - GNN features   |     +------------------+
                            +------------------+

Development Guide
=================

This section provides detailed guidance for extending and customizing the RL module.

Adding a New RL Algorithm
-------------------------

**Step 1: Create the algorithm file**

Create a new file in ``algorithms/``:

.. code-block:: python

   # fusion/modules/rl/algorithms/my_algorithm.py
   """Custom RL algorithm implementation."""

   from typing import Any
   import numpy as np

   class MyAlgorithm:
       """
       Custom RL algorithm for path selection.

       :param engine_props: Simulation engine properties
       :param rl_props: RL-specific properties
       """

       def __init__(self, engine_props: dict[str, Any], rl_props: Any) -> None:
           self.engine_props = engine_props
           self.rl_props = rl_props
           self.q_table: dict[tuple, np.ndarray] = {}

       def select_action(self, state: tuple, epsilon: float) -> int:
           """
           Select an action using epsilon-greedy policy.

           :param state: Current state representation
           :param epsilon: Exploration rate
           :return: Selected action index
           """
           if np.random.random() < epsilon:
               return np.random.randint(self.engine_props["k_paths"])

           if state not in self.q_table:
               self.q_table[state] = np.zeros(self.engine_props["k_paths"])
           return int(np.argmax(self.q_table[state]))

       def update(
           self,
           state: tuple,
           action: int,
           reward: float,
           next_state: tuple,
           alpha: float,
           gamma: float,
       ) -> None:
           """
           Update Q-values using the Bellman equation.

           :param state: Current state
           :param action: Action taken
           :param reward: Reward received
           :param next_state: Resulting state
           :param alpha: Learning rate
           :param gamma: Discount factor
           """
           if state not in self.q_table:
               self.q_table[state] = np.zeros(self.engine_props["k_paths"])
           if next_state not in self.q_table:
               self.q_table[next_state] = np.zeros(self.engine_props["k_paths"])

           current_q = self.q_table[state][action]
           max_next_q = np.max(self.q_table[next_state])
           new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
           self.q_table[state][action] = new_q

       def save(self, path: str) -> None:
           """Save the model to disk."""
           import pickle
           with open(path, "wb") as f:
               pickle.dump(self.q_table, f)

       def load(self, path: str) -> None:
           """Load the model from disk."""
           import pickle
           with open(path, "rb") as f:
               self.q_table = pickle.load(f)

**Step 2: Export in __init__.py**

Add to ``algorithms/__init__.py``:

.. code-block:: python

   from .my_algorithm import MyAlgorithm

   __all__ = [
       # ... existing exports
       "MyAlgorithm",
   ]

**Step 3: Register the algorithm**

Add to ``args/general_args.py``:

.. code-block:: python

   VALID_PATH_ALGORITHMS = [
       "q_learning",
       "epsilon_greedy",
       "ucb",
       "ppo",
       "a2c",
       "dqn",
       "qr_dqn",
       "my_algorithm",  # Add your algorithm
   ]

**Step 4: Add to registry (if using registry pattern)**

Add to ``args/registry_args.py`` if needed:

.. code-block:: python

   ALGORITHM_REGISTRY["my_algorithm"] = {
       "class": MyAlgorithm,
       "setup_fn": setup_my_algorithm,
       "load_fn": load_my_algorithm,
   }

Adding a New Policy
-------------------

Policies are used for path selection decisions. To add a new policy:

**Step 1: Create the policy file**

.. code-block:: python

   # fusion/modules/rl/policies/my_policy.py
   """Custom path selection policy."""

   from typing import Any
   import numpy as np
   from fusion.modules.rl.policies.base import PathPolicy


   class MyPolicy(PathPolicy):
       """
       Custom policy for path selection.

       :param model_path: Path to trained model (optional)
       """

       def __init__(self, model_path: str | None = None) -> None:
           self.model_path = model_path
           self.model = None
           if model_path:
               self.load(model_path)

       def select_path(
           self,
           state: dict[str, Any],
           action_mask: np.ndarray | None = None,
       ) -> int:
           """
           Select a path given the current state.

           :param state: Current network state
           :param action_mask: Boolean mask of valid actions
           :return: Selected path index, or -1 if blocked
           """
           if action_mask is not None and not action_mask.any():
               return -1  # All paths blocked

           # Your selection logic here
           if self.model is not None:
               scores = self.model.predict(state)
               if action_mask is not None:
                   scores[~action_mask] = -np.inf
               return int(np.argmax(scores))

           # Fallback: select first valid path
           if action_mask is not None:
               valid_indices = np.where(action_mask)[0]
               return int(valid_indices[0]) if len(valid_indices) > 0 else -1
           return 0

       def load(self, path: str) -> None:
           """Load model from disk."""
           # Implementation depends on your model type
           pass

       def save(self, path: str) -> None:
           """Save model to disk."""
           pass

**Step 2: Export in __init__.py**

Add to ``policies/__init__.py``:

.. code-block:: python

   from .my_policy import MyPolicy

   __all__ = [
       # ... existing exports
       "MyPolicy",
   ]

Adding Optuna Hyperparameters for a New Algorithm
-------------------------------------------------

To add hyperparameter search for your algorithm:

**Edit utils/hyperparams.py:**

.. code-block:: python

   def _my_algorithm_hyperparams(
       sim_dict: dict[str, Any], trial: Trial
   ) -> dict[str, Any]:
       """Define hyperparameter search space for MyAlgorithm."""
       params: dict[str, Any] = {}

       # Learning rate with log-uniform distribution
       params["alpha_start"] = trial.suggest_float(
           "alpha_start", 1e-4, 1e-1, log=True
       )
       params["alpha_end"] = trial.suggest_float(
           "alpha_end", 1e-5, params["alpha_start"], log=True
       )

       # Exploration parameters
       params["epsilon_start"] = trial.suggest_float(
           "epsilon_start", 0.1, 1.0
       )
       params["epsilon_end"] = trial.suggest_float(
           "epsilon_end", 0.01, 0.1
       )

       # Algorithm-specific parameters
       params["my_custom_param"] = trial.suggest_categorical(
           "my_custom_param", ["option_a", "option_b", "option_c"]
       )

       return params

   # Add to the dispatcher in get_optuna_hyperparams()
   def get_optuna_hyperparams(sim_dict: dict[str, Any], trial: Trial) -> dict[str, Any]:
       alg = sim_dict["path_algorithm"].lower()

       if alg == "my_algorithm":
           return _my_algorithm_hyperparams(sim_dict, trial)
       # ... existing algorithm handlers

Testing Your Changes
--------------------

**Unit tests:**

Create ``algorithms/tests/test_my_algorithm.py``:

.. code-block:: python

   """Tests for MyAlgorithm."""

   import pytest
   import numpy as np
   from fusion.modules.rl.algorithms.my_algorithm import MyAlgorithm


   @pytest.fixture
   def algorithm():
       """Create algorithm instance for testing."""
       engine_props = {"k_paths": 4, "max_iters": 100}
       rl_props = type("RLProps", (), {"num_nodes": 14})()
       return MyAlgorithm(engine_props, rl_props)


   def test_select_action_exploration(algorithm):
       """Test that exploration works."""
       state = (0, 1, 100)
       actions = [algorithm.select_action(state, epsilon=1.0) for _ in range(100)]
       # With epsilon=1.0, should see variety in actions
       assert len(set(actions)) > 1


   def test_select_action_exploitation(algorithm):
       """Test that exploitation works."""
       state = (0, 1, 100)
       # Set up Q-values so action 2 is best
       algorithm.q_table[state] = np.array([0.1, 0.2, 0.9, 0.3])
       action = algorithm.select_action(state, epsilon=0.0)
       assert action == 2


   def test_update_q_values(algorithm):
       """Test Q-value updates."""
       state = (0, 1, 100)
       next_state = (0, 1, 90)
       algorithm.update(state, action=0, reward=1.0, next_state=next_state,
                       alpha=0.1, gamma=0.9)
       assert state in algorithm.q_table
       assert algorithm.q_table[state][0] > 0

**Run tests:**

.. code-block:: bash

   pytest fusion/modules/rl/algorithms/tests/test_my_algorithm.py -v

Testing
=======

.. code-block:: bash

   # Run all RL module tests
   pytest fusion/modules/rl/ -v

   # Run specific submodule tests
   pytest fusion/modules/rl/environments/tests/ -v
   pytest fusion/modules/rl/adapter/tests/ -v
   pytest fusion/modules/rl/algorithms/tests/ -v

   # Run with coverage
   pytest fusion/modules/rl/ -v --cov=fusion.modules.rl

Error Handling
==============

The module provides a custom exception hierarchy:

.. code-block:: python

   from fusion.modules.rl.errors import (
       RLError,                  # Base exception
       RLConfigurationError,     # Invalid configuration
       AlgorithmNotFoundError,   # Unknown algorithm
       ModelLoadError,           # Model loading failed
       TrainingError,            # Training process failed
       RLEnvironmentError,       # Environment issues
       AgentError,               # Agent operation failed
       InvalidActionError,       # Invalid action attempted
       RouteSelectionError,      # Route selection failed
   )

Related Documentation
=====================

- :ref:`ml-module` - Traditional ML utilities (separate from RL)
- :ref:`modules-directory` - Overview of all FUSION modules
- :ref:`core-module` - Simulation engine integration

.. seealso::

   - `Stable-Baselines3 Documentation <https://stable-baselines3.readthedocs.io/>`_
   - `RLZoo3 Documentation <https://rl-baselines3-zoo.readthedocs.io/>`_
   - `Gymnasium Documentation <https://gymnasium.farama.org/>`_
   - `Optuna Documentation <https://optuna.readthedocs.io/>`_
