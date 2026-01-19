.. _rl-algorithms:

=====================
RL Algorithms Package
=====================

.. admonition:: At a Glance
   :class: tip

   :Purpose: RL algorithm implementations for network optimization
   :Location: ``fusion/modules/rl/algorithms/``
   :Key Files: ``q_learning.py``, ``bandits.py``, ``base_drl.py``
   :Prerequisites: Basic RL theory (Q-learning, bandits, policy gradients)

This package provides the actual RL algorithms used by agents. It includes
traditional methods (Q-learning, bandits) and integration classes for deep RL
via Stable-Baselines3.

Algorithm Categories
====================

FUSION supports three categories of RL algorithms:

.. code-block:: text

   +------------------------------------------------------------------+
   |                    ALGORITHM CATEGORIES                          |
   +------------------------------------------------------------------+
   |                                                                  |
   |  Traditional RL          Multi-Armed Bandits      Deep RL        |
   |  ----------------        ------------------       --------       |
   |  - Q-Learning            - Epsilon-Greedy         - PPO          |
   |    (tabular)             - UCB                    - A2C          |
   |                                                   - DQN          |
   |                                                   - QR-DQN       |
   |                                                                  |
   |  State: (src, dst)       State: (src, dst)        State: obs vec |
   |  Action: path index      Action: arm index        Action: int    |
   |  Updates: Q-table        Updates: value est.      Updates: SB3   |
   |                                                                  |
   +------------------------------------------------------------------+

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Category
     - Algorithms
     - State Space
     - Best For
   * - Traditional
     - Q-Learning
     - Discrete (tabular)
     - Small networks, interpretable
   * - Bandits
     - Epsilon-Greedy, UCB
     - Contextual (src, dst)
     - Fast learning, simple
   * - Deep RL
     - PPO, A2C, DQN, QR-DQN
     - Continuous (vectors)
     - Large networks, complex features

Quick Start: Using Q-Learning
=============================

Q-Learning maintains a table of Q-values for each (source, destination, path,
congestion_level) combination.

Step 1: Initialize Q-Learning
-----------------------------

.. code-block:: python

   from fusion.modules.rl.algorithms import QLearning

   # Q-learning needs rl_props with network info and engine_props with config
   q_learner = QLearning(rl_props=rl_props, engine_props=engine_props)

   # Q-tables are automatically initialized:
   # - routes_matrix: Q-values for path selection
   # - cores_matrix: Q-values for core selection

**What happens at initialization:**

1. Creates ``QProps`` to hold Q-tables and statistics
2. Initializes routes_matrix with shape ``(num_nodes, num_nodes, k_paths, path_levels)``
3. Initializes cores_matrix for multi-core scenarios
4. Populates Q-tables with initial paths from shortest path computation

Step 2: Get Best Action
-----------------------

.. code-block:: python

   # Get congestion levels for available paths
   congestion_list = rl_helper.classify_paths(paths_list)

   # Get action with highest Q-value
   best_index, best_path = q_learner.get_max_curr_q(
       cong_list=congestion_list,
       matrix_flag="routes_matrix",  # or "cores_matrix"
   )

   # best_index: index of path with highest Q-value
   # best_path: the actual path (list of nodes)

Step 3: Update Q-Values
-----------------------

After executing the action and receiving a reward:

.. code-block:: python

   q_learner.update_q_matrix(
       reward=1.0,                          # Reward from allocation
       level_index=congestion_level,        # Current congestion level
       network_spectrum_dict=spectrum_db,   # Current network state
       flag="path",                         # "path" or "core"
       trial=current_trial,
       iteration=current_iteration,
   )

**The Q-learning update rule:**

.. code-block:: python

   # Bellman equation
   delta = reward + gamma * max_future_q
   td_error = current_q - delta
   new_q = (1 - learn_rate) * current_q + learn_rate * delta

Step 4: Save the Model
----------------------

Models are automatically saved at configured intervals:

.. code-block:: python

   # Automatic saving happens in update_q_matrix when:
   # - iteration % save_step == 0, OR
   # - iteration == max_iters - 1

   # Models saved to: logs/q_learning/{network}/{date}/{time}/
   # Files:
   # - rewards_e{erlang}_routes_c{cores}_t{trial}_iter_{iter}.npy
   # - state_vals_e{erlang}_routes_c{cores}_t{trial}.json

Quick Start: Using Bandits
==========================

Bandits are simpler than Q-learning - they don't consider future rewards,
just immediate value estimates.

Epsilon-Greedy Bandit
---------------------

.. code-block:: python

   from fusion.modules.rl.algorithms import EpsilonGreedyBandit

   # Create bandit for path selection
   bandit = EpsilonGreedyBandit(
       rl_props=rl_props,
       engine_props=engine_props,
       is_path=True,  # True for path selection, False for core selection
   )

   # Set exploration rate
   bandit.epsilon = 0.1  # 10% random exploration

   # Select an arm (path)
   action = bandit.select_path_arm(source=0, dest=5)

   # After allocation, update with reward
   bandit.update(
       arm=action,
       reward=1.0,
       iteration=current_iter,
       trial=current_trial,
   )

**How epsilon-greedy works:**

.. code-block:: python

   if random() < epsilon:
       return random_arm()      # Explore
   else:
       return best_value_arm()  # Exploit

UCB Bandit
----------

Upper Confidence Bound adds an exploration bonus based on uncertainty:

.. code-block:: python

   from fusion.modules.rl.algorithms import UCBBandit

   # Create UCB bandit
   ucb = UCBBandit(
       rl_props=rl_props,
       engine_props=engine_props,
       is_path=True,
   )

   # Select arm (automatically balances exploration/exploitation)
   action = ucb.select_path_arm(source=0, dest=5)

   # Update after allocation
   ucb.update(arm=action, reward=1.0, iteration=iter, trial=trial)

**How UCB works:**

.. code-block:: python

   # UCB formula
   ucb_value = estimated_value + sqrt(c * log(total_counts) / arm_counts)

   # Arms with fewer selections get higher bonus (exploration)
   # Arms with high values are preferred (exploitation)

Quick Start: Deep RL Integration
================================

Deep RL algorithms (PPO, A2C, DQN, QR-DQN) are thin wrappers that provide
observation and action spaces. The actual training happens via Stable-Baselines3.

.. note::

   These classes don't implement the algorithms - they configure the spaces
   for SB3. The heavy lifting is done by SB3's implementations.

Creating a DRL Algorithm
------------------------

.. code-block:: python

   from fusion.modules.rl.algorithms import PPO, DQN

   # Create PPO configuration
   ppo = PPO(rl_props=rl_props, engine_obj=engine_obj)

   # Get spaces for SB3
   obs_space = ppo.get_obs_space()    # gymnasium.spaces.Dict
   action_space = ppo.get_action_space()  # gymnasium.spaces.Discrete

   # These spaces are used by the environment
   # SB3 handles the actual learning

Using with Stable-Baselines3
----------------------------

.. code-block:: python

   from stable_baselines3 import PPO as SB3_PPO
   from fusion.modules.rl.gymnasium_envs import GeneralSimEnv

   # Create environment (uses algorithm spaces internally)
   env = GeneralSimEnv(sim_dict=config)

   # Create SB3 model
   model = SB3_PPO("MultiInputPolicy", env, verbose=1)

   # Train
   model.learn(total_timesteps=10000)

   # The algorithm class configured the spaces
   # SB3 does the actual training

Understanding Properties Classes
================================

The algorithms module includes several properties classes that hold state
and configuration.

RLProps
-------

State container for RL simulations (used by environments and agents):

.. code-block:: python

   from fusion.modules.rl.algorithms import RLProps

   rl_props = RLProps()

   # Network configuration
   rl_props.k_paths = 3              # Candidate paths
   rl_props.cores_per_link = 7       # Cores per fiber
   rl_props.spectral_slots = 320     # Slots per core
   rl_props.num_nodes = 14           # Network nodes

   # Current request state
   rl_props.source = 0               # Source node
   rl_props.destination = 5          # Destination node
   rl_props.paths_list = [...]       # Available paths

   # Selection state (set by agent)
   rl_props.chosen_path_index = 0
   rl_props.chosen_path_list = [0, 1, 5]

QProps
------

Q-learning specific properties:

.. code-block:: python

   from fusion.modules.rl.algorithms import QProps

   q_props = QProps()

   # Epsilon (exploration rate)
   q_props.epsilon = 0.1
   q_props.epsilon_start = 1.0
   q_props.epsilon_end = 0.01
   q_props.epsilon_list = []  # Track over time

   # Q-tables
   q_props.routes_matrix = np.array(...)  # Path Q-values
   q_props.cores_matrix = np.array(...)   # Core Q-values

   # Statistics tracking
   q_props.rewards_dict = {"routes_dict": {...}, "cores_dict": {...}}
   q_props.errors_dict = {"routes_dict": {...}, "cores_dict": {...}}

BanditProps
-----------

Bandit-specific properties:

.. code-block:: python

   from fusion.modules.rl.algorithms import BanditProps

   bandit_props = BanditProps()

   # Rewards for each episode
   bandit_props.rewards_matrix = []  # [[r1, r2, ...], [r1, r2, ...], ...]

   # Action counts and values
   bandit_props.counts_list = []
   bandit_props.state_values_list = []

Model Persistence
=================

The module provides classes for saving and loading trained models.

Saving Q-Learning Models
------------------------

.. code-block:: python

   from fusion.modules.rl.algorithms import QLearningModelPersistence

   # Save is typically called automatically by QLearning.save_model()
   # But can be called directly:
   QLearningModelPersistence.save_model(
       q_dict=q_values_dict,        # Q-values as dict
       rewards_avg=rewards_array,   # Average rewards
       erlang=100.0,                # Traffic load
       cores_per_link=7,
       base_str="routes",           # or "cores"
       trial=0,
       iteration=1000,
       save_dir="logs/q_learning/NSFNet/2024-01-15/10-30-00",
   )

   # Saved files:
   # - rewards_e100.0_routes_c7_t1_iter_1000.npy
   # - state_vals_e100.0_routes_c7_t1.json

Saving Bandit Models
--------------------

.. code-block:: python

   from fusion.modules.rl.algorithms import BanditModelPersistence

   BanditModelPersistence.save_model(
       state_values_dict=bandit.values,
       erlang=100.0,
       cores_per_link=7,
       save_dir="logs/epsilon_greedy_bandit/...",
       is_path=True,
       trial=0,
   )

   # Saved file:
   # - state_vals_e100.0_routes_c7_t1.json

Loading Models
--------------

.. code-block:: python

   # Load bandit model
   state_values = BanditModelPersistence.load_model(
       train_fp="epsilon_greedy_bandit/NSFNet/.../state_vals_e100.0_routes_c7_t1.json"
   )

   # Load Q-learning model (usually via agent.load_model())
   # The Q-tables are loaded into the algorithm object

Extending the Algorithms Module
===============================

Tutorial: Adding a New Bandit Algorithm
---------------------------------------

Let's add Thompson Sampling bandit.

**Step 1: Create the class in bandits.py**

.. code-block:: python

   class ThompsonSamplingBandit:
       """
       Thompson Sampling bandit algorithm.

       Uses Beta distribution to model uncertainty about arm values.
       """

       def __init__(
           self,
           rl_props: object,
           engine_props: dict,
           is_path: bool,
       ) -> None:
           self.props = BanditProps()
           self.engine_props = engine_props
           self.rl_props = rl_props
           self.is_path = is_path
           self.iteration = 0

           self.source: int | None = None
           self.dest: int | None = None

           if is_path:
               self.n_arms = engine_props["k_paths"]
           else:
               self.n_arms = engine_props["cores_per_link"]

           self.num_nodes = rl_props.num_nodes

           # Beta distribution parameters (successes, failures)
           self.alpha, self.beta = self._init_beta_params()

       def _init_beta_params(self) -> tuple[dict, dict]:
           """Initialize Beta distribution parameters."""
           alpha = {}
           beta = {}
           for src in range(self.num_nodes):
               for dst in range(self.num_nodes):
                   if src == dst:
                       continue
                   key = (src, dst)
                   alpha[key] = np.ones(self.n_arms)
                   beta[key] = np.ones(self.n_arms)
           return alpha, beta

**Step 2: Add action selection**

.. code-block:: python

       def select_path_arm(self, source: int, dest: int) -> int:
           """Select arm using Thompson Sampling."""
           self.source = source
           self.dest = dest
           key = (source, dest)

           # Sample from Beta distribution for each arm
           samples = np.random.beta(self.alpha[key], self.beta[key])

           return int(np.argmax(samples))

**Step 3: Add update method**

.. code-block:: python

       def update(
           self,
           arm: int,
           reward: float,
           iteration: int,
           trial: int,
       ) -> None:
           """Update Beta parameters based on reward."""
           key = (self.source, self.dest)

           # Bernoulli reward: success (1) or failure (0)
           if reward > 0:
               self.alpha[key][arm] += 1
           else:
               self.beta[key][arm] += 1

           self.iteration = iteration

           # Track rewards
           if self.iteration >= len(self.props.rewards_matrix):
               self.props.rewards_matrix.append([])
           self.props.rewards_matrix[self.iteration].append(reward)

           # Save model periodically
           save_model(
               iteration=iteration,
               algorithm="thompson_sampling_bandit",
               self=self,
               trial=trial,
           )

**Step 4: Export in __init__.py**

.. code-block:: python

   from .bandits import EpsilonGreedyBandit, UCBBandit, ThompsonSamplingBandit

   __all__ = [
       # ...
       "ThompsonSamplingBandit",
   ]

**Step 5: Add to agents**

In ``base_agent.py``:

.. code-block:: python

   elif self.algorithm == "thompson_sampling_bandit":
       from fusion.modules.rl.algorithms.bandits import ThompsonSamplingBandit
       self.algorithm_obj = ThompsonSamplingBandit(
           rl_props=self.rl_props,
           engine_props=self.engine_props,
           is_path=is_path,
       )

Tutorial: Adding a New DRL Algorithm
------------------------------------

DRL algorithms are wrappers that configure spaces for SB3.

**Step 1: Create the class**

.. code-block:: python

   # sac.py
   """Soft Actor-Critic (SAC) algorithm integration."""

   from fusion.modules.rl.algorithms.base_drl import BaseDRLAlgorithm


   class SAC(BaseDRLAlgorithm):
       """
       Soft Actor-Critic for reinforcement learning.

       Inherits observation and action space handling from BaseDRLAlgorithm.
       """

       def get_action_space(self):
           """SAC typically uses continuous actions, but we use discrete."""
           # Override if SAC needs different action space
           return super().get_action_space()

**Step 2: Export and integrate**

.. code-block:: python

   # In __init__.py
   from .sac import SAC

   __all__ = [..., "SAC"]

   # In base_agent.py setup_env()
   elif self.algorithm == "sac":
       self.algorithm_obj = SAC(rl_props=self.rl_props, engine_obj=self.engine_props)

Testing
=======

Running Tests
-------------

.. code-block:: bash

   # Run algorithm tests
   pytest fusion/modules/tests/rl/test_algorithm_props.py -v
   pytest fusion/modules/tests/rl/test_q_learning.py -v
   pytest fusion/modules/tests/rl/test_bandits.py -v

   # Run with coverage
   pytest fusion/modules/tests/rl/ -v --cov=fusion.modules.rl.algorithms

Writing Algorithm Tests
-----------------------

.. code-block:: python

   import pytest
   import numpy as np
   from unittest.mock import MagicMock

   from fusion.modules.rl.algorithms import EpsilonGreedyBandit


   @pytest.fixture
   def mock_rl_props():
       props = MagicMock()
       props.num_nodes = 5
       return props


   @pytest.fixture
   def bandit(mock_rl_props):
       engine_props = {
           "k_paths": 3,
           "cores_per_link": 7,
           "max_iters": 100,
           "save_step": 50,
           "num_requests": 10,
       }
       return EpsilonGreedyBandit(mock_rl_props, engine_props, is_path=True)


   def test_select_path_arm_returns_valid_action(bandit):
       """select_path_arm should return action in valid range."""
       bandit.epsilon = 0.0  # Greedy selection

       action = bandit.select_path_arm(source=0, dest=1)

       assert 0 <= action < bandit.n_arms


   def test_epsilon_greedy_explores_with_high_epsilon(bandit):
       """High epsilon should lead to diverse actions."""
       bandit.epsilon = 1.0  # Always explore

       actions = [bandit.select_path_arm(0, 1) for _ in range(100)]

       # Should see multiple different actions
       assert len(set(actions)) > 1

Common Issues
=============

**"rl_props must have num_nodes"**

.. code-block:: python

   # RLProps needs to be properly initialized
   rl_props = RLProps()
   rl_props.num_nodes = 14  # Set this before creating algorithms

**Q-table shape mismatch**

The Q-table shape depends on network configuration:

.. code-block:: python

   # routes_matrix shape: (num_nodes, num_nodes, k_paths, path_levels)
   # Make sure engine_props matches rl_props:
   assert rl_props.num_nodes == expected_nodes
   assert rl_props.k_paths == engine_props["k_paths"]

**Model saving path errors**

Models save to ``logs/{algorithm}/{network}/{date}/{time}/``:

.. code-block:: python

   # Ensure engine_props has required fields:
   engine_props = {
       "network": "NSFNet",
       "date": "2024-01-15",
       "sim_start": "10-30-00",
       "erlang": 100.0,
       "cores_per_link": 7,
       # ...
   }

File Reference
==============

.. code-block:: text

   fusion/modules/rl/algorithms/
   |-- __init__.py          # Public exports
   |-- README.md            # Module documentation
   |-- algorithm_props.py   # RLProps, QProps, BanditProps, PPOProps
   |-- persistence.py       # BanditModelPersistence, QLearningModelPersistence
   |-- base_drl.py          # BaseDRLAlgorithm (DRL base class)
   |-- q_learning.py        # QLearning
   |-- bandits.py           # EpsilonGreedyBandit, UCBBandit
   |-- ppo.py               # PPO (SB3 wrapper)
   |-- a2c.py               # A2C (SB3 wrapper)
   |-- dqn.py               # DQN (SB3 wrapper)
   `-- qr_dqn.py            # QrDQN (SB3 wrapper)

**What to import:**

.. code-block:: python

   # Algorithms
   from fusion.modules.rl.algorithms import (
       QLearning,
       EpsilonGreedyBandit,
       UCBBandit,
       PPO,
       A2C,
       DQN,
       QrDQN,
   )

   # Properties
   from fusion.modules.rl.algorithms import (
       RLProps,
       QProps,
       BanditProps,
   )

   # Persistence
   from fusion.modules.rl.algorithms import (
       BanditModelPersistence,
       QLearningModelPersistence,
   )

   # Base class (for extending)
   from fusion.modules.rl.algorithms import BaseDRLAlgorithm

Related Documentation
=====================

- :ref:`rl-agents` - Agents that use these algorithms
- :ref:`rl-module` - Parent RL module documentation
- `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/>`_ - Deep RL library
