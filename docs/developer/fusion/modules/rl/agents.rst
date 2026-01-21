.. _rl-agents:

=================
RL Agents Package
=================

.. admonition:: At a Glance
   :class: tip

   :Purpose: RL agents that select actions during simulation
   :Location: ``fusion/modules/rl/agents/``
   :Key Files: ``base_agent.py``, ``path_agent.py``
   :Prerequisites: Understanding of RL algorithms (Q-learning, bandits, DRL)

.. warning::

   **Legacy Path Only**

   This agents package is used by the **legacy simulation path** (``GeneralSimEnv``,
   ``SDNController``). If you're using the new orchestrator path with ``UnifiedSimEnv``,
   you don't use these agents directly - the environment handles action selection
   through the :ref:`rl-adapter`.

   - **Legacy path**: Uses ``PathAgent`` directly
   - **Orchestrator path**: Uses ``RLSimulationAdapter`` + SB3 models

What Are Agents?
================

In FUSION's RL module, an **agent** is an object that:

1. **Holds an algorithm** (Q-learning, bandit, PPO, etc.)
2. **Selects actions** based on the current state
3. **Updates** the algorithm based on rewards
4. **Manages hyperparameters** (learning rate, epsilon decay)

Think of agents as the "brain" that wraps an algorithm and provides a consistent
interface for the simulation to interact with.

.. code-block:: text

   +------------------+     +------------------+     +------------------+
   | Simulation       |---->| Agent            |---->| Algorithm        |
   | (requests state) |     | (coordinates)    |     | (does the math)  |
   +------------------+     +------------------+     +------------------+
                                    |
                                    | manages
                                    v
                            +------------------+
                            | Hyperparameters  |
                            | (alpha, epsilon) |
                            +------------------+

Current Implementation Status
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Agent
     - Status
     - Description
   * - ``PathAgent``
     - **Implemented**
     - Selects which path to use for a request
   * - ``CoreAgent``
     - Placeholder
     - Will select which fiber core to use (multi-core fibers)
   * - ``SpectrumAgent``
     - Placeholder
     - Will select spectrum slots (currently uses heuristics)

Quick Start: Using PathAgent
============================

This tutorial shows how to use ``PathAgent`` with the legacy simulation path.

Step 1: Create the Agent
------------------------

.. code-block:: python

   from fusion.modules.rl.agents import PathAgent

   # Create agent with your chosen algorithm
   path_agent = PathAgent(
       path_algorithm="q_learning",  # or "epsilon_greedy_bandit", "ppo", etc.
       rl_props=rl_props,            # RL properties object
       rl_help_obj=rl_helper,        # RL helper for utilities
   )

**Available algorithms:**

- ``q_learning`` - Tabular Q-learning (good for small state spaces)
- ``epsilon_greedy_bandit`` - Multi-armed bandit with epsilon-greedy
- ``ucb_bandit`` - Upper Confidence Bound bandit
- ``ppo`` - Proximal Policy Optimization (deep RL)
- ``a2c`` - Advantage Actor-Critic (deep RL)
- ``dqn`` - Deep Q-Network (deep RL)
- ``qr_dqn`` - Quantile Regression DQN (deep RL)

Step 2: Initialize the Environment
----------------------------------

Before using the agent, set up its environment:

.. code-block:: python

   # Set engine properties (simulation configuration)
   path_agent.engine_props = {
       "max_iters": 1000,
       "k_paths": 3,
       "reward": 1.0,
       "penalty": -1.0,
       "gamma": 0.9,
       "path_algorithm": "q_learning",
       # ... other properties
   }

   # Initialize the algorithm and hyperparameters
   path_agent.setup_env(is_path=True)

**What happens in setup_env:**

1. Creates the reward tracking array
2. Initializes hyperparameter configuration
3. Creates the algorithm object (Q-learning, bandit, etc.)

Step 3: Select a Route
----------------------

During simulation, ask the agent to select a route:

.. code-block:: python

   # For Q-learning
   path_agent.get_route()

   # For bandits
   path_agent.get_route(route_obj=route_object)

   # For deep RL (PPO, DQN, etc.)
   path_agent.get_route(route_obj=route_object, action=selected_action)

   # After get_route(), these are populated:
   chosen_path = path_agent.rl_props.chosen_path_list
   chosen_index = path_agent.rl_props.chosen_path_index

**How route selection works:**

- **Q-learning**: Uses epsilon-greedy over Q-values
- **Bandits**: Uses bandit-specific selection (epsilon-greedy or UCB)
- **Deep RL**: Uses the action provided by the SB3 model

Step 4: Update After Allocation
-------------------------------

After the simulation tries to allocate the request, update the agent:

.. code-block:: python

   path_agent.update(
       was_allocated=True,           # Did allocation succeed?
       network_spectrum_dict=spectrum_db,  # Current spectrum state
       iteration=current_iter,       # Current iteration number
       path_length=len(chosen_path), # Length of selected path
       trial=current_trial,          # Current trial number
   )

**What happens in update:**

1. Calculates reward based on allocation success
2. Updates the algorithm (Q-values, bandit estimates, etc.)
3. Updates hyperparameters if using per-step decay

Step 5: End Iteration
---------------------

At the end of each iteration (episode), call:

.. code-block:: python

   path_agent.end_iter()

This updates episodic hyperparameters (alpha, epsilon decay).

Understanding the Class Hierarchy
=================================

.. code-block:: text

   BaseAgent
   |-- PathAgent     (implemented)
   |-- CoreAgent     (placeholder)
   `-- SpectrumAgent (placeholder)

BaseAgent
---------

The base class provides common functionality:

.. code-block:: python

   class BaseAgent:
       def __init__(self, algorithm, rl_props, rl_help_obj):
           self.algorithm = algorithm        # Algorithm name string
           self.rl_props = rl_props          # RL properties
           self.rl_help_obj = rl_help_obj    # Helper utilities
           self.algorithm_obj = None         # The actual algorithm instance
           self.engine_props = None          # Simulation configuration

       def setup_env(self, is_path: bool):
           """Initialize the algorithm based on self.algorithm"""

       def get_reward(self, was_allocated, dynamic, core_index, req_id):
           """Calculate reward/penalty for an action"""

       def load_model(self, model_path, file_prefix, **kwargs):
           """Load a trained model from disk"""

PathAgent
---------

Extends BaseAgent with path-specific functionality:

.. code-block:: python

   class PathAgent(BaseAgent):
       def __init__(self, path_algorithm, rl_props, rl_help_obj):
           super().__init__(path_algorithm, rl_props, rl_help_obj)
           self.iteration = None
           self.level_index = None           # For Q-learning congestion levels
           self.congestion_list = None       # Path congestion data
           self.state_action_pair = None     # (source, dest) tuple
           self.action_index = None          # Selected path index

       def get_route(self, **kwargs):
           """Select a route using the configured algorithm"""

       def update(self, was_allocated, network_spectrum_dict, ...):
           """Update agent after allocation attempt"""

       def end_iter(self):
           """End iteration and update episodic hyperparameters"""

Reward Calculation
==================

The agent calculates rewards based on allocation success:

Static Rewards
--------------

Simple success/failure rewards from configuration:

.. code-block:: python

   # In engine_props:
   # reward = 1.0, penalty = -1.0

   reward = agent.get_reward(
       was_allocated=True,
       dynamic=False,
       core_index=None,
       req_id=None,
   )
   # Returns: 1.0 (success) or -1.0 (failure)

Dynamic Rewards
---------------

Rewards that vary based on context:

.. code-block:: python

   reward = agent.get_reward(
       was_allocated=True,
       dynamic=True,
       core_index=2,       # Which core was used
       req_id=150,         # Which request number
   )
   # Returns: reward adjusted by core index and request progress

**Dynamic reward formula:**

.. code-block:: python

   # For success:
   decay_factor = 1 + decay_factor * core_index
   core_decay = reward / decay_factor
   request_ratio = (num_requests - req_id) / num_requests
   request_weight = request_ratio ** core_beta
   dynamic_reward = core_decay * request_weight

   # For failure:
   penalty_factor = 1 + gamma * core_index / req_id
   dynamic_penalty = penalty * penalty_factor

Hyperparameter Management
=========================

Agents manage hyperparameters through ``HyperparamConfig``:

.. code-block:: python

   # Hyperparameters are automatically managed
   # Access current values:
   current_alpha = agent.hyperparam_obj.current_alpha      # Learning rate
   current_epsilon = agent.hyperparam_obj.current_epsilon  # Exploration rate

Decay Strategies
----------------

Hyperparameters can decay over time:

**Episodic decay** - Updates at end of each iteration:

.. code-block:: python

   # In end_iter():
   if alpha_strategy in EPISODIC_STRATEGIES:
       hyperparam_obj.update_alpha()
   if epsilon_strategy in EPISODIC_STRATEGIES:
       hyperparam_obj.update_eps()

**Per-step decay** - Updates after each action:

.. code-block:: python

   # In update():
   if alpha_strategy not in EPISODIC_STRATEGIES:
       hyperparam_obj.update_alpha()

Extending the Agents Module
===========================

Tutorial: Implementing CoreAgent
--------------------------------

Here's how you would implement ``CoreAgent`` (currently a placeholder):

**Step 1: Define the class**

.. code-block:: python

   # core_agent.py
   from typing import Any
   from fusion.modules.rl.agents.base_agent import BaseAgent
   from fusion.modules.rl.errors import InvalidActionError


   class CoreAgent(BaseAgent):
       """Agent for intelligent core assignment in multi-core fibers."""

       def __init__(
           self,
           core_algorithm: str,
           rl_props: Any,
           rl_help_obj: Any,
       ) -> None:
           super().__init__(core_algorithm, rl_props, rl_help_obj)
           self.selected_core: int | None = None

**Step 2: Add core selection method**

.. code-block:: python

       def get_core(self, available_cores: list[int], **kwargs: Any) -> int:
           """
           Select a core for the current request.

           :param available_cores: List of cores with available spectrum
           :return: Selected core index
           """
           if not available_cores:
               raise InvalidActionError("No cores available for assignment")

           if self.algorithm == "q_learning":
               return self._ql_core_selection(available_cores)
           elif self.algorithm in ("epsilon_greedy_bandit", "ucb_bandit"):
               return self._bandit_core_selection(available_cores)
           elif self.algorithm in ("ppo", "a2c", "dqn", "qr_dqn"):
               return self._drl_core_selection(available_cores, kwargs["action"])
           else:
               raise InvalidActionError(f"Algorithm '{self.algorithm}' not supported")

**Step 3: Implement algorithm-specific selection**

.. code-block:: python

       def _ql_core_selection(self, available_cores: list[int]) -> int:
           """Q-learning based core selection."""
           assert self.hyperparam_obj is not None
           assert self.algorithm_obj is not None

           # Epsilon-greedy selection
           if np.random.random() < self.hyperparam_obj.current_epsilon:
               # Explore: random core
               self.selected_core = np.random.choice(available_cores)
           else:
               # Exploit: best Q-value core
               if hasattr(self.algorithm_obj, "get_best_core"):
                   self.selected_core = self.algorithm_obj.get_best_core(
                       available_cores=available_cores,
                       source=self.rl_props.source,
                       dest=self.rl_props.destination,
                   )
               else:
                   self.selected_core = available_cores[0]

           return self.selected_core

**Step 4: Add update method**

.. code-block:: python

       def update(
           self,
           was_allocated: bool,
           iteration: int,
           trial: int,
       ) -> None:
           """Update agent after core assignment attempt."""
           self._ensure_initialized()

           reward = self.get_reward(
               was_allocated=was_allocated,
               dynamic=self.engine_props["dynamic_reward"],
               core_index=self.selected_core,
               req_id=iteration,
           )

           if self.algorithm == "q_learning":
               self.algorithm_obj.update_core_q_values(
                   reward=reward,
                   core=self.selected_core,
                   source=self.rl_props.source,
                   dest=self.rl_props.destination,
               )
           elif self.algorithm in ("epsilon_greedy_bandit", "ucb_bandit"):
               self.algorithm_obj.update(
                   reward=reward,
                   arm=self.selected_core,
                   iteration=iteration,
                   trial=trial,
               )

**Step 5: Update __init__.py**

.. code-block:: python

   # In agents/__init__.py
   from .core_agent import CoreAgent  # Now imports real implementation

Tutorial: Adding a New Algorithm to an Agent
--------------------------------------------

To add support for a new algorithm (e.g., SARSA):

**Step 1: Create the algorithm class**

First, create ``fusion/modules/rl/algorithms/sarsa.py``:

.. code-block:: python

   class SARSA:
       """SARSA (State-Action-Reward-State-Action) algorithm."""

       def __init__(self, rl_props, engine_props):
           self.rl_props = rl_props
           self.engine_props = engine_props
           self.q_table = {}
           self.learn_rate = engine_props.get("alpha_start", 0.1)

       def select_action(self, state, epsilon):
           # Epsilon-greedy selection
           pass

       def update(self, state, action, reward, next_state, next_action):
           # SARSA update rule
           pass

**Step 2: Add to BaseAgent.setup_env()**

.. code-block:: python

   # In base_agent.py, setup_env():
   elif self.algorithm == "sarsa":
       from fusion.modules.rl.algorithms.sarsa import SARSA
       self.algorithm_obj = SARSA(
           rl_props=self.rl_props,
           engine_props=self.engine_props,
       )

**Step 3: Add to PathAgent methods**

.. code-block:: python

   # In path_agent.py, get_route():
   elif self.algorithm == "sarsa":
       self._sarsa_route()

   # Add the new method:
   def _sarsa_route(self) -> None:
       """Select route using SARSA algorithm."""
       state = (self.rl_props.source, self.rl_props.destination)
       self.rl_props.chosen_path_index = self.algorithm_obj.select_action(
           state=state,
           epsilon=self.hyperparam_obj.current_epsilon,
       )
       self.rl_props.chosen_path_list = self.rl_props.paths_list[
           self.rl_props.chosen_path_index
       ]

**Step 4: Update the update() method**

.. code-block:: python

   # In path_agent.py, update():
   elif self.algorithm == "sarsa":
       self.algorithm_obj.update(
           state=self.state_action_pair,
           action=self.action_index,
           reward=reward,
           next_state=next_state,
           next_action=next_action,
       )

**Step 5: Add to valid algorithms list**

.. code-block:: python

   # In fusion/modules/rl/args/general_args.py
   VALID_ALGORITHMS = [
       "q_learning",
       "epsilon_greedy_bandit",
       "ucb_bandit",
       "sarsa",  # Add new algorithm
       # ...
   ]

Testing
=======

Running Tests
-------------

.. code-block:: bash

   # Run all agent tests (if they exist)
   pytest fusion/modules/rl/agents/tests/ -v

   # Run with coverage
   pytest fusion/modules/rl/agents/tests/ -v --cov=fusion.modules.rl.agents

Writing Tests for Agents
------------------------

.. code-block:: python

   import pytest
   from unittest.mock import MagicMock
   from fusion.modules.rl.agents import PathAgent


   @pytest.fixture
   def mock_rl_props():
       """Create mock RL properties."""
       props = MagicMock()
       props.source = 0
       props.destination = 5
       props.k_paths = 3
       props.paths_list = [["0", "1", "5"], ["0", "2", "5"], ["0", "3", "4", "5"]]
       props.chosen_path_index = None
       props.chosen_path_list = None
       return props


   @pytest.fixture
   def path_agent(mock_rl_props):
       """Create PathAgent for testing."""
       agent = PathAgent(
           path_algorithm="epsilon_greedy_bandit",
           rl_props=mock_rl_props,
           rl_help_obj=MagicMock(),
       )
       agent.engine_props = {
           "max_iters": 100,
           "k_paths": 3,
           "reward": 1.0,
           "penalty": -1.0,
           "gamma": 0.9,
           "path_algorithm": "epsilon_greedy_bandit",
       }
       return agent


   def test_setup_env_creates_algorithm(path_agent):
       """setup_env should create the algorithm object."""
       path_agent.setup_env(is_path=True)

       assert path_agent.algorithm_obj is not None
       assert path_agent.hyperparam_obj is not None
       assert path_agent.reward_penalty_list is not None


   def test_get_reward_returns_correct_values(path_agent):
       """get_reward should return configured reward/penalty."""
       path_agent.setup_env(is_path=True)

       success_reward = path_agent.get_reward(
           was_allocated=True, dynamic=False, core_index=None, req_id=None
       )
       failure_penalty = path_agent.get_reward(
           was_allocated=False, dynamic=False, core_index=None, req_id=None
       )

       assert success_reward == 1.0
       assert failure_penalty == -1.0

Common Issues
=============

**"engine_props must be set before calling setup_env"**

.. code-block:: python

   # Wrong:
   agent = PathAgent(...)
   agent.setup_env(is_path=True)  # Error!

   # Right:
   agent = PathAgent(...)
   agent.engine_props = {...}     # Set this first
   agent.setup_env(is_path=True)

**"Algorithm 'xyz' is not supported"**

Check that your algorithm is in the supported list and spelled correctly:

- ``q_learning`` (not ``qlearning`` or ``Q_learning``)
- ``epsilon_greedy_bandit`` (not ``epsilon_greedy``)
- ``ppo``, ``a2c``, ``dqn``, ``qr_dqn`` (lowercase)

**"algorithm_obj must be initialized"**

Always call ``setup_env()`` before using the agent:

.. code-block:: python

   agent.setup_env(is_path=True)  # This creates algorithm_obj

File Reference
==============

.. code-block:: text

   fusion/modules/rl/agents/
   |-- __init__.py        # Public exports
   |-- base_agent.py      # BaseAgent class
   |-- path_agent.py      # PathAgent (implemented)
   |-- core_agent.py      # CoreAgent (placeholder)
   |-- spectrum_agent.py  # SpectrumAgent (placeholder)
   `-- README.md          # Module documentation

**What to import:**

.. code-block:: python

   # Main agent class
   from fusion.modules.rl.agents import PathAgent

   # Base class (for extending)
   from fusion.modules.rl.agents import BaseAgent

   # Placeholders (will raise NotImplementedError)
   from fusion.modules.rl.agents import CoreAgent, SpectrumAgent

Related Documentation
=====================

- :ref:`rl-module` - Parent RL module documentation
- :ref:`rl-adapter` - Adapter for orchestrator path (alternative to agents)
- ``fusion/modules/rl/algorithms/`` - Algorithm implementations
- ``fusion/modules/rl/utils/hyperparams.py`` - Hyperparameter configuration
