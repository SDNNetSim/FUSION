.. _rl-args:

===========================
RL Arguments/Config Package
===========================

.. admonition:: At a Glance
   :class: tip

   :Purpose: Configuration constants, algorithm definitions, and observation spaces
   :Location: ``fusion/modules/rl/args/``
   :Key Files: ``general_args.py``, ``observation_args.py``, ``constants.py``
   :Prerequisites: None (reference module)

This package defines the configuration constants used throughout the RL module.
It's a reference module - you import from it, but rarely need to modify it.

What This Module Provides
=========================

.. code-block:: text

   +------------------------------------------------------------------+
   |                    ARGS MODULE CONTENTS                          |
   +------------------------------------------------------------------+
   |                                                                  |
   |  Algorithm Definitions       Observation Spaces      Defaults    |
   |  ---------------------       ------------------      --------    |
   |  - Which algorithms          - obs_1 through obs_8  - Epsilon    |
   |    are valid                 - Feature lists        - Learning   |
   |  - Path vs Core vs DRL       - Validation sets        rate       |
   |  - Decay strategies                                 - Batch size |
   |                                                                  |
   +------------------------------------------------------------------+

Quick Reference
===============

Valid Algorithms
----------------

.. code-block:: python

   from fusion.modules.rl.args import (
       VALID_PATH_ALGORITHMS,
       VALID_CORE_ALGORITHMS,
       VALID_DRL_ALGORITHMS,
   )

   # Path selection: all algorithms
   VALID_PATH_ALGORITHMS = [
       "q_learning",
       "epsilon_greedy_bandit",
       "ucb_bandit",
       "ppo",
       "a2c",
       "dqn",
       "qr_dqn",
   ]

   # Core selection: traditional only
   VALID_CORE_ALGORITHMS = [
       "q_learning",
       "epsilon_greedy_bandit",
       "ucb_bandit",
   ]

   # Deep RL only
   VALID_DRL_ALGORITHMS = [
       "ppo",
       "a2c",
       "dqn",
       "qr_dqn",
   ]

Observation Spaces
------------------

.. code-block:: python

   from fusion.modules.rl.args import OBS_DICT

   # obs_1: Minimal - just routing
   OBS_DICT["obs_1"] = ["source", "destination"]

   # obs_4: Standard - routing with constraints
   OBS_DICT["obs_4"] = [
       "source", "destination",
       "request_bandwidth", "holding_time"
   ]

   # obs_8: Complete - all features
   OBS_DICT["obs_8"] = [
       "source", "destination",
       "request_bandwidth", "holding_time",
       "slots_needed", "path_lengths",
       "paths_cong", "available_slots",
       "is_feasible"
   ]

Understanding Observation Spaces
================================

The RL module defines 8 observation space configurations, from minimal to complete:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Space
     - Features
     - Use Case
   * - ``obs_1``
     - source, destination
     - Minimal routing, small state space
   * - ``obs_2``
     - + bandwidth
     - Bandwidth-aware routing
   * - ``obs_3``
     - + holding_time
     - Time-sensitive routing
   * - ``obs_4``
     - + bandwidth, holding_time
     - Standard routing (recommended start)
   * - ``obs_5``
     - + slots_needed, path_lengths
     - Resource-aware routing
   * - ``obs_6``
     - + paths_cong
     - Congestion-aware routing
   * - ``obs_7``
     - + available_slots
     - Availability-aware routing
   * - ``obs_8``
     - + is_feasible
     - Complete (action masking friendly)

Choosing an Observation Space
-----------------------------

.. code-block:: python

   # For simple experiments, start with obs_4
   engine_props["obs_space"] = "obs_4"

   # For production/complex scenarios, use obs_8
   engine_props["obs_space"] = "obs_8"

   # Check what features are included
   from fusion.modules.rl.args import OBS_DICT
   features = OBS_DICT["obs_8"]
   print(f"obs_8 includes: {features}")

Feature Descriptions
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Feature
     - Description
   * - ``source``
     - Source node ID (normalized to [0, 1])
   * - ``destination``
     - Destination node ID (normalized to [0, 1])
   * - ``request_bandwidth``
     - Requested bandwidth (normalized)
   * - ``holding_time``
     - Connection duration (normalized)
   * - ``slots_needed``
     - Spectrum slots required per path
   * - ``path_lengths``
     - Physical length of each candidate path
   * - ``paths_cong``
     - Congestion level per path [0, 1]
   * - ``available_slots``
     - Available slots ratio per path [0, 1]
   * - ``is_feasible``
     - Boolean mask indicating feasible paths

Using Constants in Your Code
============================

Validating Algorithm Names
--------------------------

.. code-block:: python

   from fusion.modules.rl.args import VALID_PATH_ALGORITHMS

   def setup_agent(algorithm: str):
       if algorithm not in VALID_PATH_ALGORITHMS:
           raise ValueError(
               f"Invalid algorithm '{algorithm}'. "
               f"Valid options: {VALID_PATH_ALGORITHMS}"
           )
       # ... proceed with setup

Checking Algorithm Category
---------------------------

.. code-block:: python

   from fusion.modules.rl.args import VALID_DRL_ALGORITHMS

   def is_deep_rl(algorithm: str) -> bool:
       """Check if algorithm is a deep RL method."""
       return algorithm in VALID_DRL_ALGORITHMS

   # Usage
   if is_deep_rl("ppo"):
       # Use SB3 for training
       pass
   else:
       # Use traditional RL update
       pass

Using Episodic Strategies
-------------------------

.. code-block:: python

   from fusion.modules.rl.args import EPISODIC_STRATEGIES

   # EPISODIC_STRATEGIES = ["exp_decay", "linear_decay"]

   def should_update_epsilon_at_episode_end(strategy: str) -> bool:
       """Check if strategy updates at episode boundaries."""
       return strategy in EPISODIC_STRATEGIES

The New Constants Module
========================

For new code, ``constants.py`` provides type-safe enums and better organization:

Using Enums (Type-Safe)
-----------------------

.. code-block:: python

   from fusion.modules.rl.args.constants import (
       AlgorithmType,
       EpisodicStrategy,
       ObservationFeature,
   )

   # Type-safe algorithm checking
   def setup_algorithm(algo_type: AlgorithmType):
       if algo_type == AlgorithmType.PPO:
           return setup_ppo()
       elif algo_type == AlgorithmType.Q_LEARNING:
           return setup_q_learning()

   # Using enum values with string-based APIs
   algorithm_name = AlgorithmType.PPO.value  # "ppo"

Available Enums
---------------

**AlgorithmType:**

.. code-block:: python

   class AlgorithmType(Enum):
       Q_LEARNING = "q_learning"
       EPSILON_GREEDY_BANDIT = "epsilon_greedy_bandit"
       UCB_BANDIT = "ucb_bandit"
       PPO = "ppo"
       A2C = "a2c"
       DQN = "dqn"
       QR_DQN = "qr_dqn"

**EpisodicStrategy:**

.. code-block:: python

   class EpisodicStrategy(Enum):
       EXPONENTIAL_DECAY = "exp_decay"
       LINEAR_DECAY = "linear_decay"

**ObservationFeature:**

.. code-block:: python

   class ObservationFeature(Enum):
       SOURCE = "source"
       DESTINATION = "destination"
       REQUEST_BANDWIDTH = "request_bandwidth"
       HOLDING_TIME = "holding_time"
       SLOTS_NEEDED = "slots_needed"
       PATH_LENGTHS = "path_lengths"
       PATH_CONGESTION = "paths_cong"
       AVAILABLE_SLOTS = "available_slots"
       IS_FEASIBLE = "is_feasible"

Default Configurations
----------------------

.. code-block:: python

   from fusion.modules.rl.args.constants import (
       DEFAULT_EPSILON,
       DEFAULT_EPSILON_MIN,
       DEFAULT_EPSILON_DECAY,
       DEFAULT_LEARNING_RATE,
       DEFAULT_DISCOUNT_FACTOR,
       DEFAULT_BATCH_SIZE,
       DEFAULT_BUFFER_SIZE,
       DEFAULT_OBSERVATION_SPACE,
   )

   # Defaults:
   # DEFAULT_EPSILON = 1.0
   # DEFAULT_EPSILON_MIN = 0.01
   # DEFAULT_EPSILON_DECAY = 0.995
   # DEFAULT_LEARNING_RATE = 0.001
   # DEFAULT_DISCOUNT_FACTOR = 0.99
   # DEFAULT_BATCH_SIZE = 32
   # DEFAULT_BUFFER_SIZE = 10000
   # DEFAULT_OBSERVATION_SPACE = "obs_4"

Algorithm Registry
==================

The ``registry_args.py`` module provides a registry for DRL algorithms:

.. code-block:: python

   from fusion.modules.rl.args.registry_args import get_algorithm_registry

   registry = get_algorithm_registry()

   # Registry structure:
   # {
   #     "ppo": {"class": PPO, "load": None, "setup": setup_ppo},
   #     "a2c": {"class": A2C, "load": None, "setup": setup_a2c},
   #     "dqn": {"class": DQN, "load": None, "setup": setup_dqn},
   #     "qr_dqn": {"class": QrDQN, "load": None, "setup": setup_qr_dqn},
   # }

   # Get setup function for an algorithm
   setup_fn = registry["ppo"]["setup"]
   model = setup_fn(env, config)

Extending the Args Module
=========================

Tutorial: Adding a New Observation Feature
------------------------------------------

**Step 1: Add to ObservationFeature enum**

In ``constants.py``:

.. code-block:: python

   class ObservationFeature(Enum):
       # ... existing features ...
       LATENCY = "latency"  # New feature

**Step 2: Add to VALID_OBSERVATION_FEATURES**

In ``observation_args.py``:

.. code-block:: python

   VALID_OBSERVATION_FEATURES: set[str] = {
       # ... existing features ...
       "latency",
   }

**Step 3: Create new observation space using it**

In ``observation_args.py``:

.. code-block:: python

   OBS_DICT["obs_9"] = [
       "source",
       "destination",
       "request_bandwidth",
       "holding_time",
       "latency",  # New feature
   ]

**Step 4: Implement feature extraction**

In your observation builder (e.g., adapter or environment):

.. code-block:: python

   if "latency" in features:
       obs["latency"] = self._compute_latency(paths)

Tutorial: Adding a New Algorithm
--------------------------------

**Step 1: Add to AlgorithmType enum**

In ``constants.py``:

.. code-block:: python

   class AlgorithmType(Enum):
       # ... existing algorithms ...
       SAC = "sac"  # Soft Actor-Critic

**Step 2: Add to algorithm lists**

In ``general_args.py``:

.. code-block:: python

   VALID_PATH_ALGORITHMS: list[str] = [
       # ... existing algorithms ...
       "sac",
   ]

   VALID_DRL_ALGORITHMS: list[str] = [
       # ... existing algorithms ...
       "sac",
   ]

**Step 3: Add to registry (if DRL)**

In ``registry_args.py``:

.. code-block:: python

   from fusion.modules.rl.algorithms.sac import SAC
   from fusion.modules.rl.utils.setup import setup_sac

   _algorithm_registry = {
       # ... existing algorithms ...
       "sac": {
           "class": SAC,
           "load": None,
           "setup": setup_sac,
       },
   }

Backward Compatibility Notes
============================

The args module maintains backward compatibility through:

1. **Abbreviated names**: ``paths_cong`` instead of ``path_congestion``
2. **Dual exports**: ``OBS_DICT`` and ``OBSERVATION_SPACE_DEFINITIONS``
3. **Alias variables**: ``VALID_DEEP_REINFORCEMENT_LEARNING_ALGORITHMS = VALID_DRL_ALGORITHMS``

When writing new code:

- Use the full names when possible (e.g., ``OBSERVATION_SPACE_DEFINITIONS``)
- Use enums from ``constants.py`` for type safety
- Import from the package level (``from fusion.modules.rl.args import ...``)

File Reference
==============

.. code-block:: text

   fusion/modules/rl/args/
   |-- __init__.py          # Package exports
   |-- README.md            # Module documentation
   |-- general_args.py      # Algorithm lists, strategies
   |-- observation_args.py  # Observation space definitions
   |-- constants.py         # Enums, defaults (new code)
   `-- registry_args.py     # DRL algorithm registry

**What to import:**

.. code-block:: python

   # Most common imports (from package)
   from fusion.modules.rl.args import (
       VALID_PATH_ALGORITHMS,
       VALID_DRL_ALGORITHMS,
       EPISODIC_STRATEGIES,
       OBS_DICT,
       VALID_OBSERVATION_FEATURES,
   )

   # For new code (type-safe enums)
   from fusion.modules.rl.args.constants import (
       AlgorithmType,
       EpisodicStrategy,
       ObservationFeature,
       DEFAULT_EPSILON,
       DEFAULT_LEARNING_RATE,
   )

   # For algorithm setup
   from fusion.modules.rl.args.registry_args import get_algorithm_registry

Related Documentation
=====================

- :ref:`rl-algorithms` - Algorithm implementations that use these constants
- :ref:`rl-agents` - Agents that validate against these constants
- :ref:`rl-adapter` - Adapter that uses observation space definitions
