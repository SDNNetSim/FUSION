================
fusion.modules
================

Algorithm implementations for routing, spectrum assignment, SNR calculation, ML, and RL.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

The ``fusion.modules`` package contains pluggable algorithm implementations organized
into five main categories:

* **Routing**: Path computation algorithms (k-shortest paths, fragmentation-aware, etc.)
* **Spectrum**: Spectrum assignment strategies (First-Fit, Best-Fit, etc.)
* **SNR**: Signal-to-noise ratio calculation methods
* **ML**: Machine learning algorithms and utilities
* **RL**: Reinforcement learning agents, algorithms, and environments

All modules follow a plugin architecture using registries for easy extensibility.

Routing Algorithms
==================

.. automodule:: fusion.modules.routing
   :members:
   :undoc-members:
   :show-inheritance:

Congestion-Aware Routing
------------------------

.. automodule:: fusion.modules.routing.congestion_aware
   :members:
   :undoc-members:
   :show-inheritance:

Fragmentation-Aware Routing
----------------------------

.. automodule:: fusion.modules.routing.fragmentation_aware
   :members:
   :undoc-members:
   :show-inheritance:

Least Congested Routing
------------------------

.. automodule:: fusion.modules.routing.least_congested
   :members:
   :undoc-members:
   :show-inheritance:

NLI-Aware Routing
-----------------

.. automodule:: fusion.modules.routing.nli_aware
   :members:
   :undoc-members:
   :show-inheritance:

Crosstalk-Aware Routing
-----------------------

.. automodule:: fusion.modules.routing.xt_aware
   :members:
   :undoc-members:
   :show-inheritance:

Routing Utilities
-----------------

.. automodule:: fusion.modules.routing.utils
   :members:
   :undoc-members:
   :show-inheritance:

Routing Registry
----------------

.. automodule:: fusion.modules.routing.registry
   :members:
   :undoc-members:
   :show-inheritance:

Spectrum Assignment Algorithms
===============================

.. automodule:: fusion.modules.spectrum
   :members:
   :undoc-members:
   :show-inheritance:

First-Fit
---------

.. automodule:: fusion.modules.spectrum.first_fit
   :members:
   :undoc-members:
   :show-inheritance:

Best-Fit
--------

.. automodule:: fusion.modules.spectrum.best_fit
   :members:
   :undoc-members:
   :show-inheritance:

Last-Fit
--------

.. automodule:: fusion.modules.spectrum.last_fit
   :members:
   :undoc-members:
   :show-inheritance:

Light Path Slicing
------------------

.. automodule:: fusion.modules.spectrum.light_path_slicing
   :members:
   :undoc-members:
   :show-inheritance:

Spectrum Utilities
------------------

.. automodule:: fusion.modules.spectrum.utils
   :members:
   :undoc-members:
   :show-inheritance:

Spectrum Registry
-----------------

.. automodule:: fusion.modules.spectrum.registry
   :members:
   :undoc-members:
   :show-inheritance:

SNR Calculation
===============

.. automodule:: fusion.modules.snr
   :members:
   :undoc-members:
   :show-inheritance:

SNR Models
----------

.. automodule:: fusion.modules.snr.snr
   :members:
   :undoc-members:
   :show-inheritance:

SNR Utilities
-------------

.. automodule:: fusion.modules.snr.utils
   :members:
   :undoc-members:
   :show-inheritance:

SNR Registry
------------

.. automodule:: fusion.modules.snr.registry
   :members:
   :undoc-members:
   :show-inheritance:

Machine Learning
================

.. automodule:: fusion.modules.ml
   :members:
   :undoc-members:
   :show-inheritance:

Feature Engineering
-------------------

.. automodule:: fusion.modules.ml.feature_engineering
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing
-------------

.. automodule:: fusion.modules.ml.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation
----------

.. automodule:: fusion.modules.ml.evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Model I/O
---------

.. automodule:: fusion.modules.ml.model_io
   :members:
   :undoc-members:
   :show-inheritance:

ML Visualization
----------------

.. automodule:: fusion.modules.ml.visualization
   :members:
   :undoc-members:
   :show-inheritance:

ML Constants
------------

.. automodule:: fusion.modules.ml.constants
   :members:
   :undoc-members:
   :show-inheritance:

ML Registry
-----------

.. automodule:: fusion.modules.ml.registry
   :members:
   :undoc-members:
   :show-inheritance:

Reinforcement Learning
======================

.. automodule:: fusion.modules.rl
   :members:
   :undoc-members:
   :show-inheritance:

Model Manager
-------------

.. automodule:: fusion.modules.rl.model_manager
   :members:
   :undoc-members:
   :show-inheritance:

Workflow Runner
---------------

.. automodule:: fusion.modules.rl.workflow_runner
   :members:
   :undoc-members:
   :show-inheritance:

RL Agents
---------

.. automodule:: fusion.modules.rl.agents
   :members:
   :undoc-members:
   :show-inheritance:

Base Agent
^^^^^^^^^^

.. automodule:: fusion.modules.rl.agents.base_agent
   :members:
   :undoc-members:
   :show-inheritance:

Path Agent
^^^^^^^^^^

.. automodule:: fusion.modules.rl.agents.path_agent
   :members:
   :undoc-members:
   :show-inheritance:

Core Agent
^^^^^^^^^^

.. automodule:: fusion.modules.rl.agents.core_agent
   :members:
   :undoc-members:
   :show-inheritance:

Spectrum Agent
^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.agents.spectrum_agent
   :members:
   :undoc-members:
   :show-inheritance:

RL Algorithms
-------------

.. automodule:: fusion.modules.rl.algorithms
   :members:
   :undoc-members:
   :show-inheritance:

Algorithm Properties
^^^^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.algorithms.algorithm_props
   :members:
   :undoc-members:
   :show-inheritance:

Bandits
^^^^^^^

.. automodule:: fusion.modules.rl.algorithms.bandits
   :members:
   :undoc-members:
   :show-inheritance:

Q-Learning
^^^^^^^^^^

.. automodule:: fusion.modules.rl.algorithms.q_learning
   :members:
   :undoc-members:
   :show-inheritance:

Base Deep RL
^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.algorithms.base_drl
   :members:
   :undoc-members:
   :show-inheritance:

DQN
^^^

.. automodule:: fusion.modules.rl.algorithms.dqn
   :members:
   :undoc-members:
   :show-inheritance:

QR-DQN
^^^^^^

.. automodule:: fusion.modules.rl.algorithms.qr_dqn
   :members:
   :undoc-members:
   :show-inheritance:

A2C
^^^

.. automodule:: fusion.modules.rl.algorithms.a2c
   :members:
   :undoc-members:
   :show-inheritance:

PPO
^^^

.. automodule:: fusion.modules.rl.algorithms.ppo
   :members:
   :undoc-members:
   :show-inheritance:

Algorithm Persistence
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.algorithms.persistence
   :members:
   :undoc-members:
   :show-inheritance:

Gymnasium Environments
----------------------

.. automodule:: fusion.modules.rl.gymnasium_envs
   :members:
   :undoc-members:
   :show-inheritance:

General Simulation Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.gymnasium_envs.general_sim_env
   :members:
   :undoc-members:
   :show-inheritance:

Feature Extractors
------------------

.. automodule:: fusion.modules.rl.feat_extrs
   :members:
   :undoc-members:
   :show-inheritance:

Base Feature Extractor
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.feat_extrs.base_feature_extractor
   :members:
   :undoc-members:
   :show-inheritance:

Path GNN
^^^^^^^^

.. automodule:: fusion.modules.rl.feat_extrs.path_gnn
   :members:
   :undoc-members:
   :show-inheritance:

Cached Path GNN
^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.feat_extrs.path_gnn_cached
   :members:
   :undoc-members:
   :show-inheritance:

Graphormer
^^^^^^^^^^

.. automodule:: fusion.modules.rl.feat_extrs.graphormer
   :members:
   :undoc-members:
   :show-inheritance:

Policies
--------

.. automodule:: fusion.modules.rl.policies
   :members:
   :undoc-members:
   :show-inheritance:

Pointer Policy
^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.policies.pointer_policy
   :members:
   :undoc-members:
   :show-inheritance:

RL Utilities
------------

Callbacks
^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Deep RL Utilities
^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.deep_rl
   :members:
   :undoc-members:
   :show-inheritance:

General Utilities
^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.general_utils
   :members:
   :undoc-members:
   :show-inheritance:

Gym Environments
^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.gym_envs
   :members:
   :undoc-members:
   :show-inheritance:

Hyperparameters
^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.hyperparams
   :members:
   :undoc-members:
   :show-inheritance:

Observation Space
^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.observation_space
   :members:
   :undoc-members:
   :show-inheritance:

RL Zoo Integration
^^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.rl_zoo
   :members:
   :undoc-members:
   :show-inheritance:

Simulation Data
^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.sim_data
   :members:
   :undoc-members:
   :show-inheritance:

Simulation Environment
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.sim_env
   :members:
   :undoc-members:
   :show-inheritance:

Simulation Filters
^^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.sim_filters
   :members:
   :undoc-members:
   :show-inheritance:

Topology Utilities
^^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.topology
   :members:
   :undoc-members:
   :show-inheritance:

GNN Caching
^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.cache_gnn_once
   :members:
   :undoc-members:
   :show-inheritance:

Unity Hyperparameters
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.utils.unity_hyperparams
   :members:
   :undoc-members:
   :show-inheritance:

RL Visualization
----------------

.. automodule:: fusion.modules.rl.visualization
   :members:
   :undoc-members:
   :show-inheritance:

RL Metrics
^^^^^^^^^^

.. automodule:: fusion.modules.rl.visualization.rl_metrics
   :members:
   :undoc-members:
   :show-inheritance:

RL Plots
^^^^^^^^

.. automodule:: fusion.modules.rl.visualization.rl_plots
   :members:
   :undoc-members:
   :show-inheritance:

RL Plot Plugin
^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.visualization.rl_plugin
   :members:
   :undoc-members:
   :show-inheritance:

RL Processors
^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.visualization.rl_processors
   :members:
   :undoc-members:
   :show-inheritance:

Stable-Baselines3 Integration
------------------------------

.. automodule:: fusion.modules.rl.sb3
   :members:
   :undoc-members:
   :show-inheritance:

Environment Registration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fusion.modules.rl.sb3.register_env
   :members:
   :undoc-members:
   :show-inheritance:

RL Registry
-----------

.. automodule:: fusion.modules.rl.registry
   :members:
   :undoc-members:
   :show-inheritance:

RL Errors
---------

.. automodule:: fusion.modules.rl.errors
   :members:
   :undoc-members:
   :show-inheritance:
