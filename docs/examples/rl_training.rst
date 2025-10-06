============================
Reinforcement Learning Training  
============================

Train RL agents to learn optimal routing and spectrum assignment policies.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

Train deep RL agents (DQN, PPO, A2C) using Stable-Baselines3 for intelligent
network resource allocation.

Quick Start
===========

.. code-block:: bash

   # Train DQN agent for path selection
   fusion-train --algorithm dqn --task path_selection --timesteps 100000

   # Evaluate trained agent
   fusion-eval --model trained_models/dqn_model.zip

Configuration
=============

RL Training Setup
-----------------

.. code-block:: ini

   [rl_settings]
   is_training = True
   path_algorithm = dqn
   n_trials = 1
   total_timesteps = 100000

   # Environment
   obs_space = obs_3
   super_channel_space = 3
   render_mode = None

   # Hyperparameters
   alpha_start = 0.0003
   gamma = 0.99
   epsilon_start = 1.0
   epsilon_end = 0.01

   # Feature extraction
   feature_extractor = path_gnn
   gnn_type = graph_conv

Training Workflow
=================

Step 1: Train Agent
-------------------

.. code-block:: python

   from fusion.modules.rl.workflow_runner import train_rl_agent
   from fusion.configs.config import Config

   config = Config.from_file('rl_config.ini')
   results = train_rl_agent(config)

   print(f"Training complete! Model saved to {results.model_path}")

Step 2: Evaluate Performance
-----------------------------

.. code-block:: python

   from fusion.modules.rl.workflow_runner import evaluate_rl_agent

   metrics = evaluate_rl_agent(
       model_path='trained_models/dqn_model.zip',
       config=config,
       num_episodes=100
   )

   print(f"Mean blocking: {metrics['mean_blocking']:.2%}")
   print(f"Mean reward: {metrics['mean_reward']:.2f}")

Available Algorithms
====================

* **DQN**: Deep Q-Network (discrete actions)
* **PPO**: Proximal Policy Optimization
* **A2C**: Advantage Actor-Critic
* **Q-Learning**: Tabular Q-learning
* **Bandits**: Multi-armed bandits

See Also
========

* :doc:`../user_guide/reinforcement_learning` - Complete RL guide
* :doc:`../api/modules` - RL module API reference
