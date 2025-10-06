========================
Reinforcement Learning
========================

This guide covers training and deploying reinforcement learning (RL) agents in FUSION for autonomous network optimization and resource allocation.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

FUSION's RL module enables training intelligent agents that learn optimal network control policies through interaction with the simulation environment. RL agents can outperform traditional heuristics by learning from experience and adapting to complex network dynamics.

What is Reinforcement Learning?
--------------------------------

Reinforcement learning is a machine learning paradigm where an agent learns to make decisions by:

1. **Observing** the current network state
2. **Taking actions** (routing, spectrum allocation decisions)
3. **Receiving rewards** based on blocking probability and resource utilization
4. **Learning** which actions lead to better long-term outcomes

Unlike supervised ML, RL doesn't require labeled training data - agents learn through trial and error.

Why Use RL for Optical Networks?
---------------------------------

**Adaptive Decision-Making**:
   RL agents adapt to changing traffic patterns and network conditions

**Multi-Objective Optimization**:
   Balance competing objectives: minimize blocking, optimize spectrum usage, reduce fragmentation

**End-to-End Learning**:
   Learn complete routing and spectrum allocation policies without hand-crafted heuristics

**State-of-the-Art Performance**:
   RL agents can achieve 20-40% lower blocking probability than traditional algorithms

Supported RL Algorithms
========================

Deep Reinforcement Learning
----------------------------

FUSION integrates with Stable-Baselines3 for state-of-the-art deep RL algorithms:

**PPO (Proximal Policy Optimization)**:
   Stable, robust policy gradient method - **recommended for beginners**

   .. code-block:: bash

      fusion-train --agent_type rl --algorithm ppo --n_steps 2048

**A2C (Advantage Actor-Critic)**:
   Fast, synchronous actor-critic algorithm

   .. code-block:: bash

      fusion-train --agent_type rl --algorithm a2c

**DQN (Deep Q-Network)**:
   Value-based method for discrete action spaces

   .. code-block:: bash

      fusion-train --agent_type rl --algorithm dqn --buffer_size 100000

**QR-DQN (Quantile Regression DQN)**:
   Distributional RL variant of DQN for better value estimation

   .. code-block:: bash

      fusion-train --agent_type rl --algorithm qr_dqn

Classical RL Methods
--------------------

FUSION also supports classical RL algorithms:

**Q-Learning**:
   Tabular method for small state/action spaces

**Multi-Armed Bandits**:
   - Epsilon-Greedy
   - UCB (Upper Confidence Bound)
   - Thompson Sampling

   .. code-block:: bash

      fusion-train --agent_type rl --algorithm epsilon_greedy --epsilon 0.1

RL Workflow
===========

Basic Training Workflow
-----------------------

**1. Configure RL Training**

Create a configuration file with RL settings:

.. code-block:: ini

   [general_settings]
   erlang_start = 400
   erlang_stop = 400
   num_requests = 10000

   [rl_settings]
   # Agent configuration
   agent_type = path  # path, spectrum, or core
   algorithm = ppo

   # Training parameters
   n_timesteps = 500000
   n_steps = 2048
   batch_size = 64
   learning_rate = 0.0003

   # Observation space
   obs_space = path_based  # or spectrum_based, hybrid

   # Device
   device = cuda  # or cpu

**2. Train the Agent**

.. code-block:: bash

   fusion-train --agent_type rl --config rl_training.ini

Training runs for the specified timesteps, saving checkpoints periodically.

**3. Evaluate the Agent**

.. code-block:: bash

   fusion-evaluate --agent_path logs/rl_models/ppo_agent.zip \
                   --config eval_config.ini \
                   --n_eval_episodes 100

**4. Deploy in Production**

.. code-block:: ini

   [rl_settings]
   deploy_model = True
   agent_path = logs/rl_models/ppo_agent_best.zip

.. code-block:: bash

   fusion-sim --config production.ini

RL Agents and Tasks
===================

Path Selection Agent
--------------------

Learns to select the best path from K candidate routes:

.. code-block:: ini

   [rl_settings]
   agent_type = path
   obs_space = path_based

**Observation Space**:
   - Path lengths
   - Link congestion on each path
   - Available spectrum on each path
   - Path diversity metrics

**Action Space**:
   Discrete choice among K candidate paths (typically K=3-10)

**Reward**:
   - +1 for successful allocation
   - -1 for blocked request
   - Bonus for low fragmentation

Spectrum Allocation Agent
--------------------------

Learns intelligent spectrum slot assignment:

.. code-block:: ini

   [rl_settings]
   agent_type = spectrum
   obs_space = spectrum_based

**Observation Space**:
   - Spectrum occupancy matrix
   - Fragmentation metrics
   - Request bandwidth requirements

**Action Space**:
   Choose starting slot index for allocation

**Reward**:
   - Success/failure
   - Fragmentation penalty
   - Spectrum efficiency bonus

Core Selection Agent
--------------------

For multi-core fiber networks, selects optimal fiber core:

.. code-block:: ini

   [rl_settings]
   agent_type = core
   obs_space = core_based

   [topology_settings]
   cores_per_link = 4

**Observation Space**:
   - Per-core spectrum utilization
   - Inter-core crosstalk levels
   - Core congestion metrics

**Action Space**:
   Select which core to use (0 to cores_per_link-1)

Observation Space Configuration
================================

Types of Observations
---------------------

**Path-Based Observations**:
   Focus on routing decisions

   .. code-block:: python

      obs = {
          'path_lengths': [450.2, 680.5, 920.1],  # km
          'path_congestion': [0.35, 0.67, 0.89],  # utilization
          'available_slots': [150, 80, 45],  # free slots
      }

**Spectrum-Based Observations**:
   Focus on allocation decisions

   .. code-block:: python

      obs = {
          'spectrum_state': np.array([...]),  # 2D occupancy matrix
          'fragmentation_index': 0.42,
          'request_bandwidth': 400.0  # Gbps
      }

**Hybrid Observations**:
   Combine multiple aspects for joint optimization

Graph Neural Network Features
------------------------------

FUSION supports advanced GNN-based feature extraction:

.. code-block:: ini

   [rl_settings]
   feature_extractor = path_gnn  # or graphormer

**Path GNN**:
   Graph convolutional network that processes network topology and state

**Graphormer**:
   Transformer-based graph representation learning

.. code-block:: python

   from fusion.modules.rl.feat_extrs.path_gnn import PathGNN

   feature_extractor = PathGNN(
       node_features=16,
       edge_features=8,
       hidden_dim=128,
       num_layers=3
   )

Training Configuration
======================

Hyperparameters
---------------

**Learning Rate**:
   Controls training speed and stability

   .. code-block:: ini

      learning_rate = 0.0003  # Typical range: 1e-5 to 1e-3

**Batch Size**:
   Number of samples per update

   .. code-block:: ini

      batch_size = 64  # Typical: 32-256

**Number of Steps** (PPO/A2C):
   Rollout length before policy update

   .. code-block:: ini

      n_steps = 2048  # Typical: 128-4096

**Replay Buffer Size** (DQN):
   Memory size for experience replay

   .. code-block:: ini

      buffer_size = 100000  # Typical: 10k-1M

**Discount Factor (Gamma)**:
   How much to value future rewards

   .. code-block:: ini

      gamma = 0.99  # Range: 0.9-0.999

**Entropy Coefficient**:
   Encourages exploration

   .. code-block:: ini

      ent_coef = 0.01  # Range: 0.0-0.1

Advanced Training Options
-------------------------

**Curriculum Learning**:
   Gradually increase task difficulty

   .. code-block:: ini

      [rl_settings]
      curriculum_learning = True
      erlang_schedule = [300, 400, 500, 600]  # Increasing load

**Hyperparameter Optimization**:
   Automatic tuning with Optuna

   .. code-block:: ini

      [rl_settings]
      optimize_hyperparameters = True
      optuna_trials = 100

   .. code-block:: bash

      fusion-train --agent_type rl \
                   --config config.ini \
                   --optimize_hyperparameters \
                   --n_trials 100

**Multi-Processing**:
   Parallel environment rollouts for faster training

   .. code-block:: ini

      [rl_settings]
      n_envs = 8  # Number of parallel environments

**Reward Shaping**:
   Custom reward functions

   .. code-block:: python

      def custom_reward(request_blocked, spectrum_efficiency, path_length):
          reward = -10.0 if request_blocked else 1.0
          reward += 0.1 * spectrum_efficiency
          reward -= 0.01 * path_length
          return reward

Monitoring Training
===================

TensorBoard Integration
-----------------------

FUSION automatically logs training metrics to TensorBoard:

.. code-block:: bash

   tensorboard --logdir logs/rl_tensorboard/

Monitor:
   - Episode rewards
   - Success rate
   - Policy loss
   - Value function loss
   - Learning rate schedule
   - Entropy

Custom Callbacks
----------------

Track custom metrics during training:

.. code-block:: python

   from fusion.modules.rl.utils.callbacks import EpisodicRewardCallback

   # Automatically tracks:
   # - Mean episode reward
   # - Blocking probability per episode
   # - Spectrum utilization

   callback = EpisodicRewardCallback(verbose=1)

Checkpointing
-------------

Save agent checkpoints during training:

.. code-block:: ini

   [rl_settings]
   save_freq = 10000  # Save every 10k steps
   checkpoint_dir = logs/rl_checkpoints/

Best model is automatically saved based on validation performance.

Evaluation and Deployment
==========================

Evaluating Trained Agents
--------------------------

**Offline Evaluation**:
   Test on validation scenarios

   .. code-block:: bash

      fusion-evaluate --agent_path logs/ppo_agent.zip \
                      --config validation.ini \
                      --n_eval_episodes 100

**Online Evaluation**:
   Deploy in full simulation

   .. code-block:: bash

      fusion-sim --config production.ini

**A/B Testing**:
   Compare multiple agents

   .. code-block:: python

      from fusion.modules.rl.evaluation import compare_agents

      results = compare_agents(
          agents=[
              'logs/ppo_agent.zip',
              'logs/dqn_agent.zip',
              'baseline_heuristic'
          ],
          config='test_config.ini',
          n_episodes=200
      )

Model Persistence
-----------------

Save and load trained agents:

.. code-block:: python

   from stable_baselines3 import PPO

   # Save
   model = PPO(policy="MlpPolicy", env=env)
   model.learn(total_timesteps=500000)
   model.save("logs/my_ppo_agent")

   # Load
   loaded_model = PPO.load("logs/my_ppo_agent")

Transfer Learning
-----------------

Fine-tune agents on new topologies or traffic patterns:

.. code-block:: python

   # Load pre-trained agent
   agent = PPO.load("logs/nsfnet_ppo_agent")

   # Continue training on new topology
   new_env = create_environment(config_path="cost239_config.ini")
   agent.set_env(new_env)
   agent.learn(total_timesteps=100000)  # Fine-tuning

Advanced RL Techniques
======================

Policy Architectures
--------------------

**MLP Policy** (default):
   Fully-connected neural network

   .. code-block:: python

      policy = "MlpPolicy"

**CNN Policy**:
   For image-like spectrum representations

   .. code-block:: python

      policy = "CnnPolicy"

**Multi-Input Policy**:
   Handle heterogeneous observations

   .. code-block:: python

      from fusion.modules.rl.policies import MultiInputPolicy

**Pointer Networks**:
   Attention-based policy for variable-length action spaces

   .. code-block:: python

      from fusion.modules.rl.policies.pointer_policy import PointerPolicy

Reward Engineering
------------------

Design effective reward signals:

**Sparse vs Dense Rewards**:

   .. code-block:: python

      # Sparse: Simple but slower learning
      reward = 1.0 if success else -1.0

      # Dense: Faster learning with shaped rewards
      reward = (
          success_bonus
          + spectrum_efficiency_bonus
          - fragmentation_penalty
          - path_length_penalty
      )

**Normalize Rewards**:
   Keep rewards in reasonable range

   .. code-block:: python

      from stable_baselines3.common.vec_env import VecNormalize

      env = VecNormalize(env, norm_reward=True)

Multi-Agent RL
--------------

Train multiple agents for different network regions:

.. code-block:: python

   from fusion.modules.rl.multi_agent import MultiAgentTrainer

   trainer = MultiAgentTrainer(
       num_agents=4,
       agent_class=PPO,
       network_partition='regional'
   )

   trainer.train(total_timesteps=1000000)

Visualization and Analysis
===========================

RL Performance Plots
--------------------

.. code-block:: python

   from fusion.modules.rl.visualization.rl_plots import plot_training_progress

   plot_training_progress(
       log_dir='logs/rl_tensorboard/',
       save_path='plots/training_curves.png'
   )

Generates:
   - Reward curves
   - Success rate over time
   - Loss curves
   - Exploration metrics

Policy Visualization
--------------------

Understand agent decisions:

.. code-block:: python

   from fusion.modules.rl.visualization import visualize_policy_decisions

   visualize_policy_decisions(
       agent=trained_agent,
       env=test_env,
       n_episodes=10,
       save_path='plots/policy_heatmap.png'
   )

Action Distribution Analysis
----------------------------

Analyze which actions the agent prefers:

.. code-block:: python

   from fusion.modules.rl.visualization.rl_metrics import analyze_action_distribution

   action_stats = analyze_action_distribution(
       agent=agent,
       env=env,
       n_steps=10000
   )

   print(f"Path selection distribution: {action_stats['path_distribution']}")

Best Practices
==============

Training Strategy
-----------------

1. **Start Simple**: Begin with PPO and MLP policy
2. **Tune Hyperparameters**: Use Optuna for systematic tuning
3. **Validate Regularly**: Track performance on held-out scenarios
4. **Curriculum Learning**: Gradually increase difficulty
5. **Ensemble Agents**: Combine multiple trained agents

Computational Efficiency
------------------------

**GPU Acceleration**:
   Use CUDA for faster training

   .. code-block:: ini

      device = cuda

**Vectorized Environments**:
   Parallelize experience collection

   .. code-block:: ini

      n_envs = 16

**GNN Feature Caching**:
   Pre-compute graph embeddings

   .. code-block:: ini

      cache_gnn_features = True

Debugging RL Training
---------------------

**Agent Not Learning**:
   - Reduce learning rate
   - Simplify observation space
   - Check reward scaling
   - Increase exploration (higher entropy coefficient)

**Unstable Training**:
   - Reduce learning rate
   - Increase batch size
   - Use gradient clipping
   - Normalize observations and rewards

**Slow Convergence**:
   - Increase learning rate (carefully)
   - Use shaped rewards
   - Employ curriculum learning
   - Tune network architecture

Common Issues
=============

Import Errors
-------------

**Error**: ``ModuleNotFoundError: No module named 'stable_baselines3'``

**Solution**:

.. code-block:: bash

   pip install stable-baselines3[extra]

**Error**: ``ModuleNotFoundError: No module named 'torch'``

**Solution**:

.. code-block:: bash

   pip install torch  # Or with CUDA: pip install torch --index-url ...

Out of Memory
-------------

**GPU OOM during training**:

- Reduce batch size
- Reduce n_steps
- Reduce network size
- Use CPU training
- Enable gradient checkpointing

**System RAM exhaustion**:

- Reduce replay buffer size (DQN)
- Reduce n_envs
- Disable feature caching

Poor Agent Performance
----------------------

**Agent performs worse than baseline**:

- Train longer (more timesteps)
- Tune hyperparameters
- Improve reward function
- Enrich observation space
- Try different algorithms

**Agent overfits to training scenarios**:

- Increase training diversity
- Add noise to observations
- Use domain randomization
- Validate on multiple topologies

Integration Examples
====================

Custom RL Environment
---------------------

Create a custom gym environment:

.. code-block:: python

   from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv

   class CustomOpticalEnv(SimEnv):
       def __init__(self, custom_callback, sim_dict):
           super().__init__(
               render_mode=None,
               custom_callback=custom_callback,
               sim_dict=sim_dict
           )

       def _calculate_reward(self, blocked, spectrum_efficiency):
           # Custom reward logic
           reward = -10.0 if blocked else 5.0
           reward += 2.0 * spectrum_efficiency
           return reward

Custom Feature Extractor
-------------------------

Implement domain-specific features:

.. code-block:: python

   from fusion.modules.rl.feat_extrs.base_feature_extractor import (
       BaseFeatureExtractor
   )

   class CustomFeatureExtractor(BaseFeatureExtractor):
       def forward(self, observations):
           # Extract custom features
           features = self.extract_topology_features(observations)
           features = self.gnn(features)
           return features

Next Steps
==========

- :doc:`../examples/rl_training` - Complete RL training walkthrough
- :doc:`../concepts/machine_learning_optical` - RL theory for optical networks
- :doc:`visualization` - Visualize RL training and agent behavior
- :doc:`machine_learning` - Combine RL with supervised ML

See Also
========

* :doc:`running_simulations` - Deploy trained RL agents
* :doc:`configuration_reference` - Complete RL configuration options
* :doc:`../developer/extending` - Implement custom RL algorithms
* :doc:`data_management` - Manage RL training artifacts
