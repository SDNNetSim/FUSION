General Arguments
==================

This module defines the lists of valid algorithm names used for policy selection
in path and core assignment, as well as supported decay strategies.

.. list-table:: Valid Algorithm Settings
   :header-rows: 1

   * - Variable
     - Description
     - Values

   * - EPISODIC_STRATEGIES
     - Supported decay methods applied episodically.
     - ``['exp_decay', 'linear_decay']``

   * - VALID_PATH_ALGORITHMS
     - Algorithms usable by path agents.
     - ``['q_learning', 'epsilon_greedy_bandit', 'ucb_bandit', 'ppo', 'a2c', 'dqn', 'qr_dqn']``

   * - VALID_CORE_ALGORITHMS
     - Algorithms usable by core agents.
     - ``['q_learning', 'epsilon_greedy_bandit', 'ucb_bandit']``

   * - VALID_DRL_ALGORITHMS
     - Deep reinforcement learning algorithms (SB3-based).
     - ``['ppo', 'a2c', 'dqn', 'qr_dqn']``