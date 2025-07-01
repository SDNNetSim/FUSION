Registry Arguments
===================


This module provides the registry that maps reinforcement learning algorithm names to
their setup and implementation classes.

.. list-table:: ALGORITHM_REGISTRY Structure
   :header-rows: 1

   * - Algorithm Key
     - Setup Function
     - Load Function
     - Class Reference

   * - ``ppo``
     - ``setup_ppo``
     - ``None``
     - ``PPO``

   * - ``a2c``
     - ``setup_a2c``
     - ``None``
     - ``A2C``

   * - ``dqn``
     - ``setup_dqn``
     - ``None``
     - ``DQN``

   * - ``qr_dqn``
     - ``setup_qr_dqn``
     - ``None``
     - ``QrDQN``
