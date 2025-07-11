Agents
==============

This module outlines the brain of our reinforcement learning system — the agents.
Each agent encapsulates a unique decision-making strategy for routing, core selection,
or spectrum assignment. We provide a common base interface, ``BaseAgent``, which is extended
by bandit agents, tabular Q-learning agents, and deep RL agents built on top of Stable Baselines3.
Whether it’s choosing the shortest path or optimizing over a GNN-encoded state space,
this module defines how the system thinks

.. toctree::

    base_agent
    core_agent
    path_agent
    spectrum_agent
