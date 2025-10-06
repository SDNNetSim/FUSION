==========
User Guide
==========

Comprehensive guides for using FUSION's full capabilities, from running simulations to deploying advanced ML/RL agents.

Overview
========

The User Guide provides in-depth documentation for all FUSION features and workflows. Whether you're running your first simulation or training state-of-the-art reinforcement learning agents, this guide has you covered.

What You'll Learn
=================

This guide covers:

- **Running Simulations**: Execute simulations with various algorithms and configurations
- **Machine Learning**: Train and deploy ML models for intelligent network optimization
- **Reinforcement Learning**: Build RL agents that learn optimal control policies
- **Visualization**: Create publication-quality plots and analyze results
- **CLI Tools**: Master the command-line interface for all operations
- **Configuration**: Understand all configuration options and best practices
- **Data Management**: Organize, version, and share simulation data and artifacts

Who Should Read This
====================

**Researchers**: Learn to design experiments, run simulations, and analyze results

**ML/RL Practitioners**: Understand how to integrate learning algorithms into optical networks

**Network Engineers**: Use FUSION to evaluate and optimize network designs

**Students**: Follow comprehensive guides from basics to advanced topics

Guide Structure
===============

The User Guide is organized into focused sections:

Running Simulations
-------------------

.. toctree::
   :maxdepth: 1

   running_simulations

Complete guide to executing simulations:

- Basic simulation workflows
- Algorithm selection (routing, spectrum allocation)
- Network topology configuration
- Performance optimization and troubleshooting
- Snapshot mode for detailed analysis
- Batch processing for large-scale experiments

**Start here if you want to:** Run optical network simulations and evaluate algorithm performance

Machine Learning
----------------

.. toctree::
   :maxdepth: 1

   machine_learning

Integrate supervised learning into FUSION:

- ML workflow: data collection → training → deployment
- Supported algorithms (Decision Trees, Random Forests, Gradient Boosting, SVM)
- Feature engineering for optical networks
- Model evaluation and validation
- Hyperparameter tuning
- Production deployment best practices

**Start here if you want to:** Use ML to make smarter routing and allocation decisions

Reinforcement Learning
----------------------

.. toctree::
   :maxdepth: 1

   reinforcement_learning

Train autonomous RL agents:

- RL algorithms (PPO, A2C, DQN, QR-DQN, Bandits)
- Agent types: path selection, spectrum allocation, core selection
- Observation space design
- Reward engineering
- Training with Stable-Baselines3
- GNN-based feature extractors (PathGNN, Graphormer)
- Transfer learning and fine-tuning

**Start here if you want to:** Develop self-learning network control agents

Visualization
-------------

.. toctree::
   :maxdepth: 1

   visualization

Analyze and present results:

- Plotting blocking probability, utilization, and performance metrics
- Network topology visualization
- Spectrum heatmaps and fragmentation analysis
- ML/RL training curves and feature importance
- Publication-quality figure generation
- Interactive plots with Plotly

**Start here if you want to:** Create plots for papers, presentations, or analysis

CLI Reference
-------------

.. toctree::
   :maxdepth: 1

   cli_reference

Complete command-line interface documentation:

- ``fusion-sim``: Run simulations
- ``fusion-train``: Train ML/RL agents
- ``fusion-evaluate``: Evaluate models
- ``fusion-plot``: Generate visualizations
- ``fusion-gui``: Launch graphical interface
- Command-line arguments and options
- Environment variables and shell completion

**Start here if you want to:** Master FUSION's command-line tools

Configuration Reference
-----------------------

.. toctree::
   :maxdepth: 1

   configuration_reference

Complete configuration options reference:

- All configuration parameters with types and defaults
- Section-by-section documentation
- Example configurations for common scenarios
- Best practices for configuration management
- Validation and error handling

**Start here if you want to:** Understand all available configuration options

Data Management
---------------

.. toctree::
   :maxdepth: 1

   data_management

Manage simulation data and artifacts:

- Directory structure and organization
- Result file formats (JSON, CSV, HDF5)
- Network snapshots and state captures
- ML/RL training data and model storage
- Data versioning with Git/DVC
- Backup strategies and storage optimization

**Start here if you want to:** Organize and manage simulation outputs and models

Quick Navigation
================

Common Tasks
------------

**I want to run my first simulation:**
   1. :doc:`../getting_started/quickstart`
   2. :doc:`running_simulations`

**I want to train an ML model:**
   1. :doc:`machine_learning` (Data Collection section)
   2. :doc:`machine_learning` (Training section)
   3. :doc:`machine_learning` (Deployment section)

**I want to train an RL agent:**
   1. :doc:`reinforcement_learning` (RL Workflow section)
   2. :doc:`reinforcement_learning` (Training Configuration section)

**I want to create plots for my paper:**
   1. :doc:`running_simulations` (generate data)
   2. :doc:`visualization` (create plots)

**I need help with configuration:**
   1. :doc:`configuration_reference` (find the option)
   2. :doc:`../getting_started/configuration` (basic tutorial)

**I need CLI help:**
   1. :doc:`cli_reference` (complete command reference)

By Topic
--------

**Optical Networking:**
   - :doc:`../concepts/optical_networking_basics`
   - :doc:`../concepts/flex_grid_networks`
   - :doc:`../concepts/modulation_formats`

**Algorithms:**
   - :doc:`../concepts/resource_allocation` (RSA/RMCSA algorithms)
   - :doc:`machine_learning` (ML algorithms)
   - :doc:`reinforcement_learning` (RL algorithms)

**Advanced Topics:**
   - :doc:`reinforcement_learning` (GNN feature extractors)
   - :doc:`machine_learning` (Hyperparameter optimization)
   - :doc:`visualization` (Custom plots)

Next Steps
==========

After mastering the User Guide, explore:

- :doc:`../examples/index` - Step-by-step tutorials and examples
- :doc:`../api/index` - API documentation for programmatic usage
- :doc:`../developer/index` - Contribute to FUSION or extend functionality

Getting Help
============

If you encounter issues:

1. Check the **Common Issues** section in each guide
2. Review :doc:`../reference/faq`
3. Search :doc:`../reference/troubleshooting`
4. Open an issue on GitHub

The User Guide is continuously updated. Feedback and contributions are welcome!