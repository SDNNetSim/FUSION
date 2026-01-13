====================================
FUSION Documentation
====================================

.. image:: https://img.shields.io/badge/version-6.0.0-blue.svg
   :alt: Version

.. image:: https://img.shields.io/badge/python-3.11-blue.svg
   :alt: Python Version

**Flexible Unified System for Intelligent Optical Networking**

Welcome to FUSION, an open-source simulation framework for **Software Defined Elastic Optical Networks (SD-EONs)**.
FUSION integrates artificial intelligence to enhance network optimization, performance, and decision-making processes,
paving the way for next-generation optical networking research.

.. note::
   **New to FUSION?** Start with our :doc:`getting_started/quickstart` guide to run your first simulation in minutes!

Key Features
============

* **Flexible Network Simulation**: Simulate elastic optical networks with flex-grid spectrum allocation
* **AI-Powered Optimization**: Integration with machine learning and reinforcement learning algorithms
* **Software Defined Networking**: SDN controller for dynamic network management
* **Multiple Algorithms**: Routing, spectrum assignment, modulation selection, and core allocation
* **Visualization Tools**: Comprehensive plotting and analysis capabilities
* **Extensible Architecture**: Easy integration of custom algorithms and network topologies
* **Performance Optimized**: Parallel execution support for large-scale simulations

Quick Links
===========

* :doc:`getting_started/installation` - Get FUSION installed on your system
* :doc:`getting_started/quickstart` - Run your first simulation
* :doc:`user_guide/running_simulations` - Learn how to use FUSION
* :doc:`concepts/optical_networking_basics` - Understand the fundamentals
* :doc:`examples/basic_simulation` - See practical examples
* :doc:`api/index` - Explore the API reference
* :doc:`developer/contributing` - Contribute to the project

What's New in v6.0
==================

* **Modernized Architecture**: Restructured codebase with improved modularity
* **Enhanced RL Support**: Integration with Stable-Baselines3 and Gymnasium
* **Improved Visualization**: New plotting system with plugin architecture
* **Better Testing**: Comprehensive test coverage with pytest
* **Type Safety**: Full mypy type checking support
* **Developer Tools**: Pre-commit hooks, automated linting, and code quality checks

Citation
========

If you use FUSION in your research, please cite:

.. code-block:: bibtex

   @INPROCEEDINGS{10898199,
     author={McCann, Ryan and Rezaee, Arash and Vokkarane, Vinod M.},
     booktitle={2024 IEEE International Conference on Advanced Networks and
                Telecommunications Systems (ANTS)},
     title={FUSION: A Flexible Unified Simulator for Intelligent Optical Networking},
     year={2024},
     pages={1-6},
     doi={10.1109/ANTS63515.2024.10898199}
   }

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/index
   getting_started/installation
   getting_started/quickstart
   getting_started/first_simulation
   getting_started/configuration
   getting_started/next_steps

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/running_simulations
   user_guide/machine_learning
   user_guide/reinforcement_learning
   user_guide/visualization
   user_guide/cli_reference
   user_guide/configuration_reference
   user_guide/data_management

.. toctree::
   :maxdepth: 2
   :caption: Concepts & Theory

   concepts/index
   concepts/optical_networking_basics
   concepts/wdm_vs_eon
   concepts/flex_grid_networks
   concepts/sdn_overview
   concepts/resource_allocation
   concepts/machine_learning_optical
   concepts/network_topologies
   concepts/modulation_formats

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials

   examples/index
   examples/basic_simulation
   examples/custom_topology
   examples/ml_decision_tree
   examples/rl_training
   examples/advanced_visualization
   examples/batch_simulations
   examples/performance_optimization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/core
   api/modules
   api/sim
   api/configs
   api/cli
   api/io
   api/utils
   api/interfaces
   api/analysis
   api/reporting
   api/visualization
   api/unity

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/index
   developer/contributing
   developer/development_setup
   developer/workflow
   developer/testing
   developer/coding_standards
   developer/style_guide
   developer/architecture
   developer/extending
   developer/release_process

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index
   reference/bibliography
   reference/glossary
   reference/faq
   reference/troubleshooting
   reference/helpful_links

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Community & Support
===================

* **GitHub**: `SDNNetSim/FUSION <https://github.com/SDNNetSim/FUSION>`_
* **Issues**: `Report a bug or request a feature <https://github.com/SDNNetSim/FUSION/issues>`_
* **Contributing**: See our :doc:`developer/contributing` guide

License
=======

FUSION is released under the MIT License. See the LICENSE file in the repository for details.

Acknowledgments
===============

This project is brought to you by the efforts of:

* **Arash Rezaee** - Networking Systems Lead and Co-Founder
* **Ryan McCann** - Technical Architect, AI Lead, and Co-Founder
* **Vinod M. Vokkarane** - Principal Investigator

We welcome contributions from the community to help make this project even better!

.. note::
   This documentation is for FUSION version 6.0.0. If you're looking for documentation
   for an older version, please visit the appropriate branch in the GitHub repository.
