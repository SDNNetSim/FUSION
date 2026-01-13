===
FAQ
===

Frequently Asked Questions about FUSION.

.. contents:: Table of Contents
   :local:
   :depth: 2

General Questions
=================

What is FUSION?
---------------

FUSION (Flexible Unified System for Intelligent Optical Networking) is an open-source
simulation framework for Software-Defined Elastic Optical Networks with integrated AI capabilities.

Who should use FUSION?
-----------------------

- Researchers studying optical network algorithms
- Graduate students learning about EONs and SDM
- Network engineers evaluating design decisions
- Anyone interested in optical networking simulation

What can FUSION simulate?
--------------------------

- Elastic optical networks (EONs) with flex-grid spectrum
- Space-division multiplexing (SDM) with multicore fibers
- Machine learning-based network optimization
- Reinforcement learning agent training
- Dynamic traffic scenarios

Installation & Setup
====================

What are the system requirements?
----------------------------------

- Python 3.10 or higher
- 4GB+ RAM (8GB+ recommended for RL training)
- Linux, macOS, or Windows

Do I need a GPU?
----------------

No, but GPU acceleration speeds up RL training significantly. FUSION supports CUDA if available.

How do I install FUSION?
-------------------------

.. code-block:: bash

   git clone https://github.com/SDNNetSim/FUSION.git
   cd FUSION
   pip install -e .

See :doc:`../getting_started/installation` for details.

Configuration
=============

How do I create a configuration file?
--------------------------------------

Use the provided templates:

.. code-block:: bash

   cp fusion/configs/templates/minimal.ini my_config.ini

Or see :doc:`../getting_started/configuration` for a complete guide.

Can I use YAML or TOML instead of INI?
---------------------------------------

Currently, FUSION uses INI format. YAML/TOML support is planned for future releases.

How do I change the network topology?
--------------------------------------

.. code-block:: ini

   [topology_settings]
   network = NSFNet  # or COST239, USNET, Pan-European
   # or path to custom JSON file
   network = path/to/custom_topology.json

See :doc:`../examples/custom_topology` for creating custom topologies.

Simulation
==========

Why is my simulation slow?
---------------------------

Try these optimizations:

1. Reduce ``num_requests`` and ``max_iters``
2. Increase ``erlang_step``
3. Enable ``thread_erlangs = True``
4. Use fewer traffic load points

See :doc:`../examples/performance_optimization` for more.

Why am I getting 100% blocking?
--------------------------------

Your network may be under-provisioned:

- Increase ``c_band`` (more spectrum)
- Lower ``erlang_start`` (lighter traffic)
- Check that routing algorithm finds paths

Why am I getting 0% blocking?
------------------------------

Your network may be over-provisioned:

- Decrease ``c_band`` (less spectrum)
- Increase ``erlang_start`` (heavier traffic)

How do I save results?
-----------------------

Results are automatically saved to ``output/`` directory by default.
Configure with:

.. code-block:: ini

   [file_settings]
   output_dir = my_results
   file_type = json  # or csv, pickle

Machine Learning
================

Do I need ML/RL knowledge to use FUSION?
----------------------------------------

No! FUSION works with traditional algorithms (first-fit, k-shortest path, etc.).
ML/RL features are optional advanced capabilities.

How do I train an RL agent?
----------------------------

See :doc:`../examples/rl_training` for a complete walkthrough.

Can I use my own ML model?
---------------------------

Yes! Implement the agent interface and register it. See :doc:`../developer/extending`.

Troubleshooting
===============

"Module not found" error
-------------------------

Ensure FUSION is installed:

.. code-block:: bash

   pip install -e .

"Numpy architecture" error
---------------------------

Reinstall numpy for your architecture:

.. code-block:: bash

   pip install --force-reinstall numpy

"Graph not connected" error
----------------------------

Your custom topology has disconnected components. Use NetworkX to check:

.. code-block:: python

   import networkx as nx
   print(nx.is_connected(G))

See :doc:`troubleshooting` for more solutions.

Contributing
============

How can I contribute?
---------------------

See :doc:`../developer/contributing` for guidelines.

Can I add my own algorithm?
----------------------------

Yes! See :doc:`../developer/extending` for the plugin system.

Where do I report bugs?
------------------------

Open an issue on `GitHub Issues <https://github.com/SDNNetSim/FUSION/issues>`_.

Research & Citation
===================

How do I cite FUSION?
----------------------

See :doc:`bibliography` for BibTeX citation.

Can I use FUSION for my thesis/paper?
--------------------------------------

Absolutely! FUSION is open-source (MIT License). Please cite our IEEE ANTS paper.

Still Have Questions?
=====================

- Check :doc:`troubleshooting` for common issues
- Browse :doc:`helpful_links` for external resources
- Ask on `GitHub Discussions <https://github.com/SDNNetSim/FUSION/discussions>`_
