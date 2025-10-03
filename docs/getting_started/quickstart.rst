==========
Quick Start
==========

Get up and running with FUSION in under 5 minutes! This guide will help you run
your first simulation to verify your installation.

.. contents:: Table of Contents
   :local:
   :depth: 2

Prerequisites
=============

Before starting, ensure you have:

* ✅ FUSION installed (see :doc:`installation`)
* ✅ Virtual environment activated
* ✅ Basic understanding of command-line interfaces

5-Minute Quick Start
====================

Step 1: Navigate to the FUSION Directory
-----------------------------------------

.. code-block:: bash

   cd /path/to/FUSION
   source venv/bin/activate  # Activate your virtual environment

Step 2: Use an Example Configuration
-------------------------------------

FUSION comes with example configuration files to get you started:

.. code-block:: bash

   # Navigate to the ini directory
   cd ini/example_ini

   # List available examples
   ls -la

You'll see example configuration files. Let's copy one to use:

.. code-block:: bash

   # Copy an example configuration
   cp example_config.ini ../run_ini/config.ini

Step 3: Run Your First Simulation
----------------------------------

From the project root directory:

.. code-block:: bash

   # Return to project root
   cd ../..

   # Run the simulation
   python -m fusion.sim.run_simulation

Or use the CLI entry point:

.. code-block:: bash

   fusion-sim --config ini/run_ini/config.ini

Step 4: Check the Results
--------------------------

Once the simulation completes, you'll find the results in:

.. code-block:: bash

   data/
   ├── input/     # Simulation input data (topology, requests, etc.)
   │   └── <date>/<time>/
   └── output/    # Simulation results
       └── <date>/<time>/
           ├── s1/   # Results for erlang value 1
           ├── s2/   # Results for erlang value 2
           └── ...

View the results:

.. code-block:: bash

   # List output directories
   ls -la data/output/

   # View a summary of results
   cat data/output/<date>/<time>/s1/summary.json

Understanding the Output
========================

Each simulation run creates organized output:

Input Directory Structure
-------------------------

.. code-block:: text

   data/input/<timestamp>/
   ├── topology.json          # Network topology used
   ├── requests.json          # Traffic requests generated
   └── config_snapshot.ini    # Configuration used

Output Directory Structure
--------------------------

.. code-block:: text

   data/output/<timestamp>/
   ├── s1/                    # First erlang value
   │   ├── summary.json       # Key metrics
   │   ├── blocking.csv       # Blocking probability
   │   ├── spectrum_usage.csv # Spectrum utilization
   │   └── ...
   ├── s2/                    # Second erlang value
   └── ...

Key Metrics
-----------

The ``summary.json`` file contains important metrics:

* **blocking_probability**: Percentage of blocked requests
* **bandwidth_blocking**: Bandwidth blocking ratio
* **spectrum_utilization**: How efficiently spectrum is used
* **average_hops**: Average path length
* **simulation_time**: Time taken to complete

What Just Happened?
===================

Your simulation just:

1. **Loaded Configuration**: Read parameters from the INI file
2. **Generated Network**: Created an optical network topology
3. **Generated Traffic**: Created connection requests
4. **Ran Simulation**: Processed requests using routing and spectrum assignment algorithms
5. **Collected Statistics**: Gathered performance metrics
6. **Saved Results**: Wrote output to the data directory

Quick Configuration Changes
============================

Try modifying these parameters in ``ini/run_ini/config.ini``:

Change Network Topology
------------------------

.. code-block:: ini

   [network_settings]
   topology_name = NSFNet      # Try: COST239, USNET, Pan-European

Change Traffic Load
-------------------

.. code-block:: ini

   [simulation_settings]
   min_erlang = 50
   max_erlang = 300
   erlang_step = 50

Change Routing Algorithm
-------------------------

.. code-block:: ini

   [routing_settings]
   algorithm = k_shortest_paths
   k_paths = 3                 # Number of candidate paths

Change Spectrum Assignment
---------------------------

.. code-block:: ini

   [spectrum_settings]
   algorithm = first_fit        # Try: best_fit, random_fit

Running Different Scenarios
===========================

Basic Simulation (No AI)
-------------------------

.. code-block:: bash

   fusion-sim --config ini/run_ini/config.ini

Machine Learning Simulation
----------------------------

.. code-block:: bash

   # Enable ML in config.ini
   [ml_settings]
   ml_training = True
   ml_model = decision_tree

   # Run
   fusion-sim --config ini/run_ini/config.ini

Reinforcement Learning Simulation
----------------------------------

.. code-block:: bash

   # Configure RL in config.ini
   [rl_settings]
   is_training = True
   path_algorithm = dqn

   # Run
   fusion-train --config ini/run_ini/config.ini

Visualizing Results
===================

Generate plots from your simulation:

.. code-block:: bash

   # Use the visualization CLI
   python -m fusion.visualization.cli plot \
       --data data/output/<timestamp> \
       --plot-type blocking \
       --output plots/

Common Issues
=============

**Issue: "Config file not found"**

Solution: Ensure you're running from the project root and the config file exists:

.. code-block:: bash

   ls -la ini/run_ini/config.ini

**Issue: "Module not found"**

Solution: Activate your virtual environment:

.. code-block:: bash

   source venv/bin/activate

**Issue: "Permission denied"**

Solution: Ensure data directories are writable:

.. code-block:: bash

   mkdir -p data/input data/output
   chmod -R u+w data/

Next Steps
==========

Great job! You've successfully run your first FUSION simulation. Here's what to explore next:

* :doc:`first_simulation` - Detailed walkthrough of a simulation
* :doc:`configuration` - Learn all configuration options
* :doc:`../user_guide/running_simulations` - Advanced simulation techniques
* :doc:`../examples/basic_simulation` - More complete examples
* :doc:`../concepts/optical_networking_basics` - Understand the theory

Tips for Success
================

.. tip::
   **Start Simple**: Begin with default configurations and gradually customize

.. tip::
   **Save Configurations**: Keep successful configs for future reference

.. tip::
   **Document Changes**: Comment your config files to track what you modified

.. tip::
   **Compare Results**: Run multiple simulations with different parameters to see the impact

.. warning::
   Large simulations (high erlang, many requests) can take significant time and memory.
   Start small and scale up!

Getting Help
============

If you encounter issues:

* Check :doc:`../reference/troubleshooting`
* Review :doc:`../reference/faq`
* Search `GitHub Issues <https://github.com/SDNNetSim/FUSION/issues>`_

Happy simulating! 🚀
