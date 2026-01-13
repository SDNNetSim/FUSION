==================
Basic Simulation
==================

This example demonstrates how to run a basic optical network simulation using FUSION.
You'll learn how to configure a simulation, run it, and interpret the results.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

This example covers:

* Creating a minimal configuration file
* Running a simulation via CLI
* Understanding the output
* Analyzing blocking probability results

Prerequisites
=============

Ensure FUSION is installed:

.. code-block:: bash

   pip install -e .

Quick Start (30 seconds)
========================

The fastest way to run a simulation:

.. code-block:: bash

   fusion-sim --network NSFNet --erlang_start 300 --erlang_stop 500 \
              --erlang_step 100 --num_requests 100 --max_iters 2

This command runs a simulation on the NSFNet topology with traffic loads from 300 to 500 Erlangs.

Step-by-Step Guide
==================

Step 1: Create Configuration File
----------------------------------

Create a file named ``basic_config.ini``:

.. code-block:: ini

   [general_settings]
   # Traffic load parameters
   erlang_start = 300
   erlang_stop = 900
   erlang_step = 300

   # Simulation control
   max_iters = 4
   num_requests = 500

   # Routing and spectrum assignment
   k_paths = 3
   route_method = k_shortest_path
   allocation_method = first_fit

   # Output settings
   print_step = 50

   [topology_settings]
   # Network topology
   network = NSFNet
   cores_per_link = 1

   [spectrum_settings]
   # Spectral resources (320 slots = 4 THz C-band)
   c_band = 320

   [file_settings]
   # Output format
   file_type = json

**Configuration Breakdown:**

* **erlang_start/stop/step**: Traffic load range (300-900 Erlangs in 300 Erlang steps)
* **max_iters**: Number of iterations per traffic load (4 iterations)
* **num_requests**: Connection requests per iteration (500 requests)
* **k_paths**: Number of candidate paths (3 shortest paths)
* **route_method**: Routing algorithm (k-shortest path)
* **allocation_method**: Spectrum assignment algorithm (first-fit)
* **network**: Network topology (NSFNet - 14 nodes)
* **c_band**: Number of frequency slots (320 slots)

Step 2: Run the Simulation
---------------------------

Execute the simulation using the CLI:

.. code-block:: bash

   fusion-sim --config basic_config.ini

**Alternative: Using Python API**

.. code-block:: python

   from fusion.configs.config import Config
   from fusion.sim.run_simulation import run_simulation

   # Load configuration
   config = Config.from_file("basic_config.ini")

   # Run simulation
   results = run_simulation(config)

   print(f"Simulation complete! Results saved to {results.output_dir}")

Step 3: Monitor Progress
-------------------------

During execution, you'll see progress output:

.. code-block:: text

   [INFO] Starting simulation with NSFNet topology
   [INFO] Traffic load: 300 Erlangs
   [INFO] Iteration 1/4: Processing 500 requests...
   [INFO]   Request 50/500 - Blocking: 2.0%
   [INFO]   Request 100/500 - Blocking: 3.5%
   ...
   [INFO] Iteration 1/4 complete - Blocking: 4.2%
   [INFO] Iteration 2/4: Processing 500 requests...
   ...

Step 4: Analyze Results
-----------------------

Results are saved in the ``output/`` directory:

.. code-block:: text

   output/
   ├── simulation_results.json    # Raw simulation data
   ├── blocking_stats.json        # Blocking probability per load
   ├── plots/
   │   ├── blocking_vs_load.png  # Blocking probability curve
   │   └── spectrum_usage.png    # Spectrum utilization
   └── logs/
       └── simulation.log         # Detailed execution log

**Blocking Probability Results:**

.. code-block:: json

   {
     "300": {
       "mean_blocking": 0.042,
       "std_blocking": 0.008,
       "bandwidth_blocking": 0.039
     },
     "600": {
       "mean_blocking": 0.158,
       "std_blocking": 0.012,
       "bandwidth_blocking": 0.152
     },
     "900": {
       "mean_blocking": 0.312,
       "std_blocking": 0.018,
       "bandwidth_blocking": 0.307
     }
   }

Understanding the Results
=========================

Key Metrics
-----------

**Blocking Probability**
   Percentage of connection requests that could not be established due to insufficient resources.
   Lower is better (0% = no blocking, 100% = all requests blocked).

**Bandwidth Blocking**
   Blocking weighted by requested bandwidth. Large requests contribute more to this metric.

**Standard Deviation**
   Variation across iterations. Lower values indicate more consistent results.

Typical Blocking Curves
------------------------

For a well-provisioned network:

* **Low load (< 400 Erlangs)**: Blocking < 5% (acceptable)
* **Medium load (400-800 Erlangs)**: Blocking 5-20% (congestion starting)
* **High load (> 800 Erlangs)**: Blocking > 20% (network saturated)

Visualization
=============

Generate plots using the visualization module:

.. code-block:: python

   from fusion.visualization.plot_stats import plot_blocking_probability

   # Plot blocking vs. traffic load
   plot_blocking_probability(
       results_file="output/simulation_results.json",
       output_file="blocking_curve.png"
   )

Or use the CLI:

.. code-block:: bash

   fusion-plot --input output/simulation_results.json \
               --plot blocking \
               --output blocking_curve.png

Common Variations
=================

Different Routing Algorithms
----------------------------

Try different routing strategies:

.. code-block:: ini

   # Fragmentation-aware routing
   route_method = fragmentation_aware_ksp

   # Congestion-aware routing
   route_method = congestion_aware_routing

   # NLI-aware routing (considers nonlinear impairments)
   route_method = nli_aware_routing

Different Spectrum Assignment
------------------------------

.. code-block:: ini

   # Best-fit (minimize fragmentation)
   allocation_method = best_fit

   # Last-fit (pack from end)
   allocation_method = last_fit

Different Topologies
--------------------

.. code-block:: ini

   # Small network (6 nodes)
   network = COST239

   # Medium network (14 nodes)
   network = NSFNet

   # Large network (24 nodes)
   network = USNET

   # European network (28 nodes)
   network = Pan-European

Multicore Networks
------------------

For space-division multiplexing (SDM):

.. code-block:: ini

   [topology_settings]
   cores_per_link = 7      # 7-core fiber

   [general_settings]
   core_allocation = first_fit  # or random_core, least_congested

Troubleshooting
===============

Simulation Runs Too Slowly
---------------------------

Reduce computational load:

.. code-block:: ini

   num_requests = 200        # Fewer requests
   max_iters = 2             # Fewer iterations
   erlang_step = 500         # Larger steps

All Requests Blocked
--------------------

Network may be under-provisioned:

.. code-block:: ini

   c_band = 640              # Double spectrum
   erlang_start = 100        # Lower starting load

No Blocking at All
------------------

Network may be over-provisioned:

.. code-block:: ini

   c_band = 160              # Reduce spectrum
   erlang_start = 800        # Higher starting load

Next Steps
==========

Now that you've run a basic simulation, explore:

* :doc:`custom_topology` - Create custom network topologies
* :doc:`ml_decision_tree` - Use machine learning for routing
* :doc:`rl_training` - Train reinforcement learning agents
* :doc:`advanced_visualization` - Create publication-quality plots
* :doc:`batch_simulations` - Run multiple scenarios in parallel

See Also
========

* :doc:`../getting_started/quickstart` - Quick start guide
* :doc:`../user_guide/running_simulations` - Complete simulation guide
* :doc:`../user_guide/configuration_reference` - All configuration options
* :doc:`../concepts/optical_networking_basics` - Optical networking fundamentals
