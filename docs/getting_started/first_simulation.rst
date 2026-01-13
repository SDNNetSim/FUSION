==================
Your First Simulation
==================

This guide provides a complete walkthrough of running a simulation in FUSION,
from setup through analysis of results.

.. contents:: Table of Contents
   :local:
   :depth: 3

Introduction
============

In this tutorial, you'll learn:

* How to set up a complete simulation scenario
* Understanding configuration parameters
* Running the simulation
* Interpreting output data
* Visualizing and analyzing results

What We'll Simulate
===================

We'll simulate an elastic optical network with:

* **Topology**: NSFNet (14 nodes, 21 links)
* **Traffic**: Dynamic connection requests
* **Routing**: K-shortest paths algorithm
* **Spectrum Assignment**: First-fit algorithm
* **Objective**: Measure blocking probability vs. traffic load

Prerequisites
=============

Ensure you have:

* FUSION installed and virtual environment activated
* Completed the :doc:`quickstart` guide
* Basic understanding of optical networks (see :doc:`../concepts/optical_networking_basics`)

Step-by-Step Walkthrough
=========================

Step 1: Understanding the Scenario
-----------------------------------

We'll simulate a network operator managing an elastic optical network.
Connection requests arrive dynamically, and the network must allocate
resources (route and spectrum) efficiently to minimize blocking.

.. image:: /_static/simulation_workflow.svg
   :alt: Flowchart showing FUSION simulation workflow from configuration loading through request processing, routing decisions, spectrum assignment, resource checking, and statistics collection
   :align: center
   :width: 70%

**Key Parameters:**

* Network has limited spectrum (e.g., 320 slots)
* Requests have different bandwidth requirements
* Requests arrive and depart (holding time)
* We'll measure performance under increasing traffic load

Step 2: Create Configuration File
----------------------------------

Navigate to the configuration directory:

.. code-block:: bash

   cd ini/example_ini

Create or copy a configuration file:

.. code-block:: bash

   cp example_config.ini ../run_ini/config.ini

Edit the configuration file ``ini/run_ini/config.ini``:

.. code-block:: ini

   # ============================================================
   # FUSION Simulation Configuration
   # Scenario: NSFNet with Dynamic Traffic
   # ============================================================

   [network_settings]
   # Network topology to simulate
   topology_name = NSFNet

   # Number of spectrum slots per link
   num_spectrum_slots = 320

   # Slot width in GHz
   slot_width = 12.5

   # Number of cores (for multi-core fiber)
   num_cores = 1

   [simulation_settings]
   # Number of connection requests to simulate per erlang value
   num_requests = 10000

   # Traffic load range (in Erlangs)
   min_erlang = 100
   max_erlang = 500
   erlang_step = 100

   # Seed for reproducibility
   random_seed = 42

   [routing_settings]
   # Routing algorithm
   algorithm = k_shortest_paths

   # Number of candidate paths
   k_paths = 3

   # Weighting metric
   weight_metric = hops

   [spectrum_settings]
   # Spectrum assignment algorithm
   algorithm = first_fit

   # Modulation selection
   modulation_selection = distance_adaptive

   [request_settings]
   # Request bandwidth distribution (in Gbps)
   min_bandwidth = 25
   max_bandwidth = 400

   # Mean holding time (time units)
   mean_holding_time = 100

   # Distribution type
   arrival_distribution = poisson
   holding_time_distribution = exponential

   [output_settings]
   # Save detailed logs
   save_detailed_logs = true

   # Save network snapshots
   save_snapshots = false

   # Output format
   output_format = json

Step 3: Understanding the Configuration
----------------------------------------

Let's break down key sections:

**Network Settings**

* ``topology_name``: Which network to simulate (NSFNet, COST239, etc.)
* ``num_spectrum_slots``: Total spectrum resources available
* ``slot_width``: Granularity of spectrum allocation

**Simulation Settings**

* ``num_requests``: How many requests to generate
* ``min/max_erlang``: Traffic load range to test
* ``erlang_step``: Increments between traffic loads

**Traffic loads** (Erlangs): Measure of average network utilization

* 100 Erlangs ≈ Light traffic
* 300 Erlangs ≈ Moderate traffic
* 500+ Erlangs ≈ Heavy traffic

**Routing Settings**

* ``k_shortest_paths``: Finds K alternative paths
* ``k_paths = 3``: Consider up to 3 different routes
* ``weight_metric``: How to rank paths (hops, distance, etc.)

**Spectrum Settings**

* ``first_fit``: Allocate lowest available spectrum slot
* ``distance_adaptive``: Choose modulation based on path length

Step 4: Run the Simulation
---------------------------

From the project root directory:

.. code-block:: bash

   # Make sure you're in the FUSION root directory
   cd /path/to/FUSION

   # Activate virtual environment
   source venv/bin/activate

   # Run the simulation
   python -m fusion.sim.run_simulation --config ini/run_ini/config.ini

You'll see output like:

.. code-block:: text

   ╔══════════════════════════════════════════════════════╗
   ║         FUSION - Optical Network Simulator           ║
   ║                    Version 6.0.0                     ║
   ╚══════════════════════════════════════════════════════╝

   Loading configuration from: ini/run_ini/config.ini
   ✓ Configuration loaded successfully

   Initializing network...
   ✓ Topology: NSFNet (14 nodes, 21 links)
   ✓ Spectrum: 320 slots @ 12.5 GHz

   Running simulations...
   [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 100 Erlangs | 10000/10000 requests
   [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 200 Erlangs | 10000/10000 requests
   [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 300 Erlangs | 10000/10000 requests
   [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 400 Erlangs | 10000/10000 requests
   [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 500 Erlangs | 10000/10000 requests

   Simulation completed successfully!
   Results saved to: data/output/2025/10/03/14_30_25_123456/

**Simulation Time**: Depending on your system, this may take 5-30 minutes.

Step 5: Explore the Output
---------------------------

The simulation creates organized output:

.. code-block:: bash

   cd data/output/2025/10/03/14_30_25_123456/

   # List simulation runs
   ls -la

You'll see:

.. code-block:: text

   s1/    # 100 Erlangs
   s2/    # 200 Erlangs
   s3/    # 300 Erlangs
   s4/    # 400 Erlangs
   s5/    # 500 Erlangs

Each directory contains:

.. code-block:: bash

   cd s1/
   ls -la

.. code-block:: text

   summary.json              # Overall metrics
   blocking_probability.csv  # Blocking data
   bandwidth_blocking.csv    # BW blocking data
   spectrum_utilization.csv  # Spectrum usage
   modulation_usage.csv      # Modulation formats used
   path_statistics.csv       # Routing statistics

Step 6: Analyze the Results
----------------------------

Let's examine the key output files:

**Summary Metrics**

.. code-block:: bash

   cat s1/summary.json

.. code-block:: json

   {
     "erlang": 100,
     "total_requests": 10000,
     "blocked_requests": 125,
     "blocking_probability": 0.0125,
     "bandwidth_blocking": 0.0143,
     "avg_spectrum_utilization": 0.42,
     "avg_hops": 2.3,
     "simulation_time_seconds": 45.2
   }

**Key Metrics Explained:**

* **blocking_probability** (1.25%): Percentage of requests that couldn't be served
* **bandwidth_blocking** (1.43%): Percentage of bandwidth blocked
* **avg_spectrum_utilization** (42%): How much spectrum is used on average
* **avg_hops** (2.3): Average path length

**Blocking vs. Traffic Load**

Let's compare across all erlang values:

.. code-block:: bash

   # Extract blocking probabilities
   for dir in s*/; do
       erlang=$(jq '.erlang' "$dir/summary.json")
       blocking=$(jq '.blocking_probability' "$dir/summary.json")
       echo "Erlang $erlang: Blocking = $blocking"
   done

Output:

.. code-block:: text

   Erlang 100: Blocking = 0.0125
   Erlang 200: Blocking = 0.0287
   Erlang 300: Blocking = 0.0523
   Erlang 400: Blocking = 0.0892
   Erlang 500: Blocking = 0.1456

**Observation**: As traffic load increases, blocking probability increases.

Step 7: Visualize the Results
------------------------------

Create visualizations to better understand the results:

**Generate Blocking Probability Plot**

.. code-block:: bash

   python -m fusion.visualization.cli plot \\
       --data data/output/2025/10/03/14_30_25_123456 \\
       --plot-type blocking \\
       --output plots/blocking.png

This creates a line graph showing blocking probability vs. traffic load.

**Generate Spectrum Utilization Plot**

.. code-block:: bash

   python -m fusion.visualization.cli plot \\
       --data data/output/2025/10/03/14_30_25_123456 \\
       --plot-type spectrum_utilization \\
       --output plots/spectrum.png

**Generate Modulation Usage Plot**

.. code-block:: bash

   python -m fusion.visualization.cli plot \\
       --data data/output/2025/10/03/14_30_25_123456 \\
       --plot-type modulation_usage \\
       --output plots/modulation.png

Step 8: Interpret the Results
------------------------------

**What Do These Results Mean?**

1. **Low Blocking at Low Load** (100-200 Erlangs)

   * Network has sufficient resources
   * Most requests are successfully provisioned
   * Good performance

2. **Increasing Blocking at Moderate Load** (300-400 Erlangs)

   * Network resources becoming constrained
   * Spectrum fragmentation may occur
   * Performance degrading

3. **High Blocking at Heavy Load** (500+ Erlangs)

   * Network is overloaded
   * Many requests blocked
   * May need more resources or better algorithms

**Factors Affecting Blocking:**

* **Network Topology**: Some topologies have more alternate paths
* **Spectrum Resources**: More slots = lower blocking
* **Routing Algorithm**: Better algorithms find more efficient paths
* **Spectrum Assignment**: Better algorithms reduce fragmentation
* **Traffic Mix**: Bandwidth distribution affects resource usage

Comparing Different Algorithms
===============================

Let's compare First-Fit vs. Best-Fit spectrum assignment:

Run with Best-Fit
-----------------

Edit ``ini/run_ini/config.ini``:

.. code-block:: ini

   [spectrum_settings]
   algorithm = best_fit  # Changed from first_fit

Run again:

.. code-block:: bash

   python -m fusion.sim.run_simulation --config ini/run_ini/config.ini

Compare Results
---------------

.. code-block:: bash

   # Compare blocking probabilities
   echo "First-Fit:"
   cat data/output/<first_fit_timestamp>/s3/summary.json | jq '.blocking_probability'

   echo "Best-Fit:"
   cat data/output/<best_fit_timestamp>/s3/summary.json | jq '.blocking_probability'

You might observe that different algorithms perform differently under various loads!

Common Issues and Solutions
===========================

**Issue: Simulation runs slowly**

* Reduce ``num_requests`` to 1000 for testing
* Use fewer erlang values
* Run with ``parallel=True`` in batch mode

**Issue: All requests blocked**

* Check if ``num_spectrum_slots`` is reasonable (e.g., 320)
* Verify ``max_erlang`` isn't too high
* Ensure topology is connected

**Issue: No variation in results**

* Check that ``random_seed`` is not causing deterministic behavior
* Ensure traffic load range is appropriate
* Verify request parameters allow diversity

Best Practices
==============

.. tip::
   **Start Small**: Begin with fewer requests (1000-5000) for faster iterations

.. tip::
   **Use Seeds**: Set ``random_seed`` for reproducible results during development

.. tip::
   **Save Configs**: Archive successful configurations with descriptive names

.. tip::
   **Document**: Add comments to config files explaining your choices

.. tip::
   **Iterate**: Run multiple scenarios, changing one parameter at a time

Next Steps
==========

Congratulations! You've successfully run and analyzed your first FUSION simulation.

Explore further:

* :doc:`configuration` - Learn all configuration options in detail
* :doc:`../user_guide/running_simulations` - Advanced simulation techniques
* :doc:`../examples/custom_topology` - Create your own network topologies
* :doc:`../user_guide/machine_learning` - Add ML to your simulations
* :doc:`../user_guide/reinforcement_learning` - Train RL agents
* :doc:`../concepts/resource_allocation` - Understand the algorithms

Experiment Ideas
================

Try these variations:

1. **Different Topologies**: Compare NSFNet, COST239, and USNET
2. **K-Paths**: Test k=1, k=3, k=5 to see impact of alternate routes
3. **Spectrum Size**: Try 160, 320, 640 slots
4. **Modulation**: Compare fixed vs. distance-adaptive modulation
5. **Traffic Mix**: Vary min/max bandwidth requirements

Happy experimenting! 🔬
