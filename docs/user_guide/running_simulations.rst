==================
Running Simulations
==================

This guide covers everything you need to know about running simulations in FUSION, from basic execution to advanced batch processing and parameter tuning.

.. contents:: Table of Contents
   :local:
   :depth: 2

Basic Simulation Execution
===========================

Running a Simple Simulation
----------------------------

The most basic way to run a FUSION simulation is using the command-line interface with a configuration file:

.. code-block:: bash

   fusion-sim --config path/to/config.ini

This will:

1. Load the configuration file
2. Initialize the network topology
3. Run the simulation for all specified Erlang values
4. Save results to the output directory
5. Generate summary statistics

Quick Start Example
--------------------

Here's a minimal example to get you started:

.. code-block:: bash

   # Create a minimal configuration file
   fusion-sim --config fusion/configs/templates/minimal.ini

This runs a lightweight simulation on NSFNet with:

- Traffic load: 300-500 Erlangs
- 100 requests per iteration
- K-shortest path routing
- First-fit spectrum allocation

Understanding Simulation Output
--------------------------------

During execution, FUSION displays real-time progress:

.. code-block:: text

   [INFO] Loading configuration from minimal.ini
   [INFO] Initializing NSFNet topology (14 nodes, 21 links)
   [INFO] Starting simulation at 300 Erlangs...
   Progress: [=========>        ] 50% | Requests: 50/100
   [INFO] Blocking probability: 0.0234
   [INFO] Simulation complete. Results saved to data/results/

The simulation outputs include:

- **Blocking probability**: Percentage of blocked connection requests
- **Bandwidth blocking**: Blocked bandwidth in Gbps
- **Spectrum utilization**: Average slot usage across the network
- **Route statistics**: Path lengths and resource consumption
- **Execution time**: Performance metrics

Configuration-Based Execution
==============================

Using Configuration Files
--------------------------

FUSION uses INI-format configuration files organized into sections:

.. code-block:: ini

   [general_settings]
   # Traffic load sweep
   erlang_start = 300
   erlang_stop = 700
   erlang_step = 100

   # Simulation parameters
   num_requests = 1000
   max_iters = 5

   # Algorithms
   route_method = k_shortest_path
   allocation_method = first_fit
   k_paths = 3

   [topology_settings]
   network = NSFNet
   cores_per_link = 1

   [spectrum_settings]
   c_band = 320

   [file_settings]
   file_type = json

See :doc:`configuration_reference` for all available options.

Configuration Templates
-----------------------

FUSION provides several configuration templates:

**Minimal Configuration** (``minimal.ini``):
   Quick testing with minimal parameters

**Default Configuration** (``default.ini``):
   Comprehensive baseline setup with all common options

**Advanced Configuration** (``advanced_config_example.yml``):
   ML/RL integration, multi-core, SNR modeling

**Cross-Platform Configuration** (``cross_platform.ini``):
   Optimized for different operating systems

Load a template with:

.. code-block:: bash

   fusion-sim --config fusion/configs/templates/default.ini

Overriding Configuration Parameters
------------------------------------

You can override configuration file settings via command-line arguments:

.. code-block:: bash

   fusion-sim --config config.ini \
              --erlang_start 400 \
              --erlang_stop 800 \
              --num_requests 2000 \
              --network COST239

Command-line arguments take precedence over configuration file values.

Simulation Modes
================

Standard Simulation
-------------------

Default mode for evaluating routing and spectrum allocation algorithms:

.. code-block:: bash

   fusion-sim --config standard_config.ini

Runs the specified number of iterations at each Erlang value and computes statistics.

Snapshot Mode
-------------

Capture detailed network state at specific intervals for analysis:

.. code-block:: ini

   [general_settings]
   save_snapshots = True
   snapshot_step = 100  # Save every 100 requests

Snapshots include:

- Current spectrum allocation state
- Active connections and their paths
- Link utilization statistics
- Per-node resource usage

Batch Simulations
-----------------

Run multiple simulations with different configurations:

.. code-block:: python

   from fusion.sim.batch_runner import BatchRunner

   configs = [
       'configs/scenario1.ini',
       'configs/scenario2.ini',
       'configs/scenario3.ini'
   ]

   runner = BatchRunner(configs)
   runner.run_all()

See :doc:`../examples/batch_simulations` for detailed examples.

Advanced Execution Options
===========================

Multi-Threaded Execution
-------------------------

Speed up Erlang sweeps by processing traffic loads in parallel:

.. code-block:: ini

   [general_settings]
   thread_erlangs = True

This spawns a separate thread for each Erlang value. Ideal for multi-core systems.

.. warning::

   Thread safety is guaranteed for standard simulations, but use caution with custom modules.

Debug Mode
----------

Enable verbose logging for troubleshooting:

.. code-block:: bash

   fusion-sim --config config.ini --debug

Debug mode provides:

- Detailed request-by-request logging
- Algorithm decision explanations
- Resource allocation details
- Performance profiling information

Resuming Simulations
---------------------

If a simulation is interrupted, resume from the last checkpoint:

.. code-block:: bash

   fusion-sim --config config.ini --resume

FUSION automatically saves checkpoints every ``save_step`` requests (configurable).

Algorithm Selection
===================

Routing Algorithms
------------------

FUSION supports multiple routing strategies:

**K-Shortest Path** (``k_shortest_path``):
   Finds K diverse paths and selects the first with available resources

.. code-block:: ini

   route_method = k_shortest_path
   k_paths = 5

**Shortest Path First** (``shortest_path``):
   Always uses the shortest path (Dijkstra's algorithm)

.. code-block:: ini

   route_method = shortest_path

**Machine Learning** (``ml``):
   ML-based routing decisions (requires trained model)

.. code-block:: ini

   route_method = ml

   [ml_settings]
   deploy_model = True
   model_path = models/decision_tree.pkl

**Reinforcement Learning** (``rl``):
   RL agent selects routes (requires trained agent)

.. code-block:: ini

   route_method = rl

   [rl_settings]
   agent_path = agents/ppo_agent.zip

Spectrum Allocation Algorithms
-------------------------------

**First-Fit** (``first_fit``):
   Assigns the first available contiguous spectrum slots

**Best-Fit** (``best_fit``):
   Finds the smallest gap that fits the request

**Random-Fit** (``random_fit``):
   Randomly selects from available spectrum blocks

**ML/RL-based**:
   Intelligent allocation using trained models

.. code-block:: ini

   allocation_method = first_fit
   spectrum_priority = lowest  # or 'highest', 'random'

Network Topologies
==================

Built-in Topologies
-------------------

FUSION includes standard optical network topologies:

- **NSFNet** (14 nodes, 21 links) - US National Science Foundation Network
- **COST239** (11 nodes, 26 links) - European research network
- **USNET** (24 nodes, 43 links) - Continental US backbone
- **German17** (17 nodes, 26 links) - German backbone network
- **Euro28** (28 nodes, 41 links) - Pan-European network

Select a topology in your configuration:

.. code-block:: ini

   [topology_settings]
   network = COST239

Custom Topologies
-----------------

Define custom networks using GraphML format:

.. code-block:: xml

   <?xml version="1.0" encoding="UTF-8"?>
   <graphml>
     <graph id="CustomNet" edgedefault="undirected">
       <node id="0"/>
       <node id="1"/>
       <edge source="0" target="1">
         <data key="weight">100.0</data>
       </edge>
     </graph>
   </graphml>

Load with:

.. code-block:: ini

   [topology_settings]
   network = /path/to/custom_network.graphml

See :doc:`../examples/custom_topology` for detailed examples.

Performance Optimization
========================

Reducing Execution Time
-----------------------

**1. Reduce Iterations:**

.. code-block:: ini

   max_iters = 3  # Instead of 10+

**2. Decrease Request Count:**

.. code-block:: ini

   num_requests = 500  # Instead of 5000+

**3. Enable Threading:**

.. code-block:: ini

   thread_erlangs = True

**4. Limit Snapshot Frequency:**

.. code-block:: ini

   save_snapshots = True
   snapshot_step = 500  # Save less frequently

**5. Use Minimal Logging:**

.. code-block:: bash

   fusion-sim --config config.ini --quiet

Memory Management
-----------------

For large-scale simulations, optimize memory usage:

.. code-block:: ini

   [general_settings]
   save_snapshots = False  # Disable if not needed
   save_start_end_slots = False  # Reduce data retention

Result Interpretation
=====================

Output Files
------------

Simulations generate multiple output files:

**results.json/csv**:
   Primary results with blocking probabilities, bandwidth blocking, and statistics

**snapshots/**:
   Network state captures (if enabled)

**logs/simulation.log**:
   Detailed execution logs

**summary.txt**:
   Human-readable summary statistics

Key Metrics
-----------

**Blocking Probability (BP)**:
   Percentage of connection requests that cannot be accommodated

   .. math::

      BP = \frac{\text{Blocked Requests}}{\text{Total Requests}} \times 100

**Bandwidth Blocking Probability (BBP)**:
   Percentage of requested bandwidth that was blocked

   .. math::

      BBP = \frac{\text{Blocked Bandwidth}}{\text{Total Requested Bandwidth}} \times 100

**Spectrum Utilization**:
   Average percentage of spectrum slots in use across all links

**Average Path Length**:
   Mean number of hops for established connections

Common Issues and Solutions
============================

Simulation Won't Start
----------------------

**Configuration file not found:**

.. code-block:: text

   FileNotFoundError: config.ini not found

**Solution:** Verify the path is correct and the file exists.

**Invalid configuration parameter:**

.. code-block:: text

   ConfigurationError: Unknown route_method 'invalid_algorithm'

**Solution:** Check spelling and refer to :doc:`configuration_reference`.

Poor Performance
----------------

**Simulation runs very slowly:**

- Enable threading (``thread_erlangs = True``)
- Reduce ``num_requests`` and ``max_iters``
- Disable unnecessary snapshots

**High memory usage:**

- Disable snapshots (``save_snapshots = False``)
- Reduce simulation scale
- Use batch processing for large parameter sweeps

Unexpected Results
------------------

**All requests blocked:**

- Check spectrum resources (increase ``c_band``)
- Verify network connectivity
- Review routing algorithm configuration

**No output files generated:**

- Check write permissions in output directory
- Verify ``file_type`` setting in configuration
- Enable debug mode to see error messages

Next Steps
==========

- :doc:`machine_learning` - Integrate ML algorithms into simulations
- :doc:`reinforcement_learning` - Train and deploy RL agents
- :doc:`visualization` - Analyze and plot simulation results
- :doc:`configuration_reference` - Complete configuration options reference
- :doc:`../examples/basic_simulation` - Step-by-step simulation tutorial

See Also
========

* :doc:`../getting_started/quickstart` - Quick start guide
* :doc:`../concepts/resource_allocation` - RSA/RMCSA algorithm theory
* :doc:`../examples/performance_optimization` - Advanced tuning techniques
* :doc:`cli_reference` - Complete CLI command reference