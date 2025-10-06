=============
Configuration
=============

Learn how to configure FUSION simulations using INI configuration files
and command-line parameters.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

FUSION uses INI-format configuration files to specify simulation parameters.
This approach provides:

* **Reproducibility**: Save and reuse configurations
* **Clarity**: Human-readable parameter definitions
* **Flexibility**: Easy modification without code changes
* **Documentation**: Self-documenting simulation setups

Configuration File Structure
============================

Basic Structure
---------------

A FUSION configuration file consists of sections containing key-value pairs:

.. code-block:: ini

   [section_name]
   parameter_key = value
   another_parameter = value

   [another_section]
   parameter = value

Comments
--------

Use ``#`` or ``;`` for comments:

.. code-block:: ini

   # This is a comment
   ; This is also a comment
   parameter = value  # Inline comments work too

Configuration Sections
======================

Network Settings
----------------

Define the network topology and physical layer parameters.

.. code-block:: ini

   [network_settings]
   # Network topology
   topology_name = NSFNet

   # Spectrum parameters
   num_spectrum_slots = 320
   slot_width = 12.5
   guard_band = 1

   # Multi-core fiber settings
   num_cores = 1
   core_xt_threshold = -20

   # Physical distances (if custom topology)
   distance_unit = km

**Common Topologies:**

* ``NSFNet`` - 14 nodes, 21 links (US network)
* ``COST239`` - 11 nodes, 26 links (European network)
* ``USNET`` - 24 nodes, 43 links (US nationwide)
* ``Pan-European`` - 28 nodes, 41 links (European network)
* ``Custom`` - Define your own (see :doc:`../examples/custom_topology`)

**Spectrum Parameters:**

* ``num_spectrum_slots``: Total available slots (typically 160-640)
* ``slot_width``: Frequency width per slot in GHz (usually 12.5 or 6.25)
* ``guard_band``: Slots between connections to prevent interference

Simulation Settings
-------------------

Control simulation behavior and traffic generation.

.. code-block:: ini

   [simulation_settings]
   # Number of requests per erlang value
   num_requests = 10000

   # Traffic load range (Erlangs)
   min_erlang = 100
   max_erlang = 500
   erlang_step = 100

   # Random seed for reproducibility
   random_seed = 42

   # Simulation mode
   simulation_mode = dynamic  # or 'static' for static traffic

   # Parallel execution
   parallel = false
   num_processes = 4

**Traffic Load (Erlangs):**

* Measures average network utilization
* Higher values = more traffic
* Typical ranges: 50-600 Erlangs

**Simulation Modes:**

* ``dynamic``: Requests arrive and depart over time
* ``static``: All requests present simultaneously

Routing Settings
----------------

Configure path computation algorithms.

.. code-block:: ini

   [routing_settings]
   # Routing algorithm
   algorithm = k_shortest_paths

   # K-shortest paths parameters
   k_paths = 3
   weight_metric = hops

   # Path constraints
   max_hops = 10
   max_distance_km = 5000

**Routing Algorithms:**

* ``k_shortest_paths``: Find K shortest paths
* ``dijkstra``: Single shortest path
* ``yen``: Yen's K-shortest paths algorithm
* ``ml_routing``: Machine learning-based routing (requires training)
* ``rl_routing``: Reinforcement learning-based routing

**Weight Metrics:**

* ``hops``: Minimize number of hops
* ``distance``: Minimize physical distance
* ``available_spectrum``: Prefer paths with more free spectrum
* ``fragmentation``: Minimize spectrum fragmentation

Spectrum Assignment Settings
----------------------------

Configure spectrum allocation algorithms.

.. code-block:: ini

   [spectrum_settings]
   # Spectrum assignment algorithm
   algorithm = first_fit

   # Modulation selection
   modulation_selection = distance_adaptive

   # Spectrum packing
   enable_defragmentation = false
   defragmentation_threshold = 0.7

**Spectrum Assignment Algorithms:**

* ``first_fit``: Allocate first available slot
* ``best_fit``: Minimize fragmentation
* ``random_fit``: Random selection
* ``exact_fit``: Exact match for request size
* ``ml_spectrum``: ML-based assignment
* ``rl_spectrum``: RL-based assignment

**Modulation Selection:**

* ``distance_adaptive``: Choose based on path length
* ``fixed``: Use single modulation format (specify with ``modulation_format``)
* ``snr_based``: Choose based on signal quality

**Available Modulation Formats:**

* ``BPSK``: 1 bit/symbol, longest reach
* ``QPSK``: 2 bits/symbol
* ``8QAM``: 3 bits/symbol
* ``16QAM``: 4 bits/symbol
* ``32QAM``: 5 bits/symbol
* ``64QAM``: 6 bits/symbol, shortest reach

Request Generation Settings
----------------------------

Define traffic characteristics.

.. code-block:: ini

   [request_settings]
   # Bandwidth distribution (Gbps)
   min_bandwidth = 25
   max_bandwidth = 400
   bandwidth_distribution = uniform  # or 'normal', 'exponential'

   # Holding time
   mean_holding_time = 100
   holding_time_distribution = exponential

   # Arrival process
   arrival_distribution = poisson
   mean_arrival_rate = auto  # calculated from erlang

   # Node pair selection
   node_pair_selection = uniform  # or 'weighted', 'distance_based'

**Bandwidth Distributions:**

* ``uniform``: Equal probability across range
* ``normal``: Gaussian distribution (specify ``bandwidth_mean`` and ``bandwidth_std``)
* ``exponential``: Exponential distribution
* ``custom``: Load from file

**Holding Time:**

* Average duration a connection occupies resources
* Affects network load
* Units: arbitrary time units (typically seconds or minutes)

Machine Learning Settings
--------------------------

Configure ML-based decision making.

.. code-block:: ini

   [ml_settings]
   # Enable ML
   ml_training = false
   deploy_model = false

   # ML model type
   ml_model = decision_tree  # or 'random_forest', 'neural_network'

   # Training data
   output_train_data = true
   train_file_path = data/training/dataset_20251003

   # Model parameters
   test_size = 0.3
   cross_validation = 5

**ML Models:**

* ``decision_tree``: Fast, interpretable
* ``random_forest``: More accurate, ensemble method
* ``neural_network``: Deep learning approach
* ``gradient_boosting``: Advanced ensemble method

Reinforcement Learning Settings
--------------------------------

Configure RL agents for dynamic optimization.

.. code-block:: ini

   [rl_settings]
   # Device
   device = cpu  # or 'cuda' for GPU

   # Training mode
   is_training = false
   optimize = false

   # Path selection agent
   path_algorithm = dqn  # or 'ppo', 'a2c', 'q_learning', 'ucb_bandit'
   path_model = models/path_agent_20251003.zip

   # Core selection agent (for MCF)
   core_algorithm = first_fit
   core_model = models/core_agent_20251003.json

   # Spectrum selection agent
   spectrum_algorithm = ppo
   spectrum_model = models/spectrum_agent_20251003.zip

   # Training hyperparameters
   learning_rate = 0.0003
   discount_factor = 0.99
   epsilon_start = 1.0
   epsilon_end = 0.01
   epsilon_decay = 0.995

   # Episode configuration
   total_timesteps = 1000000
   eval_frequency = 10000

**RL Algorithms:**

* ``dqn``: Deep Q-Network
* ``ppo``: Proximal Policy Optimization
* ``a2c``: Advantage Actor-Critic
* ``q_learning``: Tabular Q-learning
* ``ucb_bandit``: Upper Confidence Bound bandit
* ``thompson_sampling``: Bayesian bandit

Output Settings
---------------

Control what data is saved and how.

.. code-block:: ini

   [output_settings]
   # Output format
   output_format = json  # or 'csv', 'pickle'

   # Detail level
   save_detailed_logs = true
   save_network_snapshots = false
   save_request_traces = true

   # Compression
   compress_output = false

   # Custom output directory
   output_dir = data/output

**Output Formats:**

* ``json``: Human-readable, widely compatible
* ``csv``: Spreadsheet-compatible
* ``pickle``: Python-native, fastest

SNR and Physical Layer Settings
--------------------------------

Configure signal quality calculations.

.. code-block:: ini

   [snr_settings]
   # Enable SNR calculations
   enable_snr = true

   # SNR model
   snr_model = gaussian_noise  # or 'ase_noise', 'nonlinear'

   # Physical parameters
   fiber_attenuation_db_km = 0.2
   amplifier_noise_figure_db = 5.0
   amplifier_spacing_km = 80

   # Thresholds (dB)
   snr_threshold_bpsk = 6.8
   snr_threshold_qpsk = 9.8
   snr_threshold_8qam = 12.6
   snr_threshold_16qam = 14.8

Command-Line Options
====================

Override configuration file settings via command line:

.. code-block:: bash

   fusion-sim \\
       --config ini/run_ini/config.ini \\
       --num-requests 5000 \\
       --min-erlang 100 \\
       --max-erlang 300 \\
       --topology NSFNet \\
       --routing-algorithm k_shortest_paths \\
       --k-paths 5 \\
       --spectrum-algorithm best_fit \\
       --output-dir results/experiment1

Common command-line options:

* ``--config PATH``: Configuration file path
* ``--topology NAME``: Override topology
* ``--num-requests N``: Override request count
* ``--parallel``: Enable parallel execution
* ``--seed N``: Set random seed
* ``--verbose``: Increase output verbosity

Configuration Best Practices
=============================

Organization
------------

.. tip::
   **Use Descriptive Names**: Name configs after their purpose

   .. code-block:: bash

      nsfnet_first_fit_100_500erl.ini
      cost239_ml_routing_comparison.ini

.. tip::
   **Version Control**: Keep configs in git

   .. code-block:: bash

      git add ini/run_ini/experiment_*.ini
      git commit -m "Add configuration for experiment X"

Documentation
-------------

.. tip::
   **Comment Everything**: Explain non-obvious choices

   .. code-block:: ini

      # Using 3 paths after testing showed diminishing returns beyond k=3
      k_paths = 3

      # High traffic to test worst-case blocking
      max_erlang = 600

Reproducibility
---------------

.. tip::
   **Set Seeds**: Always set ``random_seed`` for experiments

   .. code-block:: ini

      [simulation_settings]
      random_seed = 42  # For reproducibility

.. tip::
   **Save with Results**: Copy config to output directory

   FUSION automatically saves a snapshot of the config used.

Parameter Tuning
----------------

.. tip::
   **Change One Thing**: Modify one parameter at a time

.. tip::
   **Document Changes**: Track what you changed and why

.. tip::
   **Baseline First**: Run with defaults before optimizing

Example Configurations
======================

Basic Simulation
----------------

.. code-block:: ini

   [network_settings]
   topology_name = NSFNet
   num_spectrum_slots = 320

   [simulation_settings]
   num_requests = 10000
   min_erlang = 100
   max_erlang = 500
   erlang_step = 100

   [routing_settings]
   algorithm = k_shortest_paths
   k_paths = 3

   [spectrum_settings]
   algorithm = first_fit

ML-Based Simulation
-------------------

.. code-block:: ini

   [ml_settings]
   ml_training = true
   ml_model = random_forest
   output_train_data = true

   [routing_settings]
   algorithm = ml_routing

RL Training Scenario
--------------------

.. code-block:: ini

   [rl_settings]
   is_training = true
   path_algorithm = ppo
   total_timesteps = 1000000
   learning_rate = 0.0003

   [simulation_settings]
   num_requests = 100000

Advanced Configuration Topics
==============================

For advanced configuration topics, see:

* :doc:`../user_guide/configuration_reference` - Complete parameter reference
* :doc:`../examples/custom_topology` - Custom network topologies
* :doc:`../user_guide/machine_learning` - ML configuration
* :doc:`../user_guide/reinforcement_learning` - RL training configuration

Next Steps
==========

Now that you understand configuration, explore:

* :doc:`next_steps` - Where to go from here
* :doc:`../user_guide/running_simulations` - Advanced simulation techniques
* :doc:`../examples/basic_simulation` - Complete examples

.. seealso::

   * :doc:`../reference/faq` - Common configuration questions
   * :doc:`../reference/troubleshooting` - Configuration issues
