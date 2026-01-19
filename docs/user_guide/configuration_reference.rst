=======================
Configuration Reference
=======================

Complete reference for all FUSION configuration options in INI format.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

FUSION uses INI-format configuration files organized into sections. This reference documents all available options, their types, default values, and usage.

Configuration File Structure
============================

Basic Format
------------

.. code-block:: ini

   [section_name]
   parameter_name = value
   another_parameter = value

   [another_section]
   parameter = value

Sections:
   - ``[general_settings]`` - Core simulation parameters
   - ``[topology_settings]`` - Network topology configuration
   - ``[spectrum_settings]`` - Spectrum resource allocation
   - ``[snr_settings]`` - SNR and physical layer modeling
   - ``[file_settings]`` - Output and data management
   - ``[ml_settings]`` - Machine learning options
   - ``[rl_settings]`` - Reinforcement learning options

General Settings
================

Traffic Load Parameters
-----------------------

``erlang_start`` (float, required)
   Starting traffic load in Erlangs

   .. code-block:: ini

      erlang_start = 300

``erlang_stop`` (float, required)
   Ending traffic load in Erlangs

   .. code-block:: ini

      erlang_stop = 700

``erlang_step`` (float, required)
   Step size for traffic load sweep

   .. code-block:: ini

      erlang_step = 100

``holding_time`` (float, required)
   Average connection holding time

   .. code-block:: ini

      holding_time = 1.0

Simulation Control
------------------

``num_requests`` (int, required)
   Number of connection requests per iteration

   .. code-block:: ini

      num_requests = 1000

   Range: 100 - 100000+

``max_iters`` (int, required)
   Maximum number of iterations per Erlang value

   .. code-block:: ini

      max_iters = 5

   Range: 1 - 100

``thread_erlangs`` (bool, required)
   Enable multi-threaded Erlang sweep

   .. code-block:: ini

      thread_erlangs = True

   Options: ``True``, ``False``

Routing & Allocation
--------------------

``route_method`` (str, required)
   Routing algorithm

   .. code-block:: ini

      route_method = k_shortest_path

   Options:
      - ``k_shortest_path`` - K-shortest path algorithm
      - ``shortest_path`` - Single shortest path (Dijkstra)
      - ``ml`` - Machine learning routing
      - ``rl`` - Reinforcement learning routing

``k_paths`` (int, optional)
   Number of candidate paths (for k-shortest path)

   .. code-block:: ini

      k_paths = 3

   Default: 3, Range: 1-10

``allocation_method`` (str, required)
   Spectrum allocation algorithm

   .. code-block:: ini

      allocation_method = first_fit

   Options:
      - ``first_fit`` - First available slot
      - ``best_fit`` - Smallest suitable gap
      - ``random_fit`` - Random selection
      - ``ml`` - Machine learning allocation
      - ``rl`` - Reinforcement learning allocation

``spectrum_priority`` (str, required)
   Slot selection priority

   .. code-block:: ini

      spectrum_priority = lowest

   Options: ``lowest``, ``highest``, ``random``

Modulation & Segmentation
--------------------------

``mod_assumption`` (str, required)
   Modulation format selection method

   .. code-block:: ini

      mod_assumption = distance_adaptive

   Options:
      - ``distance_adaptive`` - Select based on distance
      - ``fixed`` - Use single modulation format
      - ``snr_adaptive`` - Select based on SNR

``mod_assumption_path`` (str, required)
   Path to modulation format configuration file

   .. code-block:: ini

      mod_assumption_path = configs/modulation_formats.json

``max_segments`` (int, required)
   Maximum transponder segments per connection

   .. code-block:: ini

      max_segments = 4

   Range: 1-10

``guard_slots`` (int, required)
   Guard slots between connections

   .. code-block:: ini

      guard_slots = 1

   Range: 0-5

``pre_calc_mod_selection`` (bool, required)
   Pre-calculate modulation format selection

   .. code-block:: ini

      pre_calc_mod_selection = True

Network Features
----------------

``dynamic_lps`` (bool, required)
   Enable dynamic lightpath switching

   .. code-block:: ini

      dynamic_lps = False

``fixed_grid`` (bool, required)
   Use fixed grid instead of flex-grid

   .. code-block:: ini

      fixed_grid = False

``request_distribution`` (str, optional)
   Traffic request distribution

   .. code-block:: ini

      request_distribution = uniform

   Options: ``uniform``, ``non_uniform``, ``gravity``

   Default: ``uniform``

Output Control
--------------

``save_snapshots`` (bool, required)
   Save network state snapshots

   .. code-block:: ini

      save_snapshots = True

``snapshot_step`` (int, required)
   Snapshot frequency (requests between snapshots)

   .. code-block:: ini

      snapshot_step = 100

``save_step`` (int, required)
   Result save frequency

   .. code-block:: ini

      save_step = 100

``print_step`` (int, required)
   Console output frequency

   .. code-block:: ini

      print_step = 100

``save_start_end_slots`` (bool, required)
   Save detailed slot allocation info

   .. code-block:: ini

      save_start_end_slots = False

Topology Settings
=================

Network Selection
-----------------

``network`` (str, required)
   Network topology

   .. code-block:: ini

      network = NSFNet

   Built-in Options:
      - ``NSFNet`` - 14 nodes, 21 links
      - ``COST239`` - 11 nodes, 26 links
      - ``USNET`` - 24 nodes, 43 links
      - ``German17`` - 17 nodes, 26 links
      - ``Euro28`` - 28 nodes, 41 links

   Or provide path to custom GraphML file:

   .. code-block:: ini

      network = /path/to/custom_topology.graphml

Physical Parameters
-------------------

``bw_per_slot`` (float, required)
   Bandwidth per spectrum slot (GHz)

   .. code-block:: ini

      bw_per_slot = 12.5

   Common values: 6.25, 12.5, 25

``cores_per_link`` (int, required)
   Number of cores in multi-core fiber

   .. code-block:: ini

      cores_per_link = 1

   Range: 1-12

``multi_fiber`` (bool, required)
   Enable multi-fiber links

   .. code-block:: ini

      multi_fiber = False

Topology Features
-----------------

``const_link_weight`` (bool, required)
   Use constant link weights (ignore distance)

   .. code-block:: ini

      const_link_weight = False

``bi_directional`` (bool, optional)
   Bidirectional links

   .. code-block:: ini

      bi_directional = True

   Default: ``True``

``is_only_core_node`` (bool, required)
   Restrict source/destination to core nodes

   .. code-block:: ini

      is_only_core_node = False

Spectrum Settings
=================

Spectrum Bands
--------------

``c_band`` (int, required)
   Number of slots in C-band

   .. code-block:: ini

      c_band = 320

   Common values: 160, 320, 400

``l_band`` (int, optional)
   Number of slots in L-band

   .. code-block:: ini

      l_band = 0

   Default: 0

``s_band`` (int, optional)
   Number of slots in S-band

   .. code-block:: ini

      s_band = 0

   Default: 0

``o_band`` (int, optional)
   Number of slots in O-band

   .. code-block:: ini

      o_band = 0

   Default: 0

``e_band`` (int, optional)
   Number of slots in E-band

   .. code-block:: ini

      e_band = 0

   Default: 0

SNR Settings
============

Physical Layer Model
--------------------

``snr_type`` (str, required)
   SNR calculation method

   .. code-block:: ini

      snr_type = calculated

   Options:
      - ``calculated`` - Physics-based SNR model
      - ``margin`` - Fixed SNR margin
      - ``none`` - No SNR consideration

``input_power`` (float, required)
   Input power per channel (dBm)

   .. code-block:: ini

      input_power = 0.0

   Range: -10 to 10 dBm

``egn_model`` (bool, required)
   Use Enhanced Gaussian Noise (EGN) model

   .. code-block:: ini

      egn_model = True

Noise Parameters
----------------

``beta`` (float, required)
   Beta parameter for nonlinear noise

   .. code-block:: ini

      beta = 1.0

``theta`` (float, required)
   Theta parameter for crosstalk

   .. code-block:: ini

      theta = 0.5

Crosstalk Settings
------------------

``xt_type`` (str, required)
   Crosstalk model type

   .. code-block:: ini

      xt_type = homogeneous

   Options: ``homogeneous``, ``heterogeneous``, ``none``

``xt_noise`` (bool, required)
   Include crosstalk noise

   .. code-block:: ini

      xt_noise = True

``requested_xt`` (str, required)
   Requested crosstalk level

   .. code-block:: ini

      requested_xt = -20dB

``phi`` (str, required)
   Phase parameter for crosstalk

   .. code-block:: ini

      phi = 0.0

File Settings
=============

Output Format
-------------

``file_type`` (str, required)
   Output file format

   .. code-block:: ini

      file_type = json

   Options: ``json``, ``csv``, ``hdf5``

``run_id`` (str, optional)
   Custom run identifier

   .. code-block:: ini

      run_id = experiment_v1

   Default: Auto-generated timestamp

ML Settings
===========

Deployment
----------

``deploy_model`` (bool, required)
   Deploy trained ML model

   .. code-block:: ini

      deploy_model = False

``ml_model`` (str, optional)
   Path to trained model file

   .. code-block:: ini

      ml_model = logs/ml_models/random_forest_500.joblib

Training
--------

``ml_training`` (bool, optional)
   Enable ML training mode

   .. code-block:: ini

      ml_training = False

   Default: ``False``

``output_train_data`` (bool, optional)
   Save training data during simulation

   .. code-block:: ini

      output_train_data = True

   Default: ``False``

``train_file_path`` (str, optional)
   Path for training data output

   .. code-block:: ini

      train_file_path = data/ml_training/

``test_size`` (float, optional)
   Fraction of data for testing

   .. code-block:: ini

      test_size = 0.2

   Range: 0.1-0.5, Default: 0.2

RL Settings
===========

Agent Configuration
-------------------

``obs_space`` (str, optional)
   Observation space type

   .. code-block:: ini

      obs_space = path_based

   Options: ``path_based``, ``spectrum_based``, ``hybrid``

``device`` (str, optional)
   Training device

   .. code-block:: ini

      device = cuda

   Options: ``cpu``, ``cuda``, ``mps``

   Default: ``cpu``

Model Paths
-----------

``path_algorithm`` (str, optional)
   Path selection algorithm/model

   .. code-block:: ini

      path_algorithm = ppo

``path_model`` (str, optional)
   Path to trained path selection model

   .. code-block:: ini

      path_model = logs/rl_models/path_agent.zip

``core_algorithm`` (str, optional)
   Core selection algorithm

   .. code-block:: ini

      core_algorithm = ppo

``core_model`` (str, optional)
   Path to trained core selection model

   .. code-block:: ini

      core_model = logs/rl_models/core_agent.zip

``spectrum_algorithm`` (str, optional)
   Spectrum allocation algorithm

   .. code-block:: ini

      spectrum_algorithm = ppo

``spectrum_model`` (str, optional)
   Path to trained spectrum agent

   .. code-block:: ini

      spectrum_model = logs/rl_models/spectrum_agent.zip

Training Parameters
-------------------

``is_training`` (bool, optional)
   Enable training mode

   .. code-block:: ini

      is_training = True

   Default: ``False``

``n_trials`` (int, optional)
   Number of training trials

   .. code-block:: ini

      n_trials = 5

``optimize_hyperparameters`` (bool, optional)
   Enable hyperparameter optimization

   .. code-block:: ini

      optimize_hyperparameters = False

``optuna_trials`` (int, optional)
   Number of Optuna optimization trials

   .. code-block:: ini

      optuna_trials = 100

Learning Parameters
-------------------

``gamma`` (float, optional)
   Discount factor

   .. code-block:: ini

      gamma = 0.99

   Range: 0.9-0.999

``epsilon_start`` (float, optional)
   Initial epsilon (exploration rate)

   .. code-block:: ini

      epsilon_start = 1.0

``epsilon_end`` (float, optional)
   Final epsilon

   .. code-block:: ini

      epsilon_end = 0.01

``epsilon_update`` (str, optional)
   Epsilon decay schedule

   .. code-block:: ini

      epsilon_update = linear

   Options: ``linear``, ``exponential``

``alpha_start`` (float, optional)
   Initial learning rate

   .. code-block:: ini

      alpha_start = 0.001

``alpha_end`` (float, optional)
   Final learning rate

   .. code-block:: ini

      alpha_end = 0.0001

``alpha_update`` (str, optional)
   Learning rate schedule

   .. code-block:: ini

      alpha_update = cosine

   Options: ``linear``, ``exponential``, ``cosine``

``decay_rate`` (float, optional)
   Decay rate for exponential schedules

   .. code-block:: ini

      decay_rate = 0.995

Network Architecture
--------------------

``feature_extractor`` (str, optional)
   Feature extraction method

   .. code-block:: ini

      feature_extractor = path_gnn

   Options: ``mlp``, ``path_gnn``, ``graphormer``

``gnn_type`` (str, optional)
   Graph neural network type

   .. code-block:: ini

      gnn_type = gcn

   Options: ``gcn``, ``gat``, ``graphsage``

``layers`` (int, optional)
   Number of network layers

   .. code-block:: ini

      layers = 3

``emb_dim`` (int, optional)
   Embedding dimension

   .. code-block:: ini

      emb_dim = 128

``heads`` (int, optional)
   Number of attention heads (for GAT)

   .. code-block:: ini

      heads = 4

Reward Shaping
--------------

``reward`` (int, optional)
   Reward for successful allocation

   .. code-block:: ini

      reward = 1

``penalty`` (int, optional)
   Penalty for blocking

   .. code-block:: ini

      penalty = -1

``dynamic_reward`` (bool, optional)
   Use dynamic reward calculation

   .. code-block:: ini

      dynamic_reward = False

``core_beta`` (float, optional)
   Core crosstalk penalty weight

   .. code-block:: ini

      core_beta = 0.5

Advanced Options
----------------

``super_channel_space`` (int, optional)
   Super-channel spacing slots

   .. code-block:: ini

      super_channel_space = 1

``path_levels`` (int, optional)
   Path priority levels

   .. code-block:: ini

      path_levels = 3

``conf_param`` (int, optional)
   Confidence parameter

   .. code-block:: ini

      conf_param = 2

``cong_cutoff`` (float, optional)
   Congestion threshold

   .. code-block:: ini

      cong_cutoff = 0.8

``render_mode`` (str, optional)
   Rendering mode for visualization

   .. code-block:: ini

      render_mode = none

   Options: ``none``, ``human``, ``rgb_array``

Example Configurations
======================

Minimal Configuration
---------------------

.. code-block:: ini

   [general_settings]
   erlang_start = 300
   erlang_stop = 500
   erlang_step = 100
   num_requests = 100
   max_iters = 2
   route_method = k_shortest_path
   allocation_method = first_fit

   [topology_settings]
   network = NSFNet

   [spectrum_settings]
   c_band = 320

   [file_settings]
   file_type = json

Production Configuration
------------------------

.. code-block:: ini

   [general_settings]
   erlang_start = 300
   erlang_stop = 800
   erlang_step = 50
   num_requests = 5000
   max_iters = 10
   thread_erlangs = True
   route_method = k_shortest_path
   k_paths = 5
   allocation_method = first_fit
   spectrum_priority = lowest
   save_snapshots = True
   snapshot_step = 500

   [topology_settings]
   network = NSFNet
   bw_per_slot = 12.5
   cores_per_link = 1

   [spectrum_settings]
   c_band = 320

   [snr_settings]
   snr_type = calculated
   egn_model = True

   [file_settings]
   file_type = json

ML Training Configuration
--------------------------

.. code-block:: ini

   [general_settings]
   erlang_start = 400
   erlang_stop = 600
   num_requests = 10000
   route_method = k_shortest_path
   allocation_method = first_fit

   [ml_settings]
   output_train_data = True
   train_file_path = data/ml_training/
   deploy_model = False

   [file_settings]
   file_type = csv

RL Training Configuration
--------------------------

.. code-block:: ini

   [general_settings]
   erlang_start = 400
   erlang_stop = 400
   num_requests = 10000

   [rl_settings]
   is_training = True
   obs_space = path_based
   path_algorithm = ppo
   device = cuda
   gamma = 0.99
   feature_extractor = path_gnn
   layers = 3
   emb_dim = 128

Best Practices
==============

Configuration Management
------------------------

1. **Use version control**: Track config files with Git
2. **Descriptive names**: Name files by experiment purpose
3. **Document changes**: Add comments explaining non-standard values
4. **Template reuse**: Create templates for common scenarios

Performance Tuning
------------------

1. **Start small**: Test with minimal config first
2. **Enable threading**: Use ``thread_erlangs = True`` for speed
3. **Disable snapshots**: If not needed for analysis
4. **Optimize iterations**: Balance accuracy vs speed

Validation
----------

FUSION validates all configuration options on load. Common errors:

- Missing required parameters
- Invalid parameter types
- Out-of-range values
- Incompatible option combinations

See Also
========

* :doc:`running_simulations` - Using configurations
* :doc:`cli_reference` - Command-line overrides
* :doc:`../getting_started/configuration` - Configuration basics
* :doc:`data_management` - Output management
