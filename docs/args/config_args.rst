
Configuration Parameters
========================

This section provides a reference for all configuration file parameters.
Parameters are grouped by section as defined in `example_config.ini`. This file
can be passed using the `--config_path` argument when running the simulator.

.. automodule:: arg_scripts.config_args
    :members:
    :undoc-members:

General Settings
----------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - holding_time
     - float
     - Mean connection holding time (in seconds)
   * - mod_assumption
     - str
     - Modulation assumption strategy (e.g., DEFAULT, dynamic)
   * - mod_assumption_path
     - str
     - Path to modulation format definition JSON
   * - erlang_start
     - int
     - Starting Erlang value (offered load)
   * - erlang_stop
     - int
     - Final Erlang value
   * - erlang_step
     - int
     - Step size between Erlang values
   * - max_iters
     - int
     - Number of iterations per Erlang
   * - guard_slots
     - int
     - Number of guard slots
   * - max_segments
     - int
     - Maximum number of segments per path
   * - thread_erlangs
     - bool
     - Run Erlangs in parallel threads
   * - dynamic_lps
     - bool
     - Enable dynamic lightpath selection
   * - fixed_grid
     - bool
     - Use a fixed ITU grid for spectrum
   * - pre_calc_mod_selection
     - bool
     - Pre-compute modulation selection per path
   * - spectrum_priority
     - str
     - Spectrum prioritization policy
   * - num_requests
     - int
     - Number of requests per simulation iteration
   * - request_distribution
     - dict
     - Distribution of request bandwidths
   * - allocation_method
     - str
     - Spectrum allocation method (e.g., first_fit)
   * - k_paths
     - int
     - Number of candidate paths per source-destination pair
   * - route_method
     - str
     - Routing method used
   * - save_snapshots
     - bool
     - Save internal state snapshots
   * - snapshot_step
     - int
     - Interval between snapshots
   * - print_step
     - int
     - Console log print frequency
   * - save_step
     - int
     - Statistics file save interval
   * - save_start_end_slots
     - bool
     - Save slot indices for allocations

Topology Settings
-----------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - network
     - str
     - Name of the network topology
   * - bw_per_slot
     - float
     - Bandwidth per slot (GHz)
   * - cores_per_link
     - int
     - Number of spatial cores per fiber
   * - const_link_weight
     - bool
     - Use constant link weights
   * - is_only_core_node
     - bool
     - Allow multiple cores only on core nodes
   * - multi_fiber
     - bool
     - Enable multi-fiber links
   * - bi_directional
     - bool
     - Treat links as bidirectional

Spectrum Settings
-----------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - c_band
     - int
     - Number of slots in the C-band
   * - o_band, e_band, s_band, l_band
     - int
     - Optional bands and their slot counts

SNR Settings
------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - snr_type
     - str
     - SNR estimation method (e.g., BER-based)
   * - xt_type
     - str
     - Cross-talk model type
   * - beta, theta
     - float
     - Cross-talk model coefficients
   * - input_power
     - float
     - Input power in Watts
   * - egn_model
     - bool
     - Use Enhanced GN model
   * - phi
     - dict
     - Nonlinear coefficients for each modulation format
   * - xt_noise
     - bool
     - Enable XT-aware noise
   * - requested_xt
     - dict
     - Required XT thresholds per modulation

RL Settings
-----------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - obs_space
     - str
     - Observation space type
   * - n_trials
     - int
     - Number of training trials
   * - device
     - str
     - Computation device (cpu/cuda)
   * - optimize_hyperparameters
     - bool
     - Enable Optuna-based optimization
   * - optuna_trials
     - int
     - Number of optimization trials
   * - is_training
     - bool
     - Enable training mode
   * - path_algorithm, core_algorithm, spectrum_algorithm
     - str
     - Algorithm type for each decision point
   * - path_model, core_model, spectrum_model
     - str
     - File paths to pretrained models
   * - super_channel_space
     - int
     - Superchannel slot count
   * - alpha_start, alpha_end, alpha_update
     - float, str
     - Alpha values and update method
   * - gamma
     - float
     - Discount factor
   * - epsilon_start, epsilon_end, epsilon_update
     - float, str
     - Epsilon values and decay policy
   * - path_levels
     - int
     - Path grouping levels
   * - decay_rate
     - float
     - Decay constant for scheduling
   * - feature_extractor, gnn_type
     - str
     - GNN architecture and input extraction
   * - layers, emb_dim, heads
     - int
     - GNN model depth and complexity
   * - conf_param
     - float
     - Confidence bound parameter
   * - cong_cutoff
     - float
     - Congestion threshold
   * - reward, penalty
     - float
     - RL reward shaping
   * - dynamic_reward
     - bool
     - Enable adaptive reward
   * - core_beta
     - float
     - Congestion penalty on core usage

ML Settings
-----------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - deploy_model
     - bool
     - Deploy pre-trained model
   * - output_train_data
     - bool
     - Save simulation data for training
   * - ml_training
     - bool
     - Enable machine learning model training
   * - ml_model
     - str
     - Model type (e.g., SVM, RF)
   * - train_file_path
     - str
     - Path to dataset
   * - test_size
     - float
     - Proportion of dataset used for testing

File Settings
-------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - file_type
     - str
     - Output format (e.g., csv, json)
