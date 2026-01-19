.. _configuration-examples:

======================
Configuration Examples
======================

This section provides complete, ready-to-use configuration examples for different
simulation scenarios. Each example includes the full configuration file and
explanation of key parameters.

.. contents:: Configuration Types
   :local:
   :depth: 2

Basic Simulation
================

Standard K-Shortest Path Simulation
-----------------------------------

The most common simulation type - routing requests using K-shortest paths with
first-fit spectrum allocation.

**Use Case:** Baseline performance evaluation, comparison studies

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; Basic K-Shortest Path Simulation
   ; Purpose: Standard routing with first-fit allocation
   ; =============================================================

   [general_settings]
   ; --- Traffic Parameters ---
   erlang_start = 300          ; Starting traffic load
   erlang_stop = 1200          ; Ending traffic load
   erlang_step = 100           ; Increment between loads
   num_requests = 1000         ; Requests per iteration
   holding_time = 3600         ; Average holding time (seconds)

   ; --- Simulation Control ---
   max_iters = 5               ; Statistical iterations per Erlang point

   ; --- Routing Configuration ---
   network = NSFNet            ; Network topology
   k_paths = 3                 ; Number of candidate paths
   route_method = k_shortest_path
   allocation_method = first_fit

   ; --- Output Control ---
   print_step = 0              ; 0 = no progress printing
   save_snapshots = False

   [topology_settings]
   cores_per_link = 1          ; Single-core fiber
   bw_per_slot = 12.5          ; 12.5 GHz per slot

   [spectrum_settings]
   c_band = 320                ; 320 slots = 4 THz C-band

   [file_settings]
   file_type = json            ; Output format

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path basic_simulation.ini \
       --run_id baseline_nsfnet

**Key Parameters Explained:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Explanation
   * - erlang_start/stop/step
     - Defines the traffic load sweep range (300 to 1200 Erlangs)
   * - max_iters
     - Number of independent trials for statistical confidence
   * - k_paths
     - Number of candidate routes to compute (more = better blocking, higher computation)
   * - route_method
     - Algorithm for computing paths (k_shortest_path is standard)
   * - allocation_method
     - How spectrum slots are assigned (first_fit is simplest)

Multi-Core Fiber Simulation
---------------------------

Simulation with space-division multiplexing (SDM) using multiple cores per fiber.

**Use Case:** Evaluating SDM networks, crosstalk studies

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; Multi-Core Fiber Simulation
   ; Purpose: SDM network with multiple cores per link
   ; =============================================================

   [general_settings]
   erlang_start = 1000         ; Higher load for SDM capacity
   erlang_stop = 3000
   erlang_step = 200
   num_requests = 2000
   holding_time = 3600
   max_iters = 5
   network = Pan-European
   k_paths = 4
   route_method = k_shortest_path
   allocation_method = first_fit

   [topology_settings]
   ; Multi-core configuration
   cores_per_link = 7          ; 7-core fiber
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [snr_settings]
   ; Crosstalk awareness
   snr_type = None             ; No SNR calculation
   xt_type = None              ; Crosstalk disabled for basic SDM

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path multicore_simulation.ini \
       --run_id sdm_pan_european

SNR-Aware Simulation
====================

Physical Layer Impairment-Aware Routing
---------------------------------------

Simulation that considers signal quality degradation along the optical path.

**Use Case:** Realistic network modeling, modulation format selection

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; SNR-Aware Simulation
   ; Purpose: Physical layer impairment-aware routing
   ; =============================================================

   [general_settings]
   erlang_start = 300
   erlang_stop = 900
   erlang_step = 100
   num_requests = 1000
   holding_time = 3600
   max_iters = 5
   network = NSFNet
   k_paths = 5                 ; More paths for SNR-aware selection
   route_method = k_shortest_path
   allocation_method = first_fit

   ; SNR-specific routing options
   mod_assumption = REACH_BASED    ; Modulation based on reach

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [snr_settings]
   ; Enable SNR calculation
   snr_type = snr_calculation
   snr_check = True            ; Check SNR before allocation

   ; Physical parameters
   input_power = 0.001         ; Launch power (W)
   beta = 0.5                  ; Spontaneous emission factor
   theta = 1.0                 ; Noise figure factor

   ; Modulation thresholds (dB)
   ; Each modulation format has minimum SNR requirement
   ; Format: [bits_per_symbol, min_snr_db, max_reach_km]

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path snr_aware.ini \
       --run_id snr_study

**SNR Parameters Explained:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Explanation
   * - snr_type
     - ``snr_calculation`` enables SNR computation, ``None`` disables
   * - snr_check
     - If True, verify SNR meets modulation threshold before allocation
   * - input_power
     - Optical launch power (impacts both signal and nonlinear noise)
   * - beta
     - Amplifier spontaneous emission factor (affects ASE noise)
   * - mod_assumption
     - ``REACH_BASED`` selects modulation by distance, ``XTAR_ASSUMPTIONS`` for crosstalk-aware

Crosstalk-Aware Simulation
==========================

Inter-Core Crosstalk (XT) Modeling
----------------------------------

Simulation that models crosstalk between fiber cores in SDM networks.

**Use Case:** Multi-core fiber crosstalk studies, XT-aware resource allocation

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; Crosstalk-Aware Simulation (XTAR)
   ; Purpose: Inter-core crosstalk modeling in SDM networks
   ; =============================================================

   [general_settings]
   erlang_start = 2000         ; High load for XT impact
   erlang_stop = 2100
   erlang_step = 100
   num_requests = 1500
   holding_time = 3600
   max_iters = 4
   network = Pan-European
   k_paths = 4
   mod_assumption = XTAR_ASSUMPTIONS  ; Crosstalk-aware modulation

   ; XT-aware routing
   route_method = xt_aware
   allocation_method = priority_first
   spectrum_priority = CSB     ; Core-Spectrum-Bandwidth priority

   [topology_settings]
   cores_per_link = 7          ; 7-core fiber (typical for XT studies)
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [snr_settings]
   ; Enable crosstalk calculation
   snr_type = xt_calculation
   xt_type = with_length       ; Length-dependent crosstalk

   ; Crosstalk parameters
   ; coupling_coefficient determined by core spacing and fiber design
   ; Typical values: -30 to -40 dB/km for adjacent cores

   [ml_settings]
   ; Optional: ML model for crosstalk prediction
   ml_training = True
   ml_model = decision_tree

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path crosstalk_aware.ini \
       --run_id xtar_study

**Crosstalk Parameters Explained:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Explanation
   * - xt_type
     - ``with_length`` models XT as function of distance, ``without_length`` uses fixed
   * - route_method=xt_aware
     - Uses XT-aware routing algorithm
   * - spectrum_priority
     - ``CSB`` = Core-Spectrum-Bandwidth, ``SCB`` = Spectrum-Core-Bandwidth
   * - mod_assumption=XTAR_ASSUMPTIONS
     - Uses crosstalk-aware modulation format selection

Reinforcement Learning Configurations
=====================================

Online RL Training (Legacy)
---------------------------

Configuration for training RL agents using online simulation interaction.

**Use Case:** Training RL routing policies, algorithm development

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; Online RL Training Configuration (Legacy System)
   ; Purpose: Train RL agent through simulation interaction
   ; =============================================================

   [general_settings]
   erlang_start = 400
   erlang_stop = 800
   erlang_step = 100
   num_requests = 500          ; Shorter episodes for faster training
   holding_time = 3600
   max_iters = 1               ; Single iteration per Erlang (RL handles exploration)
   network = NSFNet
   k_paths = 4

   ; RL routing mode
   route_method = k_shortest_path

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [rl_settings]
   ; --- Training Mode ---
   is_training = True          ; Enable training mode
   path_algorithm = dqn        ; DQN agent (options: dqn, ppo, a2c)

   ; --- Hyperparameters ---
   learning_rate = 0.0003
   gamma = 0.99                ; Discount factor
   epsilon = 1.0               ; Initial exploration rate
   epsilon_decay = 0.995       ; Exploration decay per episode
   epsilon_min = 0.01          ; Minimum exploration

   ; --- Network Architecture ---
   hidden_layers = [128, 64]   ; Neural network layers
   batch_size = 32
   memory_size = 10000         ; Replay buffer size

   ; --- Training Control ---
   target_update_freq = 100    ; Steps between target network updates
   save_freq = 1000            ; Steps between checkpoints

   ; --- Path Selection ---
   path_levels = 4             ; Action space size (matches k_paths)

   ; --- Device ---
   device = auto               ; auto/cpu/cuda

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path rl_training.ini \
       --run_id rl_dqn_training

Offline RL with Behavior Cloning
--------------------------------

Configuration for training RL policies from pre-collected datasets (v6.0+).

**Use Case:** Learning from heuristic demonstrations, imitation learning

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; Offline RL Configuration (Behavior Cloning)
   ; Purpose: Train policy from offline dataset
   ; =============================================================

   [general_settings]
   erlang_start = 500
   erlang_stop = 1000
   erlang_step = 100
   num_requests = 1000
   holding_time = 3600
   max_iters = 5
   network = NSFNet
   k_paths = 4

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [policy_settings]
   ; Use offline RL policy
   policy_type = bc             ; Behavior cloning
   policy_name = bc_nsfnet_v1

   [offline_rl_settings]
   ; Model paths
   bc_model_path = models/bc/nsfnet_policy.pt
   device = cpu

   ; Fallback when BC fails
   fallback_policy = ksp_ff     ; Fall back to KSP + First-Fit

   [dataset_logging_settings]
   ; Log dataset for future training
   log_offline_dataset = True
   dataset_output_path = data/offline_rl/bc_dataset.parquet
   epsilon_mix = 0.1            ; 10% random exploration for diversity

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path offline_rl_bc.ini \
       --run_id bc_evaluation

Offline RL with IQL
-------------------

Configuration for Implicit Q-Learning (IQL) policy evaluation.

**Use Case:** Conservative offline RL, avoiding out-of-distribution actions

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; Offline RL Configuration (IQL)
   ; Purpose: Conservative offline RL evaluation
   ; =============================================================

   [general_settings]
   erlang_start = 500
   erlang_stop = 1000
   erlang_step = 100
   num_requests = 1000
   holding_time = 3600
   max_iters = 5
   network = NSFNet
   k_paths = 4

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [policy_settings]
   ; Use IQL policy
   policy_type = iql
   policy_name = iql_nsfnet_conservative

   [offline_rl_settings]
   iql_model_path = models/iql/nsfnet_policy.pt
   device = cpu
   fallback_policy = ksp_ff

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path offline_rl_iql.ini \
       --run_id iql_evaluation

Supervised Learning Configuration
=================================

ML Model Training Mode
----------------------

Configuration for training supervised ML models (decision trees, etc.) for
routing decisions.

**Use Case:** Feature analysis, interpretable routing models

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; Supervised Learning Configuration
   ; Purpose: Train ML model for routing decisions
   ; =============================================================

   [general_settings]
   erlang_start = 500
   erlang_stop = 1000
   erlang_step = 100
   num_requests = 2000         ; More data for ML training
   holding_time = 3600
   max_iters = 3
   network = NSFNet
   k_paths = 4
   route_method = k_shortest_path
   allocation_method = first_fit

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [ml_settings]
   ; Enable ML training
   ml_training = True
   ml_model = decision_tree    ; Options: decision_tree, random_forest, svm

   ; Model parameters
   max_depth = 10
   min_samples_split = 5

   ; Train/test split
   test_size = 0.2             ; 20% for testing

   ; Model output
   deploy_model = False        ; True to use trained model for inference

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path ml_training.ini \
       --run_id ml_dt_training

ML Model Inference Mode
-----------------------

Configuration for using a trained ML model for routing decisions.

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; ML Inference Configuration
   ; Purpose: Use trained ML model for routing
   ; =============================================================

   [general_settings]
   erlang_start = 500
   erlang_stop = 1500
   erlang_step = 100
   num_requests = 1000
   holding_time = 3600
   max_iters = 5
   network = NSFNet
   k_paths = 4
   route_method = k_shortest_path
   allocation_method = first_fit

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [ml_settings]
   ml_training = False         ; Inference mode
   deploy_model = True         ; Use trained model
   model_path = models/ml/decision_tree_nsfnet.pkl

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path ml_inference.ini \
       --run_id ml_evaluation

Survivability Configurations
============================

1+1 Protection (No Failures)
----------------------------

Configuration for dedicated path protection without failure injection.

**Use Case:** Baseline protection overhead measurement

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; 1+1 Protection Configuration
   ; Purpose: Dedicated path protection baseline
   ; =============================================================

   [general_settings]
   erlang_start = 200          ; Lower load due to protection overhead
   erlang_stop = 600
   erlang_step = 50
   num_requests = 1000
   holding_time = 3600
   max_iters = 5
   network = NSFNet
   k_paths = 4

   ; Protection routing
   route_method = 1plus1_protection

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [protection_settings]
   ; 1+1 dedicated protection
   protection_switchover_ms = 50    ; Switchover time in milliseconds
   node_disjoint_protection = True  ; Node-disjoint backup paths

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path protection_1plus1.ini \
       --run_id protection_baseline

Link Failure Experiment
-----------------------

Configuration for simulating single link failures.

**Use Case:** Network resilience evaluation, protection effectiveness

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; Link Failure Experiment
   ; Purpose: Evaluate network resilience to link failures
   ; =============================================================

   [general_settings]
   erlang_start = 300
   erlang_stop = 800
   erlang_step = 100
   num_requests = 2000         ; More requests to capture failure event
   holding_time = 3600
   max_iters = 10              ; More iterations for failure statistics
   network = NSFNet
   k_paths = 4
   route_method = 1plus1_protection

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [failure_settings]
   ; Single link failure
   failure_type = link

   ; When failure occurs (request index)
   t_fail_arrival_index = 500

   ; How long failure lasts (in arrivals)
   t_repair_after_arrivals = 1000

   ; Which link fails (optional, random if not specified)
   ; failed_link = ["node_A", "node_B"]

   [protection_settings]
   protection_switchover_ms = 50
   restoration_latency_ms = 100
   revert_to_primary = True    ; Revert to primary after repair

   [reporting_settings]
   export_csv = True
   csv_output_path = results/link_failure/

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path link_failure.ini \
       --run_id link_fail_experiment

Node Failure Experiment
-----------------------

Configuration for simulating node failures (more severe than link failures).

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; Node Failure Experiment
   ; Purpose: Evaluate network resilience to node failures
   ; =============================================================

   [general_settings]
   erlang_start = 200
   erlang_stop = 600
   erlang_step = 50
   num_requests = 2000
   holding_time = 3600
   max_iters = 10
   network = Pan-European      ; Larger network for node failure
   k_paths = 5                 ; More paths for redundancy
   route_method = 1plus1_protection

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [failure_settings]
   failure_type = node
   t_fail_arrival_index = 500
   t_repair_after_arrivals = 1500

   ; Node to fail (optional)
   ; failed_node = "node_5"

   [protection_settings]
   protection_switchover_ms = 50
   node_disjoint_protection = True  ; Essential for node failure survival

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path node_failure.ini \
       --run_id node_fail_experiment

SRLG Failure Experiment
-----------------------

Configuration for Shared Risk Link Group failures (multiple simultaneous failures).

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; SRLG Failure Experiment
   ; Purpose: Evaluate resilience to correlated failures
   ; =============================================================

   [general_settings]
   erlang_start = 200
   erlang_stop = 500
   erlang_step = 50
   num_requests = 2000
   holding_time = 3600
   max_iters = 10
   network = USbackbone60      ; Large network with SRLG definitions
   k_paths = 5
   route_method = 1plus1_protection

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [failure_settings]
   failure_type = srlg         ; Shared Risk Link Group
   t_fail_arrival_index = 500
   t_repair_after_arrivals = 2000

   ; SRLG to fail (defined in network topology)
   ; srlg_id = "conduit_1"

   [protection_settings]
   protection_switchover_ms = 50
   srlg_disjoint_protection = True  ; SRLG-disjoint backup paths

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path srlg_failure.ini \
       --run_id srlg_fail_experiment

CI/CD and Testing Configurations
================================

Cross-Platform CI Configuration
-------------------------------

Configuration optimized for continuous integration pipelines.

**Use Case:** Automated testing, GitHub Actions, Jenkins

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; Cross-Platform CI Configuration
   ; Purpose: Fast, portable configuration for CI/CD
   ; =============================================================

   [general_settings]
   ; Minimal load for fast execution
   erlang_start = 300
   erlang_stop = 400
   erlang_step = 100
   num_requests = 100          ; Minimal for quick tests
   holding_time = 3600
   max_iters = 1               ; Single iteration
   network = NSFNet            ; Smallest standard topology
   k_paths = 2                 ; Minimal paths

   route_method = k_shortest_path
   allocation_method = first_fit

   ; No output files to avoid path issues
   print_step = 0
   save_snapshots = False

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

   [file_settings]
   file_type = json

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path fusion/configs/templates/cross_platform.ini \
       --run_id ci_test

Quick Validation Test
---------------------

Ultra-minimal configuration for syntax and import validation.

**Configuration:**

.. code-block:: ini

   ; =============================================================
   ; Quick Validation Configuration
   ; Purpose: Fastest possible validation run
   ; =============================================================

   [general_settings]
   erlang_start = 300
   erlang_stop = 300           ; Single point
   erlang_step = 100
   num_requests = 10           ; Absolute minimum
   holding_time = 3600
   max_iters = 1
   network = NSFNet
   k_paths = 1                 ; Single path

   [topology_settings]
   cores_per_link = 1
   bw_per_slot = 12.5

   [spectrum_settings]
   c_band = 320

**Run Command:**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim \
       --config_path quick_test.ini \
       --run_id validate

Using Profiles Instead of Files
-------------------------------

For quick testing without creating files:

.. code-block:: python

   from fusion.configs import ConfigRegistry

   registry = ConfigRegistry()

   # Create quick test configuration programmatically
   config_manager = registry.create_profile_config('quick_test')

   # Or combine template with custom overrides
   config_manager = registry.create_custom_config(
       'minimal',
       overrides={
           'general_settings.num_requests': 10,
           'general_settings.max_iters': 1,
       }
   )

Configuration Comparison Summary
================================

.. list-table:: Configuration Quick Reference
   :header-rows: 1
   :widths: 20 15 15 15 35

   * - Use Case
     - Template
     - key_paths
     - max_iters
     - Key Parameters
   * - Quick Test
     - minimal.ini
     - 2-3
     - 1-2
     - num_requests=100
   * - Production
     - default.ini
     - 3-4
     - 5-10
     - num_requests=1000+
   * - SNR Study
     - (custom)
     - 4-5
     - 5
     - snr_type=snr_calculation
   * - Crosstalk
     - xtar_example
     - 4
     - 4
     - xt_type=with_length, cores=7
   * - RL Training
     - (custom)
     - 4
     - 1
     - is_training=True
   * - Offline RL
     - (custom)
     - 4
     - 5
     - policy_type=bc/iql
   * - Protection
     - (custom)
     - 4-5
     - 5-10
     - route_method=1plus1_protection
   * - Failure Test
     - (custom)
     - 5
     - 10
     - failure_type=link/node/srlg
   * - CI/CD
     - cross_platform
     - 2
     - 1
     - num_requests=100

Related Documentation
=====================

- :ref:`cli-module` - Complete CLI argument reference
- :ref:`configuration-tutorials` - Step-by-step tutorials
- ``fusion/configs/templates/`` - Template files in the repository
