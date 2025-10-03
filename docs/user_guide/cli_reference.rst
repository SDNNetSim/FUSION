=============
CLI Reference
=============

Complete command-line interface reference for all FUSION commands and options.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

FUSION provides several command-line tools for different operations:

- ``fusion-sim`` - Run optical network simulations
- ``fusion-train`` - Train ML/RL agents
- ``fusion-gui`` - Launch graphical interface
- ``fusion-evaluate`` - Evaluate trained models
- ``fusion-plot`` - Generate visualizations

All commands support ``--help`` for detailed usage information.

fusion-sim
==========

Run optical network simulations with specified configuration.

Basic Usage
-----------

.. code-block:: bash

   fusion-sim --config path/to/config.ini

Common Options
--------------

**Configuration**

``--config PATH``
   Path to configuration file (INI format)

   .. code-block:: bash

      fusion-sim --config configs/nsfnet_sim.ini

``--config_path PATH``
   Alias for ``--config``

**Simulation Parameters**

``--erlang_start FLOAT``
   Starting Erlang value for traffic load sweep

   .. code-block:: bash

      fusion-sim --config config.ini --erlang_start 300

``--erlang_stop FLOAT``
   Ending Erlang value

``--erlang_step FLOAT``
   Step size for Erlang sweep

``--num_requests INT``
   Number of connection requests per iteration

   .. code-block:: bash

      fusion-sim --config config.ini --num_requests 5000

``--max_iters INT``
   Maximum iterations per Erlang value

**Network Configuration**

``--network NAME``
   Network topology to use

   Options: ``NSFNet``, ``COST239``, ``USNET``, ``German17``, ``Euro28``, or path to GraphML file

   .. code-block:: bash

      fusion-sim --config config.ini --network COST239

``--cores_per_link INT``
   Number of fiber cores per link (multi-core fibers)

``--c_band INT``
   Number of spectrum slots in C-band

   .. code-block:: bash

      fusion-sim --config config.ini --c_band 320

**Algorithm Selection**

``--route_method NAME``
   Routing algorithm

   Options: ``k_shortest_path``, ``shortest_path``, ``ml``, ``rl``

   .. code-block:: bash

      fusion-sim --config config.ini --route_method k_shortest_path

``--allocation_method NAME``
   Spectrum allocation algorithm

   Options: ``first_fit``, ``best_fit``, ``random_fit``, ``ml``, ``rl``

``--k_paths INT``
   Number of candidate paths (for k-shortest path routing)

**Output Control**

``--file_type TYPE``
   Output file format

   Options: ``json``, ``csv``, ``hdf5``

   .. code-block:: bash

      fusion-sim --config config.ini --file_type json

``--save_snapshots BOOL``
   Enable network state snapshots

``--snapshot_step INT``
   Frequency of snapshots (requests between snapshots)

``--debug``
   Enable debug logging

   .. code-block:: bash

      fusion-sim --config config.ini --debug

**Performance Options**

``--thread_erlangs``
   Run Erlang sweep in parallel threads

   .. code-block:: bash

      fusion-sim --config config.ini --thread_erlangs

``--quiet``
   Minimal console output

Examples
--------

**Basic simulation:**

.. code-block:: bash

   fusion-sim --config configs/minimal.ini

**Override config parameters:**

.. code-block:: bash

   fusion-sim --config configs/base.ini \
              --erlang_start 400 \
              --erlang_stop 800 \
              --network COST239

**High-performance simulation:**

.. code-block:: bash

   fusion-sim --config configs/production.ini \
              --thread_erlangs \
              --num_requests 10000

**Debug mode:**

.. code-block:: bash

   fusion-sim --config configs/test.ini --debug

fusion-train
============

Train machine learning or reinforcement learning agents.

Basic Usage
-----------

.. code-block:: bash

   fusion-train --agent_type rl --config training_config.ini

Required Options
----------------

``--agent_type TYPE``
   Type of agent to train

   Options: ``ml`` (machine learning), ``rl`` (reinforcement learning)

   .. code-block:: bash

      fusion-train --agent_type rl --config config.ini

ML Training Options
-------------------

``--algorithm NAME``
   ML algorithm

   Options: ``decision_tree``, ``random_forest``, ``gradient_boosting``, ``svm``

   .. code-block:: bash

      fusion-train --agent_type ml \
                   --algorithm random_forest \
                   --config ml_config.ini

``--training_data PATH``
   Path to training data CSV

``--test_size FLOAT``
   Fraction of data for testing (default: 0.2)

``--cv_folds INT``
   Number of cross-validation folds

RL Training Options
-------------------

``--algorithm NAME``
   RL algorithm

   Options: ``ppo``, ``a2c``, ``dqn``, ``qr_dqn``, ``epsilon_greedy``

   .. code-block:: bash

      fusion-train --agent_type rl \
                   --algorithm ppo \
                   --config rl_config.ini

``--n_timesteps INT``
   Total training timesteps

   .. code-block:: bash

      fusion-train --agent_type rl --n_timesteps 1000000

``--n_steps INT``
   Rollout steps (PPO/A2C)

``--batch_size INT``
   Training batch size

``--learning_rate FLOAT``
   Learning rate

``--device NAME``
   Training device

   Options: ``cpu``, ``cuda``, ``mps``

   .. code-block:: bash

      fusion-train --agent_type rl --device cuda

Advanced Options
----------------

``--optimize_hyperparameters``
   Enable automatic hyperparameter tuning with Optuna

   .. code-block:: bash

      fusion-train --agent_type rl \
                   --optimize_hyperparameters \
                   --n_trials 100

``--n_trials INT``
   Number of Optuna trials for hyperparameter search

``--n_envs INT``
   Number of parallel environments (RL only)

Examples
--------

**Train ML model:**

.. code-block:: bash

   fusion-train --agent_type ml \
                --algorithm random_forest \
                --config ml_config.ini

**Train RL agent:**

.. code-block:: bash

   fusion-train --agent_type rl \
                --algorithm ppo \
                --config rl_config.ini \
                --n_timesteps 500000 \
                --device cuda

**Hyperparameter optimization:**

.. code-block:: bash

   fusion-train --agent_type rl \
                --config config.ini \
                --optimize_hyperparameters \
                --n_trials 50

fusion-gui
==========

Launch the FUSION graphical user interface.

Basic Usage
-----------

.. code-block:: bash

   fusion-gui

Options
-------

``--config PATH``
   Optional configuration file to load

``--debug``
   Enable debug mode

``--theme NAME``
   GUI theme

   Options: ``light``, ``dark``

Examples
--------

**Launch GUI:**

.. code-block:: bash

   fusion-gui

**Launch with specific config:**

.. code-block:: bash

   fusion-gui --config configs/gui_defaults.ini

**Debug mode:**

.. code-block:: bash

   fusion-gui --debug

fusion-evaluate
===============

Evaluate trained ML/RL models on test scenarios.

Basic Usage
-----------

.. code-block:: bash

   fusion-evaluate --agent_path logs/ppo_agent.zip --config eval_config.ini

Options
-------

``--agent_path PATH``
   Path to trained agent/model file

   .. code-block:: bash

      fusion-evaluate --agent_path logs/models/best_agent.zip

``--config PATH``
   Evaluation configuration file

``--n_eval_episodes INT``
   Number of evaluation episodes (RL)

   .. code-block:: bash

      fusion-evaluate --agent_path agent.zip \
                      --config config.ini \
                      --n_eval_episodes 100

``--deterministic``
   Use deterministic policy (RL)

``--output PATH``
   Output file for evaluation results

Examples
--------

**Evaluate RL agent:**

.. code-block:: bash

   fusion-evaluate --agent_path logs/ppo_agent.zip \
                   --config validation.ini \
                   --n_eval_episodes 200

**Evaluate ML model:**

.. code-block:: bash

   fusion-evaluate --agent_path models/rf_model.joblib \
                   --config test_scenarios.ini

fusion-plot
===========

Generate plots from simulation results.

Basic Usage
-----------

.. code-block:: bash

   fusion-plot --results data/results/simulation.json --plot_type blocking

Options
-------

``--results PATH``
   Path to results file (JSON/CSV)

   .. code-block:: bash

      fusion-plot --results data/results/nsfnet.json

``--plot_type TYPE``
   Type of plot to generate

   Options: ``blocking``, ``bandwidth``, ``utilization``, ``comparison``

   .. code-block:: bash

      fusion-plot --results results.json --plot_type blocking

``--save PATH``
   Output file path

   .. code-block:: bash

      fusion-plot --results results.json \
                  --plot_type blocking \
                  --save plots/blocking_prob.png

``--format FORMAT``
   Output format

   Options: ``png``, ``pdf``, ``svg``, ``html``

``--dpi INT``
   Resolution (dots per inch)

   .. code-block:: bash

      fusion-plot --results results.json \
                  --save plot.png \
                  --dpi 300

``--interactive``
   Generate interactive plot

Examples
--------

**Basic plot:**

.. code-block:: bash

   fusion-plot --results data/results/sim.json \
               --plot_type blocking \
               --save plots/bp.png

**High-resolution PDF:**

.. code-block:: bash

   fusion-plot --results results.json \
               --plot_type blocking \
               --save paper_figure.pdf \
               --dpi 300

**Interactive HTML:**

.. code-block:: bash

   fusion-plot --results results.json \
               --plot_type comparison \
               --save interactive.html \
               --interactive

Global Options
==============

Options available across all commands:

``--help``, ``-h``
   Show help message and exit

   .. code-block:: bash

      fusion-sim --help

``--version``
   Show version and exit

   .. code-block:: bash

      fusion-sim --version

``--verbose``, ``-v``
   Increase verbosity

   .. code-block:: bash

      fusion-sim --config config.ini -v

``--quiet``, ``-q``
   Suppress output

Environment Variables
=====================

FUSION respects several environment variables:

``FUSION_CONFIG_PATH``
   Default configuration directory

   .. code-block:: bash

      export FUSION_CONFIG_PATH=/path/to/configs
      fusion-sim --config myconfig.ini  # Looks in FUSION_CONFIG_PATH

``FUSION_DATA_DIR``
   Default data output directory

``FUSION_LOG_LEVEL``
   Logging level

   Options: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``

   .. code-block:: bash

      export FUSION_LOG_LEVEL=DEBUG
      fusion-sim --config config.ini

``FUSION_DEVICE``
   Default compute device for ML/RL

   Options: ``cpu``, ``cuda``, ``mps``

Configuration File Syntax
==========================

INI Format
----------

FUSION uses INI-format configuration files:

.. code-block:: ini

   [general_settings]
   erlang_start = 300
   erlang_stop = 700
   erlang_step = 100
   num_requests = 1000

   [topology_settings]
   network = NSFNet
   cores_per_link = 1

   [spectrum_settings]
   c_band = 320

   [file_settings]
   file_type = json

See :doc:`configuration_reference` for complete options.

Command-Line Override
---------------------

Command-line arguments override configuration file values:

.. code-block:: bash

   # config.ini has erlang_start = 300
   # This overrides it to 400
   fusion-sim --config config.ini --erlang_start 400

Exit Codes
==========

All FUSION commands use standard exit codes:

- ``0`` - Success
- ``1`` - General error or interruption
- ``2`` - Command-line argument error

Check exit codes in scripts:

.. code-block:: bash

   fusion-sim --config config.ini
   if [ $? -eq 0 ]; then
       echo "Simulation succeeded"
   else
       echo "Simulation failed"
   fi

Shell Completion
================

Enable tab completion for bash/zsh:

Bash
----

.. code-block:: bash

   # Add to ~/.bashrc
   eval "$(register-python-argcomplete fusion-sim)"
   eval "$(register-python-argcomplete fusion-train)"

Zsh
---

.. code-block:: bash

   # Add to ~/.zshrc
   autoload -U bashcompinit
   bashcompinit
   eval "$(register-python-argcomplete fusion-sim)"

Common Workflows
================

Complete Simulation Workflow
-----------------------------

.. code-block:: bash

   # 1. Run simulation
   fusion-sim --config configs/experiment.ini

   # 2. Generate plots
   fusion-plot --results data/results/simulation.json \
               --plot_type blocking \
               --save plots/results.png

   # 3. Compare with baseline
   fusion-plot --results data/results/simulation.json \
               --plot_type comparison \
               --save plots/comparison.png

ML Training Workflow
--------------------

.. code-block:: bash

   # 1. Collect training data
   fusion-sim --config data_collection.ini

   # 2. Train model
   fusion-train --agent_type ml \
                --algorithm random_forest \
                --training_data data/ml_training/data.csv

   # 3. Evaluate model
   fusion-evaluate --agent_path logs/ml_models/rf_model.joblib \
                   --config validation.ini

RL Training Workflow
--------------------

.. code-block:: bash

   # 1. Train agent
   fusion-train --agent_type rl \
                --algorithm ppo \
                --config rl_training.ini \
                --n_timesteps 500000 \
                --device cuda

   # 2. Evaluate agent
   fusion-evaluate --agent_path logs/rl_models/ppo_agent.zip \
                   --config eval.ini \
                   --n_eval_episodes 100

   # 3. Deploy in simulation
   fusion-sim --config production.ini  # With RL agent path in config

Troubleshooting
===============

Command Not Found
-----------------

**Error**: ``fusion-sim: command not found``

**Solution**:

.. code-block:: bash

   # Ensure FUSION is installed
   pip install -e .

   # Or run via Python module
   python -m fusion.cli.run_sim --config config.ini

Permission Denied
-----------------

**Error**: ``Permission denied`` when saving results

**Solution**:

.. code-block:: bash

   # Check write permissions
   mkdir -p data/results
   chmod u+w data/results

Module Import Errors
--------------------

**Error**: ``ModuleNotFoundError: No module named 'xyz'``

**Solution**:

.. code-block:: bash

   # Install missing dependencies
   pip install -e .[all]  # All optional dependencies

   # Or specific groups
   pip install -e .[ml,rl,gui]

Next Steps
==========

- :doc:`configuration_reference` - Complete configuration options
- :doc:`running_simulations` - Detailed simulation guide
- :doc:`machine_learning` - ML training details
- :doc:`reinforcement_learning` - RL training details

See Also
========

* :doc:`../getting_started/quickstart` - Quick start guide
* :doc:`data_management` - Managing output data
* :doc:`../developer/contributing` - Contributing to FUSION
