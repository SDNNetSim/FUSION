.. _cli-module:

==========
CLI Module
==========

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Command-line interface for running simulations and training agents
   :Location: ``fusion/cli/``
   :Key Files: ``run_sim.py``, ``run_train.py``, ``config_setup.py``, ``main_parser.py``
   :Depends On: ``fusion.configs``, ``fusion.sim``, ``fusion.utils.logging_config``
   :Used By: End users running simulations, training pipelines, automation scripts

The CLI module provides the command-line interface for FUSION. It handles argument
parsing, configuration loading, and dispatches to the appropriate simulation or
training pipelines.

Developers work here when adding new CLI arguments, modifying configuration loading
behavior, or creating new entry point commands.

.. warning:: **CLI Arguments Override ALL Processes**

   When you specify a CLI argument (e.g., ``--k_paths=3``), it overrides that value
   for **ALL processes** in a multi-process simulation, regardless of what individual
   process sections specify in the INI file.

   Example: If your INI file has::

      [s1]
      k_paths = 5

      [s2]
      k_paths = 7

   Running with ``--k_paths=3`` will set ``k_paths=3`` for **both** s1 and s2.

Key Concepts
============

Required vs Optional Configuration
----------------------------------

Configuration options are divided into two categories:

**Required Options** (``SIM_REQUIRED_OPTIONS_DICT`` in ``fusion/configs/schema.py``)
   Must be present in the INI file's ``[general_settings]`` section. The simulation
   will fail to start if any required option is missing.

   Examples: ``erlang_start``, ``erlang_stop``, ``network``, ``num_requests``

**Optional Options** (``OPTIONAL_OPTIONS_DICT`` in ``fusion/configs/schema.py``)
   Can be omitted from the INI file. When omitted, sensible defaults are used.
   These are organized into nested sections like ``[rl_settings]``, ``[routing_settings]``.

   Examples: ``k_paths``, ``policy_type``, ``is_training``

Default Configuration File
--------------------------

The default configuration file path is:

.. code-block:: text

   <project_root>/ini/run_ini/config.ini

This is defined in ``fusion/configs/constants.py``:

.. code-block:: python

   DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "ini", "run_ini", "config.ini")

**To use a different configuration file:**

.. code-block:: bash

   # Specify absolute path
   python -m fusion.cli.run_sim run_sim --config_path /path/to/your/config.ini --run_id my_run

   # Specify relative path (resolved from project root)
   python -m fusion.cli.run_sim run_sim --config_path configs/my_experiment.ini --run_id my_run

   # Use tilde expansion
   python -m fusion.cli.run_sim run_sim --config_path ~/configs/my_config.ini --run_id my_run

**Where to put configuration files:**

- ``fusion/configs/templates/`` - Template configurations for common scenarios
- ``fusion/configs/examples/`` - Example configurations with documentation
- ``ini/run_ini/`` - Default location for user configurations

Multi-Process Configuration
---------------------------

FUSION supports running multiple simulation configurations in parallel using
process sections. Each process section is named ``s1``, ``s2``, ``s3``, etc.

.. code-block:: ini

   [general_settings]
   ; Base configuration - all processes inherit these values
   network = NSFNet
   num_requests = 10000
   k_paths = 3

   [s1]
   ; Process 1: Override k_paths for this process only
   k_paths = 5

   [s2]
   ; Process 2: Different k_paths value
   k_paths = 7

   [s3]
   ; Process 3: Uses base k_paths=3 (no override)

**How it works:**

1. ``[general_settings]`` defines the base configuration
2. Process sections (``[s1]``, ``[s2]``, etc.) inherit from ``[general_settings]``
3. Values in process sections override the base values for that process only
4. CLI arguments override ALL processes (see warning above)

**Pattern:** Process sections must match ``^s\d`` (s followed by a digit).

Legacy vs Orchestrator Systems
------------------------------

FUSION has two routing systems that coexist:

.. list-table:: Routing System Comparison
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - Legacy System
     - Orchestrator (v6.0.0)
   * - CLI Args
     - ``--route_method``, ``--k_paths``, ``--allocation_method``
     - ``--policy-type``, ``--policy-name``, ``--protection-enabled``
   * - Architecture
     - Tightly coupled (path computation + spectrum allocation)
     - Decoupled (path computation -> policy selection -> spectrum allocation)
   * - Module Location
     - ``fusion/modules/routing/``
     - ``fusion/policies/``
   * - Interface
     - ``AbstractRoutingAlgorithm``
     - ``ControlPolicy`` protocol
   * - SL/RL Support
     - Limited
     - Native support
   * - Status
     - Maintained for backward compatibility
     - New development (v6.0.0+)

**Legacy System Flow:**

.. code-block:: text

   Request -> KShortestPath.route() -> [computes paths + assigns spectrum] -> Result

**Orchestrator System Flow:**

.. code-block:: text

   Request -> Orchestrator computes K paths -> Policy selects one -> Orchestrator allocates spectrum -> Result

.. note::

   The legacy ``--route_method`` arguments will eventually be deprecated in favor of
   the orchestrator ``--policy-*`` arguments. For new experiments, prefer the
   orchestrator system.

Architecture
============

Module Structure
----------------

.. code-block:: text

   fusion/cli/
   ├── __init__.py              # Module exports and public API
   ├── config_setup.py          # Configuration loading and validation
   ├── constants.py             # Exit codes and constants
   ├── main_parser.py           # Argument parser construction
   ├── run_gui.py               # GUI entry point (not supported - raises error)
   ├── run_sim.py               # Simulation entry point
   ├── run_train.py             # Training entry point
   ├── utils.py                 # Entry point wrapper utilities
   ├── parameters/              # CLI argument definitions
   │   ├── __init__.py          # Exports all parameter functions
   │   ├── analysis.py          # Statistics, plotting, export args
   │   ├── gui.py               # GUI args (not supported)
   │   ├── network.py           # Network topology args
   │   ├── policy.py            # Orchestrator policy args (v6.0.0)
   │   ├── registry.py          # ArgumentRegistry coordinator
   │   ├── routing.py           # Legacy routing args
   │   ├── shared.py            # Common args (config, debug, output)
   │   ├── simulation.py        # Compatibility module
   │   ├── snr.py               # SNR/modulation args
   │   ├── survivability.py     # Failure injection, protection args
   │   ├── traffic.py           # Erlang, request args
   │   └── training.py          # RL/SL training args
   └── tests/
       ├── __init__.py
       ├── conftest.py
       └── test_*.py

Visual: CLI Module Interactions
-------------------------------

**How the CLI components interact:**

.. code-block:: text

   User Command Line
         │
         ▼
   ┌─────────────────┐
   │   Entry Point   │  run_sim.py / run_train.py
   │   (run_*.py)    │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  main_parser.py │  Builds ArgumentParser using registry
   │  + registry.py  │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ parameters/*.py │  Defines argument groups
   │  (modular args) │  (network, routing, traffic, etc.)
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ config_setup.py │  Loads INI file + applies CLI overrides
   └────────┬────────┘
            │
            ▼
   ┌─────────────────────────────────────────────────┐
   │              Simulation Pipeline                │
   │  ┌──────────────────┐  ┌────────────────────┐   │
   │  │  Legacy Pipeline │  │ Orchestrator (v6)  │   │
   │  │  (modules/       │  │ (policies/,        │   │
   │  │   routing/)      │  │  orchestrator/)    │   │
   │  └──────────────────┘  └────────────────────┘   │
   └─────────────────────────────────────────────────┘

**Step-by-step execution flow:**

**1. User runs command**

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim --config_path config.ini --run_id test1 --k_paths 5

**2. Entry point** (``run_sim.py``)

.. code-block:: python

   # run_sim.py main()
   stop_flag = multiprocessing.Event()
   parser = build_parser()
   args = parser.parse_args()
   return run_simulation_pipeline(args, stop_flag)

**3. Parser construction** (``main_parser.py`` + ``registry.py``)

.. code-block:: python

   # main_parser.py build_parser()
   parser = argparse.ArgumentParser()
   ArgumentRegistry.add_all_groups(parser)  # Registers all argument groups
   return parser

**4. Configuration loading** (``config_setup.py``)

.. code-block:: python

   # Resolves config path, reads INI, applies CLI overrides
   config = ConfigParser()
   config.read(args.config_path)
   process_required_options(config)
   process_optional_options(config)
   apply_cli_overrides(config, args)  # Overrides ALL processes (s1, s2, etc.)
   return config

**5. Pipeline execution**

.. code-block:: python

   # Config passed to simulation engine
   # Engine uses legacy routing OR orchestrator based on config
   engine = SimulationEngine(config)
   engine.run()

Entry Point Wrapper Pattern
---------------------------

The ``utils.py`` module provides wrapper functions for entry points:

.. code-block:: python

   # In utils.py
   def create_entry_point_wrapper(
       main_func: Callable[[], int],
       _legacy_name: str,
       _entry_point_description: str,
   ) -> tuple[Callable[[], int], Callable[[], None]]:
       """
       Create standardized entry point wrapper functions.

       Returns both a legacy compatibility function and a main entry point
       function that handles sys.exit.

       TODO (v6.1.0): Remove this function - use create_main_wrapper instead.
       """
       def legacy_main() -> int:
           return main_func()

       def main_entry() -> None:
           sys.exit(main_func())

       return legacy_main, main_entry

**Usage in run_train.py:**

.. code-block:: python

   # Create entry point functions using shared utilities
   train_main, run_train_main = create_entry_point_wrapper(
       main,
       "training",
       "Convenience function that handles the sys.exit call...",
   )

   if __name__ == "__main__":
       run_train_main()

.. note::

   The ``create_entry_point_wrapper`` function exists for backward compatibility.
   New code should use ``create_main_wrapper`` which is simpler. Both will be
   consolidated in v6.1.0.

Script Reference
================

run_sim.py
----------

:Purpose: Entry point for running network simulations
:Command: ``python -m fusion.cli.run_sim run_sim``
:Main Function: ``main(stop_flag=None) -> int``

**Exit Codes:**

- ``0``: Success
- ``1``: Error or keyboard interrupt

**Usage:**

.. code-block:: bash

   # Basic simulation
   python -m fusion.cli.run_sim run_sim --config_path config.ini --run_id my_simulation

   # With CLI overrides
   python -m fusion.cli.run_sim run_sim --config_path config.ini --run_id test --k_paths 5 --verbose

run_train.py
------------

:Purpose: Entry point for training RL/SL agents
:Command: ``python -m fusion.cli.run_train``
:Main Function: ``main() -> int``

**Exit Codes:**

- ``0`` (SUCCESS_EXIT_CODE): Training completed successfully
- ``1`` (ERROR_EXIT_CODE): Error occurred
- ``1`` (INTERRUPT_EXIT_CODE): User interrupted

**Usage:**

.. code-block:: bash

   # Train RL agent
   python -m fusion.cli.run_train --agent_type rl --config_path config.ini

   # Train SL agent
   python -m fusion.cli.run_train --agent_type sl --config_path config.ini --ml_training

run_gui.py
----------

:Purpose: GUI entry point (NOT SUPPORTED in v6.0.0)
:Status: Raises ``GUINotSupportedError`` - planned for v6.1.0

.. warning::

   The GUI is not supported in v6.0.0. Calling ``run_gui.main()`` will raise
   ``GUINotSupportedError`` with a message directing users to use the CLI instead.

Debug Output Tutorial
=====================

FUSION provides several levels of debug output control:

**1. Verbose Mode** (:code:`--verbose` or :code:`-v`)

Enables informational logging:

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim --config_path config.ini --run_id test --verbose

**2. Debug Mode** (:code:`--debug`)

Enables detailed debug logging:

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim --config_path config.ini --run_id test --debug

**3. Combining Both**

For maximum verbosity:

.. code-block:: bash

   python -m fusion.cli.run_sim run_sim --config_path config.ini --run_id test --verbose --debug

**4. Programmatic Logging Control**

The logging system uses ``fusion.utils.logging_config``:

.. code-block:: python

   from fusion.utils.logging_config import get_logger

   logger = get_logger(__name__)
   logger.debug("Debug message")    # Only shown with --debug
   logger.info("Info message")      # Shown with --verbose
   logger.warning("Warning")        # Always shown
   logger.error("Error")            # Always shown

**5. Progress Output**

Control simulation progress output with:

.. code-block:: bash

   # Print progress every 1000 requests
   python -m fusion.cli.run_sim run_sim --config_path config.ini --run_id test --print_step 1000

   # Save snapshots every 500 requests
   python -m fusion.cli.run_sim run_sim --config_path config.ini --run_id test --save_snapshots --snapshot_step 500

Development Guide
=================

Adding New CLI Arguments
------------------------

**Step 1: Choose the right file**

- ``shared.py`` - Arguments used across multiple commands
- ``network.py`` - Network topology arguments
- ``routing.py`` - Legacy routing arguments
- ``policy.py`` - Orchestrator policy arguments (v6.0.0)
- ``traffic.py`` - Traffic generation arguments
- ``training.py`` - RL/SL training arguments
- ``survivability.py`` - Failure/protection arguments
- ``analysis.py`` - Statistics/plotting arguments
- **New file** - If your arguments form a distinct new category

**Step 2: Add the argument function**

Follow the established pattern in the target file:

.. code-block:: python

   # In parameters/your_module.py
   def add_your_args(parser: argparse.ArgumentParser) -> None:
       """
       Add your argument group to the parser.

       Describe what these arguments configure and when to use them.

       :param parser: ArgumentParser instance to add arguments to
       :type parser: argparse.ArgumentParser
       :return: None
       :rtype: None
       """
       group = parser.add_argument_group("Your Configuration")
       group.add_argument(
           "--your_arg",
           type=str,
           default=None,
           help="Description of what this argument does",
       )
       group.add_argument(
           "--your_flag",
           action="store_true",
           help="Enable some feature",
       )

**Step 3: Export from ``__init__.py``**

Add your function to ``parameters/__init__.py``:

.. code-block:: python

   from .your_module import add_your_args

   __all__ = [
       # ... existing exports
       "add_your_args",
   ]

**Step 4: Register with the registry**

Add to ``registry.py`` in the appropriate method:

.. code-block:: python

   def _register_your_groups(self) -> None:
       """Register your argument groups."""
       self.register_group("your_category", add_your_args)

**Step 5: Add to schema if needed**

If your arguments should be loadable from INI files, add them to
``fusion/configs/schema.py``:

.. code-block:: python

   OPTIONAL_OPTIONS_DICT = {
       # ... existing sections
       "your_settings": {
           "your_arg": str,
           "your_flag": str_to_bool,
       },
   }

**Step 6: Write tests**

Add tests in ``cli/tests/test_your_module.py``:

.. code-block:: python

   class TestYourArgs:
       """Tests for your argument functions."""

       def test_add_your_args_adds_group(self) -> None:
           """Test that add_your_args adds argument group."""
           parser = argparse.ArgumentParser()
           add_your_args(parser)
           # Verify arguments were added
           args = parser.parse_args(["--your_arg", "value"])
           assert args.your_arg == "value"

Creating a New Parameter File
-----------------------------

When your arguments form a distinct category:

1. Create ``parameters/your_category.py`` with the pattern above
2. Add to ``parameters/__init__.py`` exports and ``__all__``
3. Register in ``registry.py``
4. Add configuration schema in ``fusion/configs/schema.py``
5. Update ``parameters/README.md`` to document the new file
6. Create tests in ``cli/tests/test_your_category.py``

Testing
=======

:Test Location: ``fusion/cli/tests/``
:Run Tests: ``pytest fusion/cli/tests/ -v``

**Test Coverage:**

- ``test_config_setup.py`` - Configuration loading and validation
- ``test_constants.py`` - Exit codes and constants
- ``test_main_parser.py`` - Argument parser construction
- ``test_registry.py`` - ArgumentRegistry functionality
- ``test_run_gui.py`` - GUI error handling (GUINotSupportedError)
- ``test_run_sim.py`` - Simulation entry point
- ``test_run_train.py`` - Training entry point
- ``test_utils.py`` - Entry point utilities

Complete CLI Argument Reference
===============================

.. warning::

   CLI arguments override INI configuration values for **ALL processes**.
   See the warning at the top of this document.

Required Arguments
------------------

These arguments must be provided on the command line or have values in the config:

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Argument
     - Type
     - Description
   * - **config_path**
     - str
     - Path to INI configuration file (required)
   * - **run_id**
     - str
     - Unique identifier for this simulation run (required)

Core Configuration (shared.py)
------------------------------

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **verbose**, **-v**
     - flag
     - False
     - Enable verbose output
   * - **debug**
     - flag
     - False
     - Enable debug mode
   * - **output_dir**
     - str
     - None
     - Directory to save output files
   * - **save_results**
     - flag
     - False
     - Save simulation results to file
   * - **plot_format**
     - str
     - None
     - Output format for plots (not currently supported - v6.1.0)

Network Configuration (network.py)
----------------------------------

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **network**
     - str
     - None
     - Network topology name (e.g., 'NSFNet', 'USbackbone60')
   * - **cores_per_link**
     - int
     - 1
     - Number of cores per fiber link
   * - **bw_per_slot**
     - float
     - None
     - Bandwidth per spectral slot in GHz
   * - **const_link_weight**
     - flag
     - False
     - Use constant link weights for routing
   * - **bi_directional**
     - flag
     - False
     - Enable bidirectional links
   * - **multi_fiber**
     - flag
     - False
     - Enable multi-fiber links
   * - **is_only_core_node**
     - flag
     - False
     - Only allow core nodes to send requests
   * - **c_band**
     - int
     - 96
     - Number of spectral slots in C-band
   * - **l_band**
     - int
     - 0
     - Number of spectral slots in L-band
   * - **s_band**
     - int
     - 0
     - Number of spectral slots in S-band
   * - **e_band**
     - int
     - 0
     - Number of spectral slots in E-band
   * - **o_band**
     - int
     - 0
     - Number of spectral slots in O-band

Legacy Routing Configuration (routing.py)
-----------------------------------------

.. note::

   These arguments are for the **legacy routing system**. For new experiments,
   consider using the orchestrator policy arguments below.

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **route_method**
     - str
     - None
     - Routing algorithm (shortest_path, k_shortest_path)
   * - **k_paths**
     - int
     - 3
     - Number of candidate paths for k-shortest path routing
   * - **allocation_method**
     - str
     - None
     - Spectrum allocation method (first_fit, best_fit, last_fit)
   * - **guard_slots**
     - int
     - 1
     - Number of guard slots between allocations
   * - **spectrum_priority**
     - str
     - None
     - Priority order for multi-band allocation (BSC, CSB)
   * - **dynamic_lps**
     - flag
     - False
     - Enable SDN dynamic lightpath switching
   * - **single_core**
     - flag
     - False
     - Force single-core allocation per request

Orchestrator Policy Configuration (policy.py) - v6.0.0
------------------------------------------------------

.. note::

   These arguments are for the **new orchestrator system** (v6.0.0). They use
   dash-style naming (e.g., ``--policy-type``) rather than underscore-style.

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 45 15 40

   * - Argument
     - Default
     - Description
   * - **policy-type**
     - heuristic
     - Policy type: heuristic, sl (supervised learning), rl (reinforcement learning)
   * - **policy-name**
     - first_feasible
     - Heuristic policy name (first_feasible, shortest, shortest_feasible, least_congested, random, random_feasible, load_balanced)
   * - **policy-model-path**
     - None
     - Path to SL/RL model file
   * - **policy-fallback**
     - first_feasible
     - Fallback policy when SL/RL fails
   * - **policy-k-paths**
     - 3
     - Number of candidate paths for policy
   * - **policy-seed**
     - None
     - Random seed for policy
   * - **policy-algorithm**
     - None
     - RL algorithm name (PPO, MaskablePPO, DQN, A2C)
   * - **policy-device**
     - cpu
     - Device for SL/RL inference (cpu, cuda, auto)
   * - **heuristic-alpha**
     - 0.5
     - Alpha for LoadBalancedPolicy (0.0=congestion, 1.0=length)
   * - **heuristic-seed**
     - None
     - Random seed for RandomFeasiblePolicy
   * - **protection-enabled**
     - False
     - Enable 1+1 protection
   * - **disjointness-type**
     - link
     - Path disjointness type (link, node)
   * - **protection-switchover-ms**
     - 50.0
     - Protection switchover latency in milliseconds
   * - **restoration-latency-ms**
     - 100.0
     - Restoration latency in milliseconds

SNR and Modulation Configuration (snr.py)
-----------------------------------------

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **mod_assumption**
     - str
     - None
     - Modulation format selection strategy
   * - **mod_assumption_path**
     - str
     - None
     - Path to modulation format configuration file
   * - **snr_type**
     - str
     - None
     - SNR calculation method (linear, nonlinear, egn)
   * - **input_power**
     - float
     - None
     - Input power in Watts
   * - **egn_model**
     - flag
     - False
     - Enable Enhanced Gaussian Noise (EGN) model

Traffic Configuration (traffic.py)
----------------------------------

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **erlang_start**
     - float
     - None
     - Starting Erlang load
   * - **erlang_stop**
     - float
     - None
     - Ending Erlang load
   * - **erlang_step**
     - float
     - None
     - Erlang load increment
   * - **holding_time**
     - float
     - None
     - Average holding time for requests
   * - **num_requests**
     - int
     - None
     - Total number of requests to generate
   * - **max_iters**
     - int
     - 3
     - Maximum number of simulation iterations
   * - **thread_erlangs**
     - flag
     - False
     - Enable multi-process Erlang processing

Training Configuration (training.py)
------------------------------------

**Reinforcement Learning (RL)**

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **path_algorithm**
     - str
     - None
     - Path selection RL algorithm (dqn, ppo, a2c, q_learning, bandits)
   * - **core_algorithm**
     - str
     - None
     - Core selection RL algorithm
   * - **spectrum_algorithm**
     - str
     - None
     - Spectrum allocation RL algorithm
   * - **path_model**
     - str
     - None
     - Path to pre-trained path selection model
   * - **core_model**
     - str
     - None
     - Path to pre-trained core selection model
   * - **spectrum_model**
     - str
     - None
     - Path to pre-trained spectrum allocation model
   * - **is_training**
     - flag
     - False
     - Enable training mode (vs inference mode)
   * - **learn_rate**
     - float
     - 0.001
     - Learning rate for RL algorithms
   * - **gamma**
     - float
     - 0.99
     - Discount factor for future rewards
   * - **epsilon_start**
     - float
     - 1.0
     - Initial epsilon for epsilon-greedy exploration
   * - **epsilon_end**
     - float
     - 0.01
     - Final epsilon for epsilon-greedy exploration
   * - **epsilon_update**
     - str
     - linear
     - Epsilon decay strategy (linear, exponential, step)
   * - **reward**
     - float
     - 1.0
     - Reward value for successful actions
   * - **penalty**
     - float
     - -1.0
     - Penalty value for unsuccessful actions
   * - **dynamic_reward**
     - flag
     - False
     - Enable dynamic reward calculation

**Supervised Learning (SL)**

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **ml_training**
     - flag
     - False
     - Enable SL training mode
   * - **ml_model**
     - str
     - None
     - SL model type (random_forest, svm, linear_regression, neural_network, decision_tree)
   * - **train_file_path**
     - str
     - None
     - Path to training data file
   * - **test_size**
     - float
     - 0.2
     - Fraction of data for testing (0.0-1.0)
   * - **output_train_data**
     - flag
     - False
     - Save training data to file
   * - **deploy_model**
     - flag
     - False
     - Deploy trained model for inference

**Feature Extraction**

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **feature_extractor**
     - str
     - None
     - Feature extraction method (graphormer, path_gnn)
   * - **gnn_type**
     - str
     - None
     - GNN architecture type (gcn, gat, sage, graphconv)
   * - **layers**
     - int
     - 3
     - Number of layers in neural network
   * - **emb_dim**
     - int
     - 64
     - Embedding dimension for neural networks
   * - **heads**
     - int
     - 8
     - Number of attention heads
   * - **obs_space**
     - str
     - None
     - Observation space representation (graph, vector, matrix, hybrid)

**Optimization**

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **optimize**
     - flag
     - False
     - Enable hyperparameter optimization
   * - **optimize_hyperparameters**
     - flag
     - False
     - Enable automated hyperparameter tuning
   * - **optuna_trials**
     - int
     - 100
     - Number of Optuna optimization trials
   * - **n_trials**
     - int
     - 10
     - Number of trials for grid/random search
   * - **device**
     - str
     - auto
     - Computing device (cpu, cuda, mps, auto)

Survivability Configuration (survivability.py)
----------------------------------------------

**Failure Injection**

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **failure_type**
     - str
     - none
     - Failure type (none, link, node, srlg, geo)
   * - **t_fail_arrival_index**
     - int
     - None
     - Request arrival index when failure occurs (-1 = midpoint)
   * - **t_repair_after_arrivals**
     - int
     - None
     - Arrivals after failure until repair
   * - **failed_link_src**
     - int
     - None
     - Source node of failed link (F1)
   * - **failed_link_dst**
     - int
     - None
     - Destination node of failed link (F1)
   * - **failed_node_id**
     - int
     - None
     - Node ID for node failure (F2)
   * - **srlg_links**
     - str
     - None
     - Link tuples in SRLG (F3), e.g., "[(0,1), (2,3)]"
   * - **geo_center_node**
     - int
     - None
     - Center node for geographic failure (F4)
   * - **geo_hop_radius**
     - int
     - None
     - Hop radius for geographic failure (F4)

**Protection (Survivability Pipeline)**

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **protection_switchover_ms**
     - float
     - None
     - 1+1 protection switchover latency (ms)
   * - **restoration_latency_ms**
     - float
     - None
     - Restoration compute + signaling latency (ms)
   * - **revert_to_primary**
     - flag
     - False
     - Revert to primary path after repair

**Offline RL (Survivability Pipeline)**

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **policy_type**
     - str
     - None
     - Survivability policy (ksp_ff, one_plus_one, bc, iql)
   * - **bc_model_path**
     - str
     - None
     - Path to Behavior Cloning model (.pt)
   * - **iql_model_path**
     - str
     - None
     - Path to IQL model (.pt)
   * - **fallback_policy**
     - str
     - None
     - Fallback when all actions masked (ksp_ff, one_plus_one)
   * - **log_offline_dataset**
     - flag
     - False
     - Enable offline dataset logging
   * - **dataset_output_path**
     - str
     - None
     - Output path for JSONL dataset
   * - **epsilon_mix**
     - float
     - None
     - Probability of selecting second-best path (0.0-1.0)

Statistics and Analysis Configuration (analysis.py)
---------------------------------------------------

**Statistics**

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **save_snapshots**
     - flag
     - False
     - Save simulation state snapshots
   * - **snapshot_step**
     - int
     - None
     - Requests between snapshots
   * - **save_step**
     - int
     - None
     - Requests between saving results
   * - **print_step**
     - int
     - None
     - Requests between progress updates
   * - **save_start_end_slots**
     - flag
     - False
     - Save detailed slot allocation info

**Plotting** (not currently supported - v6.1.0)

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **plot_results**
     - flag
     - False
     - Generate plots from results
   * - **plot_dpi**
     - int
     - None
     - Resolution (DPI) for plots
   * - **show_plots**
     - flag
     - False
     - Display plots interactively

**Export** (only JSON currently supported)

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **file_type**
     - str
     - None
     - Export format (json supported; csv, excel, tsv in v6.1.0)

**Filtering**

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **filter_mods**
     - flag
     - False
     - Filter results by modulation format
   * - **min_erlang**
     - float
     - None
     - Minimum Erlang load for analysis
   * - **max_erlang**
     - float
     - None
     - Maximum Erlang load for analysis

**Comparison**

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **compare_runs**
     - str+
     - None
     - List of run IDs to compare
   * - **baseline_run**
     - str
     - None
     - Run ID for baseline comparison
   * - **metrics**
     - str+
     - None
     - Metrics to compare (blocking_probability, path_length, execution_time, resource_utilization)
   * - **significance_test**
     - str
     - None
     - Statistical test (t_test, wilcoxon, mann_whitney)

Training Entry Point Arguments
------------------------------

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 70 5 8 17

   * - Argument
     - Type
     - Default
     - Description
   * - **agent_type**
     - str
     - required
     - Agent type to train (rl, sl)
