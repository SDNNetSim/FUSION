.. _sim-module:

==========
Sim Module
==========

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Simulation orchestration, batch execution, and workflow pipelines
   :Location: ``fusion/sim/``
   :Key Files: ``batch_runner.py``, ``input_setup.py``, ``network_simulator.py`` (legacy)
   :Depends On: ``fusion.core``, ``fusion.io``, ``fusion.modules.rl``
   :Used By: CLI entry points, experiment scripts

The ``sim`` module is the **orchestration layer** for FUSION simulations. It manages
*how* simulations are run (batch processing, multi-process execution, workflow pipelines)
while delegating the *what* (actual simulation logic) to ``fusion.core``.

.. important::

   **This module is NOT the simulation engine.** It orchestrates simulation runs.
   The actual discrete event simulation happens in ``fusion.core.simulation``.

**When you work here:**

- Adding new batch execution modes
- Modifying multi-process coordination
- Creating new workflow pipelines (training, evaluation)
- Changing how simulation inputs are prepared

Understanding the Module Landscape
==================================

FUSION has several modules with overlapping names that can cause confusion. This section
clarifies what each does and when to use it.

.. warning::

   **Common Confusion Points:**

   - ``fusion/sim/`` vs ``fusion/core/`` - orchestration vs simulation logic
   - ``fusion/sim/ml_pipeline.py`` vs ``fusion/pipelines/`` - training workflows vs RSA pipelines
   - ``fusion/sim/utils/`` vs ``fusion/utils/`` - sim-specific vs general utilities
   - ``fusion/sim/input_setup.py`` vs ``fusion/io/`` - prep vs I/O operations

Module Comparison Table
-----------------------

.. list-table:: Module Responsibilities
   :header-rows: 1
   :widths: 18 25 30 27

   * - Module
     - Purpose
     - Contains
     - Example Use
   * - ``fusion/sim/``
     - **Orchestration**
     - Batch runners, workflow pipelines, input preparation
     - "Run 10 simulations with different Erlang loads"
   * - ``fusion/core/``
     - **Simulation Engine**
     - Event processing, request handling, statistics
     - "Process this request arrival event"
   * - ``fusion/pipelines/``
     - **RSA Pipelines**
     - Routing strategies, protection, disjoint paths
     - "Find k-shortest paths with 1+1 protection"
   * - ``fusion/io/``
     - **Data I/O**
     - Topology loading, data export, file operations
     - "Load NSFNet topology from file"
   * - ``fusion/reporting/``
     - **Presentation**
     - Result formatting, aggregation, console output
     - "Display simulation results with confidence intervals"

sim vs core: The Key Difference
-------------------------------

.. code-block:: text

   +------------------+                    +-------------------+
   |   fusion/sim     |  orchestrates -->  |   fusion/core     |
   | (BatchRunner)    |                    | (SimulationEngine)|
   +------------------+                    +-------------------+
          |                                        |
          | "Run these 5 Erlang loads"             | "Process 10,000 requests"
          | "Use 4 parallel processes"             | "Handle arrivals/departures"
          | "Prepare input files"                  | "Track blocking statistics"

**Analogy:** ``sim`` is the *project manager* coordinating multiple simulations.
``core`` is the *worker* doing the actual simulation work.

ml_pipeline.py vs pipelines Module
----------------------------------

This is a naming collision that causes confusion:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Item
     - Location
     - Purpose
   * - ``ml_pipeline.py``
     - ``fusion/sim/ml_pipeline.py``
     - **Workflow**: ML model training orchestration (placeholder)
   * - ``train_pipeline.py``
     - ``fusion/sim/train_pipeline.py``
     - **Workflow**: RL agent training orchestration
   * - ``pipelines`` module
     - ``fusion/pipelines/``
     - **RSA**: Routing strategies, protection algorithms

The ``pipelines`` module handles **RSA (Routing and Spectrum Assignment)** algorithms.
The ``*_pipeline.py`` files in ``sim/`` handle **workflow orchestration**.

.. note::

   The naming is historical. Future refactoring may rename the workflow files to
   ``ml_workflow.py`` and ``train_workflow.py`` to reduce confusion.

input_setup.py vs io Module
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Item
     - Purpose
     - When Used
   * - ``input_setup.py``
     - **Prepare** simulation inputs (calls io functions)
     - Before simulation starts
   * - ``fusion/io/``
     - **Load/Save** data (topology, results)
     - During I/O operations

``input_setup.py`` is a **consumer** of ``fusion/io``. It calls ``create_network()``,
``create_pt()``, and ``create_bw_info()`` from io to prepare everything needed
before a simulation run.

Utils Duplication
-----------------

There are TWO utils packages with different scopes:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Package
     - Purpose
     - Key Functions
   * - ``fusion/utils/``
     - **General** utilities (logging, config, OS)
     - ``get_logger()``, ``setup_logger()``
   * - ``fusion/sim/utils/``
     - **Simulation-specific** utilities (network, spectrum)
     - ``find_path_length()``, ``find_free_slots()``

The sim-specific utils contain 31+ functions for:

- Network analysis (path length, congestion, fragmentation)
- Spectrum management (free slots, super-channels)
- Data processing (matrix operations, scaling)
- Simulation helpers (erlang values, timestamps)

.. tip::

   When adding new utilities:

   - General-purpose (logging, paths, config) -> ``fusion/utils/``
   - Simulation/network-specific -> ``fusion/sim/utils/``

Multi-Processing Architecture
=============================

.. warning::

   **Terminology Correction:** FUSION uses **multi-processing** (separate processes),
   NOT multi-threading. The codebase may have legacy references to "threads" but
   the actual implementation uses ``multiprocessing.Pool`` and ``multiprocessing.Process``.

.. important::

   **Multi-Processing Limitations (v6.x)**

   Multi-processing is **NOT fully supported** across all configurations:

   - RL training/inference may not work correctly with parallel execution
   - ML pipelines are currently single-process only
   - Some utility functions (e.g., ``modify_multiple_json_values``) are single-process only
   - Protection/failure scenarios have limited parallel support

   **Planned for v6.2+**: Full multi-processing support across all features.

Modern Approach (batch_runner.py)
---------------------------------

The recommended approach uses ``multiprocessing.Pool`` for task-based parallelism:

.. code-block:: text

   BatchRunner.run(parallel=True)
           |
           v
   +------------------+
   | multiprocessing  |
   |     Pool(4)      |
   +------------------+
           |
   +-------+-------+-------+-------+
   |       |       |       |       |
   v       v       v       v       v
   Task 1  Task 2  Task 3  Task 4  Task 5
   E=100   E=150   E=200   E=250   E=300
           |
           v
   Results aggregated

**Characteristics:**

- Each Erlang load becomes a separate task
- Pool manages process lifecycle automatically
- Clean serialization via Pool
- Progress tracking via ``Manager().dict()``

Legacy Approach (network_simulator.py)
--------------------------------------

The legacy approach spawns one process per configuration:

.. code-block:: text

   NetworkSimulator.run()
           |
   +-------+-------+-------+
   |       |       |       |
   v       v       v       v
   Process Process Process Process
   Config1 Config2 Config3 Config4
           |
           v
   Each runs sequential Erlangs internally

**Characteristics:**

- Manual process spawning with ``multiprocessing.Process``
- Sequential Erlang execution within each process
- Complex state passing (deepcopy to avoid pickling issues)
- Manual queue/event management

.. note::

   **Which to use?** Use ``BatchRunner`` for new code. ``NetworkSimulator`` exists
   for backward compatibility with legacy experiment scripts.

Data Flow and Architecture
==========================

High-Level Flow
---------------

.. code-block:: text

   CLI / Experiment Script
            |
            v
   +--------------------+
   | fusion/sim         |
   | (Orchestration)    |
   +--------------------+
            |
            +---> input_setup.create_input()
            |          |
            |          v
            |     fusion/io (load topology, create PT)
            |
            +---> BatchRunner / NetworkSimulator
                       |
                       v
              +-------------------+
              | fusion/core       |
              | SimulationEngine  |
              +-------------------+
                       |
                       +---> SDNOrchestrator (new) / LegacyHandler (old)
                       |          |
                       |          v
                       |     fusion/pipelines (routing, spectrum)
                       |
                       v
              +-------------------+
              | Results           |
              +-------------------+
                       |
                       v
              fusion/reporting (format, aggregate, export)

Step-by-Step Execution
----------------------

1. **Configuration Parsing**

   .. code-block:: python

      # CLI parses config file
      config = load_config("simulation.ini")

2. **Input Preparation** (``input_setup.py``)

   .. code-block:: python

      # Creates bandwidth info, topology, physical topology
      engine_props = create_input(base_fp, engine_props)

      # Internally calls:
      # - fusion.io.generate.create_bw_info()
      # - fusion.io.structure.create_network()
      # - fusion.io.generate.create_pt()

3. **Batch Execution** (``batch_runner.py``)

   .. code-block:: python

      runner = BatchRunner(config)
      results = runner.run(parallel=True)

      # For each Erlang load:
      # - Creates SimulationEngine from fusion.core
      # - Calls engine.run()
      # - Collects results

4. **Simulation** (``fusion.core``)

   .. code-block:: python

      engine = SimulationEngine(engine_props)
      engine.run()

      # Processes 10,000+ request events
      # Uses SDNOrchestrator for routing/spectrum

5. **Results**

   .. code-block:: python

      # Results returned to BatchRunner
      # Can be aggregated, exported, reported

Key Data Structures
-------------------

**engine_props (dict):**

.. code-block:: python

   engine_props = {
       # Network
       "network": "NSFNet",
       "topology": nx.Graph,           # NetworkX graph
       "cores_per_link": 7,

       # Traffic
       "erlang": 300.0,
       "arrival_rate": 0.06,
       "holding_time": 5000.0,

       # Spectrum
       "mod_per_bw": {...},            # Modulation -> bandwidth mapping
       "topology_info": {...},          # Physical topology with cores

       # Execution
       "thread_num": "s1",              # Process identifier
       "progress_dict": {...},          # Shared progress state
   }

**Batch Results (list[dict]):**

.. code-block:: python

   results = [
       {
           "erlang": 100.0,
           "elapsed_time": 45.2,
           "stats": {
               "blocking_probability": 0.0023,
               "total_requests": 10000,
           }
       },
       # ... more Erlang results
   ]

Components
==========

batch_runner.py (Modern)
------------------------

:Purpose: Modern batch simulation orchestrator with parallel execution
:Key Class: ``BatchRunner``
:Key Function: ``run_batch_simulation()``

.. code-block:: python

   from fusion.sim import BatchRunner, run_batch_simulation

   # Object-oriented approach
   runner = BatchRunner(config)
   results = runner.run(parallel=True)

   # Convenience function
   results = run_batch_simulation(config, parallel=True)

**Key Methods:**

- ``prepare_simulation()`` - Creates input data and topology
- ``run_single_erlang()`` - Runs one Erlang load
- ``run_parallel_batch()`` - Parallel execution via Pool
- ``run_sequential_batch()`` - Sequential execution
- ``run()`` - Main entry point

network_simulator.py (Legacy)
-----------------------------

:Purpose: Legacy multi-process control for backward compatibility
:Status: **Deprecated** - use ``batch_runner.py`` for new code
:Key Class: ``NetworkSimulator``

.. code-block:: python

   from fusion.sim.network_simulator import NetworkSimulator

   # Legacy usage
   simulator = NetworkSimulator()
   simulator.run(sims_dict)

run_simulation.py (Compatibility)
---------------------------------

:Purpose: Backward-compatible entry points
:Key Functions: ``run_simulation()``, ``run_simulation_pipeline()``

.. code-block:: python

   from fusion.sim import run_simulation

   # Legacy single-run interface
   result = run_simulation(config)

   # Internally calls BatchRunner with parallel=False

input_setup.py
--------------

:Purpose: Prepare all input data before simulation
:Key Functions: ``create_input()``, ``save_input()``

.. code-block:: python

   from fusion.sim.input_setup import create_input

   # Prepares:
   # - Bandwidth info (modulation assumptions)
   # - Network topology
   # - Physical topology (cores, fiber properties)
   engine_props = create_input(base_fp, engine_props)

**Integration with io module:**

.. code-block:: text

   input_setup.create_input()
           |
           +---> fusion.io.generate.create_bw_info()
           +---> fusion.io.structure.create_network()
           +---> fusion.io.generate.create_pt()

Workflow Pipelines
==================

.. warning::

   **Beta Status:** The ML and evaluation pipelines are in beta. They contain
   placeholder implementations that will be expanded in future versions.

train_pipeline.py
-----------------

:Purpose: Bridge between new config system and RL training workflow
:Status: Functional (bridges to legacy RL)

.. code-block:: python

   from fusion.sim import train_rl_agent

   # Launches RL training via legacy workflow
   train_rl_agent(config)

   # Internally:
   # 1. Extracts config path
   # 2. Creates RL environment
   # 3. Calls fusion.modules.rl.workflow_runner.run()

ml_pipeline.py
--------------

:Purpose: ML model training orchestration
:Status: **Placeholder** - not implemented

.. code-block:: python

   from fusion.sim.ml_pipeline import train_ml_model

   # Currently just logs the config
   train_ml_model(config)

.. note::

   This file will be expanded as supervised/unsupervised learning features
   are developed. Currently it only logs that it was invoked.

evaluate_pipeline.py
--------------------

:Purpose: Evaluation workflow for models and algorithms
:Status: **Beta** - contains placeholder implementations
:Key Class: ``EvaluationPipeline``

.. code-block:: python

   from fusion.sim import EvaluationPipeline

   pipeline = EvaluationPipeline(config)
   results = pipeline.run_full_evaluation(eval_config)

**Placeholder Functions:**

The following functions have placeholder implementations:

- ``_run_rl_episode()`` - Returns random dummy results (needs RL env integration)
- ``_generate_comparison_plots()`` - Stub for visualization
- ``_generate_excel_report()`` - Stub for Excel export

These will be implemented as the evaluation framework matures.

Legacy vs Orchestrator
======================

FUSION has two architectural approaches that coexist:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - Legacy (v5.x)
     - Orchestrator (v6.x+)
   * - **Location**
     - ``fusion/core/simulation.py`` internal
     - ``fusion/core/orchestrator.py``
   * - **Pattern**
     - Monolithic handling
     - Pipeline-based coordination
   * - **RSA Logic**
     - Embedded in engine
     - Delegated to pipelines
   * - **Extensibility**
     - Modify engine directly
     - Add new pipelines
   * - **RL Integration**
     - Legacy adapter
     - Clean policy interface

**SDNOrchestrator (New):**

.. code-block:: python

   # Thin coordination layer
   # Does NOT implement algorithm logic
   # Delegates to pipelines

   orchestrator = SDNOrchestrator(config, pipelines, policy)
   result = orchestrator.handle_arrival(request, network_state)

**Rules for SDNOrchestrator:**

1. No algorithm logic (K-shortest-path, first-fit, etc.)
2. No direct numpy access
3. No hardcoded slicing/grooming logic
4. Receives NetworkState per call, never stores it

Beta Features and TODOs
=======================

.. warning::

   **Features in Beta (v6.x):**

   - ML training pipelines (placeholder implementations)
   - Evaluation pipelines (partial implementation)
   - Protection/failure scenarios (limited testing)
   - Multi-process RL training

.. important::

   **Known Limitations:**

   1. **Multi-processing not fully supported:**

      - RL training/inference may fail in parallel mode
      - Some configs require ``parallel=False``
      - Planned fix: v6.2+

   2. **Hardcoded values in some utilities:**

      - 256 spectral slots assumed in some functions
      - 6 cores per link assumed in spectrum utilities
      - Will be parameterized in future versions

   3. **Placeholder implementations:**

      - ``ml_pipeline.py`` - Just logs, no training
      - ``_run_rl_episode()`` - Returns random values
      - ``_generate_comparison_plots()`` - Stub

Development Guide
=================

Getting Started
---------------

1. Read the `Understanding the Module Landscape`_ section
2. Understand the difference between ``sim`` and ``core``
3. Examine ``batch_runner.py`` for the modern execution pattern
4. Run the tests to see example usage

Common Tasks
------------

**Adding a new execution mode**

1. Add method to ``BatchRunner`` in ``batch_runner.py``
2. Update ``run()`` to route to new mode based on config
3. Add tests in ``tests/test_batch_runner.py``

**Creating a new workflow pipeline**

1. Create new file ``{name}_pipeline.py`` in ``fusion/sim/``
2. Follow pattern from ``evaluate_pipeline.py``
3. Add to ``__init__.py`` exports
4. Add CLI integration if needed

**Modifying input preparation**

1. Edit ``input_setup.py``
2. Update calls to ``fusion/io`` functions as needed
3. Ensure backward compatibility with existing configs

Configuration
-------------

Batch execution options:

.. code-block:: ini

   [simulation]
   erlang_start = 100
   erlang_stop = 500
   erlang_step = 50
   parallel = true
   num_processes = 4

Testing
=======

:Test Location: ``fusion/sim/tests/``
:Run Tests: ``pytest fusion/sim/tests/ -v``

**Test files:**

- ``test_batch_runner.py`` - Modern batch execution
- ``test_network_simulator.py`` - Legacy execution
- ``test_run_simulation.py`` - Compatibility functions
- ``test_train_pipeline.py`` - RL training bridge

**Utils tests:**

- ``tests/test_network.py`` - Path/congestion utilities
- ``tests/test_spectrum.py`` - Spectrum utilities
- ``tests/test_data_utils.py`` - Data processing

.. code-block:: bash

   # Run all sim tests
   pytest fusion/sim/tests/ -v

   # Run with coverage
   pytest --cov=fusion.sim fusion/sim/tests/

Related Documentation
=====================

- :ref:`core-module` - Core simulation engine
- :ref:`io-module` - Data input/output
- :ref:`reporting-module` - Result formatting and export
- ``fusion/pipelines/`` - RSA pipeline implementations
- ``fusion/modules/rl/`` - Reinforcement learning integration
