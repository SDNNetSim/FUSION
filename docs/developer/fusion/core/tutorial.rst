.. _core-tutorial:

========
Tutorial
========

This tutorial guides you through the core module step by step. By the end,
you'll understand how simulations work, how to configure them, and how
to extend the system.

Prerequisites
=============

Before starting, ensure you have:

1. FUSION installed and working (``make test-new`` passes)
2. Basic understanding of optical networking concepts
3. Read the :doc:`architecture` document

Learning Path
=============

.. code-block:: text

   [Start Here]
        |
        v
   1. Run Your First Simulation (CLI)
        |
        v
   2. Understand Configuration Files
        |
        v
   3. Understand the Simulation Flow
        |
        v
   4. Work with Metrics and Results
        |
        v
   5. Add Custom Functionality
        |
        v
   [Ready to Contribute!]

Part 1: Run Your First Simulation
=================================

FUSION simulations are typically run via the CLI with INI configuration files.

Step 1.1: Using an Existing Config Template
-------------------------------------------

The simplest way to run a simulation is using an existing template:

.. code-block:: bash

   # Run with the minimal template
   python -m fusion.cli.run_sim --config_path fusion/configs/templates/minimal.ini

This runs a simulation with:

- NSFNet topology (14 nodes)
- Erlang range 300-500 (step 100)
- 2 iterations per Erlang value
- 100 requests per iteration

Step 1.2: Understanding the Output
----------------------------------

The simulation outputs:

1. **Console logs**: Progress messages showing Erlang values being processed
2. **JSON results**: Saved to ``data/output/{network}/{date}/{time}/{thread}/``

Example output structure:

.. code-block:: text

   data/output/NSFNet/0115/14_30_45_123456/s1/
   |-- 300_erlang.json
   |-- 400_erlang.json
   `-- 500_erlang.json

Each JSON file contains blocking statistics, resource utilization, and
detailed metrics for that traffic load.

Part 2: Understanding Configuration
===================================

Step 2.1: INI File Structure
----------------------------

FUSION uses INI configuration files with multiple sections:

.. code-block:: ini

   [general_settings]
   # Traffic load parameters
   erlang_start = 300
   erlang_stop = 500
   erlang_step = 100

   # Simulation control
   max_iters = 5
   num_requests = 1000
   holding_time = 0.2

   # Algorithms
   route_method = k_shortest_path
   allocation_method = first_fit

   [topology_settings]
   network = NSFNet
   cores_per_link = 7
   bw_per_slot = 12.5

   [routing_settings]
   k_paths = 3

   [spectrum_settings]
   c_band = 320
   guard_slots = 1

**Key sections:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Section
     - Purpose
   * - ``[general_settings]``
     - Traffic load, iterations, algorithms, feature flags
   * - ``[topology_settings]``
     - Network topology, fiber configuration
   * - ``[routing_settings]``
     - Path computation parameters
   * - ``[spectrum_settings]``
     - Spectrum slots, guard bands
   * - ``[snr_settings]``
     - Signal quality parameters
   * - ``[file_settings]``
     - Output format and paths

Step 2.2: Key Configuration Parameters
--------------------------------------

**Traffic Load:**

.. code-block:: ini

   # Erlang = arrival_rate * holding_time
   erlang_start = 300      # Starting traffic load
   erlang_stop = 600       # Ending traffic load
   erlang_step = 100       # Increment between loads
   holding_time = 0.2      # Average connection duration

**Simulation Control:**

.. code-block:: ini

   max_iters = 10          # Iterations per Erlang (for statistical confidence)
   num_requests = 1000     # Requests per iteration
   print_step = 100        # Console output frequency
   save_step = 5           # How often to save intermediate results

**Algorithm Selection:**

.. code-block:: ini

   route_method = k_shortest_path    # or: xt_aware, 1plus1_protection
   allocation_method = first_fit     # or: best_fit, last_fit
   k_paths = 3                       # Candidate paths for k-shortest

**Feature Flags:**

.. code-block:: ini

   is_grooming_enabled = False   # Traffic grooming
   max_segments = 1              # Slicing (>1 enables multi-segment)
   dynamic_lps = False           # Dynamic lightpath sizing

Step 2.3: Create a Custom Configuration
---------------------------------------

Create ``my_experiment.ini``:

.. code-block:: ini

   [general_settings]
   # High traffic load experiment
   erlang_start = 400
   erlang_stop = 800
   erlang_step = 50

   # More iterations for statistical significance
   max_iters = 10
   num_requests = 5000
   holding_time = 0.2

   # Enable grooming
   is_grooming_enabled = True

   # Routing
   route_method = k_shortest_path
   allocation_method = first_fit

   # Bandwidth distribution (JSON format)
   request_distribution = {"100": 0.5, "200": 0.3, "400": 0.2}

   # Output
   print_step = 500
   log_level = INFO

   [topology_settings]
   network = USbackbone60
   cores_per_link = 7

   [routing_settings]
   k_paths = 5

   [spectrum_settings]
   c_band = 320
   guard_slots = 1

   [file_settings]
   file_type = json

Run it:

.. code-block:: bash

   python -m fusion.cli.run_sim --config_path my_experiment.ini

Part 3: Understanding the Simulation Flow
=========================================

Step 3.1: High-Level Flow
-------------------------

When you run a simulation, this is what happens:

.. code-block:: text

   1. CLI parses config file
          |
          v
   2. BatchRunner created with config
          |
          v
   3. For each Erlang value (300, 400, 500, ...):
          |
          +-- SimulationEngine created
          |
          +-- Topology loaded (NSFNet, USbackbone60, etc.)
          |
          +-- For each iteration (0, 1, 2, ...):
          |       |
          |       +-- Requests generated (Poisson process)
          |       |
          |       +-- For each request (arrival/departure):
          |       |       |
          |       |       +-- SDNController.allocate() [legacy]
          |       |       |   OR
          |       |       +-- SDNOrchestrator.handle_arrival() [v6.0+]
          |       |       |
          |       |       +-- SimStats.iter_update()
          |       |
          |       +-- Calculate blocking statistics
          |       |
          |       +-- Check confidence interval
          |
          +-- Save results to JSON

Step 3.2: Request Processing (Legacy Path)
------------------------------------------

The legacy path uses ``SDNController`` for request processing:

.. code-block:: text

   Request arrives
        |
        v
   SDNController.allocate()
        |
        +-- Routing: Find k-shortest paths
        |
        +-- For each candidate path:
        |       |
        |       +-- Spectrum Assignment: Find free slots
        |       |
        |       +-- SNR Validation (if enabled)
        |       |
        |       +-- If successful: allocate and return
        |
        +-- If all paths fail: mark as blocked

Step 3.3: Request Processing (Orchestrator Path)
------------------------------------------------

The orchestrator path (``use_orchestrator=True``) uses a pipeline architecture:

.. code-block:: text

   Request arrives
        |
        v
   SDNOrchestrator.handle_arrival()
        |
        +-- Stage 1: Grooming (if enabled)
        |       Can this request use existing lightpath capacity?
        |
        +-- Stage 2: Routing
        |       Find candidate paths via adapter
        |
        +-- Stage 3: Spectrum Assignment
        |       Find free spectrum via adapter
        |
        +-- Stage 4: SNR Validation (if enabled)
        |       Check signal quality via adapter
        |
        +-- Stage 5: Slicing (if enabled)
        |       Split request across multiple lightpaths
        |
        +-- Stage 6: Protection (if enabled)
        |       Establish backup path for 1+1 protection
        |
        +-- Return AllocationResult

Step 3.4: Enabling the Orchestrator
-----------------------------------

To use the orchestrator path instead of legacy:

**Option 1: Environment variable**

.. code-block:: bash

   export FUSION_USE_ORCHESTRATOR=true
   python -m fusion.cli.run_sim --config_path my_config.ini

**Option 2: Config parameter**

.. code-block:: ini

   [general_settings]
   use_orchestrator = True

Part 4: Working with Metrics and Results
========================================

Step 4.1: Understanding Output Files
------------------------------------

Each Erlang simulation produces a JSON file with structure:

.. code-block:: json

   {
     "sim_end_time": "0115_14_30_45_123456",
     "blocking_mean": 0.0523,
     "blocking_variance": 0.0012,
     "ci_rate_block": 0.0034,
     "ci_percent_block": 6.5,
     "bit_rate_blocking_mean": 0.0412,
     "iter_stats": {
       "1": {
         "trans_mean": 2.3,
         "hops_mean": 3.1,
         "lengths_mean": 425.6,
         "sim_block_list": [0.048, 0.052, 0.057],
         "mods_used_dict": {"QPSK": 450, "16-QAM": 320}
       }
     },
     "link_usage": {
       "(0, 1)": {"total_allocations": 523}
     }
   }

**Key metrics:**

- ``blocking_mean``: Average blocking probability across iterations
- ``blocking_variance``: Variance of blocking probability
- ``ci_rate_block``: Confidence interval half-width
- ``ci_percent_block``: CI as percentage of mean
- ``bit_rate_blocking_mean``: Bandwidth-weighted blocking

Step 4.2: Analyzing Results Programmatically
--------------------------------------------

Load and analyze results:

.. code-block:: python

   import json
   from pathlib import Path

   # Find result files
   results_dir = Path("data/output/NSFNet/0115/14_30_45_123456/s1")

   # Load and analyze
   for result_file in sorted(results_dir.glob("*_erlang.json")):
       with open(result_file) as f:
           data = json.load(f)

       erlang = result_file.stem.replace("_erlang", "")
       blocking = data.get("blocking_mean", 0)
       ci = data.get("ci_percent_block", 0)

       print(f"Erlang {erlang}: Blocking = {blocking:.4f} (CI: {ci:.1f}%)")

Step 4.3: Confidence Interval Stopping
--------------------------------------

FUSION can stop early when results are statistically significant:

.. code-block:: ini

   [general_settings]
   blocking_type_ci = True   # Enable CI-based stopping
   max_iters = 20            # Maximum iterations (will stop early if CI met)

The simulation stops when the 95% confidence interval is narrow enough
(typically <5% of the mean).

Part 5: Adding Custom Functionality
===================================

Step 5.1: Adding a Custom Metric
--------------------------------

To track a new metric, modify three files:

**1. Add field to StatsProps** (``fusion/core/properties.py``):

.. code-block:: python

   class StatsProps:
       def __init__(self):
           # ... existing fields ...
           self.my_custom_list: list[float] = []

**2. Update collection in SimStats** (``fusion/core/metrics.py``):

.. code-block:: python

   def iter_update(self, request_data, sdn_data, ...):
       # ... existing code ...

       # Add your custom metric
       custom_value = self._calculate_custom_metric(request_data)
       self.stats_props.my_custom_list.append(custom_value)

**3. Include in output** (``fusion/core/persistence.py``):

.. code-block:: python

   def _prepare_iteration_stats(self, stats_props, iteration):
       # ... existing code ...

       if stats_props.my_custom_list:
           iter_stats["my_custom_mean"] = statistics.mean(
               stats_props.my_custom_list
           )

Step 5.2: Registering a New Routing Algorithm
---------------------------------------------

FUSION uses registries for pluggable algorithms. To add a new routing method:

**1. Create algorithm class** (``fusion/modules/routing/my_routing.py``):

.. code-block:: python

   from fusion.interfaces.routing import AbstractRoutingAlgorithm

   class MyCustomRouting(AbstractRoutingAlgorithm):
       """My custom routing algorithm."""

       def get_route(
           self,
           source: str,
           destination: str,
           topology: nx.Graph,
           **kwargs,
       ) -> list[list[str]]:
           # Your routing logic here
           paths = []
           # ... compute paths ...
           return paths

**2. Register in registry** (``fusion/modules/routing/registry.py``):

.. code-block:: python

   from fusion.modules.routing.my_routing import MyCustomRouting

   ROUTING_REGISTRY = {
       "k_shortest_path": KShortestPathRouting,
       "xt_aware": XTAwareRouting,
       "my_custom": MyCustomRouting,  # Add your algorithm
   }

**3. Use in config**:

.. code-block:: ini

   [general_settings]
   route_method = my_custom

Step 5.3: Understanding the Adapter Pattern
-------------------------------------------

The orchestrator uses adapters to wrap legacy code. If you want to understand
how legacy and orchestrator paths connect, see:

- ``fusion/core/adapters/routing_adapter.py`` - Wraps legacy routing
- ``fusion/core/adapters/spectrum_adapter.py`` - Wraps legacy spectrum assignment

Adapters convert between:

- Legacy: Mutable ``*Props`` classes (RoutingProps, SpectrumProps)
- Orchestrator: Immutable ``*Result`` dataclasses (RouteResult, SpectrumResult)

See :doc:`adapters` for detailed documentation.

Next Steps
==========

After completing this tutorial:

1. Explore ``fusion/modules/`` for algorithm implementations
2. Review test files in ``fusion/core/tests/`` for usage patterns
3. Check ``fusion/configs/templates/`` for more configuration examples

See Also
========

- :doc:`architecture` - Deep dive into both architectures
- :doc:`orchestrator` - Pipeline flow documentation
- :doc:`metrics` - Metrics collection details
- :ref:`configs-module` - Configuration system documentation
