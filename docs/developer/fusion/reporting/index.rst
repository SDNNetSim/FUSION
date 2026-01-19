.. _reporting-module:

================
Reporting Module
================

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Format, aggregate, and export simulation results for human consumption and ML training
   :Location: ``fusion/reporting/``
   :Key Files: ``simulation_reporter.py``, ``aggregation.py``, ``csv_export.py``, ``statistics.py``, ``dataset_logger.py``
   :Depends On: ``fusion.utils.logging_config``, ``numpy``
   :Used By: ``fusion.core.simulation``, offline RL training pipelines

The ``reporting`` module is the **presentation layer** for FUSION simulation results. It handles
how results are displayed to users, exported for analysis, and logged for ML training. This module
separates presentation concerns from data collection (``fusion.core.metrics``) and network analysis
(``fusion.analysis``).

**When you work here:**

- Adding new console output formats for simulation progress
- Creating new export formats (e.g., Parquet, HDF5)
- Adding new statistical aggregation methods for multi-seed experiments
- Extending offline RL dataset logging with new fields
- Adding grooming-specific reporting features

Module Differentiation
======================

Understanding how ``reporting`` differs from related modules is crucial:

.. list-table:: Module Responsibilities
   :header-rows: 1
   :widths: 15 25 30 30

   * - Module
     - Primary Purpose
     - What It Does
     - When To Use
   * - ``fusion.io``
     - Data Input/Output
     - Loads network topologies, generates physical layer parameters, exports raw simulation data
     - Loading networks, saving raw JSON/CSV data
   * - ``fusion.analysis``
     - Network Analysis
     - Analyzes network state (utilization, congestion, bottlenecks) during/after simulation
     - Computing network metrics, identifying problems
   * - ``fusion.reporting``
     - Presentation Layer
     - Formats results for humans, aggregates multi-seed statistics, logs ML training data
     - Console output, statistical summaries, RL datasets
   * - ``fusion.core.metrics``
     - Statistics Collection
     - Collects raw statistics during simulation (blocking, hops, SNR, etc.)
     - Tracking every metric during simulation

**Data Flow Between Modules:**

.. code-block:: text

   Simulation Engine
         |
         v
   +---------------------+       +--------------------+       +--------------------+
   | fusion.core.metrics | ----> | fusion.analysis    | ----> | fusion.reporting   |
   | (collect raw stats) |       | (analyze network)  |       | (format & export)  |
   +---------------------+       +--------------------+       +--------------------+
         |                                                           |
         v                                                           v
   +---------------------+                                  +--------------------+
   | fusion.io.exporter  |                                  | Console Output     |
   | (raw data files)    |                                  | CSV Summaries      |
   +---------------------+                                  | RL Datasets        |
                                                            +--------------------+

**Key Distinction from io module:**

- ``fusion.io``: Low-level file operations (read/write JSON, CSV) with no semantic understanding
- ``fusion.reporting``: High-level presentation with semantic understanding (aggregation, formatting, confidence intervals)

**Key Distinction from analysis module:**

- ``fusion.analysis``: Computes derived metrics FROM network state (utilization, congestion)
- ``fusion.reporting``: Presents statistics TO users (formatting, export, display)

statistics.py vs metrics.py
===========================

This is a common source of confusion. Here is how they differ:

.. list-table:: Statistics vs Metrics
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - ``fusion.core.metrics.py`` (SimStats)
     - ``fusion.reporting.statistics.py`` (GroomingStatistics)
   * - **Purpose**
     - Core statistics collection engine
     - Grooming-specific statistics only
   * - **Scope**
     - ALL simulation metrics (blocking, hops, SNR, transponders, resource usage)
     - ONLY traffic grooming metrics (grooming rates, lightpath utilization, bandwidth savings)
   * - **When Used**
     - Every simulation, always enabled
     - Only when grooming is enabled (``is_grooming_enabled=True``)
   * - **Size**
     - 2,000+ lines (comprehensive)
     - 275 lines (focused)
   * - **Key Class**
     - ``SimStats``
     - ``GroomingStatistics``, ``SimulationStatistics``
   * - **Location**
     - ``fusion/core/metrics.py``
     - ``fusion/reporting/statistics.py``

**Why the separation?**

1. ``SimStats`` is a large, monolithic class that handles all core simulation statistics
2. ``GroomingStatistics`` is a specialized class for traffic grooming experiments
3. Keeping grooming stats in ``reporting`` maintains separation of concerns
4. Future plan: Split ``SimStats`` into focused modules (see :ref:`core-metrics`)

**Usage Example:**

.. code-block:: python

   # Core metrics - always used
   from fusion.core.metrics import SimStats

   stats = SimStats(engine_props, sim_info)
   stats.iter_update(request_data, sdn_data, network_spectrum)

   # Grooming statistics - only when grooming enabled
   from fusion.reporting import GroomingStatistics

   if engine_props.get("is_grooming_enabled"):
       grooming_stats = GroomingStatistics()
       grooming_stats.update_grooming_outcome(was_groomed, partial, bandwidth, new_lps)

Key Concepts
============

Multi-Seed Aggregation
   Running the same experiment with different random seeds and combining results
   with statistical analysis (mean, standard deviation, 95% confidence intervals).

Confidence Interval (CI95)
   The range within which the true value lies with 95% probability.
   Calculated as ``CI95 = 1.96 * std / sqrt(n)`` where n is the sample count.

Comparison Table
   Side-by-side comparison of baseline vs RL policy results, showing improvement
   percentages with statistical significance.

Offline RL Dataset
   JSONL files containing (state, action, reward, next_state, action_mask) tuples
   for training offline RL algorithms (BC, IQL, CQL).

Grooming Statistics
   Metrics specific to traffic grooming: grooming rate, lightpath utilization,
   bandwidth savings, transponder blocking.

Architecture
============

Module Structure
----------------

.. code-block:: text

   fusion/reporting/
   |-- __init__.py              # Public API exports
   |-- simulation_reporter.py   # Console output formatting
   |-- aggregation.py           # Multi-seed statistical aggregation
   |-- csv_export.py            # CSV export utilities
   |-- statistics.py            # Grooming-specific statistics
   |-- dataset_logger.py        # Offline RL dataset logging (JSONL)
   |-- README.md                # Module documentation
   `-- tests/                   # Unit tests
       |-- test_simulation_reporter.py
       |-- test_aggregation.py
       |-- test_csv_export.py
       |-- test_statistics.py
       `-- test_dataset_logger.py

Data Flow
---------

**Console Reporting Flow:**

.. code-block:: text

   SimStats (core) --> SimulationReporter --> Console/Log
                       |
                       +-> report_iteration_stats()
                       +-> report_simulation_complete()
                       +-> report_blocking_statistics()

**Multi-Seed Aggregation Flow:**

.. code-block:: text

   Seed 1 Results --+
   Seed 2 Results --+--> aggregate_seed_results() --> Mean, Std, CI95
   Seed 3 Results --+                                      |
                                                           v
                                              export_aggregated_results() --> CSV

**Offline RL Dataset Flow:**

.. code-block:: text

   Simulation --> DatasetLogger.log_transition() --> JSONL File
                                                          |
                                                          v
                                                    load_dataset() --> RL Training

Components
==========

simulation_reporter.py
----------------------

:Purpose: Format and display simulation progress and results
:Key Classes: ``SimulationReporter``

Handles all console output and logging for simulation progress. Integrates with
the Python logging system for proper message delivery.

.. code-block:: python

   from fusion.reporting import SimulationReporter

   reporter = SimulationReporter(verbose=True)

   # Report iteration progress
   reporter.report_iteration_stats(
       iteration=5,
       max_iterations=100,
       erlang=50.0,
       blocking_list=[0.01, 0.02, 0.015],
       print_flag=True
   )

   # Report final results
   reporter.report_simulation_complete(
       erlang=50.0,
       iterations_completed=100,
       confidence_interval=95.0
   )

**Key Methods:**

- ``report_iteration_stats()`` - Progress for each iteration
- ``report_simulation_start()`` - Log startup information
- ``report_simulation_complete()`` - Final results with CI
- ``report_blocking_statistics()`` - Detailed blocking breakdown
- ``create_summary_report()`` - Generate formatted summary string

aggregation.py
--------------

:Purpose: Aggregate results across multiple random seeds with statistical analysis
:Key Functions: ``aggregate_seed_results()``, ``create_comparison_table()``, ``format_comparison_for_display()``

Computes mean, standard deviation, and 95% confidence intervals across multiple
seed runs. Essential for statistically valid comparisons.

.. code-block:: python

   from fusion.reporting import (
       aggregate_seed_results,
       create_comparison_table,
       format_comparison_for_display
   )

   # Results from multiple seeds
   results = [
       {"bp_overall": 0.10, "hops_mean": 3.2, "seed": 42},
       {"bp_overall": 0.11, "hops_mean": 3.1, "seed": 43},
       {"bp_overall": 0.09, "hops_mean": 3.3, "seed": 44},
   ]

   # Aggregate with CI95
   aggregated = aggregate_seed_results(results, metric_keys=["bp_overall", "hops_mean"])
   # Returns: {
   #     "bp_overall": {"mean": 0.10, "std": 0.01, "ci95_lower": 0.089, "ci95_upper": 0.111, "n": 3},
   #     "hops_mean": {"mean": 3.2, "std": 0.1, ...}
   # }

   # Compare baseline vs RL
   comparison = create_comparison_table(baseline_results, rl_results, metrics=["bp_overall"])
   print(format_comparison_for_display(comparison))
   # Output:
   # Metric         | Baseline           | RL                 | Improvement
   # -------------------------------------------------------------------------------
   # bp_overall     | 0.1050 +/- 0.0071  | 0.0850 +/- 0.0071  | +19.05%

csv_export.py
-------------

:Purpose: Export simulation results to CSV format for analysis tools
:Key Functions: ``export_results_to_csv()``, ``export_aggregated_results()``, ``export_comparison_table()``, ``append_result_to_csv()``

Provides CSV export utilities with smart column ordering and support for
incremental logging during long experiments.

.. code-block:: python

   from fusion.reporting import (
       export_results_to_csv,
       export_aggregated_results,
       append_result_to_csv
   )

   # Export multiple results
   export_results_to_csv(results, "output/all_results.csv")

   # Export aggregated statistics
   export_aggregated_results(
       aggregated,
       "output/summary.csv",
       metadata={"topology": "NSFNet", "policy": "baseline"}
   )

   # Append single result (for incremental logging)
   append_result_to_csv(result, "output/running_results.csv")

statistics.py
-------------

:Purpose: Grooming-specific statistics collection and reporting
:Key Classes: ``GroomingStatistics``, ``SimulationStatistics``
:Key Functions: ``generate_grooming_report()``, ``export_grooming_stats_csv()``

Tracks metrics specific to traffic grooming experiments.

.. code-block:: python

   from fusion.reporting import (
       GroomingStatistics,
       generate_grooming_report,
       export_grooming_stats_csv
   )

   # Create grooming statistics tracker
   grooming_stats = GroomingStatistics()

   # Update on each request
   grooming_stats.update_grooming_outcome(
       was_groomed=True,
       was_partially_groomed=False,
       bandwidth=100.0,
       new_lightpaths=0
   )

   # Update on lightpath release
   grooming_stats.update_lightpath_release(
       _lightpath_id=1,
       utilization=0.75,
       _lifetime=120.0
   )

   # Generate report
   report = generate_grooming_report(grooming_stats)
   print(report)

   # Export to CSV
   export_grooming_stats_csv(grooming_stats, "output/grooming_stats.csv")

**Metrics Tracked:**

- Grooming outcomes (fully groomed, partially groomed, not groomed)
- Lightpath lifecycle (created, released, active, utilization)
- Bandwidth efficiency (groomed vs new lightpath bandwidth)
- Transponder usage (blocking counts, per-node usage)

dataset_logger.py
-----------------

:Purpose: Log simulation transitions for offline RL training
:Key Classes: ``DatasetLogger``
:Key Functions: ``load_dataset()``, ``filter_by_window()``

Logs (state, action, reward, next_state, action_mask) tuples in JSONL format
for training offline RL algorithms (Behavior Cloning, IQL, CQL).

.. code-block:: python

   from fusion.reporting import DatasetLogger

   # Use as context manager
   with DatasetLogger("datasets/training.jsonl", engine_props) as logger:
       for request in requests:
           # ... process request ...
           logger.log_transition(
               state=state_dict,
               action=action_idx,
               reward=reward,
               next_state=next_state_dict,
               action_mask=mask_list,
               meta={"request_id": request.id}
           )

   # Load for training
   from fusion.reporting.dataset_logger import load_dataset

   for transition in load_dataset("datasets/training.jsonl"):
       state = transition["state"]
       action = transition["action"]
       # ... use for training ...

**JSONL Format:**

.. code-block:: json

   {"t": 456, "seed": 42, "state": {"src": 0, "dst": 13}, "action": 0, "reward": 1.0, "next_state": null, "action_mask": [true, false], "meta": {"request_id": 123}}

Dependencies
============

This Module Depends On
----------------------

- ``fusion.utils.logging_config`` - Consistent logging across the module
- External: ``numpy`` - Statistical calculations (mean, std, CI95)
- External: ``json``, ``csv``, ``pathlib`` - File operations
- External: ``statistics`` - Python standard library statistics

Modules That Depend On This
---------------------------

- ``fusion.core.simulation`` - Uses ``SimulationReporter`` for console output
- ``fusion.modules.rl`` - Uses ``DatasetLogger`` for offline RL data collection
- Analysis scripts - Use aggregation and export functions

Development Guide
=================

Getting Started
---------------

1. Read the `Key Concepts`_ section above
2. Understand the `Module Differentiation`_ to know where this module fits
3. Examine ``simulation_reporter.py`` for console output patterns
4. Run tests to see example inputs and expected outputs

Common Tasks
------------

**Adding a new export format**

1. Create a new export function in ``csv_export.py`` (or create a new file for complex formats)
2. Follow the existing pattern: accept results dict, output path
3. Create parent directories with ``Path.mkdir(parents=True, exist_ok=True)``
4. Add tests in ``tests/test_csv_export.py``

**Adding new console output**

1. Add a new method to ``SimulationReporter`` in ``simulation_reporter.py``
2. Use the logger (``self.logger``) for output, not print statements
3. Follow existing formatting patterns for consistency
4. Add tests in ``tests/test_simulation_reporter.py``

**Adding new grooming metrics**

1. Add new fields to ``GroomingStatistics.__init__()`` in ``statistics.py``
2. Add update logic in the appropriate method (``update_grooming_outcome`` or new method)
3. Include in ``to_dict()`` for serialization
4. Update ``generate_grooming_report()`` to display new metrics
5. Add tests in ``tests/test_statistics.py``

**Extending offline RL dataset**

1. Add new fields to the transition dict in ``DatasetLogger.log_transition()``
2. Update ``load_dataset()`` if special handling needed
3. Document new fields in the JSONL format section
4. Add tests in ``tests/test_dataset_logger.py``

Code Patterns
-------------

**Statistical Aggregation Pattern**

.. code-block:: python

   def aggregate_something(results: list[dict]) -> dict:
       """Aggregate results with statistical analysis."""
       import numpy as np

       values = [r["metric"] for r in results]
       n = len(values)

       return {
           "mean": np.mean(values),
           "std": np.std(values, ddof=1),  # Sample std
           "ci95_lower": np.mean(values) - 1.96 * np.std(values, ddof=1) / np.sqrt(n),
           "ci95_upper": np.mean(values) + 1.96 * np.std(values, ddof=1) / np.sqrt(n),
           "n": n
       }

**JSONL Logging Pattern**

.. code-block:: python

   import json

   def log_entry(entry: dict, filepath: str) -> None:
       """Append entry to JSONL file."""
       with open(filepath, "a", encoding="utf-8") as f:
           f.write(json.dumps(entry) + "\n")
           f.flush()  # Ensure written even on crash

Configuration
-------------

Enable offline RL dataset logging in configuration:

.. code-block:: ini

   [dataset_logging]
   log_offline_dataset = true
   dataset_output_path = datasets/offline_data.jsonl

Testing
=======

:Test Location: ``fusion/reporting/tests/``
:Run Tests: ``pytest fusion/reporting/tests/ -v``
:Coverage Target: 80%+

**Test files:**

- ``test_simulation_reporter.py`` - Console output formatting
- ``test_aggregation.py`` - Statistical aggregation and comparison tables
- ``test_csv_export.py`` - CSV export functionality
- ``test_statistics.py`` - Grooming statistics tracking
- ``test_dataset_logger.py`` - JSONL logging and loading

**Running tests:**

.. code-block:: bash

   # Run all reporting tests
   pytest fusion/reporting/tests/ -v

   # Run with coverage
   pytest --cov=fusion.reporting fusion/reporting/tests/

   # Run specific test file
   pytest fusion/reporting/tests/test_aggregation.py -v

Related Documentation
=====================

- :ref:`core-metrics` - Core statistics collection (SimStats)
- :ref:`analysis-module` - Network analysis utilities
- :ref:`io-module` - Data input/output operations
- ``fusion.modules.rl`` - Reinforcement learning integration
