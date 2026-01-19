.. _core-metrics:

=============
Metrics Guide
=============

This document covers the metrics collection, ML training data, and persistence
systems in the core module.

Overview
========

FUSION collects comprehensive statistics during simulation:

.. code-block:: text

   +-------------------+     +-------------------+     +-------------------+
   |   SimStats        | --> | StatsPersistence  | --> |   JSON/CSV File   |
   | (collect metrics) |     | (save to file)    |     |   (output)        |
   +-------------------+     +-------------------+     +-------------------+

   +-------------------+     +-------------------+
   | MLMetricsCollector| --> |   CSV File        |
   | (ML training data)|     |   (training data) |
   +-------------------+     +-------------------+

**Key distinction:**

- ``metrics.py`` / ``SimStats``: Runtime statistics collection and blocking calculations
- ``persistence.py`` / ``StatsPersistence``: Saving and loading statistics to files
- ``ml_metrics.py`` / ``MLMetricsCollector``: SL/USL training data (NOT RL metrics)

.. important::

   The organization of ``metrics.py`` (2,027 lines) will be improved in a future
   version. The file should be split into focused modules. This project is open
   source and welcomes contributions to help with this refactoring.

metrics.py - Statistics Collection
==================================

The ``SimStats`` class is the central statistics collector for all simulation metrics.

SimStats Class
--------------

:Location: ``fusion/core/metrics.py``
:Size: 2,027 lines (needs refactoring)
:Key Class: ``SimStats``

**Initialization:**

.. code-block:: python

   from fusion.core import SimStats

   stats = SimStats(
       engine_props=engine_props,
       sim_info="experiment_001",
   )

   # Initialize for new iteration
   stats.init_iter_stats()

**Key methods:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``init_iter_stats()``
     - Initialize statistics for new iteration
   * - ``iter_update()``
     - Process single request outcome
   * - ``calculate_blocking_statistics()``
     - Compute blocking rates after iteration
   * - ``calculate_confidence_interval()``
     - Check if results are statistically significant
   * - ``finalize_iteration_statistics()``
     - Complete iteration and prepare for next

Metrics Tracked
---------------

**Blocking Metrics:**

.. code-block:: python

   # Request blocking
   stats_props.simulation_blocking_list: list[float]  # Per-iteration blocking rates
   stats_props.block_reasons_dict: dict[str, int]     # Reasons for blocking

   # Bit-rate blocking
   stats_props.simulation_bitrate_blocking_list: list[float]

   # Bandwidth-specific blocking
   stats_props.bandwidth_blocking_dict: dict[str, int]  # Blocked per bandwidth class

**Resource Utilization:**

.. code-block:: python

   # Spectrum usage
   stats_props.cores_dict: dict[int, int]              # Allocations per core
   stats_props.link_usage_dict: dict[str, dict]        # Per-link usage
   stats_props.frag_dict: dict[str, float]             # Fragmentation scores

   # Lightpath utilization
   stats_props.lp_bw_utilization_dict: dict[str, float]
   stats_props.sim_lp_utilization_list: list[float]

**Path Metrics:**

.. code-block:: python

   stats_props.hops_list: list[float]      # Hop counts
   stats_props.lengths_list: list[float]   # Path lengths in km
   stats_props.route_times_list: list[float]  # Routing computation time

**Signal Quality:**

.. code-block:: python

   stats_props.snr_list: list[float]       # SNR values
   stats_props.crosstalk_list: list[float] # Crosstalk values

**Transponders:**

.. code-block:: python

   stats_props.transponders_list: list[int]  # Transponders per request

**Survivability (v6.0+):**

.. code-block:: python

   stats_props.protection_switchovers: int    # Successful switchovers
   stats_props.protection_failures: int       # Both paths failed
   stats_props.failure_induced_blocks: int    # Dropped due to failures
   stats_props.switchover_times: list[float]  # Switchover timestamps

Usage Flow
----------

.. code-block:: python

   # 1. Create stats collector
   stats = SimStats(engine_props, sim_info)

   # 2. For each iteration
   for iteration in range(max_iters):
       stats.init_iter_stats()

       # 3. Process each request
       for request in requests:
           # ... process request ...
           stats.iter_update(
               request_data=request_dict,
               sdn_data=sdn_props,
               network_spectrum_dict=spectrum_dict,
           )

       # 4. Calculate iteration statistics
       stats.calculate_blocking_statistics()
       stats.finalize_iteration_statistics()

       # 5. Check convergence
       if stats.calculate_confidence_interval():
           break  # Statistically significant

   # 6. Save results
   persistence = StatsPersistence(engine_props, sim_info)
   persistence.save_stats(stats_dict, stats.stats_props, blocking_stats)

Confidence Interval Calculation
-------------------------------

FUSION uses confidence intervals to determine when results are statistically
significant:

.. code-block:: python

   def calculate_confidence_interval(self) -> bool:
       """
       Calculate 95% confidence interval for blocking probability.

       Returns True if CI width is below threshold (results are reliable).
       """
       blocking_list = self.stats_props.simulation_blocking_list

       if len(blocking_list) < 3:
           return False  # Need at least 3 samples

       mean = statistics.mean(blocking_list)
       stdev = statistics.stdev(blocking_list)
       n = len(blocking_list)

       # 95% CI using t-distribution
       t_value = 1.96  # For large n
       ci_half_width = t_value * (stdev / math.sqrt(n))

       # Check if CI is narrow enough
       ci_percent = (ci_half_width / mean) * 100 if mean > 0 else float('inf')
       return ci_percent < self.engine_props.get('ci_target', 5.0)

StatsProps Class
----------------

The ``StatsProps`` class holds all collected statistics:

.. code-block:: python

   class StatsProps:
       """Statistics collection container."""

       def __init__(self) -> None:
           # Snapshot tracking
           self.snapshots_dict: dict[int, dict] = {}

           # Resource usage
           self.cores_dict: dict[int, int] = {}
           self.link_usage_dict: dict[str, dict] = {}
           self.frag_dict: dict[str, float] = {}

           # Blocking
           self.block_reasons_dict: dict[str, int] = {
               "distance": 0,
               "congestion": 0,
               "xt_threshold": 0,
               "failure": 0,
           }

           # Per-simulation metrics
           self.simulation_blocking_list: list[float] = []
           self.simulation_bitrate_blocking_list: list[float] = []

           # ... many more fields

persistence.py - Data Storage
=============================

The ``StatsPersistence`` class handles saving and loading simulation statistics.

StatsPersistence Class
----------------------

:Location: ``fusion/core/persistence.py``
:Size: 254 lines
:Key Class: ``StatsPersistence``

**Purpose:**

- Save statistics to JSON files
- Load previously saved statistics
- Format data for backward compatibility

**Usage:**

.. code-block:: python

   from fusion.core import StatsPersistence

   # Create persistence handler
   persistence = StatsPersistence(
       engine_props=engine_props,
       sim_info="experiment_001",
   )

   # Save statistics
   persistence.save_stats(
       stats_dict={"key": "value"},
       stats_props=stats.stats_props,
       blocking_stats={
           "block_mean": 0.05,
           "block_variance": 0.001,
           "block_ci": 0.002,
           "iteration": 10,
       },
   )

   # Load statistics
   loaded = persistence.load_stats("path/to/file.json")

Output File Format
------------------

Statistics are saved in JSON format:

.. code-block:: json

   {
     "sim_end_time": "0115_14_30_45_123456",
     "blocking_mean": 0.05,
     "blocking_variance": 0.001,
     "ci_rate_block": 0.002,
     "ci_percent_block": 4.0,
     "bit_rate_blocking_mean": 0.03,
     "lightpath_utilization": 0.75,
     "link_usage": {
       "(0, 1)": {"total_allocations": 100}
     },
     "iter_stats": {
       "1": {
         "trans_mean": 2.5,
         "trans_min": 1,
         "trans_max": 5,
         "hops_mean": 3.2,
         "lengths_mean": 450.5,
         "sim_block_list": [0.04, 0.05, 0.06],
         "mods_used_dict": {"QPSK": 50, "16-QAM": 30}
       }
     }
   }

Output File Location
--------------------

Files are saved to:

.. code-block:: text

   {PROJECT_ROOT}/{base_file_path}/output/{sim_info}/{thread_num}/{erlang}_erlang.json

Example:

.. code-block:: text

   /project/data/output/experiment_001/s1/300_erlang.json

Difference: metrics.py vs persistence.py
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - metrics.py (SimStats)
     - persistence.py (StatsPersistence)
   * - Purpose
     - Collect and calculate statistics
     - Save and load statistics
   * - Timing
     - During simulation
     - End of simulation
   * - Data
     - Live, mutable
     - Final, serialized
   * - Format
     - Python objects
     - JSON/CSV files
   * - Scope
     - Statistics, CI, blocking rates
     - File I/O operations

ml_metrics.py - ML Training Data
================================

The ``MLMetricsCollector`` collects training data for supervised and unsupervised
learning models.

.. important::

   Despite the "ML" naming, this module is specifically for **supervised learning
   (SL)** and **unsupervised learning (USL)** data collection. Reinforcement
   learning (RL) metrics are handled separately in ``fusion.rl``.

.. note::

   This file is **new and not heavily tested yet**. Contributions welcome!

MLMetricsCollector Class
------------------------

:Location: ``fusion/core/ml_metrics.py``
:Size: 163 lines
:Key Class: ``MLMetricsCollector``

**Purpose:**

- Collect features for ML model training
- Track network state at allocation time
- Generate training datasets

**Usage:**

.. code-block:: python

   from fusion.core import MLMetricsCollector

   # Enable in config
   engine_props['output_train_data'] = True

   # Create collector
   ml_metrics = MLMetricsCollector(
       engine_props=engine_props,
       sim_info="experiment_001",
   )

   # After each allocation
   ml_metrics.update_train_data(
       old_request_info_dict=original_request,
       request_info_dict=processed_request,
       network_spectrum_dict=spectrum_dict,
       current_transponders=num_transponders,
   )

   # Save at end
   ml_metrics.save_train_data(
       iteration=current_iter,
       max_iterations=max_iters,
   )

Training Data Features
----------------------

The collector generates these features:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Feature
     - Description
   * - ``old_bandwidth``
     - Original request bandwidth before slicing
   * - ``path_length``
     - Physical path length in km
   * - ``longest_reach``
     - Maximum reach of available modulation
   * - ``average_congestion``
     - Average spectrum utilization on path
   * - ``num_segments``
     - Number of lightpaths (transponders) used

Output Format
-------------

Training data is saved to CSV:

.. code-block:: text

   {PROJECT_ROOT}/data/output/{sim_info}/{erlang}_train_data.csv

.. code-block:: text

   old_bandwidth,path_length,longest_reach,average_congestion,num_segments
   100,450.5,1200,0.45,1
   200,800.2,1000,0.62,2
   400,350.1,2000,0.38,3

When Data is Saved
------------------

**Simulation statistics** (via ``StatsPersistence``) can be saved at configurable
intervals using the ``save_step`` parameter, not just at the end.

**ML training data** (via ``MLMetricsCollector``) is saved only on the last
iteration by default to capture the complete dataset:

.. code-block:: python

   def save_train_data(self, iteration: int, max_iterations: int, ...):
       if iteration == (max_iterations - 1):
           # ML training data saves on last iteration by default
           save_df = pd.DataFrame(self.train_data_list)
           save_df.to_csv(output_path, index=False)

Best Practices
==============

Collecting Custom Metrics
-------------------------

To add custom metrics:

1. Add field to ``StatsProps`` class:

.. code-block:: python

   # In properties.py
   class StatsProps:
       def __init__(self):
           # ... existing fields ...
           self.my_custom_list: list[float] = []

2. Update in ``SimStats.iter_update()``:

.. code-block:: python

   # In metrics.py
   def iter_update(self, request_data, sdn_data, ...):
       # ... existing updates ...
       self.stats_props.my_custom_list.append(custom_value)

3. Include in ``StatsPersistence._prepare_iteration_stats()``:

.. code-block:: python

   # In persistence.py
   def _prepare_iteration_stats(self, stats_props, iteration):
       # ... existing code ...
       if stats_props.my_custom_list:
           iter_stats["my_custom_mean"] = mean(stats_props.my_custom_list)

Performance Considerations
--------------------------

1. **Memory Usage**
   - Lists grow during simulation
   - Consider periodic aggregation for long simulations
   - Clear lists between iterations when possible

2. **Computation**
   - Confidence interval calculation can be expensive
   - Only calculate when needed (not every request)
   - Use numpy for large array operations

3. **File I/O**
   - JSON files can be large for long simulations
   - Consider compression for large datasets
   - Save incrementally for crash recovery

Future Improvements
-------------------

The following improvements are planned:

1. **Split metrics.py** into focused modules:

   - ``blocking_stats.py`` - Blocking calculations
   - ``resource_stats.py`` - Resource utilization
   - ``survivability_stats.py`` - Recovery metrics
   - ``confidence.py`` - Statistical analysis

2. **Add streaming output** for long simulations

3. **Improve ML features** for better model training

See Also
========

- :doc:`simulation` - How simulation uses metrics
- :doc:`data_structures` - StatsProps documentation
- ``fusion.analysis`` - Post-simulation analysis tools
- ``fusion.rl`` - RL-specific metrics (separate from ml_metrics.py)
